#!/usr/bin/env python3
"""
Batch SD3 Medium + Canny + Depth ControlNet style transfer, with an
optional ground-truth-mask-driven inpainting mode.

Example (canny + depth style transfer):
    python batch_canny_depth_control_dir.py \
        --prompt "photorealistic airport photograph" \
        --scale 0.25 \
        -content_dir /data/input \
        --format .jpeg \
        --output_dir /data/output

Example (GT-mask inpainting mode):
    python batch_canny_depth_control_dir.py \
        --prompt "photorealistic airport photograph" \
        --scale 0.75 \
        -content_dir /data/input \
        --format .jpeg \
        --output_dir /data/output \
        --use_gt_mask

When --use_gt_mask is set, each <name>.txt sitting next to <name>.jpeg
is expected to contain a line whose last 8 whitespace-separated values
are normalized (x, y) polygon corner coordinates (top-left, bottom-left,
top-right, bottom-right) in [0, 1]. Those points are rasterized into a
binary mask, and only the masked region is regenerated via
StableDiffusion3ControlNetInpaintingPipeline while the rest of the image
is preserved.

This inpainting mode is multi-conditioning: the pipeline's controlnet is
an SD3MultiControlNetModel combining
    - alimama-creative/SD3-Controlnet-Inpainting, fed the source image,
    - InstantX/SD3-Controlnet-Canny, fed a canny edge map of the source
      image, and
    - InstantX/SD3-Controlnet-Depth, fed a depth map of the source image.
For each of these three conditioning images, the pipeline independently
blacks out the masked region and appends the mask as an extra channel
before encoding, so every branch "sees" the same mask both as a pixel-
level blackout and (for the branch trained with it) as an explicit
channel. Since InstantX's canny/depth checkpoints were never trained
with that extra channel, this script pads their input convs with a
zero-initialized channel so they can accept the same mask-augmented
tensor without changing their behavior on their native 16 channels.

All models are loaded directly from the Hugging Face Hub:
    stabilityai/stable-diffusion-3-medium-diffusers
    InstantX/SD3-Controlnet-Canny
    InstantX/SD3-Controlnet-Depth
    Intel/dpt-hybrid-midas
    alimama-creative/SD3-Controlnet-Inpainting   (only when --use_gt_mask)

Note: each input image is generated at its own native resolution
(rounded to the nearest multiple of 16, which SD3's VAE + patchified
transformer requires) rather than being forced to a fixed size.
"""

import argparse
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.pipelines import StableDiffusion3ControlNetInpaintingPipeline
from transformers import DPTForDepthEstimation, DPTImageProcessor


DEFAULT_SD3_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
DEFAULT_CANNY_MODEL = "InstantX/SD3-Controlnet-Canny"
DEFAULT_DEPTH_MODEL = "InstantX/SD3-Controlnet-Depth"
DEFAULT_DEPTH_ESTIMATOR_MODEL = "Intel/dpt-hybrid-midas"
DEFAULT_INPAINT_MODEL = "alimama-creative/SD3-Controlnet-Inpainting"

DEFAULT_STEPS = 28
DEFAULT_CANNY_SCALE = 1.0
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_INPAINT_SCALE = 0.95
DEFAULT_SEED = 1234
DEFAULT_NEGATIVE_PROMPT = "hallucinated details, artificial edges, extra geometry, random artifacts"

# SD3's VAE downsamples by 8x and its transformer uses a patch size of 2,
# so both generation dimensions must be divisible by 16.
SD3_DIM_MULTIPLE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch SD3 Medium style transfer with Canny + Depth ControlNet, or GT-mask inpainting."
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt controlling the transferred style/content appearance.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help=(
            "Negative prompt passed to the pipeline (e.g. quality/NSFW terms "
            f"to steer away from). Default: {DEFAULT_NEGATIVE_PROMPT!r}."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Prompt-transfer strength in [0, 1]. 0 is conservative/no prompt transfer; 1 is maximum.",
    )
    parser.add_argument(
        "-content_dir",
        "--content_dir",
        required=True,
        type=Path,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--format",
        required=True,
        help="Image extension to process, e.g. .jpeg, jpeg, .png.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=Path,
        help="Directory receiving the generated images.",
    )

    parser.add_argument(
        "--sd3_model",
        type=str,
        default=DEFAULT_SD3_MODEL,
        help=f"SD3 Medium repo ID or local path (default: {DEFAULT_SD3_MODEL}).",
    )
    parser.add_argument(
        "--canny_model",
        type=str,
        default=DEFAULT_CANNY_MODEL,
        help=f"Canny ControlNet repo ID or local path (default: {DEFAULT_CANNY_MODEL}).",
    )
    parser.add_argument(
        "--depth_model",
        type=str,
        default=DEFAULT_DEPTH_MODEL,
        help=f"Depth ControlNet repo ID or local path (default: {DEFAULT_DEPTH_MODEL}).",
    )
    parser.add_argument(
        "--depth_estimator_model",
        type=str,
        default=DEFAULT_DEPTH_ESTIMATOR_MODEL,
        help=(
            "Depth-estimation repo ID or local path used to compute depth maps "
            f"on the fly (default: {DEFAULT_DEPTH_ESTIMATOR_MODEL})."
        ),
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of inference steps (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--canny-scale",
        type=float,
        default=DEFAULT_CANNY_SCALE,
        help=f"Canny ControlNet strength (default: {DEFAULT_CANNY_SCALE}).",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=DEFAULT_DEPTH_SCALE,
        help=f"Depth ControlNet strength (default: {DEFAULT_DEPTH_SCALE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )

    parser.add_argument(
        "--use_gt_mask",
        action="store_true",
        help=(
            "Switch to inpainting mode: for each <name>.<format> read a "
            "matching <name>.txt containing normalized polygon corner "
            "points, rasterize it into a mask, and regenerate only that "
            "region with StableDiffusion3ControlNetInpaintingPipeline "
            "instead of the canny+depth ControlNets."
        ),
    )
    parser.add_argument(
        "--inpaint_model",
        type=str,
        default=DEFAULT_INPAINT_MODEL,
        help=f"SD3 inpainting ControlNet repo ID or local path (default: {DEFAULT_INPAINT_MODEL}).",
    )
    parser.add_argument(
        "--inpaint-scale",
        type=float,
        default=DEFAULT_INPAINT_SCALE,
        help=f"Inpainting ControlNet conditioning strength (default: {DEFAULT_INPAINT_SCALE}).",
    )
    parser.add_argument(
        "--skip-missing-mask",
        action="store_true",
        help=(
            "In --use_gt_mask mode, skip (instead of aborting on) images "
            "whose matching .txt mask file is missing or malformed."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Install debug hooks into the pipeline's internal methods to "
            "print/log: what is passed as initial `latents` to the SD3 "
            "transformer, how each controlnet branch's output is scaled "
            "and combined before being fed to the transformer, and "
            "whether control_image/control_mask leak into latent "
            "initialization. Also saves a decoded preview of the initial "
            "(pre-denoising) latents next to each output image."
        ),
    )

    args = parser.parse_args()

    if not 0.0 <= args.scale <= 1.0:
        parser.error("--scale must be between 0 and 1.")

    if not 0.0 <= args.canny_scale:
        parser.error("--canny-scale must be >= 0.")

    if not 0.0 <= args.depth_scale:
        parser.error("--depth-scale must be >= 0.")

    if not 0.0 <= args.inpaint_scale:
        parser.error("--inpaint-scale must be >= 0.")

    if args.steps <= 0:
        parser.error("--steps must be > 0.")

    return args


def normalize_format(fmt):
    fmt = fmt.strip().lower()
    if not fmt.startswith("."):
        fmt = "." + fmt
    return fmt


def transfer_to_guidance(scale):
    # SD3 guidance_scale <= 1 disables classifier-free guidance.
    # Map the user-facing [0, 1] transfer parameter to [1, 9].
    return 1.0 + 8.0 * scale


def round_to_multiple(value, multiple=SD3_DIM_MULTIPLE):
    """Round to the nearest multiple, never rounding down to 0."""
    rounded = int(round(value / multiple)) * multiple
    return max(rounded, multiple)


def native_generation_size(image):
    """Return (width, height) for `image`, rounded to a multiple of 16."""
    width, height = image.size
    return round_to_multiple(width), round_to_multiple(height)


def make_canny(image):
    image_np = np.asarray(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)


def load_depth_estimator(args):
    processor = DPTImageProcessor.from_pretrained(args.depth_estimator_model)
    model = DPTForDepthEstimation.from_pretrained(args.depth_estimator_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return processor, model, device


def prepare_depth(image, processor, model, device, size):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {name: value.to(device) for name, value in inputs.items()}

    with torch.inference_mode():
        depth = model(**inputs).predicted_depth

    depth = depth.unsqueeze(1)
    depth = F.interpolate(
        depth,
        size=size,
        mode="bicubic",
        align_corners=False,
    )[0, 0]
    depth = depth - depth.min()
    depth = depth / depth.max().clamp(min=1e-6)
    depth = (depth * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    depth = Image.fromarray(depth, mode="L")
    return Image.merge("RGB", (depth, depth, depth))


# ---------------------------------------------------------------------------
# GT-mask loading (used only in --use_gt_mask mode)
# ---------------------------------------------------------------------------

def load_gt_points_from_txt(img_path, w, h):
    txt_path = os.path.splitext(str(img_path))[0] + ".txt"

    if not os.path.exists(txt_path):
        return None

    with open(txt_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()

    if not line:
        return None

    values = line.split()

    if len(values) < 13:
        return None

    coords = list(map(float, values[-8:]))

    (
        x_tl,
        y_tl,
        x_bl,
        y_bl,
        x_tr,
        y_tr,
        x_br,
        y_br,
    ) = coords

    gt_pts = np.array(
        [
            [x_tl * w, y_tl * h],
            [x_tr * w, y_tr * h],
            [x_br * w, y_br * h],
            [x_bl * w, y_bl * h],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)

    return gt_pts


def load_gt_mask(img_path, w, h):
    gt_pts = load_gt_points_from_txt(img_path, w, h)

    if gt_pts is None:
        raise ValueError(f"Could not read GT runway points: {img_path}")

    gt_mask = np.zeros((h, w), dtype=np.uint8)

    cv2.fillPoly(gt_mask, [np.round(gt_pts).astype(np.int32)], 255)

    return gt_mask > 0


def gt_mask_to_pil(gt_mask_bool):
    """Convert a boolean (H, W) mask into a single-channel 'L' PIL image
    with 255 for the masked (to-be-inpainted) region and 0 elsewhere."""
    mask_uint8 = (gt_mask_bool.astype(np.uint8)) * 255
    return Image.fromarray(mask_uint8, mode="L")


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def match_controlnet_input_channels(controlnet, target_in_channels):
    """Expand `controlnet`'s conditioning patch-embed conv
    (`pos_embed_input.proj`) to accept `target_in_channels` input
    channels if it currently accepts fewer, zero-padding the new
    channel(s) so they contribute nothing to the conv's output.

    This is needed because StableDiffusion3ControlNetInpaintingPipeline
    builds one mask-concatenated conditioning tensor (base latent
    channels + mask channel(s)) and feeds that same tensor to every
    controlnet in an SD3MultiControlNetModel list, not just the
    mask-aware one. Plain pretrained controlnets (canny, depth) have a
    `pos_embed_input.proj` sized for the base latent channel count only,
    so without this they raise a channel-count mismatch. Zero-padding
    the extra channel(s) preserves each plain controlnet's original,
    already-trained behavior on its native channels; the mask channel
    simply multiplies against zero weights and adds nothing.
    """
    old_conv = controlnet.pos_embed_input.proj

    if old_conv.in_channels >= target_in_channels:
        return controlnet

    new_conv = torch.nn.Conv2d(
        target_in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
        dtype=old_conv.weight.dtype,
        device=old_conv.weight.device,
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, : old_conv.in_channels] = old_conv.weight
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    controlnet.pos_embed_input.proj = new_conv
    return controlnet


def flatten_tensors(obj):
    """Recursively yield every torch.Tensor leaf inside `obj`.

    `obj` may be a Tensor, None, a (possibly nested) list/tuple (e.g. a
    tuple wrapping a per-transformer-block list of tensors, which is the
    common controlnet return shape), a dict-like object (anything with
    `.items()`), a diffusers BaseOutput-style object (anything with
    `.to_tuple()`), or a plain dataclass/object (fallback: `vars(obj)`).
    A flat `isinstance(x, (list, tuple))` + `torch.is_tensor` filter
    silently returns nothing for any of the nested cases, which is why
    the norm lists were coming back empty.
    """
    if obj is None:
        return
    if torch.is_tensor(obj):
        yield obj
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            yield from flatten_tensors(item)
        return
    if hasattr(obj, "items"):
        for _, v in obj.items():
            yield from flatten_tensors(v)
        return
    if hasattr(obj, "to_tuple"):
        yield from flatten_tensors(obj.to_tuple())
        return
    if hasattr(obj, "__dict__"):
        for v in vars(obj).values():
            yield from flatten_tensors(v)


def install_debug_hooks(pipe, debug_state):
    """Monkey-patch several of `pipe`'s bound methods to print/log what
    they actually receive and return at runtime, and stash results into
    `debug_state` (a plain dict) for the caller to inspect afterward.

    This answers three questions empirically instead of by reading source:
      1. What is passed as the initial `latents` to the SD3 transformer?
         -> hooks `pipe.prepare_latents`.
      2. How is each controlnet branch's output scaled/combined before
         being handed to the transformer?
         -> hooks each net in `pipe.controlnet.nets` (or `pipe.controlnet`
            itself if it's a single, non-multi controlnet) and, for multi,
            the combining `SD3MultiControlNetModel.forward` too.
      3. Does control_image / control_mask leak into latent initialization
         or scheduler prep?
         -> hooks `pipe.prepare_image_with_mask` (only present on the
            inpainting pipeline) so its calls/outputs are visible
            alongside the `prepare_latents` calls, in call order, so you
            can see whether they ever touch the same tensor.
    """

    debug_state.setdefault("initial_latents", None)
    debug_state.setdefault("prepare_latents_calls", 0)
    debug_state.setdefault("prepare_image_with_mask_calls", 0)

    # --- 1. Initial latents passed to the transformer -----------------
    if hasattr(pipe, "prepare_latents"):
        orig_prepare_latents = pipe.prepare_latents

        def debug_prepare_latents(*a, **kw):
            call_idx = debug_state["prepare_latents_calls"]
            debug_state["prepare_latents_calls"] += 1

            latents_arg = kw.get("latents", None)
            if latents_arg is None and len(a) >= 8:
                latents_arg = a[7]

            result = orig_prepare_latents(*a, **kw)

            print(
                f"[debug] prepare_latents() call #{call_idx}: "
                f"`latents` argument was "
                f"{'None -> freshly sampled Gaussian noise' if latents_arg is None else 'PROVIDED (not fresh noise!)'}; "
                f"returned shape={tuple(result.shape)}, "
                f"mean={result.float().mean().item():.5f}, "
                f"std={result.float().std().item():.5f}"
            )

            # Only the very first call is the actual denoising start point
            # for a given `pipe(...)` invocation; stash it so the caller
            # can decode/compare it after generation finishes.
            if call_idx == 0 or debug_state["initial_latents"] is None:
                debug_state["initial_latents"] = result.detach().clone()

            return result

        pipe.prepare_latents = debug_prepare_latents

    # --- 3. Does control_image/control_mask leak into latents? --------
    if hasattr(pipe, "prepare_image_with_mask"):
        orig_prepare_image_with_mask = pipe.prepare_image_with_mask

        def debug_prepare_image_with_mask(*a, **kw):
            call_idx = debug_state["prepare_image_with_mask_calls"]
            debug_state["prepare_image_with_mask_calls"] += 1

            result = orig_prepare_image_with_mask(*a, **kw)

            # Last channel is the mask channel appended inside
            # prepare_image_with_mask (see diffusers source: it does
            # `control_image = torch.cat([image_latents, mask], dim=1)`
            # after computing `mask = 1 - mask`).
            mask_channel = result[:, -1:, :, :]
            same_object_as_latents = (
                debug_state["initial_latents"] is not None
                and result.shape == debug_state["initial_latents"].shape
                and torch.equal(result, debug_state["initial_latents"])
            )
            print(
                f"[debug] prepare_image_with_mask() call #{call_idx}: "
                f"control_image tensor shape={tuple(result.shape)}, "
                f"mask-channel mean={mask_channel.float().mean().item():.5f} "
                f"(post internal `1 - mask`: ~0 -> fully masked/regenerate, "
                f"~1 -> fully preserved), "
                f"identical to current initial `latents` tensor? {same_object_as_latents}"
            )

            return result

        pipe.prepare_image_with_mask = debug_prepare_image_with_mask

    # --- 2. Per-branch controlnet scaling/combination ------------------
    controlnet = pipe.controlnet

    def wrap_branch_forward(net, branch_index):
        orig_forward = net.forward

        def debug_branch_forward(*a, **kw):
            scale = kw.get("conditioning_scale", None)
            if scale is None and len(a) > 2:
                scale = a[2]
            out = orig_forward(*a, **kw)
            tensors = list(flatten_tensors(out))
            if tensors:
                norms = [t.float().norm().item() for t in tensors]
                print(
                    f"[debug] controlnet branch {branch_index}: "
                    f"conditioning_scale={scale}, "
                    f"{len(tensors)} block-sample tensor(s), "
                    f"norm(s)={['%.4f' % n for n in norms]}"
                )
            else:
                print(
                    f"[debug] controlnet branch {branch_index}: "
                    f"conditioning_scale={scale}, "
                    f"found NO tensors in return value -- return type is "
                    f"{type(out)!r}, repr(out)[:200]={repr(out)[:200]!r}"
                )
            return out

        net.forward = debug_branch_forward

    if isinstance(controlnet, SD3MultiControlNetModel):
        for i, net in enumerate(controlnet.nets):
            wrap_branch_forward(net, i)

        orig_multi_forward = controlnet.forward

        def debug_multi_forward(*a, **kw):
            result = orig_multi_forward(*a, **kw)
            tensors = list(flatten_tensors(result))
            if tensors:
                norms = [t.float().norm().item() for t in tensors]
                print(
                    f"[debug] SD3MultiControlNetModel combined output: "
                    f"{len(tensors)} tensor(s) (sum of all branches above, "
                    f"this is what reaches the transformer), "
                    f"norm(s)={['%.4f' % n for n in norms]}"
                )
            else:
                print(
                    f"[debug] SD3MultiControlNetModel combined output: "
                    f"found NO tensors -- return type is {type(result)!r}, "
                    f"repr(result)[:200]={repr(result)[:200]!r}"
                )
            return result

        controlnet.forward = debug_multi_forward
    else:
        wrap_branch_forward(controlnet, 0)

    # --- 2 (cont.) How the transformer combines it ---------------------
    if hasattr(pipe, "transformer"):
        orig_transformer_forward = pipe.transformer.forward
        max_calls_to_log = 2  # cond + uncond pass of the first denoising step
        debug_state.setdefault("transformer_calls", 0)

        def debug_transformer_forward(*a, **kw):
            call_idx = debug_state["transformer_calls"]
            debug_state["transformer_calls"] += 1

            if call_idx < max_calls_to_log:
                print(f"[debug] transformer.forward() call #{call_idx}, all kwargs received:")
                controlnet_like_keys = []
                for k, v in kw.items():
                    tensors = list(flatten_tensors(v))
                    if tensors:
                        norms = [t.float().norm().item() for t in tensors]
                        print(
                            f"    {k}: type={type(v).__name__}, "
                            f"{len(tensors)} tensor(s), "
                            f"norm(s)={['%.4f' % n for n in norms]}"
                        )
                        if any(n > 1e-6 for n in norms):
                            controlnet_like_keys.append(k)
                    else:
                        print(f"    {k}: type={type(v).__name__} (no tensors found), value={v!r}"[:150])
                print(
                    f"    -> kwarg(s) actually carrying non-zero tensor "
                    f"content (candidates for the controlnet residual): "
                    f"{controlnet_like_keys}"
                )
            elif call_idx == max_calls_to_log:
                print(
                    f"[debug] transformer.forward(): suppressing further "
                    f"per-call logs after {max_calls_to_log} calls "
                    f"(pattern repeats identically for every remaining step)"
                )

            return orig_transformer_forward(*a, **kw)

        pipe.transformer.forward = debug_transformer_forward


def decode_latents_preview(pipe, latents, path):
    """Best-effort decode of a (typically noisy, pre-denoising) latents
    tensor into a viewable PNG, purely for visual sanity-checking in
    --debug mode. Uses the same shift/scale convention
    prepare_image_with_mask uses when going the other direction."""
    try:
        with torch.inference_mode():
            unscaled = latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
            decoded = pipe.vae.decode(unscaled.to(pipe.vae.dtype), return_dict=False)[0]
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded_np = (
            decoded[0].float().cpu().permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)
        Image.fromarray(decoded_np).save(path)
        print(f"[debug] saved decoded initial-latents preview to {path}")
    except Exception as exc:  # noqa: BLE001 - debug-only, never fatal
        print(f"[debug] could not decode initial-latents preview: {exc}")


def load_pipeline(args):
    dtype = torch.float16

    if args.use_gt_mask:
        inpaint_controlnet = SD3ControlNetModel.from_pretrained(
            args.inpaint_model,
            use_safetensors=True,
            extra_conditioning_channels=1,
        )

        canny_controlnet = SD3ControlNetModel.from_pretrained(
            args.canny_model,
            torch_dtype=dtype,
        )

        depth_controlnet = SD3ControlNetModel.from_pretrained(
            args.depth_model,
            torch_dtype=dtype,
        )

        # For each entry in control_image, StableDiffusion3ControlNetInpaintingPipeline
        # independently blacks out the masked region of that specific
        # conditioning image, VAE-encodes it, and concatenates the mask as
        # an extra channel -- applied identically to every branch, not just
        # the inpainting one. canny/depth were pretrained without that
        # extra channel, so pad their input convs to match (zero-
        # initialized: doesn't change their behavior on the original 16
        # channels, and the masked region is already blacked out at the
        # pixel level before their branch ever sees it).
        target_in_channels = inpaint_controlnet.pos_embed_input.proj.in_channels
        match_controlnet_input_channels(canny_controlnet, target_in_channels)
        match_controlnet_input_channels(depth_controlnet, target_in_channels)

        # Order just needs to match control_image / controlnet_conditioning_scale
        # below; all three branches are treated the same way internally.
        controlnet = SD3MultiControlNetModel(
            [inpaint_controlnet, canny_controlnet, depth_controlnet]
        )

        pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
            args.sd3_model,
            controlnet=controlnet,
            torch_dtype=dtype,
        )
        pipe.text_encoder.to(dtype)
        pipe.controlnet.to(dtype)
        pipe = pipe.to("cuda")
    else:
        canny_controlnet = SD3ControlNetModel.from_pretrained(
            args.canny_model,
            torch_dtype=dtype,
        )

        depth_controlnet = SD3ControlNetModel.from_pretrained(
            args.depth_model,
            torch_dtype=dtype,
        )

        controlnet = SD3MultiControlNetModel(
            [canny_controlnet, depth_controlnet]
        )

        pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
            args.sd3_model,
            controlnet=controlnet,
            torch_dtype=dtype,
        )

        pipe = pipe.to("cuda")

    # Force ordinary PyTorch attention. No FlashAttention/xFormers/
    # Transformer Engine dependency is required.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    return pipe


# ---------------------------------------------------------------------------
# Per-image processing
# ---------------------------------------------------------------------------

def process_image_control(pipe, depth_processor, depth_model, depth_device, input_path, output_path, args, guidance_scale):
    """Original canny+depth ControlNet style-transfer path."""
    image = Image.open(input_path).convert("RGB")

    # Keep this image's own aspect ratio/resolution instead of forcing a
    # fixed size; just round up/down to the nearest multiple of 16 since
    # SD3 requires that.
    gen_width, gen_height = native_generation_size(image)
    if (gen_width, gen_height) != image.size:
        image = image.resize((gen_width, gen_height), Image.Resampling.LANCZOS)

    canny = make_canny(image)
    depth = prepare_depth(
        image,
        depth_processor,
        depth_model,
        depth_device,
        (gen_height, gen_width),
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            control_image=[canny, depth],
            controlnet_conditioning_scale=[
                args.canny_scale,
                args.depth_scale,
            ],
            height=gen_height,
            width=gen_width,
            num_inference_steps=args.steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    result.images[0].save(output_path)


def process_image_inpaint(pipe, depth_processor, depth_model, depth_device, input_path, output_path, args, guidance_scale, debug_state=None):
    """GT-mask-driven inpainting path, multi-conditioned on the mask
    (inpainting ControlNet), edges (canny ControlNet), and scene geometry
    (depth ControlNet)."""
    image = Image.open(input_path).convert("RGB")

    gen_width, gen_height = native_generation_size(image)
    if (gen_width, gen_height) != image.size:
        image = image.resize((gen_width, gen_height), Image.Resampling.LANCZOS)

    try:
        gt_mask_bool = load_gt_mask(input_path, gen_width, gen_height)
    except ValueError as exc:
        if args.skip_missing_mask:
            print(f"  Skipping (no/invalid GT mask): {exc}")
            return
        raise

    mask_image = gt_mask_to_pil(gt_mask_bool)
    canny = make_canny(image)
    depth = prepare_depth(
        image,
        depth_processor,
        depth_model,
        depth_device,
        (gen_height, gen_width),
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            # Order matches the [inpaint_controlnet, canny_controlnet,
            # depth_controlnet] order the multi-controlnet was built with.
            # The pipeline independently blacks out the masked region of
            # each of these three images and mask-channel-augments them
            # before feeding each to its corresponding controlnet branch,
            # using the same control_mask for all three.
            control_image=[image, canny, depth],
            control_mask=mask_image,
            height=gen_height,
            width=gen_width,
            num_inference_steps=args.steps,
            controlnet_conditioning_scale=[
                args.inpaint_scale,
                args.canny_scale,
                args.depth_scale,
            ],
            guidance_scale=guidance_scale,
            generator=generator,
        )

    result.images[0].save(output_path)

    if args.debug and debug_state is not None and debug_state.get("initial_latents") is not None:
        debug_dir = output_path.parent / "debug_previews"
        debug_dir.mkdir(parents=True, exist_ok=True)
        preview_path = debug_dir / f"{output_path.stem}_initial_latents.png"
        decode_latents_preview(pipe, debug_state["initial_latents"], preview_path)
        print(
            f"[debug] If this preview already resembles the source image "
            f"or the final output rather than unstructured noise/static, "
            f"that's strong evidence diffusion isn't actually running "
            f"from a fresh noise sample for this image."
        )


def main():
    args = parse_args()

    content_dir = args.content_dir
    output_dir = args.output_dir
    image_format = normalize_format(args.format)

    if not content_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {content_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Match only files whose extension exactly matches --format.
    input_files = sorted(
        p for p in content_dir.iterdir()
        if p.is_file() and p.suffix.lower() == image_format
    )

    if not input_files:
        raise SystemExit(
            f"No files with format '{image_format}' found in {content_dir}"
        )

    depth_processor, depth_model, depth_device = load_depth_estimator(args)

    pipe = load_pipeline(args)
    guidance_scale = transfer_to_guidance(args.scale)

    debug_state = {}
    if args.debug:
        install_debug_hooks(pipe, debug_state)
        print("[debug] Debug hooks installed on prepare_latents, "
              "prepare_image_with_mask (if present), each controlnet "
              "branch, the multi-controlnet combiner, and transformer.forward.")

    print(f"Found {len(input_files)} input file(s).")
    print(f"Mode:                  {'GT-mask inpainting (mask+canny+depth)' if args.use_gt_mask else 'canny+depth control'}")
    print(f"Prompt transfer scale: {args.scale}")
    print(f"SD3 guidance scale:    {guidance_scale:.3f}")
    print(f"Negative prompt:       {args.negative_prompt!r}")
    print("Generation size:       native per-image (rounded to multiple of 16)")

    for index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name

        print(
            f"[{index}/{len(input_files)}] "
            f"{input_path.name} -> {output_path}"
        )

        if args.use_gt_mask:
            process_image_inpaint(
                pipe,
                depth_processor,
                depth_model,
                depth_device,
                input_path,
                output_path,
                args,
                guidance_scale,
                debug_state=debug_state,
            )
        else:
            process_image_control(
                pipe,
                depth_processor,
                depth_model,
                depth_device,
                input_path,
                output_path,
                args,
                guidance_scale,
            )

    print(f"Finished. Wrote {len(input_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()