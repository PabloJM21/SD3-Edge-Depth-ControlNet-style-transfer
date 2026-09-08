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
    - alimama-creative/SD3-Controlnet-Inpainting (consumes control_mask
      and drives which pixels get regenerated),
    - InstantX/SD3-Controlnet-Canny (consumes a canny edge map of the
      source image, to keep structural edges consistent inside and
      around the masked region), and
    - InstantX/SD3-Controlnet-Depth (consumes a depth map of the source
      image, to keep scene geometry/perspective consistent).

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

        # StableDiffusion3ControlNetInpaintingPipeline concatenates the
        # mask onto the conditioning latents once and feeds that same
        # tensor to every controlnet in the list. canny/depth were
        # pretrained without that extra channel, so pad their input convs
        # to match (zero-initialized: no change to their behavior).
        target_in_channels = inpaint_controlnet.pos_embed_input.proj.in_channels
        match_controlnet_input_channels(canny_controlnet, target_in_channels)
        match_controlnet_input_channels(depth_controlnet, target_in_channels)

        # The inpainting controlnet must come first: only it has the extra
        # mask-conditioning channel, and StableDiffusion3ControlNetInpaintingPipeline
        # forwards control_mask to that branch specifically.
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


def process_image_inpaint(pipe, depth_processor, depth_model, depth_device, input_path, output_path, args, guidance_scale):
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