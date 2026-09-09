#!/usr/bin/env python3
"""
Batch SD3 Medium + Canny + Depth ControlNet style transfer, with an
optional ground-truth-mask-driven inpainting mode.
"""

import argparse
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from diffusers import (
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3Pipeline,
    StableDiffusion3InpaintPipeline,
)
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

SD3_DIM_MULTIPLE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch SD3 Medium style transfer with Canny + Depth ControlNet, or GT-mask inpainting."
    )

    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative_prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("-content_dir", "--content_dir", required=True, type=Path)
    parser.add_argument("--format", required=True)
    parser.add_argument("--output_dir", required=True, type=Path)

    parser.add_argument("--sd3_model", type=str, default=DEFAULT_SD3_MODEL)
    parser.add_argument("--canny_model", type=str, default=DEFAULT_CANNY_MODEL)
    parser.add_argument("--depth_model", type=str, default=DEFAULT_DEPTH_MODEL)
    parser.add_argument("--depth_estimator_model", type=str, default=DEFAULT_DEPTH_ESTIMATOR_MODEL)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--canny-scale", type=float, default=DEFAULT_CANNY_SCALE)
    parser.add_argument("--depth-scale", type=float, default=DEFAULT_DEPTH_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    parser.add_argument(
        "--use_inpaint_pipeline",
        action="store_true",
        help="Use the SD3 inpainting pipeline and the matching .txt GT mask.",
    )
    parser.add_argument("--inpaint_model", type=str, default=DEFAULT_INPAINT_MODEL)
    parser.add_argument("--inpaint-scale", type=float, default=DEFAULT_INPAINT_SCALE)
    parser.add_argument("--skip-missing-mask", action="store_true")

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
    return fmt if fmt.startswith(".") else "." + fmt


def transfer_to_guidance(scale):
    return 1.0 + 8.0 * scale


def round_to_multiple(value, multiple=SD3_DIM_MULTIPLE):
    rounded = int(round(value / multiple)) * multiple
    return max(rounded, multiple)


def native_generation_size(image):
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
    depth = F.interpolate(depth, size=size, mode="bicubic", align_corners=False)[0, 0]
    depth = depth - depth.min()
    depth = depth / depth.max().clamp(min=1e-6)
    depth = (depth * 255.0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    depth = Image.fromarray(depth, mode="L")
    return Image.merge("RGB", (depth, depth, depth))


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

    x_tl, y_tl, x_bl, y_bl, x_tr, y_tr, x_br, y_br = coords

    return np.array(
        [
            [x_tl * w, y_tl * h],
            [x_tr * w, y_tr * h],
            [x_br * w, y_br * h],
            [x_bl * w, y_bl * h],
        ],
        dtype=np.float32,
    ).reshape(-1, 1, 2)


def load_gt_mask(img_path, w, h):
    gt_pts = load_gt_points_from_txt(img_path, w, h)

    if gt_pts is None:
        raise ValueError(f"Could not read GT runway points: {img_path}")

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(gt_mask, [np.round(gt_pts).astype(np.int32)], 255)

    return gt_mask > 0


def gt_mask_to_pil(gt_mask_bool):
    mask_uint8 = (gt_mask_bool.astype(np.uint8)) * 255
    return Image.fromarray(mask_uint8, mode="L")


def match_controlnet_input_channels(controlnet, target_in_channels):
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
      3. Does control_image / control_mask leak into latents?
         -> hooks `pipe.prepare_image_with_mask` (only present on the
            inpainting pipeline) so its calls/outputs are visible
            alongside the `prepare_latents` calls, in call order, so you
            can see whether they ever touch the same tensor.
    """
    debug_state.setdefault("initial_latents", None)
    debug_state.setdefault("prepare_latents_calls", 0)
    debug_state.setdefault("prepare_image_with_mask_calls", 0)

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

            if call_idx == 0 or debug_state["initial_latents"] is None:
                debug_state["initial_latents"] = result.detach().clone()

            return result

        pipe.prepare_latents = debug_prepare_latents

    if hasattr(pipe, "prepare_image_with_mask"):
        orig_prepare_image_with_mask = pipe.prepare_image_with_mask

        def debug_prepare_image_with_mask(*a, **kw):
            call_idx = debug_state["prepare_image_with_mask_calls"]
            debug_state["prepare_image_with_mask_calls"] += 1

            result = orig_prepare_image_with_mask(*a, **kw)

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

    controlnet = pipe.controlnet


    def wrap_branch_forward(net, branch_index):
        orig_forward = net.forward

        def debug_branch_forward(*a, **kw):
            scale = kw.get("conditioning_scale", None)
            if scale is None and len(a) > 2:
                scale = a[2]

            out = orig_forward(*a, **kw)

            print(
                f"\n[DEBUG-CN] branch {branch_index}: "
                f"scale={scale}, return_type={type(out).__name__}"
            )

            if hasattr(out, "keys"):
                print(f"[DEBUG-CN] branch {branch_index}: keys={list(out.keys())}")

            if hasattr(out, "to_tuple"):
                tup = out.to_tuple()
                print(f"[DEBUG-CN] branch {branch_index}: tuple_len={len(tup)}")

                for i, x in enumerate(tup):
                    if torch.is_tensor(x):
                        print(
                            f"[DEBUG-CN] branch {branch_index}: "
                            f"tuple[{i}] shape={tuple(x.shape)} "
                            f"mean={x.float().mean().item():.6f} "
                            f"std={x.float().std().item():.6f} "
                            f"norm={x.float().norm().item():.4f} "
                            f"finite={torch.isfinite(x).all().item()}"
                        )
                    elif isinstance(x, (list, tuple)):
                        print(
                            f"[DEBUG-CN] branch {branch_index}: "
                            f"tuple[{i}] type={type(x).__name__} "
                            f"len={len(x)}"
                        )
                        for j, y in enumerate(x):
                            if torch.is_tensor(y):
                                print(
                                    f"[DEBUG-CN] branch {branch_index}: "
                                    f"tuple[{i}][{j}] shape={tuple(y.shape)} "
                                    f"mean={y.float().mean().item():.6f} "
                                    f"std={y.float().std().item():.6f} "
                                    f"norm={y.float().norm().item():.4f} "
                                    f"finite={torch.isfinite(y).all().item()}"
                                )

            return out

        net.forward = debug_branch_forward

        

    if isinstance(controlnet, SD3MultiControlNetModel):
        for i, net in enumerate(controlnet.nets):
            wrap_branch_forward(net, i)

        orig_multi_forward = controlnet.forward

        def debug_multi_forward(*a, **kw):
            result = orig_multi_forward(*a, **kw)

            print(
                f"\n[DEBUG-MULTI] return_type={type(result).__name__}"
            )

            if hasattr(result, "keys"):
                print(f"[DEBUG-MULTI] keys={list(result.keys())}")

            if hasattr(result, "to_tuple"):
                tup = result.to_tuple()
                print(f"[DEBUG-MULTI] tuple_len={len(tup)}")

                for i, x in enumerate(tup):
                    if torch.is_tensor(x):
                        print(
                            f"[DEBUG-MULTI] tuple[{i}] shape={tuple(x.shape)} "
                            f"mean={x.float().mean().item():.6f} "
                            f"std={x.float().std().item():.6f} "
                            f"norm={x.float().norm().item():.4f} "
                            f"finite={torch.isfinite(x).all().item()}"
                        )
                    elif isinstance(x, (list, tuple)):
                        print(
                            f"[DEBUG-MULTI] tuple[{i}] "
                            f"type={type(x).__name__} len={len(x)}"
                        )

                        for j, y in enumerate(x):
                            if torch.is_tensor(y):
                                print(
                                    f"[DEBUG-MULTI] tuple[{i}][{j}] "
                                    f"shape={tuple(y.shape)} "
                                    f"mean={y.float().mean().item():.6f} "
                                    f"std={y.float().std().item():.6f} "
                                    f"norm={y.float().norm().item():.4f} "
                                    f"finite={torch.isfinite(y).all().item()}"
                                )

            return result

        controlnet.forward = debug_multi_forward
    else:
        wrap_branch_forward(controlnet, 0)

    if hasattr(pipe, "transformer"):
        orig_transformer_forward = pipe.transformer.forward
        max_calls_to_log = 2
        debug_state.setdefault("transformer_calls", 0)

        def debug_transformer_forward(*a, **kw):
            call_idx = debug_state["transformer_calls"]
            debug_state["transformer_calls"] += 1

            if call_idx < 2:
                print(
                    f"\n[DEBUG-TRANSFORMER] call #{call_idx}"
                )

                print(
                    f"[DEBUG-TRANSFORMER] positional args={len(a)}, "
                    f"kwargs={list(kw.keys())}"
                )

                for k, v in kw.items():
                    if torch.is_tensor(v):
                        print(
                            f"[DEBUG-TRANSFORMER] {k}: "
                            f"shape={tuple(v.shape)} "
                            f"mean={v.float().mean().item():.6f} "
                            f"std={v.float().std().item():.6f} "
                            f"norm={v.float().norm().item():.4f} "
                            f"finite={torch.isfinite(v).all().item()}"
                        )
                    elif isinstance(v, (list, tuple)):
                        print(
                            f"[DEBUG-TRANSFORMER] {k}: "
                            f"{type(v).__name__}, len={len(v)}"
                        )

                        for i, x in enumerate(v):
                            if torch.is_tensor(x):
                                print(
                                    f"[DEBUG-TRANSFORMER] "
                                    f"{k}[{i}]: "
                                    f"shape={tuple(x.shape)} "
                                    f"mean={x.float().mean().item():.6f} "
                                    f"std={x.float().std().item():.6f} "
                                    f"norm={x.float().norm().item():.4f} "
                                    f"finite={torch.isfinite(x).all().item()}"
                                )

            return orig_transformer_forward(*a, **kw)

        pipe.transformer.forward = debug_transformer_forward


def decode_latents_preview(pipe, latents, path):
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
    except Exception as exc:
        print(f"[debug] could not decode initial-latents preview: {exc}")


def pipeline_branch(args):
    controls = sum(
        x > 0 for x in (args.canny_scale, args.depth_scale, args.inpaint_scale)
    )
    if args.use_inpaint_pipeline:
        if controls == 0:
            return "plain_inpaint"
        return "control_inpaint_single" if controls == 1 else "control_inpaint_multi"
    controls = sum(x > 0 for x in (args.canny_scale, args.depth_scale))
    if controls == 0:
        return "plain"
    return "control_single" if controls == 1 else "control_multi"


def load_pipeline(args):
    dtype = torch.float16
    branch = pipeline_branch(args)

    if branch == "plain":
        return StableDiffusion3Pipeline.from_pretrained(
            args.sd3_model, torch_dtype=dtype
        ).to("cuda")

    if branch == "plain_inpaint":
        return StableDiffusion3InpaintPipeline.from_pretrained(
            args.sd3_model, torch_dtype=dtype
        ).to("cuda")

    # Inpaint + ControlNet branch
    if args.use_inpaint_pipeline:
        nets = []
        inpaint_controlnet = None

        # Optional inpaint ControlNet (image + mask conditioning)
        if args.inpaint_scale > 0:
            inpaint_controlnet = SD3ControlNetModel.from_pretrained(
                args.inpaint_model,
                use_safetensors=True,
                extra_conditioning_channels=1,
            )
            nets.append(inpaint_controlnet)

        # Canny ControlNet
        if args.canny_scale > 0:
            nets.append(
                SD3ControlNetModel.from_pretrained(
                    args.canny_model, torch_dtype=dtype
                )
            )

        # Depth ControlNet
        if args.depth_scale > 0:
            nets.append(
                SD3ControlNetModel.from_pretrained(
                    args.depth_model, torch_dtype=dtype
                )
            )

        # --- Channel matching logic ---
        # Case 1: inpaint_controlnet present -> use its in_channels as target
        if inpaint_controlnet is not None and len(nets) > 1:
            target_in_channels = inpaint_controlnet.pos_embed_input.proj.in_channels
            for net in nets[1:]:
                match_controlnet_input_channels(net, target_in_channels)

        # Case 2: no inpaint_controlnet, but we still use control_mask
        # StableDiffusion3ControlNetInpaintingPipeline will build a 17-channel
        # control tensor (RGB + mask), so we must expand all nets to 17 channels.
        elif inpaint_controlnet is None and len(nets) > 0:
            # Original canny/depth nets expect 16 channels; we need 17.
            target_in_channels = nets[0].pos_embed_input.proj.in_channels + 1
            for net in nets:
                match_controlnet_input_channels(net, target_in_channels)

        controlnet = nets[0] if len(nets) == 1 else SD3MultiControlNetModel(nets)

        pipe = StableDiffusion3ControlNetInpaintingPipeline.from_pretrained(
            args.sd3_model,
            controlnet=controlnet,
            torch_dtype=dtype,
        )
        pipe.text_encoder.to(dtype)
        pipe.controlnet.to(dtype)
        return pipe.to("cuda")

    # Plain ControlNet (no inpaint pipeline)
    nets = []
    if args.canny_scale > 0:
        nets.append(
            SD3ControlNetModel.from_pretrained(
                args.canny_model, torch_dtype=dtype
            )
        )
    if args.depth_scale > 0:
        nets.append(
            SD3ControlNetModel.from_pretrained(
                args.depth_model, torch_dtype=dtype
            )
        )

    controlnet = nets[0] if len(nets) == 1 else SD3MultiControlNetModel(nets)

    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.sd3_model,
        controlnet=controlnet,
        torch_dtype=dtype,
    )
    return pipe.to("cuda")



def process_image_control(pipe, depth_processor, depth_model, depth_device, input_path, output_path, args, guidance_scale):
    image = Image.open(input_path).convert("RGB")

    gen_width, gen_height = native_generation_size(image)
    if (gen_width, gen_height) != image.size:
        image = image.resize((gen_width, gen_height), Image.Resampling.LANCZOS)

    control_images = []
    scales = []

    if args.canny_scale > 0:
        control_images.append(make_canny(image))
        scales.append(args.canny_scale)

    if args.depth_scale > 0:
        control_images.append(
            prepare_depth(
                image,
                depth_processor,
                depth_model,
                depth_device,
                (gen_height, gen_width),
            )
        )
        scales.append(args.depth_scale)

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            control_image=control_images[0] if len(control_images) == 1 else control_images,
            controlnet_conditioning_scale=scales[0] if len(scales) == 1 else scales,
            height=gen_height,
            width=gen_width,
            num_inference_steps=args.steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    result.images[0].save(output_path)


def process_image_inpaint(pipe, depth_processor, depth_model, depth_device, input_path, output_path, args, guidance_scale, debug_state=None):
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
    mask_image = ImageOps.invert(mask_image)

    branch = pipeline_branch(args)

    if branch == "plain_inpaint":
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                image=image,
                mask_image=mask_image,
                height=gen_height,
                width=gen_width,
                num_inference_steps=args.steps,
                guidance_scale=guidance_scale,
                strength=args.scale,
                generator=generator,
            )
    else:
        control_images = []
        scales = []

        if args.inpaint_scale > 0:
            control_images.append(image)
            scales.append(args.inpaint_scale)

        if args.canny_scale > 0:
            control_images.append(make_canny(image))
            scales.append(args.canny_scale)

        if args.depth_scale > 0:
            control_images.append(
                prepare_depth(
                    image,
                    depth_processor,
                    depth_model,
                    depth_device,
                    (gen_height, gen_width),
                )
            )
            scales.append(args.depth_scale)

        generator = torch.Generator(device="cuda").manual_seed(args.seed)

        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                control_image=control_images[0] if len(control_images) == 1 else control_images,
                control_mask=mask_image,
                height=gen_height,
                width=gen_width,
                num_inference_steps=args.steps,
                controlnet_conditioning_scale=scales[0] if len(scales) == 1 else scales,
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

    input_files = sorted(
        p for p in content_dir.iterdir()
        if p.is_file() and p.suffix.lower() == image_format
    )

    if not input_files:
        raise SystemExit(
            f"No files with format '{image_format}' found in {content_dir}"
        )

    needs_depth = args.depth_scale > 0
    if needs_depth:
        depth_processor, depth_model, depth_device = load_depth_estimator(args)
    else:
        depth_processor = depth_model = depth_device = None

    pipe = load_pipeline(args)
    guidance_scale = transfer_to_guidance(args.scale)

    debug_state = {}
    if args.debug:
        install_debug_hooks(pipe, debug_state)
        print("[debug] Debug hooks installed on prepare_latents, "
              "prepare_image_with_mask (if present), each controlnet "
              "branch, the multi-controlnet combiner, and transformer.forward.")

    print(f"Found {len(input_files)} input file(s).")
    print(f"Mode:                  {'GT-mask inpainting (mask+canny+depth)' if args.use_inpaint_pipeline else 'canny+depth control'}")
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

        if args.use_inpaint_pipeline:
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