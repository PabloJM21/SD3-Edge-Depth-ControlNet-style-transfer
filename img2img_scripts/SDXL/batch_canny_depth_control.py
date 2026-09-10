#!/usr/bin/env python3
"""
Batch SDXL img2img + Canny + Depth ControlNet style transfer.

Example:
    python batch_canny_depth_control_dir.py \
        --prompt "photorealistic airport photograph" \
        --scale 0.25 \
        -content_dir /data/input \
        --format .jpeg \
        --output_dir /data/output

All models are loaded directly from the Hugging Face Hub:
    stabilityai/stable-diffusion-xl-base-1.0
    diffusers/controlnet-canny-sdxl-1.0
    diffusers/controlnet-depth-sdxl-1.0
    madebyollin/sdxl-vae-fp16-fix
    Intel/dpt-hybrid-midas

Note: each input image is generated at its own native resolution
rather than being forced to a fixed size.
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
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
)
from transformers import DPTForDepthEstimation, DPTImageProcessor


DEFAULT_SDXL_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_CANNY_MODEL = "diffusers/controlnet-canny-sdxl-1.0"
DEFAULT_DEPTH_MODEL = "diffusers/controlnet-depth-sdxl-1.0"
DEFAULT_VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
DEFAULT_DEPTH_ESTIMATOR_MODEL = "Intel/dpt-hybrid-midas"

DEFAULT_STEPS = 28
DEFAULT_CANNY_SCALE = 1.0
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_SEED = 1234
DEFAULT_NEGATIVE_PROMPT = "hallucinated details, artificial edges, extra geometry, random artifacts"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch SDXL img2img style transfer with Canny + Depth ControlNet."
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
        "--sdxl_model",
        type=str,
        default=DEFAULT_SDXL_MODEL,
        help=f"SDXL base repo ID or local path (default: {DEFAULT_SDXL_MODEL}).",
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
        "--vae_model",
        type=str,
        default=DEFAULT_VAE_MODEL,
        help=f"SDXL VAE repo ID or local path (default: {DEFAULT_VAE_MODEL}).",
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

    args = parser.parse_args()

    if not 0.0 <= args.scale <= 1.0:
        parser.error("--scale must be between 0 and 1.")

    if not 0.0 <= args.canny_scale:
        parser.error("--canny-scale must be >= 0.")

    if not 0.0 <= args.depth_scale:
        parser.error("--depth-scale must be >= 0.")

    if args.steps <= 0:
        parser.error("--steps must be > 0.")

    return args


def normalize_format(fmt):
    fmt = fmt.strip().lower()
    if not fmt.startswith("."):
        fmt = "." + fmt
    return fmt


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


def load_pipeline(args):
    dtype = torch.float16

    canny_controlnet = ControlNetModel.from_pretrained(
        args.canny_model,
        torch_dtype=dtype,
    )

    depth_controlnet = ControlNetModel.from_pretrained(
        args.depth_model,
        torch_dtype=dtype,
    )

    use_canny = args.canny_scale > 0.0
    use_depth = args.depth_scale > 0.0

    if not (use_canny or use_depth):
        raise SystemExit("At least one ControlNet must be enabled (set --canny-scale or --depth-scale > 0).")

    if use_canny and use_depth:
        controlnet = [canny_controlnet, depth_controlnet]
    elif use_canny:
        controlnet = canny_controlnet
    else:
        controlnet = depth_controlnet

    vae = AutoencoderKL.from_pretrained(
        args.vae_model,
        torch_dtype=dtype,
    )

    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
        args.sdxl_model,
        controlnet=controlnet,
        vae=vae,
        torch_dtype=dtype,
    )

    pipe = pipe.to("cuda")

    # Force ordinary PyTorch attention. No FlashAttention/xFormers/
    # Transformer Engine dependency is required.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    return pipe


def process_image(pipe, depth_processor, depth_model, depth_device, input_path, output_path, args, guidance_scale):
    image = Image.open(input_path).convert("RGB")

    canny = make_canny(image)
    depth = prepare_depth(
        image,
        depth_processor,
        depth_model,
        depth_device,
        (image.height, image.width),
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    use_canny = args.canny_scale > 0.0
    use_depth = args.depth_scale > 0.0
    control_images = []
    control_scales = []

    if use_canny:
        control_images.append(canny)
        control_scales.append(args.canny_scale)
    if use_depth:
        control_images.append(depth)
        control_scales.append(args.depth_scale)

    if len(control_images) == 1:
        control_image_arg = control_images[0]
        conditioning_scale_arg = control_scales[0]
    else:
        control_image_arg = control_images
        conditioning_scale_arg = control_scales

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=image,
            strength=args.scale,
            control_image=control_image_arg,
            controlnet_conditioning_scale=conditioning_scale_arg,
            num_inference_steps=args.steps,
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
    guidance_scale = 5.0

    print(f"Found {len(input_files)} input file(s).")
    print(f"Prompt transfer scale: {args.scale}")
    print(f"SDXL guidance scale:   {guidance_scale:.3f}")
    print(f"Negative prompt:       {args.negative_prompt!r}")
    print("Generation size:       native per-image")

    for index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name

        print(
            f"[{index}/{len(input_files)}] "
            f"{input_path.name} -> {output_path}"
        )

        process_image(
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