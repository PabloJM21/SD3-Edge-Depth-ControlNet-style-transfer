#!/usr/bin/env python3
"""
Batch SD3 Medium + Canny + Depth ControlNet style transfer.

Example:
    python batch_canny_depth_control_dir.py \
        --prompt "photorealistic airport photograph" \
        --scale 0.25 \
        -content_dir /data/input \
        --format .jpeg \
        --output_dir /data/output

Expected local model layout:
    /cluster/models/sd3-medium
    /cluster/models/sd3-canny
    /cluster/models/sd3-depth
"""

import argparse
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel


DEFAULT_SD3_MODEL = "/cluster/models/sd3-medium"
DEFAULT_CANNY_MODEL = "/cluster/models/sd3-canny"
DEFAULT_DEPTH_MODEL = "/cluster/models/sd3-depth"

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 28
DEFAULT_CANNY_SCALE = 1.0
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_SEED = 1234


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch SD3 Medium style transfer with Canny + Depth ControlNet."
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help="Prompt controlling the transferred style/content appearance.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help="Prompt-transfer strength in [0, 1]. 0 is conservative/no prompt transfer; 1 is maximum.",
    )
    parser.add_argument(
        "-content_dir",
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
        type=Path,
        default=Path(DEFAULT_SD3_MODEL),
        help=f"Local SD3 Medium model path (default: {DEFAULT_SD3_MODEL}).",
    )
    parser.add_argument(
        "--canny_model",
        type=Path,
        default=Path(DEFAULT_CANNY_MODEL),
        help=f"Local SD3 Canny ControlNet path (default: {DEFAULT_CANNY_MODEL}).",
    )
    parser.add_argument(
        "--depth_model",
        type=Path,
        default=Path(DEFAULT_DEPTH_MODEL),
        help=f"Local SD3 Depth ControlNet path (default: {DEFAULT_DEPTH_MODEL}).",
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
        "--width",
        type=int,
        default=DEFAULT_WIDTH,
        help=f"Generation width (default: {DEFAULT_WIDTH}).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_HEIGHT,
        help=f"Generation height (default: {DEFAULT_HEIGHT}).",
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

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be > 0.")

    return args


def normalize_format(fmt):
    fmt = fmt.strip().lower()
    if not fmt.startswith("."):
        fmt = "." + fmt
    return fmt


def transfer_to_guidance(scale):
    # SD3 guidance_scale <= 1 disables classifier-free guidance.
    # Map the user-facing [0, 1] transfer parameter to [1, 5].
    return 1.0 + 4.0 * scale


def make_canny(image):
    image_np = np.asarray(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)


def prepare_depth(path, size):
    depth = Image.open(path).convert("L")
    depth = depth.resize(size, Image.Resampling.BILINEAR)
    return Image.merge("RGB", (depth, depth, depth))


def load_pipeline(args):
    dtype = torch.float16

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


def process_image(pipe, input_path, output_path, args, guidance_scale):
    image = Image.open(input_path).convert("RGB")
    image = image.resize(
        (args.width, args.height),
        Image.Resampling.LANCZOS,
    )

    canny = make_canny(image)
    depth = prepare_depth(
        input_path,
        (args.width, args.height),
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            control_image=[canny, depth],
            controlnet_conditioning_scale=[
                args.canny_scale,
                args.depth_scale,
            ],
            height=args.height,
            width=args.width,
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

    pipe = load_pipeline(args)
    guidance_scale = transfer_to_guidance(args.scale)

    print(f"Found {len(input_files)} input file(s).")
    print(f"Prompt transfer scale: {args.scale}")
    print(f"SD3 guidance scale:    {guidance_scale:.3f}")

    for index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name

        print(
            f"[{index}/{len(input_files)}] "
            f"{input_path.name} -> {output_path}"
        )

        process_image(
            pipe,
            input_path,
            output_path,
            args,
            guidance_scale,
        )

    print(f"Finished. Wrote {len(input_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
