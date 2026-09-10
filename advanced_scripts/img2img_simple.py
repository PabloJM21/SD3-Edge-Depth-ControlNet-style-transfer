#!/usr/bin/env python3
"""
Batch SDXL img2img using stabilityai/stable-diffusion-xl-refiner-1.0.

Example:
    python batch_sdxl_refiner_img2img.py \
        --prompt "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k" \
        --scale 0.5 \
        --content_dir /data/input \
        --format .jpeg \
        --output_dir /data/output
"""

import argparse
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from PIL import Image
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image


DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-refiner-1.0"
DEFAULT_STEPS = 28
DEFAULT_SEED = 1234
DEFAULT_GUIDANCE_SCALE = 8.0
DEFAULT_NEGATIVE_PROMPT = "low quality, blurry, artifacts"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch SDXL img2img with SDXL Refiner."
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
        "--strength",
        type=float,
        required=True,
        help="Img2img strength in [0, 1]. Higher = more deviation from input.",
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
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"SDXL Refiner repo ID or local path (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of inference steps (default: {DEFAULT_STEPS}).",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=DEFAULT_GUIDANCE_SCALE,
        help=f"Classifier-free guidance scale (default: {DEFAULT_GUIDANCE_SCALE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )

    args = parser.parse_args()

    if not 0.0 <= args.strength <= 1.0:
        parser.error("--scale must be between 0 and 1.")

    if args.steps <= 0:
        parser.error("--steps must be > 0.")

    if args.guidance_scale <= 0.0:
        parser.error("--guidance_scale must be > 0.")

    return args


def normalize_format(fmt):
    fmt = fmt.strip().lower()
    if not fmt.startswith("."):
        fmt = "." + fmt
    return fmt


def load_pipeline(args):
    pipe = AutoPipelineForImage2Image.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Offload to CPU to save VRAM
    pipe.enable_model_cpu_offload()


    return pipe


def process_image(pipe, input_path, output_path, args):
    # load_image handles local paths as well
    init_image = load_image(str(input_path))

    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(args.seed)

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            image=init_image,
            strength=args.strength,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
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

    input_files = sorted(
        p for p in content_dir.iterdir()
        if p.is_file() and p.suffix.lower() == image_format
    )

    if not input_files:
        raise SystemExit(
            f"No files with format '{image_format}' found in {content_dir}"
        )

    pipe = load_pipeline(args)

    print(f"Found {len(input_files)} input file(s).")
    print(f"Prompt:              {args.prompt!r}")
    print(f"Negative prompt:     {args.negative_prompt!r}")
    print(f"Img2img strength:    {args.strength}")
    print(f"Guidance scale:      {args.guidance_scale}")
    print(f"Steps:               {args.steps}")
    print("Generation size:     native per-image")

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
        )

    print(f"Finished. Wrote {len(input_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
