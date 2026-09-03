#!/usr/bin/env python3
"""
Batch SD3.5 Large + Canny + Depth ControlNet + IP-Adapter style transfer.

Example:
    python batch_canny_depth_control_dir.py \
        --prompt "photorealistic airport photograph" \
        --scale 0.25 \
        -content_dir /data/input \
        --format .jpeg \
        --output_dir /data/output

All models are loaded directly from the Hugging Face Hub:
    stabilityai/stable-diffusion-3.5-large
    InstantX/SD3-Controlnet-Canny
    InstantX/SD3-Controlnet-Depth
    Intel/dpt-hybrid-midas
    google/siglip-so400m-patch14-384
    InstantX/SD3.5-Large-IP-Adapter
"""

import argparse
import inspect
import os
from pathlib import Path
import textwrap

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from transformers import (
    DPTForDepthEstimation,
    DPTImageProcessor,
    SiglipImageProcessor,
    SiglipVisionModel,
)


DEFAULT_SD3_MODEL = "stabilityai/stable-diffusion-3.5-large"
DEFAULT_CANNY_MODEL = "InstantX/SD3-Controlnet-Canny"
DEFAULT_DEPTH_MODEL = "InstantX/SD3-Controlnet-Depth"
DEFAULT_DEPTH_ESTIMATOR_MODEL = "Intel/dpt-hybrid-midas"
DEFAULT_IMAGE_ENCODER = "google/siglip-so400m-patch14-384"
DEFAULT_IP_ADAPTER_CHECKPOINT = "InstantX/SD3.5-Large-IP-Adapter"
DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter.bin"

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 28
DEFAULT_CANNY_SCALE = 1.0
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_SEED = 1234


def patch_sd3_ip_adapter_view_bug():
    """Patch known SD3 IP-Adapter view/stride bug in some diffusers versions."""
    try:
        from diffusers.models import attention_processor as ap
    except Exception:
        return

    cls = getattr(ap, "SD3IPAdapterJointAttnProcessor2_0", None)
    if cls is None:
        return

    try:
        src = inspect.getsource(cls.__call__)
    except (OSError, TypeError):
        return

    needle = ".view(batch_size, -1, attn.heads * head_dim)"
    if needle not in src:
        return

    patched_src = src.replace(
        needle,
        ".reshape(batch_size, -1, attn.heads * head_dim)",
    )

    namespace = {}
    exec(
        textwrap.dedent(patched_src),
        cls.__call__.__globals__,
        namespace,
    )
    cls.__call__ = namespace["__call__"]
    print("Applied compatibility patch for SD3 IP-Adapter attention processor.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch SD3.5 Large style transfer with Canny + Depth ControlNet + IP-Adapter."
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
        "--style_image",
        required=True,
        type=Path,
        help="Reference style image used by IP-Adapter.",
    )

    parser.add_argument(
        "--sd3_model",
        type=str,
        default=DEFAULT_SD3_MODEL,
        help=f"SD3.5 Large repo ID or local path (default: {DEFAULT_SD3_MODEL}).",
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
        "--image_encoder_model",
        type=str,
        default=DEFAULT_IMAGE_ENCODER,
        help=f"SigLIP image encoder repo ID or local path (default: {DEFAULT_IMAGE_ENCODER}).",
    )
    parser.add_argument(
        "--ip_adapter_checkpoint",
        type=str,
        default=DEFAULT_IP_ADAPTER_CHECKPOINT,
        help=(
            "SD3.5 Large IP-Adapter checkpoint repo ID or local path "
            f"(default: {DEFAULT_IP_ADAPTER_CHECKPOINT})."
        ),
    )
    parser.add_argument(
        "--ip_adapter_weight_name",
        type=str,
        default=DEFAULT_IP_ADAPTER_WEIGHT_NAME,
        help=(
            "IP-Adapter weight filename inside the checkpoint repo/path "
            f"(default: {DEFAULT_IP_ADAPTER_WEIGHT_NAME})."
        ),
    )
    parser.add_argument(
        "--ip_adapter_scale",
        type=float,
        default=0.5,
        help="IP-Adapter conditioning strength (default: 0.5).",
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

    if not 0.0 <= args.ip_adapter_scale:
        parser.error("--ip_adapter_scale must be >= 0.")

    if args.steps <= 0:
        parser.error("--steps must be > 0.")

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be > 0.")

    if not args.style_image.is_file():
        parser.error(f"Style image does not exist: {args.style_image}")

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

    feature_extractor = SiglipImageProcessor.from_pretrained(
        args.image_encoder_model,
    )
    image_encoder = SiglipVisionModel.from_pretrained(
        args.image_encoder_model,
        torch_dtype=dtype,
    )

    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.sd3_model,
        controlnet=controlnet,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    )

    pipe = pipe.to("cuda")
    pipe.load_ip_adapter(
        args.ip_adapter_checkpoint,
        weight_name=args.ip_adapter_weight_name,
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)

    # Force ordinary PyTorch attention. No FlashAttention/xFormers/
    # Transformer Engine dependency is required.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    return pipe


def process_image(
    pipe,
    depth_processor,
    depth_model,
    depth_device,
    style_image,
    input_path,
    output_path,
    args,
    guidance_scale,
):
    image = Image.open(input_path).convert("RGB")
    image = image.resize(
        (args.width, args.height),
        Image.Resampling.LANCZOS,
    )

    canny = make_canny(image)
    depth = prepare_depth(
        image,
        depth_processor,
        depth_model,
        depth_device,
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
            ip_adapter_image=style_image,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    result.images[0].save(output_path)


def main():
    patch_sd3_ip_adapter_view_bug()
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

    style_image = Image.open(args.style_image).convert("RGB")

    depth_processor, depth_model, depth_device = load_depth_estimator(args)
    pipe = load_pipeline(args)
    guidance_scale = transfer_to_guidance(args.scale)

    print(f"Found {len(input_files)} input file(s).")
    print(f"Prompt transfer scale: {args.scale}")
    print(f"SD3 guidance scale:    {guidance_scale:.3f}")
    print(f"IP-Adapter scale:      {args.ip_adapter_scale}")

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
            style_image,
            input_path,
            output_path,
            args,
            guidance_scale,
        )

    print(f"Finished. Wrote {len(input_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
