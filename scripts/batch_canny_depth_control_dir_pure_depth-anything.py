#!/usr/bin/env python3
"""
Batch SD3.5 Large + Canny + Depth ControlNet + IP-Adapter style transfer.

Example:
    python batch_canny_depth_control_dir.py \
        --prompt "photorealistic airport photograph" \
        --scale 0.25 \
        -content_dir /data/input \
        --format .jpeg \
        --output_dir /data/output \
        --style_image /data/style.jpg

If --style_image is omitted, each input image is used as its own
IP-Adapter style reference (self-style transfer).

All models are loaded directly from the Hugging Face Hub:
    stabilityai/stable-diffusion-3.5-large
    stabilityai/stable-diffusion-3.5-large-controlnet-canny
    stabilityai/stable-diffusion-3.5-large-controlnet-depth
    depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf
    google/siglip-so400m-patch14-384
    InstantX/SD3.5-Large-IP-Adapter

Note: each input image is generated at its own native resolution
(rounded to the nearest multiple of 16, which SD3.5's VAE + patchified
transformer requires) rather than being forced to a fixed size.
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
from PIL import Image
from diffusers import StableDiffusion3ControlNetPipeline
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from image_gen_aux import DepthPreprocessor
from transformers import (
    SiglipImageProcessor,
    SiglipVisionModel,
)


DEFAULT_SD3_MODEL = "stabilityai/stable-diffusion-3.5-large"
DEFAULT_CANNY_MODEL = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
DEFAULT_DEPTH_MODEL = "stabilityai/stable-diffusion-3.5-large-controlnet-depth"
DEFAULT_DEPTH_ESTIMATOR_MODEL = "depth-anything/Depth-Anything-V2-Large-hf" #"depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
DEFAULT_IMAGE_ENCODER = "google/siglip-so400m-patch14-384"
DEFAULT_IP_ADAPTER_CHECKPOINT = "InstantX/SD3.5-Large-IP-Adapter"
DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter.bin"

DEFAULT_STEPS = 28
DEFAULT_CANNY_SCALE = 1.0
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_SEED = 1234

# SD3.5's VAE downsamples by 8x and its transformer uses a patch size of 2,
# so both generation dimensions must be divisible by 16.
SD3_DIM_MULTIPLE = 16


class SD3CannyImageProcessor(VaeImageProcessor):
    """Custom preprocessing required for the SD3.5 Canny ControlNet (see
    the model card's usage snippet). Applied by hand to only the canny
    control image (see prepare_canny_tensor() / load_pipeline()) rather
    than being installed as pipe.image_processor, since that attribute is
    shared across every entry in a multi-ControlNet control_image list and
    would otherwise also corrupt the depth conditioning."""

    def __init__(self):
        super().__init__(do_normalize=False)

    def preprocess(self, image, **kwargs):
        image = super().preprocess(image, **kwargs)
        image = image * 255 * 0.5 + 0.5
        return image

    def postprocess(self, image, do_denormalize=True, **kwargs):
        do_denormalize = [True] * image.shape[0]
        image = super().postprocess(image, **kwargs, do_denormalize=do_denormalize)
        return image


# Shared instance used to preprocess only the canny control image by hand.
CANNY_IMAGE_PROCESSOR = SD3CannyImageProcessor()


def prepare_canny_tensor(canny_image, width, height):
    """Runs the SD3.5-Canny-specific preprocessing and returns a tensor.

    Passing an already-preprocessed torch.Tensor (rather than a PIL image)
    in the `control_image` list makes StableDiffusion3ControlNetPipeline's
    prepare_image() skip pipe.image_processor.preprocess() for this entry,
    so the depth entry can still use the pipeline's default processor.
    """
    return CANNY_IMAGE_PROCESSOR.preprocess(
        canny_image,
        height=height,
        width=width,
    )


def patch_sd3_ip_adapter_view_bug():
    """Patch the SD3 IP-Adapter attention processor for non-contiguous tensors."""
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

    old = ".view(batch_size, -1, attn.heads * head_dim)"

    if old not in src:
        return

    patched_src = src.replace(
        old,
        ".reshape(batch_size, -1, attn.heads * head_dim)",
    )

    namespace = {}

    exec(
        textwrap.dedent(patched_src),
        cls.__call__.__globals__,
        namespace,
    )

    cls.__call__ = namespace["__call__"]

    print(
        "Applied SD3 IP-Adapter reshape compatibility patch."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch SD3.5 Large style transfer with "
            "Canny + Depth ControlNet + IP-Adapter."
        )
    )

    parser.add_argument(
        "--prompt",
        required=True,
        help=(
            "Prompt controlling the transferred "
            "style/content appearance."
        ),
    )

    parser.add_argument(
        "--scale",
        type=float,
        required=True,
        help=(
            "Prompt-transfer strength in [0, 1]. "
            "0 is conservative/no prompt transfer; "
            "1 is maximum."
        ),
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
        type=Path,
        default=None,
        help=(
            "Reference style image used by IP-Adapter. "
            "If omitted, each input image is used as its own "
            "style image (self-style transfer)."
        ),
    )

    parser.add_argument(
        "--sd3_model",
        type=str,
        default=DEFAULT_SD3_MODEL,
        help=(
            "SD3.5 Large repo ID or local path "
            f"(default: {DEFAULT_SD3_MODEL})."
        ),
    )

    parser.add_argument(
        "--canny_model",
        type=str,
        default=DEFAULT_CANNY_MODEL,
        help=(
            "Canny ControlNet repo ID or local path "
            f"(default: {DEFAULT_CANNY_MODEL})."
        ),
    )

    parser.add_argument(
        "--depth_model",
        type=str,
        default=DEFAULT_DEPTH_MODEL,
        help=(
            "Depth ControlNet repo ID or local path "
            f"(default: {DEFAULT_DEPTH_MODEL})."
        ),
    )

    parser.add_argument(
        "--depth_estimator_model",
        type=str,
        default=DEFAULT_DEPTH_ESTIMATOR_MODEL,
        help=(
            "Metric depth-estimation repo ID or local path used "
            "to compute metric depth maps on the fly "
            f"(default: {DEFAULT_DEPTH_ESTIMATOR_MODEL})."
        ),
    )

    parser.add_argument(
        "--image_encoder_model",
        type=str,
        default=DEFAULT_IMAGE_ENCODER,
        help=(
            "SigLIP image encoder repo ID or local path "
            f"(default: {DEFAULT_IMAGE_ENCODER})."
        ),
    )

    parser.add_argument(
        "--ip_adapter_checkpoint",
        type=str,
        default=DEFAULT_IP_ADAPTER_CHECKPOINT,
        help=(
            "SD3.5 Large IP-Adapter checkpoint repo ID "
            "or local path "
            f"(default: {DEFAULT_IP_ADAPTER_CHECKPOINT})."
        ),
    )

    parser.add_argument(
        "--ip_adapter_weight_name",
        type=str,
        default=DEFAULT_IP_ADAPTER_WEIGHT_NAME,
        help=(
            "IP-Adapter weight filename inside the checkpoint "
            "repo/path "
            f"(default: {DEFAULT_IP_ADAPTER_WEIGHT_NAME})."
        ),
    )

    parser.add_argument(
        "--ip_adapter_scale",
        type=float,
        default=0.5,
        help=(
            "IP-Adapter conditioning strength "
            "(default: 0.5)."
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
        help=(
            "Canny ControlNet strength "
            f"(default: {DEFAULT_CANNY_SCALE})."
        ),
    )

    parser.add_argument(
        "--depth-scale",
        type=float,
        default=DEFAULT_DEPTH_SCALE,
        help=(
            "Depth ControlNet strength "
            f"(default: {DEFAULT_DEPTH_SCALE})."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )

    parser.add_argument(
        "--save_canny_dir",
        type=Path,
        default=None,
        help="Optional directory to save computed canny maps.",
    )

    parser.add_argument(
        "--save_depth_dir",
        type=Path,
        default=None,
        help="Optional directory to save computed depth maps.",
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

    if args.style_image is not None and not args.style_image.is_file():
        parser.error(
            f"Style image does not exist: {args.style_image}"
        )

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

    gray = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2GRAY,
    )

    edges = cv2.Canny(
        gray,
        100,
        200,
    )

    edges = np.stack(
        [edges, edges, edges],
        axis=-1,
    )

    return Image.fromarray(edges)


def load_depth_estimator(args):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    preprocessor = (
        DepthPreprocessor.from_pretrained(
            args.depth_estimator_model
        )
    )

    preprocessor = preprocessor.to(
        device
    )

    return preprocessor, device


def prepare_depth(image, depth_estimator, size):
    preprocessor, _ = depth_estimator
    depth = preprocessor(image, invert=True)[0].convert("RGB")
    return depth.resize(size, Image.Resampling.BILINEAR)


def load_pipeline(args):
    dtype = torch.float16

    canny_controlnet = (
        SD3ControlNetModel.from_pretrained(
            args.canny_model,
            torch_dtype=dtype,
        )
    )

    depth_controlnet = (
        SD3ControlNetModel.from_pretrained(
            args.depth_model,
            torch_dtype=dtype,
        )
    )

    controlnet = SD3MultiControlNetModel(
        [
            canny_controlnet,
            depth_controlnet,
        ]
    )

    feature_extractor = (
        SiglipImageProcessor.from_pretrained(
            args.image_encoder_model,
        )
    )

    image_encoder = (
        SiglipVisionModel.from_pretrained(
            args.image_encoder_model,
            torch_dtype=dtype,
        )
    )

    pipe = (
        StableDiffusion3ControlNetPipeline.from_pretrained(
            args.sd3_model,
            controlnet=controlnet,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            torch_dtype=dtype,
        )
    )

    pipe = pipe.to("cuda")

    # NOTE: pipe.image_processor is intentionally left as the default
    # VaeImageProcessor. StableDiffusion3ControlNetPipeline.prepare_image()
    # uses this single, shared processor for every entry in the
    # `control_image` list (canny AND depth) when doing multi-ControlNet.
    # The Canny model's special preprocessing (see
    # stabilityai/stable-diffusion-3.5-large-controlnet-canny) is instead
    # applied by hand to just the canny image in process_image(), so the
    # depth image still gets standard preprocessing.

    pipe.load_ip_adapter(
        args.ip_adapter_checkpoint,
        weight_name=args.ip_adapter_weight_name,
    )

    pipe.set_ip_adapter_scale(
        args.ip_adapter_scale
    )

    # Force ordinary PyTorch attention.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    return pipe


def process_image(
    pipe,
    depth_estimator,
    style_image,
    input_path,
    output_path,
    args,
    guidance_scale,
):
    image = Image.open(
        input_path
    ).convert("RGB")

    # Keep this image's own aspect ratio/resolution instead of forcing a
    # fixed size; just round up/down to the nearest multiple of 16 since
    # SD3.5 requires that.
    gen_width, gen_height = native_generation_size(image)
    if (gen_width, gen_height) != image.size:
        image = image.resize(
            (gen_width, gen_height),
            Image.Resampling.LANCZOS,
        )

    canny = make_canny(
        image
    )

    depth = prepare_depth(
        image,
        depth_estimator,
        (gen_width, gen_height),
    )

    if args.save_canny_dir is not None:
        canny.save(
            args.save_canny_dir
            / input_path.name
        )

    if args.save_depth_dir is not None:
        depth.save(
            args.save_depth_dir
            / input_path.name
        )

    canny_tensor = prepare_canny_tensor(
        canny,
        gen_width,
        gen_height,
    ).to(dtype=torch.float16)

    generator = (
        torch.Generator(
            device="cuda"
        ).manual_seed(
            args.seed
        )
    )

    # If no explicit global style image was provided, fall back to using
    # this content image itself as the IP-Adapter style reference.
    ip_adapter_image = (
        style_image
        if style_image is not None
        else image
    )

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            control_image=[canny_tensor, depth],
            controlnet_conditioning_scale=[
                args.canny_scale,
                args.depth_scale,
            ],
            ip_adapter_image=ip_adapter_image,
            height=gen_height,
            width=gen_width,
            num_inference_steps=args.steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    result.images[0].save(
        output_path
    )


def main():
    patch_sd3_ip_adapter_view_bug()
    args = parse_args()

    content_dir = args.content_dir
    output_dir = args.output_dir

    image_format = normalize_format(
        args.format
    )

    if not content_dir.is_dir():
        raise SystemExit(
            f"Input directory does not exist: "
            f"{content_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if args.save_canny_dir is not None:
        args.save_canny_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    if args.save_depth_dir is not None:
        args.save_depth_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    input_files = sorted(
        p
        for p in content_dir.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            == image_format
        )
    )

    if not input_files:
        raise SystemExit(
            f"No files with format "
            f"'{image_format}' found in "
            f"{content_dir}"
        )

    # If --style_image was given, load it once up front and reuse it for
    # every image. Otherwise leave it as None so process_image() falls
    # back to using each content image as its own style reference.
    style_image = (
        Image.open(args.style_image).convert("RGB")
        if args.style_image is not None
        else None
    )

    depth_estimator = (
        load_depth_estimator(args)
    )

    pipe = load_pipeline(args)

    guidance_scale = (
        transfer_to_guidance(
            args.scale
        )
    )

    print(
        f"Found {len(input_files)} input file(s)."
    )

    print(
        f"Prompt transfer scale: "
        f"{args.scale}"
    )

    print(
        f"SD3 guidance scale:    "
        f"{guidance_scale:.3f}"
    )

    print(
        f"IP-Adapter scale:      "
        f"{args.ip_adapter_scale}"
    )

    print(
        "Generation size:       "
        "native per-image (rounded to multiple of 16)"
    )

    if style_image is None:
        print(
            "No --style_image provided: "
            "using each input image as its own style reference."
        )

    for index, input_path in enumerate(
        input_files,
        start=1,
    ):
        output_path = (
            output_dir
            / input_path.name
        )

        print(
            f"[{index}/{len(input_files)}] "
            f"{input_path.name} -> "
            f"{output_path}"
        )

        process_image(
            pipe,
            depth_estimator,
            style_image,
            input_path,
            output_path,
            args,
            guidance_scale,
        )

    print(
        f"Finished. Wrote "
        f"{len(input_files)} file(s) "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()