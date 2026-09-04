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
        --style_image /data/style.jpg \
        --distance_csv /data/distances.csv

All models are loaded directly from the Hugging Face Hub:
    stabilityai/stable-diffusion-3.5-large
    stabilityai/stable-diffusion-3.5-large-controlnet-canny
    stabilityai/stable-diffusion-3.5-large-controlnet-depth
    depth-anything/Depth-Anything-V2-Large-hf
    google/siglip-so400m-patch14-384
    InstantX/SD3.5-Large-IP-Adapter
"""

import argparse
import csv
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
DEFAULT_DEPTH_ESTIMATOR_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
DEFAULT_IMAGE_ENCODER = "google/siglip-so400m-patch14-384"
DEFAULT_IP_ADAPTER_CHECKPOINT = "InstantX/SD3.5-Large-IP-Adapter"
DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter.bin"

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 28
DEFAULT_CANNY_SCALE = 1.0
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_SEED = 1234


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
        required=True,
        type=Path,
        help="Reference style image used by IP-Adapter.",
    )

    parser.add_argument(
        "--distance_csv",
        required=True,
        type=Path,
        help=(
            "CSV mapping image names to slant_distance "
            "in kilometers."
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
            "Depth-estimation repo ID or local path used "
            "to compute depth maps on the fly "
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

    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be > 0.")

    if not args.style_image.is_file():
        parser.error(
            f"Style image does not exist: {args.style_image}"
        )

    if not args.distance_csv.is_file():
        parser.error(
            f"Distance CSV does not exist: {args.distance_csv}"
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



def load_slant_distances(csv_path, image_format):
    distances = {}

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            name = f"{row['image_name'].strip()}.{image_format}"
            if not name:
                continue

            dist = float(row["slant_distance"])
            if dist <= 0:
                raise ValueError(f"Invalid slant distance for {name}: {dist}")

            distances[name] = dist

    return distances



def load_gt_points_from_txt(img_path, w, h):
    txt_path = os.path.splitext(
        str(img_path)
    )[0] + ".txt"

    if not os.path.exists(txt_path):
        return None

    with open(
        txt_path,
        "r",
        encoding="utf-8",
    ) as f:
        line = f.readline().strip()

    if not line:
        return None

    values = line.split()

    if len(values) < 13:
        return None

    coords = list(
        map(float, values[-8:])
    )

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
    gt_pts = load_gt_points_from_txt(
        img_path,
        w,
        h,
    )

    if gt_pts is None:
        raise ValueError(
            f"Could not read GT runway points: {img_path}"
        )

    gt_mask = np.zeros(
        (h, w),
        dtype=np.uint8,
    )

    cv2.fillPoly(
        gt_mask,
        [
            np.round(gt_pts).astype(
                np.int32
            )
        ],
        255,
    )

    return gt_mask > 0


def relative_depth_to_metric(
    depth_image,
    gt_mask,
    slant_distance_km,
):
    depth = np.asarray(
        depth_image.convert("L"),
        dtype=np.float32,
    )

    valid = (
        gt_mask
        & np.isfinite(depth)
    )

    if not np.any(valid):
        raise ValueError(
            "GT runway mask contains no valid "
            "depth pixels."
        )

    runway_depth = depth[valid]

    reference_depth = float(
        np.median(runway_depth)
    )

    if reference_depth <= 0:
        raise ValueError(
            "Invalid runway reference depth: "
            f"{reference_depth}"
        )

    metric_depth_km = (
        depth
        / reference_depth
        * slant_distance_km
    )

    metric_depth_km = np.maximum(
        metric_depth_km,
        1e-6,
    )

    return metric_depth_km


def metric_depth_to_control_image(
    metric_depth_km,
):
    inverse_depth = 1.0 / np.maximum(
        metric_depth_km,
        1e-6,
    )

    low = np.percentile(
        inverse_depth,
        1.0,
    )

    high = np.percentile(
        inverse_depth,
        99.0,
    )

    inverse_depth = np.clip(
        inverse_depth,
        low,
        high,
    )

    inverse_depth = (
        inverse_depth - low
    ) / max(
        high - low,
        1e-6,
    )

    depth_uint8 = np.round(
        inverse_depth * 255.0
    ).astype(np.uint8)

    return Image.fromarray(
        depth_uint8
    ).convert("RGB")


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


def prepare_depth(
    image,
    preprocessor,
    input_path,
    slant_distance_km,
    size,
):
    depth = preprocessor(
        image,
        invert=True,
    )[0].convert("L")

    depth = depth.resize(
        size,
        Image.Resampling.BILINEAR,
    )

    gt_mask = load_gt_mask(
        input_path,
        size[0],
        size[1],
    )

    metric_depth_km = (
        relative_depth_to_metric(
            depth,
            gt_mask,
            slant_distance_km,
        )
    )

    control_depth = (
        metric_depth_to_control_image(
            metric_depth_km
        )
    )

    return (
        control_depth,
        metric_depth_km,
    )


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
    depth_preprocessor,
    style_image,
    input_path,
    output_path,
    args,
    guidance_scale,
    slant_distance_km,
):
    image = Image.open(
        input_path
    ).convert("RGB")

    image = image.resize(
        (
            args.width,
            args.height,
        ),
        Image.Resampling.LANCZOS,
    )

    canny = make_canny(
        image
    )

    depth, metric_depth_km = prepare_depth(
        image,
        depth_preprocessor,
        input_path,
        slant_distance_km,
        (
            args.width,
            args.height,
        ),
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

        np.save(
            args.save_depth_dir
            / f"{input_path.stem}_metric_km.npy",
            metric_depth_km.astype(
                np.float32
            ),
        )

    canny_tensor = prepare_canny_tensor(
        canny,
        args.width,
        args.height,
    ).to(dtype=torch.float16)

    generator = (
        torch.Generator(
            device="cuda"
        ).manual_seed(
            args.seed
        )
    )

    with torch.inference_mode():
        result = pipe(
            prompt=args.prompt,
            control_image=[canny_tensor, depth],
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

    slant_distances = load_slant_distances(
        args.distance_csv,
        image_format
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

    style_image = (
        Image.open(
            args.style_image
        ).convert("RGB")
    )

    depth_preprocessor, _ = (
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

    for index, input_path in enumerate(
        input_files,
        start=1,
    ):
        output_path = (
            output_dir
            / input_path.name
        )

        if (
            input_path.name
            not in slant_distances
        ):
            raise SystemExit(
                "No slant_distance found "
                "in CSV for "
                f"{input_path.name}"
            )

        slant_distance_km = (
            slant_distances[
                input_path.name
            ]
        )

        txt_path = (
            input_path.with_suffix(".txt")
        )

        if not txt_path.is_file():
            raise SystemExit(
                "No runway annotation found "
                f"for {input_path.name}: "
                f"{txt_path}"
            )

        print(
            f"[{index}/{len(input_files)}] "
            f"{input_path.name} -> "
            f"{output_path}"
        )


        process_image(
            pipe,
            depth_preprocessor,
            style_image,
            input_path,
            output_path,
            args,
            guidance_scale,
            slant_distance_km,
        )

    print(
        f"Finished. Wrote "
        f"{len(input_files)} file(s) "
        f"to {output_dir}"
    )


if __name__ == "__main__":
    main()