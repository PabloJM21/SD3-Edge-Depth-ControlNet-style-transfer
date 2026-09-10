#!/usr/bin/env python3
"""
Batch SD3 Medium Img2Img + Canny/Depth ControlNet (single- or multi-conditioning),
followed by GT-mask runway extraction and SDXL RealVisXL inpainting to
re-integrate the runway into the newly generated surroundings.

Example:
    python batch_canny_depth_control_dir.py \
        --prompt "photorealistic airport photograph" \
        --scale 0.25 \
        -content_dir /data/input \
        --format .jpeg \
        --output_dir /data/output

Pipeline stages per image:
    1. SD3 Medium Img2Img + Canny/Depth ControlNet(s) generates a new
       stylized version of the whole image. Canny and/or Depth conditioning
       are used depending on which of --canny-scale / --depth-scale is > 0
       (single ControlNet if only one is enabled, multi-ControlNet if both).
    2. A ground-truth runway mask is loaded from the image's companion
       ``.txt`` polygon-annotation file (if present).
    3. The runway region of the SD3-generated image is re-generated with
       StableDiffusionXLInpaintPipeline (RealVisXL), so the runway blends
       naturally into the new background instead of being crudely pasted in.
    4. The inpainted runway is composited back onto the SD3-generated
       surroundings using the (blurred) mask.

All models are loaded directly from the Hugging Face Hub:
    stabilityai/stable-diffusion-3-medium-diffusers
    InstantX/SD3-Controlnet-Canny
    InstantX/SD3-Controlnet-Depth
    Intel/dpt-hybrid-midas
    OzzyGT/RealVisXL_V4.0_inpainting

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
from diffusers import (
    AutoencoderKL,
    StableDiffusion3ControlNetImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
)
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from transformers import DPTForDepthEstimation, DPTImageProcessor


DEFAULT_SD3_MODEL = "stabilityai/stable-diffusion-3-medium-diffusers"
DEFAULT_CANNY_MODEL = "InstantX/SD3-Controlnet-Canny"
DEFAULT_DEPTH_MODEL = "InstantX/SD3-Controlnet-Depth"
DEFAULT_DEPTH_ESTIMATOR_MODEL = "Intel/dpt-hybrid-midas"

DEFAULT_STEPS = 28
DEFAULT_CANNY_SCALE = 1.0
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_SEED = 1234
DEFAULT_NEGATIVE_PROMPT = "unnatural colors not suited for realistic landscape images" #"hallucinated details, artificial edges, extra geometry, random artifacts"

# Runway re-integration (SDXL inpainting) defaults.
DEFAULT_INPAINT_MODEL = "OzzyGT/RealVisXL_V4.0_inpainting"
DEFAULT_INPAINT_VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
DEFAULT_INPAINT_PROMPT = "Match the runway to the surrounding scene in tone, lighting, and atmosphere, while preserving all original runway geometry and surface details exactly."
DEFAULT_INPAINT_NEGATIVE_PROMPT = ""
DEFAULT_INPAINT_GUIDANCE_SCALE = 10.0
DEFAULT_INPAINT_STRENGTH = 0.8
DEFAULT_INPAINT_STEPS = 30
DEFAULT_INPAINT_MASK_BLUR = 20

# SD3's VAE downsamples by 8x and its transformer uses a patch size of 2,
# so both generation dimensions must be divisible by 16.
SD3_DIM_MULTIPLE = 16


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch SD3 Medium Img2Img style transfer with Canny + Depth ControlNet, "
        "then SDXL/RealVisXL inpainting to re-integrate the runway."
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
        help="Img2Img transfer strength in [0, 1]. 0 is conservative/no change; 1 is maximum change.",
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
        help=(
            "Canny ControlNet strength (default: {}). Set to 0 to disable Canny "
            "conditioning entirely.".format(DEFAULT_CANNY_SCALE)
        ),
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=DEFAULT_DEPTH_SCALE,
        help=(
            "Depth ControlNet strength (default: {}). Set to 0 to disable Depth "
            "conditioning entirely.".format(DEFAULT_DEPTH_SCALE)
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )

    # Runway re-integration (SDXL inpainting) options.
    parser.add_argument(
        "--inpaint_model",
        type=str,
        default=DEFAULT_INPAINT_MODEL,
        help=f"SDXL RealVisXL inpainting repo ID or local path (default: {DEFAULT_INPAINT_MODEL}).",
    )
    parser.add_argument(
        "--inpaint_vae_model",
        type=str,
        default=DEFAULT_INPAINT_VAE_MODEL,
        help=f"VAE repo ID used with the inpainting pipeline (default: {DEFAULT_INPAINT_VAE_MODEL}).",
    )
    parser.add_argument(
        "--inpaint_prompt",
        type=str,
        default=DEFAULT_INPAINT_PROMPT,
        help=f"Prompt used for runway inpainting (default: {DEFAULT_INPAINT_PROMPT!r}).",
    )
    parser.add_argument(
        "--inpaint_negative_prompt",
        type=str,
        default=DEFAULT_INPAINT_NEGATIVE_PROMPT,
        help="Negative prompt used for runway inpainting.",
    )
    parser.add_argument(
        "--inpaint_guidance_scale",
        type=float,
        default=DEFAULT_INPAINT_GUIDANCE_SCALE,
        help=f"Guidance scale for runway inpainting (default: {DEFAULT_INPAINT_GUIDANCE_SCALE}).",
    )
    parser.add_argument(
        "--inpaint_strength",
        type=float,
        default=DEFAULT_INPAINT_STRENGTH,
        help=f"Denoising strength for runway inpainting (default: {DEFAULT_INPAINT_STRENGTH}).",
    )
    parser.add_argument(
        "--inpaint_steps",
        type=int,
        default=DEFAULT_INPAINT_STEPS,
        help=f"Number of inference steps for runway inpainting (default: {DEFAULT_INPAINT_STEPS}).",
    )
    parser.add_argument(
        "--inpaint_mask_blur",
        type=int,
        default=DEFAULT_INPAINT_MASK_BLUR,
        help=(
            "Blur factor applied to the runway mask before inpainting, for a "
            f"smoother transition (default: {DEFAULT_INPAINT_MASK_BLUR})."
        ),
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

    if not (args.canny_scale > 0.0 or args.depth_scale > 0.0):
        parser.error("At least one ControlNet must be enabled (set --canny-scale or --depth-scale > 0).")

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


def load_gt_points_from_txt(img_path, w, h):
    """Read a polygon annotation (last 8 values = normalized corner coords)."""
    txt_path = os.path.splitext(img_path)[0] + ".txt"

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

    gt_pts = np.array(
        [
            [x_tl * w, y_tl * h],
            [x_tr * w, y_tr * h],
            [x_br * w, y_br * h],
            [x_bl * w, y_bl * h],
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    return gt_pts


def load_gt_mask(img_path, w, h):
    """Build a binary runway mask (H, W) float32 in {0, 1} from the GT polygon.

    Returns None if no companion .txt annotation exists for this image, so
    callers can fall back to skipping the runway re-integration step.
    """
    gt_pts = load_gt_points_from_txt(img_path, w, h)
    if gt_pts is None:
        return None

    gt_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(gt_mask, [gt_pts], 255)

    return (gt_mask > 0).astype(np.float32)


def load_pipeline(args):
    dtype = torch.float16

    use_canny = args.canny_scale > 0.0
    use_depth = args.depth_scale > 0.0

    canny_controlnet = None
    depth_controlnet = None

    if use_canny:
        canny_controlnet = SD3ControlNetModel.from_pretrained(
            args.canny_model,
            torch_dtype=dtype,
        )

    if use_depth:
        depth_controlnet = SD3ControlNetModel.from_pretrained(
            args.depth_model,
            torch_dtype=dtype,
        )

    if use_canny and use_depth:
        controlnet = SD3MultiControlNetModel([canny_controlnet, depth_controlnet])
    elif use_canny:
        controlnet = canny_controlnet
    else:
        controlnet = depth_controlnet

    pipe = StableDiffusion3ControlNetImg2ImgPipeline.from_pretrained(
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


def load_inpaint_pipeline(args):
    dtype = torch.float16

    vae = AutoencoderKL.from_pretrained(
        args.inpaint_vae_model,
        torch_dtype=dtype,
    )

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.inpaint_model,
        torch_dtype=dtype,
        vae=vae,
    )

    pipe = pipe.to("cuda")

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    return pipe


def reintegrate_runway(inpaint_pipe, generated_image, mask_image, args, seed):
    """Re-generate the runway region so it blends into `generated_image`,
    then composite the result back over the generated surroundings.

    `mask_image` is an "L" mode PIL image where the runway (the inpainting
    target) is white (255) and everything to be preserved is black (0).
    """
    # Blur the mask for a smoother transition between the inpainted runway
    # and the SD3-generated surroundings. The runway stays white/target,
    # the surroundings stay black/preserved -- blurring only softens the
    # edge between them.
    mask_blurred = inpaint_pipe.mask_processor.blur(
        mask_image, blur_factor=args.inpaint_mask_blur
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)

    with torch.inference_mode():
        inpainted = inpaint_pipe(
            prompt=args.inpaint_prompt,
            negative_prompt=args.inpaint_negative_prompt,
            image=generated_image,
            mask_image=mask_blurred,
            guidance_scale=args.inpaint_guidance_scale,
            strength=args.inpaint_strength,
            num_inference_steps=args.inpaint_steps,
            generator=generator,
        ).images[0]

    # Paste the newly inpainted runway back over the generated surroundings,
    # using the (unblurred) binary mask so only the runway region changes.
    final_image = Image.composite(inpainted, generated_image, mask_image)
    return final_image


def process_image(
    pipe,
    inpaint_pipe,
    depth_processor,
    depth_model,
    depth_device,
    input_path,
    output_path,
    args,
    guidance_scale,
):
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

    generated_image = result.images[0]
    if generated_image.size != (gen_width, gen_height):
        generated_image = generated_image.resize(
            (gen_width, gen_height), Image.Resampling.LANCZOS
        )

    gt_mask = load_gt_mask(str(input_path), gen_width, gen_height)

    if gt_mask is not None:
        # Runway (inpainting target) = white (255), preserved surroundings = black (0).
        mask_image = Image.fromarray((gt_mask * 255.0).astype(np.uint8), mode="L")

        # Naive cut-paste preview: original runway pixels pasted straight onto
        # the SD3-generated surroundings, saved before inpainting is applied.
        cutpaste_image = Image.composite(image, generated_image, mask_image)
        cutpaste_path = output_path.with_name(f"cutpaste-{output_path.name}")
        cutpaste_image.save(cutpaste_path)

        final_image = reintegrate_runway(
            inpaint_pipe, generated_image, mask_image, args, args.seed
        )
    else:
        # No GT annotation found for this image: nothing to re-integrate,
        # keep the SD3-generated image as-is.
        final_image = generated_image

    final_image.save(output_path)


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
    inpaint_pipe = load_inpaint_pipeline(args)
    guidance_scale = transfer_to_guidance(args.scale)

    use_canny = args.canny_scale > 0.0
    use_depth = args.depth_scale > 0.0
    if use_canny and use_depth:
        conditioning_mode = "multi (Canny + Depth)"
    elif use_canny:
        conditioning_mode = "single (Canny only)"
    else:
        conditioning_mode = "single (Depth only)"

    print(f"Found {len(input_files)} input file(s).")
    print(f"Img2Img strength:      {args.scale}")
    print(f"ControlNet mode:       {conditioning_mode}")
    print(f"SD3 guidance scale:    {guidance_scale:.3f}")
    print(f"Negative prompt:       {args.negative_prompt!r}")
    print("Generation size:       native per-image (rounded to multiple of 16)")
    print(f"Runway inpaint model:  {args.inpaint_model}")

    for index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name

        print(
            f"[{index}/{len(input_files)}] "
            f"{input_path.name} -> {output_path}"
        )

        process_image(
            pipe,
            inpaint_pipe,
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