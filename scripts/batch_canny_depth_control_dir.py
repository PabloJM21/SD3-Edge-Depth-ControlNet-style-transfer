#!/usr/bin/env python3
"""
Batch SD3.5 Large + Canny + Depth ControlNet + IP-Adapter style transfer.

Minimal img2img modification of the original script:
- keeps the same models and CLI arguments
- keeps --scale as the existing user-facing transfer/guidance control
- adds true img2img initialization from the content image
- adds --strength as an optional img2img denoising strength (default 0.45)
- preserves aspect ratio by center-cropping to the requested output size
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
from diffusers.models import SD3ControlNetModel, SD3MultiControlNetModel
from diffusers.utils.torch_utils import randn_tensor
from image_gen_aux import DepthPreprocessor
from transformers import SiglipImageProcessor, SiglipVisionModel


DEFAULT_SD3_MODEL = "stabilityai/stable-diffusion-3.5-large"
DEFAULT_CANNY_MODEL = "stabilityai/stable-diffusion-3.5-large-controlnet-canny"
DEFAULT_DEPTH_MODEL = "stabilityai/stable-diffusion-3.5-large-controlnet-depth"
DEFAULT_DEPTH_ESTIMATOR_MODEL = "depth-anything/Depth-Anything-V2-Large-hf"
DEFAULT_IMAGE_ENCODER = "google/siglip-so400m-patch14-384"
DEFAULT_IP_ADAPTER_CHECKPOINT = "InstantX/SD3.5-Large-IP-Adapter"
DEFAULT_IP_ADAPTER_WEIGHT_NAME = "ip-adapter.bin"

DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_STEPS = 40
DEFAULT_CANNY_SCALE = 1.0
DEFAULT_DEPTH_SCALE = 1.0
DEFAULT_SEED = 1234
DEFAULT_STRENGTH = 0.45


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

    patched_src = src.replace(old, ".reshape(batch_size, -1, attn.heads * head_dim)")
    namespace = {}
    exec(textwrap.dedent(patched_src), cls.__call__.__globals__, namespace)
    cls.__call__ = namespace["__call__"]
    print("Applied SD3 IP-Adapter reshape compatibility patch.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch SD3.5 Large img2img style transfer with Canny + Depth + IP-Adapter."
    )

    parser.add_argument("--prompt", required=True,
                        help="Prompt controlling the transferred style/content appearance.")
    parser.add_argument("--scale", type=float, required=True,
                        help="Prompt-transfer strength in [0, 1]. 0 is conservative; 1 is maximum.")
    parser.add_argument("-content_dir", "--content_dir", required=True, type=Path,
                        help="Directory containing source images.")
    parser.add_argument("--format", required=True,
                        help="Image extension to process, e.g. .jpeg, jpeg, .png.")
    parser.add_argument("--output_dir", required=True, type=Path,
                        help="Directory receiving the generated images.")
    parser.add_argument("--style_image", required=True, type=Path,
                        help="Reference style image used by IP-Adapter.")

    parser.add_argument("--sd3_model", type=str, default=DEFAULT_SD3_MODEL)
    parser.add_argument("--canny_model", type=str, default=DEFAULT_CANNY_MODEL)
    parser.add_argument("--depth_model", type=str, default=DEFAULT_DEPTH_MODEL)
    parser.add_argument("--depth_estimator_model", type=str, default=DEFAULT_DEPTH_ESTIMATOR_MODEL)
    parser.add_argument("--image_encoder_model", type=str, default=DEFAULT_IMAGE_ENCODER)
    parser.add_argument("--ip_adapter_checkpoint", type=str, default=DEFAULT_IP_ADAPTER_CHECKPOINT)
    parser.add_argument("--ip_adapter_weight_name", type=str, default=DEFAULT_IP_ADAPTER_WEIGHT_NAME)

    parser.add_argument("--ip_adapter_scale", type=float, default=0.35,
                        help="IP-Adapter conditioning strength (default: 0.35).")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--canny-scale", type=float, default=DEFAULT_CANNY_SCALE)
    parser.add_argument("--depth-scale", type=float, default=DEFAULT_DEPTH_SCALE)
    parser.add_argument("--strength", type=float, default=DEFAULT_STRENGTH,
                        help="Img2img denoising strength in [0,1]. Lower preserves content more strongly.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)

    args = parser.parse_args()

    if not 0.0 <= args.scale <= 1.0:
        parser.error("--scale must be between 0 and 1.")
    if not 0.0 <= args.canny_scale:
        parser.error("--canny-scale must be >= 0.")
    if not 0.0 <= args.depth_scale:
        parser.error("--depth-scale must be >= 0.")
    if not 0.0 <= args.ip_adapter_scale:
        parser.error("--ip_adapter_scale must be >= 0.")
    if not 0.0 <= args.strength <= 1.0:
        parser.error("--strength must be between 0 and 1.")
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
    # Preserve the original CLI behavior: scale [0,1] -> SD3 guidance [1,5].
    return 1.0 + 4.0 * scale


def resize_and_crop(image, size):
    """Resize without geometric distortion, then center-crop to the target size."""
    target_w, target_h = size
    src_w, src_h = image.size
    ratio = max(target_w / src_w, target_h / src_h)
    new_size = (round(src_w * ratio), round(src_h * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    left = (image.width - target_w) // 2
    top = (image.height - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


def make_canny(image):
    image_np = np.asarray(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)


def load_depth_estimator(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocessor = DepthPreprocessor.from_pretrained(args.depth_estimator_model)
    preprocessor = preprocessor.to(device)
    return preprocessor, device


def prepare_depth(image, preprocessor, size):
    depth = preprocessor(image, invert=True)[0].convert("RGB")
    return depth.resize(size, Image.Resampling.BILINEAR)


def load_pipeline(args):
    dtype = torch.float16

    canny_controlnet = SD3ControlNetModel.from_pretrained(
        args.canny_model, torch_dtype=dtype
    )
    depth_controlnet = SD3ControlNetModel.from_pretrained(
        args.depth_model, torch_dtype=dtype
    )
    controlnet = SD3MultiControlNetModel([canny_controlnet, depth_controlnet])

    feature_extractor = SiglipImageProcessor.from_pretrained(args.image_encoder_model)
    image_encoder = SiglipVisionModel.from_pretrained(
        args.image_encoder_model, torch_dtype=dtype
    )

    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.sd3_model,
        controlnet=controlnet,
        feature_extractor=feature_extractor,
        image_encoder=image_encoder,
        torch_dtype=dtype,
    ).to("cuda")

    pipe.load_ip_adapter(
        args.ip_adapter_checkpoint,
        weight_name=args.ip_adapter_weight_name,
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)

    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    return pipe


def encode_sd3_image(pipe, image, dtype, device):
    """
    Encode a content image using SD3's VAE convention.

    SD3 uses:
        latents = (vae_latents - shift_factor) * scaling_factor
    """
    image_tensor = pipe.image_processor.preprocess(
        image, height=image.height, width=image.width
    ).to(device=device, dtype=dtype)

    with torch.no_grad():
        latents = pipe.vae.encode(image_tensor).latent_dist.sample()

    latents = (
        latents - pipe.vae.config.shift_factor
    ) * pipe.vae.config.scaling_factor

    return latents.to(device=device, dtype=dtype)


def make_img2img_latents_and_timesteps(pipe, image, strength, steps, generator, dtype, device):
    """
    Prepare the true img2img starting point for the existing SD3 ControlNet
    text-to-image pipeline.

    The current SD3 ControlNet pipeline exposes `latents`, but not an
    `image`/`strength` img2img API. We therefore reproduce SD3's img2img
    initialization: encode the image, select the strength-dependent starting
    timestep, and add scheduler-consistent noise.
    """
    # Create the full scheduler schedule exactly as the pipeline does.
    pipe.scheduler.set_timesteps(steps, device=device)
    full_timesteps = pipe.scheduler.timesteps

    init_timestep = min(int(steps * strength), steps)
    t_start = max(steps - init_timestep, 0)

    timesteps = full_timesteps[t_start * pipe.scheduler.order:]
    if hasattr(pipe.scheduler, "set_begin_index"):
        pipe.scheduler.set_begin_index(t_start * pipe.scheduler.order)

    if len(timesteps) == 0:
        raise RuntimeError("Img2img strength produced an empty timestep schedule.")

    latent_timestep = timesteps[:1]

    init_latents = encode_sd3_image(pipe, image, dtype, device)
    noise = randn_tensor(
        init_latents.shape, generator=generator, device=device, dtype=dtype
    )
    latents = pipe.scheduler.scale_noise(init_latents, latent_timestep, noise)

    return latents, timesteps


def install_img2img_scheduler_schedule(pipe, timesteps):
    """
    The SD3 ControlNet pipeline currently has no public `strength` argument.
    It calls scheduler.set_timesteps() internally. Make that call a no-op so
    the already prepared full scheduler state and selected img2img timestep
    are retained.
    """
    original_set_timesteps = pipe.scheduler.set_timesteps

    def set_timesteps_preserve_img2img(*args, **kwargs):
        return None

    pipe.scheduler.set_timesteps = set_timesteps_preserve_img2img
    pipe.scheduler.timesteps = timesteps

    return original_set_timesteps


def process_image(
    pipe,
    depth_preprocessor,
    style_image,
    input_path,
    output_path,
    args,
    guidance_scale,
):
    original = Image.open(input_path).convert("RGB")
    image = resize_and_crop(original, (args.width, args.height))

    # All structural conditioning is generated from exactly the same
    # geometrically transformed image that is encoded for img2img.
    canny = make_canny(image)
    depth = prepare_depth(
        image, depth_preprocessor, (args.width, args.height)
    )

    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    dtype = torch.float16
    device = torch.device("cuda")

    # Prepare a true img2img latent at the strength-dependent timestep.
    latents, timesteps = make_img2img_latents_and_timesteps(
        pipe,
        image,
        args.strength,
        args.steps,
        generator,
        dtype,
        device,
    )

    # The stock SD3 ControlNet pipeline is text-to-image-only, so keep its
    # ControlNet/IP-Adapter implementation and feed it the correctly noised
    # content latent plus the truncated img2img schedule.
    original_set_timesteps = install_img2img_scheduler_schedule(pipe, timesteps)

    try:
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
                num_inference_steps=len(timesteps),
                guidance_scale=guidance_scale,
                generator=generator,
                latents=latents,
            )
    finally:
        pipe.scheduler.set_timesteps = original_set_timesteps

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

    input_files = sorted(
        p for p in content_dir.iterdir()
        if p.is_file() and p.suffix.lower() == image_format
    )

    if not input_files:
        raise SystemExit(
            f"No files with format '{image_format}' found in {content_dir}"
        )

    style_image = Image.open(args.style_image).convert("RGB")

    depth_preprocessor, _ = load_depth_estimator(args)
    pipe = load_pipeline(args)
    guidance_scale = transfer_to_guidance(args.scale)

    print(f"Found {len(input_files)} input file(s).")
    print(f"Prompt transfer scale: {args.scale}")
    print(f"SD3 guidance scale:    {guidance_scale:.3f}")
    print(f"IP-Adapter scale:      {args.ip_adapter_scale}")
    print(f"Img2img strength:      {args.strength}")
    print(f"Canny scale:           {args.canny_scale}")
    print(f"Depth scale:            {args.depth_scale}")
    print(f"Steps:                 {args.steps}")

    for index, input_path in enumerate(input_files, start=1):
        output_path = output_dir / input_path.name

        print(
            f"[{index}/{len(input_files)}] "
            f"{input_path.name} -> {output_path}"
        )

        process_image(
            pipe,
            depth_preprocessor,
            style_image,
            input_path,
            output_path,
            args,
            guidance_scale,
        )

    print(f"Finished. Wrote {len(input_files)} file(s) to {output_dir}")


if __name__ == "__main__":
    main()
