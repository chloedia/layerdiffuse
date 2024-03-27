from pathlib import Path
import torch
import argparse

from refiners.fluxion.utils import (
    load_from_safetensors,
    manual_seed,
    no_grad,
    images_to_tensor,
)
from refiners.foundationals.latent_diffusion.lora import SDLoraManager
from refiners.foundationals.latent_diffusion.stable_diffusion_xl import (
    StableDiffusion_XL,
)

from layerdiffuse.models import TransparentDecoder
from layerdiffuse.utils import load_frozen_patcher

from PIL import Image


def dtype_from_str(dtype: str) -> torch.dtype:
    match dtype:
        case "float16":
            return torch.float16
        case "bfloat16":
            return torch.bfloat16
        case "float32":
            return torch.float32
        case _:
            raise ValueError(f"Invalid dtype: {dtype}")


def load_models(
    path: Path, /, device: torch.device, dtype: torch.dtype
) -> tuple[StableDiffusion_XL, TransparentDecoder]:
    assert path.exists(), f"Checkpoint path {path} does not exist"
    checkpoints = [
        "unet.safetensors",
        "text_encoder.safetensors",
        "lda.safetensors",
        "vae_transparent_decoder.safetensors",
        "layer_xl_transparent_attn.safetensors",
    ]
    for checkpoint in checkpoints:
        assert (
            path / checkpoint
        ).is_file(), f"Checkpoint {path / checkpoint} does not exist"

    sdxl = StableDiffusion_XL(device=device, dtype=dtype)
    sdxl.clip_text_encoder.load_from_safetensors(path / "text_encoder.safetensors")
    sdxl.unet.load_from_safetensors(path / "unet.safetensors")
    sdxl.lda.load_from_safetensors(path / "lda.safetensors")

    lora_weights = load_from_safetensors(path / "layer_xl_transparent_attn.safetensors")
    lora_weights = load_frozen_patcher(lora_weights)
    manager = SDLoraManager(sdxl)
    manager.add_loras(
        "transparent-diffuse",
        tensors=lora_weights,
        unet_inclusions=["Attention", "SelfAttention"],
    )

    transparent_decoder = TransparentDecoder(device=device, dtype=dtype)
    transparent_decoder.load_from_safetensors(
        path / "vae_transparent_decoder.safetensors"
    )

    return sdxl, transparent_decoder


class Params(argparse.Namespace):
    checkpoint_path: str = "checkpoints/"
    prompt: str = "a futuristic magical panda with a purple glow, cyberpunk"
    seed: int = -1
    steps: int = 50
    device: str = "cuda"
    dtype: str = "bfloat16"
    save_path: str = "outputs/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/")
    parser.add_argument(
        "--prompt",
        type=str,
        default="a futuristic magical panda with a purple glow, cyberpunk",
    )
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--save_path", type=str, default="outputs/")
    args = parser.parse_args(namespace=Params)

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint_path)
    device = torch.device(args.device)
    dtype = dtype_from_str(args.dtype)
    sdxl, transparent_decoder = load_models(checkpoint_path, device=device, dtype=dtype)
    manual_seed(seed=args.seed)

    with no_grad():
        clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
            text=args.prompt + ", best quality, high quality",
            negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
        )
        time_ids = sdxl.default_time_ids
        x = sdxl.init_latents((1024, 1024)).to(device=device, dtype=dtype)

        for step in sdxl.steps:
            x = sdxl(
                x,
                step=step,
                clip_text_embedding=clip_text_embedding,
                pooled_text_embedding=pooled_text_embedding,
                time_ids=time_ids,
            )
        latent = x
        pixel = sdxl.lda.decode_latents(x)
        pixel.save(save_path / "original_output.png")

        pixel = images_to_tensor([pixel], dtype=dtype, device=device)
        _, transparent_image_png = transparent_decoder.run(pixel, latent)
        Image.fromarray(transparent_image_png).save(
            save_path / "transparent_output.png"
        )


if __name__ == "__main__":
    main()
