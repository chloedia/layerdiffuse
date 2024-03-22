import torch

from refiners.fluxion.utils import load_from_safetensors, manual_seed, no_grad, save_to_safetensors
from refiners.foundationals.latent_diffusion.lora import SDLoraManager
from refiners.foundationals.latent_diffusion.stable_diffusion_xl import (
    StableDiffusion_XL,
)

from layerdiffuse.models.models import UNet1024Refiners
from utils.utils import transform_keys

from PIL import Image

# Load SDXL
# sdxl = StableDiffusion_XL(device="cpu", dtype=torch.float16)
# sdxl.clip_text_encoder.load_from_safetensors("sdxl-weights/text_encoder.safetensors")
# sdxl.unet.load_from_safetensors("sdxl-weights/unet.safetensors")
# sdxl.lda.load_from_safetensors("sdxl-weights/lda.safetensors")

# Load LoRA weights from disk and inject them into target
# manager = SDLoraManager(sdxl)
ld_lora_weights = transform_keys(
    load_from_safetensors("layer_xl_transparent_attn.safetensors")
)
save_to_safetensors("layer_xl_transparent_attn_refined.safetensors", ld_lora_weights)

sci_fi_lora_weights = load_from_safetensors("sci-fi-lora.safetensors")
# manager.add_loras("ld-lora", tensors=ld_lora_weights)

print("ok")
# Load Layer diffuse decoder
ld_decoder = UNet1024Refiners()
ld_decoder.load_from_safetensors("vae_transparent_decoder.safetensors")


# Hyperparameters
prompt = "a futuristic magical panda with a purple glow, cyberpunk"
seed = 42
sdxl.set_inference_steps(50, first_step=0)
sdxl.set_self_attention_guidance(
    enable=True, scale=0.75
)  # Enable self-attention guidance to enhance the quality of the generated images

with no_grad():
    clip_text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(
        text=prompt + ", best quality, high quality",
        negative_text="monochrome, lowres, bad anatomy, worst quality, low quality",
    )
    time_ids = sdxl.default_time_ids

    manual_seed(seed=seed)

    # SDXL typically generates 1024x1024, here we use a higher resolution.
    x = sdxl.init_latents((1024, 1024)).to(sdxl.device, sdxl.dtype)
    ld_decoder.set_context("unet1024", {"latent": x})

    # Diffusion process
    for step in sdxl.steps:
        if step % 10 == 0:
            print(f"Step {step}")
        x = sdxl(
            x,
            step=step,
            clip_text_embedding=clip_text_embedding,
            pooled_text_embedding=pooled_text_embedding,
            time_ids=time_ids,
        )
    predicted_image = sdxl.lda.decode_latents(x)

    transparent_image = ld_decoder(predicted_image)

    predicted_image.save("outputs/origin_sdxl.png")
    Image.fromarray(transparent_image).save("outputs/transparent_sdxl.png")
