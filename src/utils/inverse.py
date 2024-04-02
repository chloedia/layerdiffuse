from __future__ import annotations

from torch import Tensor
from refiners.foundationals.latent_diffusion.stable_diffusion_xl import (
    StableDiffusion_XL,
)
from refiners.fluxion.utils import (
    no_grad,
)
import torch

from tqdm import tqdm
from PIL import Image
from pathlib import Path


from refiners.fluxion.utils import (
    no_grad,
    manual_seed,
)


def inversion_callback(latents: Tensor, ref_z:Tensor, step: int, offset: int = 0):
    latents[0] = ref_z[max(offset + 1, step + 1)].to(latents.device, latents.dtype)
    return latents


def inverse_ddim(sdxl : StableDiffusion_XL, prompt: str, img_path: Path, guidance_scale: int):
    no_grad().__enter__()
    manual_seed(1999)
    #calculate text embedding
    text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(prompt)

    time_ids = sdxl.default_time_ids
    #calculate image embedding
    ref_img = Image.open(img_path).convert("RGB").resize((1024, 1024))
    ref_latent = sdxl.lda.image_to_latents(ref_img)
    ref_latents: list[Tensor] = [ref_latent]
    #start loop
    for step in tqdm(sdxl.steps):
        timestep = sdxl.solver.timesteps[-step-1]
        x = torch.cat([ref_latent]*2, dim = 0)
        sdxl.set_unet_context(timestep = timestep.unsqueeze(0), clip_text_embedding= text_embedding, pooled_text_embedding=pooled_text_embedding, time_ids=time_ids)
        noise_pred = sdxl.unet(x)
        uncond_pred, cond_pred = noise_pred.chunk(2)
        noise_pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)

        timestep, next_timestep =  sdxl.solver.timesteps[-step-2] if step < sdxl.solver.num_inference_steps - 1 else torch.tensor(data=[999], device=timestep.device, dtype=timestep.dtype), timestep
        current_scale_factor,next_scale_factor = sdxl.solver.cumulative_scale_factors[timestep], sdxl.solver.cumulative_scale_factors[next_timestep]

        current_scale_factor, next_scale_factor =  current_scale_factor**2, next_scale_factor**2
        beta_prod_t = 1 - current_scale_factor
        next_original_sample = (ref_latent - beta_prod_t ** 0.5 * noise_pred) / current_scale_factor ** 0.5
        next_sample_direction = (1 - next_scale_factor) ** 0.5 * noise_pred
        ref_latent = next_scale_factor ** 0.5 * next_original_sample + next_sample_direction
        
        ref_latents.append(ref_latent)
    return torch.cat(ref_latents, dim= 0).flip(0)
        




if __name__ == "__main__":
    no_grad().__enter__()
    manual_seed(1999)

    DEVICE = torch.device("cuda")
    DTYPE = torch.bfloat16
    TEXT = "Hello World!"
    IMG_PATH = Path("./ref_image.png")
    GUIDANCE_SCALE = 5


    sdxl = StableDiffusion_XL(device=DEVICE, dtype=DTYPE)
    # sdxl.clip_text_encoder.load_from_safetensors(path / "text_encoder.safetensors")
    # sdxl.unet.load_from_safetensors(path / "unet.safetensors")
    # sdxl.lda.load_from_safetensors(path / "lda.safetensors")

    #calculate text embedding
    text_embedding, pooled_text_embedding = sdxl.compute_clip_text_embedding(TEXT)

    time_ids = sdxl.default_time_ids
    #calculate image embedding
    ref_img = Image.open(IMG_PATH).convert("RGB").resize((1024, 1024))
    ref_latent = sdxl.lda.image_to_latents(ref_img)
    #start loop
    for step in tqdm(sdxl.steps):
        timestep = sdxl.solver.timesteps[-step-1]
        x = torch.cat([ref_latent]*2, dim = 0)
        sdxl.set_unet_context(timestep = timestep.unsqueeze(0), clip_text_embedding= text_embedding, pooled_text_embedding=pooled_text_embedding, time_ids=time_ids)
        noise_pred = sdxl.unet(x)
        uncond_pred, cond_pred = noise_pred.chunk(2)
        noise_pred = uncond_pred + GUIDANCE_SCALE * (cond_pred * uncond_pred)

        timestep, next_timestep =  sdxl.solver.timesteps[-step-2] if step < sdxl.solver.num_inference_steps - 1 else torch.tensor(data=[999], device=timestep.device, dtype=timestep.dtype), timestep
        current_scale_factor,next_scale_factor = sdxl.solver.cumulative_scale_factors[timestep], sdxl.solver.cumulative_scale_factors[next_timestep]
        
        predicted_x = (ref_latent - torch.sqrt(1 - current_scale_factor**2) *noise_pred) / current_scale_factor
        #noise_factor = torch.sqrt(1 - previous_scale_factor**2)
        ref_latent = next_scale_factor**0.5 * ref_latent + (1 - next_scale_factor**2)**0.5 * predicted_x
        










        
        
    





 

    

    
