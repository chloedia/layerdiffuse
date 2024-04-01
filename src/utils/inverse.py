from __future__ import annotations
from typing import Callable

from typing import Any
from torch import Tensor
from refiners.foundationals.latent_diffusion.stable_diffusion_xl import (
    StableDiffusion_XL,
)
from refiners.fluxion.utils import (
    no_grad,
)
import torch
import numpy.typing as npt

from tqdm import tqdm


TN = Tensor | None
InversionCallback = Callable[
    [StableDiffusion_XL, int, Tensor, dict[str, Tensor]], dict[str, Tensor]
]


def _encode_text_sdxl(
    model: StableDiffusion_XL, prompt: str
) -> tuple[dict[str, Tensor], Tensor]:
    prompt_embeds, _ = model.compute_clip_text_embedding(prompt)
    prompt_embeds_2, pooled_prompt_embeds2 = model.compute_clip_text_embedding(prompt)
    prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_2), dim=-1)
    add_time_ids = model.default_time_ids
    added_cond_kwargs = {"text_embeds": pooled_prompt_embeds2, "time_ids": add_time_ids}
    return added_cond_kwargs, prompt_embeds


def _encode_text_sdxl_with_negative(
    model: StableDiffusion_XL, prompt: str
) -> tuple[dict[str, Tensor], Tensor]:
    added_cond_kwargs, prompt_embeds = _encode_text_sdxl(model, prompt)
    added_cond_kwargs_uncond, prompt_embeds_uncond = _encode_text_sdxl(model, "")
    prompt_embeds = torch.cat(
        (
            prompt_embeds_uncond,
            prompt_embeds,
        )
    )
    added_cond_kwargs = {
        "text_embeds": torch.cat(
            (added_cond_kwargs_uncond["text_embeds"], added_cond_kwargs["text_embeds"])
        ),
        "time_ids": torch.cat(
            (added_cond_kwargs_uncond["time_ids"], added_cond_kwargs["time_ids"])
        ),
    }
    return added_cond_kwargs, prompt_embeds


def _next_step(
    model: StableDiffusion_XL, model_output: Tensor, timestep: int, sample: Tensor
) -> Tensor:
    timestep, next_timestep = (
        min(
            timestep
            - model.solver.params.num_train_timesteps
            // model.solver.num_inference_steps,
            999,
        ),
        timestep,
    )
    alpha_prod_t = (
        model.solver.cumulative_scale_factors[int(timestep)]
        if timestep >= 0
        else model.solver.cumulative_scale_factors[0]
    )
    alpha_prod_t_next = model.solver.cumulative_scale_factors[int(next_timestep)]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (
        sample - beta_prod_t**0.5 * model_output
    ) / alpha_prod_t**0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
    return next_sample


def _get_noise_pred(
    model: StableDiffusion_XL,
    latent: Tensor,
    step: Tensor,
    clip_text_embedding: Tensor,
    guidance_scale: float,
    added_cond_kwargs: dict[str, Tensor],
):
    pooled_text_embedding = added_cond_kwargs["text-embed"]
    time_ids = added_cond_kwargs["time_ids"]
    latents_input = torch.cat([latent] * 2)
    timestep = model.solver.timesteps[step].unsqueeze(dim=0)
    model.set_unet_context(
        timestep=timestep,
        clip_text_embedding=clip_text_embedding,
        pooled_text_embedding=pooled_text_embedding,
        time_ids=time_ids,
    )
    noise_pred = model.unet(latents_input)
    noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_prediction_text - noise_pred_uncond
    )
    return noise_pred


def make_inversion_callback(
    zts: Tensor, offset: int = 0
) -> tuple[Tensor, InversionCallback]:
    def callback_on_step_end(
        pipeline: StableDiffusion_XL,
        i: int,
        t: Tensor,
        callback_kwargs: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        latents = callback_kwargs["latents"]
        latents[0] = zts[max(offset + 1, i + 1)].to(latents.device, latents.dtype)
        return {"latents": latents}

    return zts[offset], callback_on_step_end


def ddim_loop(
    model: StableDiffusion_XL,
    z0: Tensor,
    prompt: str,
    guidance_scale: int,
    time_ids: Tensor,
) -> Tensor:
    all_latent = [z0]
    added_cond_kwargs, text_embedding = _encode_text_sdxl_with_negative(model, prompt)
    latent = z0.clone().detach().half()
    for i in tqdm(model.steps):
        t = time_ids[len(time_ids) - i - 1]  # gros doute sur time ids
        noise_pred = _get_noise_pred(
            model, latent, t, text_embedding, guidance_scale, added_cond_kwargs
        )
        latent = _next_step(model, noise_pred, int(t), latent)
        all_latent.append(latent)
    return torch.cat(all_latent).flip(0)


def inverse_ddim(
    model: StableDiffusion_XL,
    ref_img: npt.NDArray[Any],
    prompt: str,
    guidance_scale: int,
) -> Tensor:
    with no_grad():
        # encode ref image
        ref_tensor = torch.from_numpy(ref_img).float() / 255.0  # type: ignore
        ref_tensor = (ref_tensor * 2 - 1).permute(2, 0, 1).unsqueeze(0)
        img_latent = model.lda.encode(ref_tensor) * model.lda.encoder_scale
        time_ids = model.default_time_ids
        noisy_ref = ddim_loop(model, img_latent, prompt, guidance_scale, time_ids)
    return noisy_ref
