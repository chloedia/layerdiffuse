from pathlib import Path
import torch
import safetensors
from typing import Dict, Any

import numpy as np
import os
from urllib.parse import urlparse


def load_torch_file(ckpt, safe_load=False, device=None) -> dict:  # type: ignore
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):  # type: ignore
        sd = safetensors.torch.load_file(ckpt, device=device.type)  # type: ignore
    else:
        if safe_load:
            if "weights_only" not in torch.load.__code__.co_varnames:  # type: ignore
                print(
                    "Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely."
                )
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)  # type: ignore
        else:
            raise Exception("The document is not a safetensor")

        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd  # type: ignore


def checkerboard(shape: tuple[int, int]) -> np.ndarray:  # type: ignore
    return np.indices(shape).sum(axis=0) % 2


def load_frozen_patcher(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    patch_dict: Dict[str, Any] = {}
    for k, w in state_dict.items():
        model_key, patch_type, weight_index = k.split("::")
        if model_key not in patch_dict:
            patch_dict[model_key] = {}
        if patch_type not in patch_dict[model_key]:
            patch_dict[model_key][patch_type] = [None] * 16
        patch_dict[model_key][patch_type][int(weight_index)] = w

    patch_flat: Dict[str, Any] = {}
    for model_key, v in patch_dict.items():
        for patch_type, weight_list in v.items():
            patch_flat[f"{model_key[:-7]}.up.weight"] = weight_list[1]
            patch_flat[f"{model_key[:-7]}.down.weight"] = weight_list[0]

    return patch_flat


def load_file_from_url(
    url: str,
    *,
    model_dir: str | Path,
    progress: bool = True,
    file_name: str | None = None,
) -> None:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
