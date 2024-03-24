import torch
import safetensors
from typing import Dict, Any

import numpy as np


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


def modify_dict(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    modified_dict = {}
    matrix_down = None
    for key in input_dict:
        base_key = key.split("::lora::")[0]
        if "::lora::0" in key:
            matrix_down = input_dict[key]
        elif "::lora::1" in key:
            lora_0_key = base_key + "::lora::0"
            lora_1_key = base_key + "::lora::1"
            modified_dict[lora_0_key] = input_dict[key]
            modified_dict[lora_1_key] = matrix_down
            matrix_down = None
        else:
            modified_dict[key] = input_dict[key]
    return modified_dict #type: ignore

def checkerboard(shape: tuple[int, int]) -> np.ndarray: #type: ignore
   return np.indices(shape).sum(axis=0) % 2
