import torch
import safetensors
import re
from typing import Dict, Any


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


def transform_keys(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    transformed_dict: Dict[str, Any] = {}
    for key, value in input_dict.items():
        # Extract the components of the key using regex
        match = re.match(
            r"(.*?)\.(\d+)\.(\d+)\.(\w+)\.(\d+)\.(\w+)\.(\w+)\.(\w+)::lora::(\d+)", key
        )
        if match:
            # Construct the new key format
            new_key = (
                f"lora_{match.group(1).replace('_', '')}_{match.group(2)}_{match.group(3)}_"
                f"{match.group(4)}s_{match.group(5)}_{match.group(6)}_{match.group(7)}"
            )
            if match.group(9) == "0":
                new_key += ".lora_up.weight"
            else:
                new_key += ".lora_down.weight"
            if new_key.endswith("proj_in.lora_up.weight"):
                new_key = new_key.replace("proj_in.lora_up.weight", "proj_in.alpha")
            elif new_key.endswith("proj_out.lora_up.weight"):
                new_key = new_key.replace("proj_out.lora_up.weight", "proj_out.alpha")
            transformed_dict[new_key] = value
    return transformed_dict
