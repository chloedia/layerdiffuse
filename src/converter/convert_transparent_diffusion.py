# pyright: reportPrivateUsage=false
import argparse
from pathlib import Path

import torch
from torch import nn

from refiners.fluxion.model_converter import ModelConverter
from layerdiffuse.models.models import (
    LatentTransparencyOffsetEncoderRefiners,
    UNet1024Refiners,
)
from source_layerdiffuse.lib_layerdiffusion.models import (
    LatentTransparencyOffsetEncoder,
    UNet1024,
)
from utils import load_torch_file  # type: ignore


def convert_unet1024(args: argparse.Namespace) -> None:
    target = UNet1024Refiners(out_channels=4)
    # low_cpu_mem_usage=False stops some annoying console messages us to `pip install accelerate`
    # source: nn.Module = UNet1024.from_pretrained(  # type: ignore
    #     pretrained_model_name_or_path=args.source_path_unet,
    #     low_cpu_mem_usage=False,
    # )
    source: nn.Module = UNet1024(out_channels=4)
    source.load_state_dict(  # type: ignore
        load_torch_file(args.source_path_unet)  # type: ignore
    )
    assert isinstance(source, nn.Module), "Source model is not a nn.Module"

    x = torch.randn(1, 3, 1024, 1024)
    latent = torch.randn(1, 4, 128, 128)

    target.set_context("unet1024", {"latent": latent})

    converter = ModelConverter(
        source_model=source, target_model=target, verbose=args.verbose, threshold=10
    )
    if not converter.run(source_args=(x, latent), target_args=(x,)):
        raise RuntimeError("Model conversion failed")

    converter.save_to_safetensors(path=args.output_path_unet, half=args.half)


def convert_offset_encoder(args: argparse.Namespace) -> None:
    target = LatentTransparencyOffsetEncoderRefiners()
    # low_cpu_mem_usage=False stops some annoying console messages us to `pip install accelerate`
    source: nn.Module = LatentTransparencyOffsetEncoder()

    source.load_state_dict(
        load_torch_file(args.source_path_offset)  # type: ignore
    )
    assert isinstance(source, nn.Module), "Source model is not a nn.Module"

    x = torch.randn(1, 4, 1024, 1024)

    converter = ModelConverter(
        source_model=source,
        target_model=target,
        verbose=args.verbose,
        skip_output_check=True,
    )
    if not converter.run(source_args=(x,)):
        raise RuntimeError("Model conversion failed")

    converter.save_to_safetensors(path=args.output_path_offset, half=args.half)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converts a Layer diffusers model to refiners."
    )
    parser.add_argument(
        "--unet-from",
        type=str,
        required=True,
        dest="source_path_unet",
        help="Path to the unet source model. (e.g.: 'layer-diffuser_sdxl.bin').",
    )
    parser.add_argument(
        "--offset-from",
        type=str,
        required=True,
        dest="source_path_offset",
        help="Path to the offset encoder source model. (e.g.: 'layer-diffuser_sdxl.bin').",
    )
    parser.add_argument(
        "--to-unet",
        type=str,
        dest="output_path_unet",
        default=None,
        help=(
            "Path to save the converted model. If not specified, the output path will be the source path with the"
            " extension changed to .safetensors."
        ),
    )

    parser.add_argument(
        "--to-offset",
        type=str,
        dest="output_path_offset",
        default=None,
        help=(
            "Path to save the converted model. If not specified, the output path will be the source path with the"
            " extension changed to .safetensors."
        ),
    )
    parser.add_argument("--verbose", action="store_true", dest="verbose")
    parser.add_argument("--half", action="store_true", dest="half")
    args = parser.parse_args()

    # if not args.source_path_offset:
    #     download_file(
    #         url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_encoder.safetensors",
    #         dest_folder=".",
    #         filename="vae_transparent_encoder.safetensors",
    #     )
    #     args.source_path_offset = "./vae_transparent_encoder.safetensors"

    # if not args.source_path_unet:
    #     download_file(
    #         url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors",
    #         dest_folder=".",
    #         filename="vae_transparent_decoder.safetensors",
    #     )
    #     args.source_path_unet = "./vae_transparent_decoder.safetensors"

    if args.output_path_offset is None:
        args.output_path_offset = f"{Path(args.source_path_offset).stem}.safetensors"

    if args.output_path_unet is None:
        args.output_path_unet = f"{Path(args.source_path_unet).stem}.safetensors"

    assert args.output_path_unet is not None and args.output_path_offset is not None

    print("Converting the layer diffuse encoder ...")
    convert_offset_encoder(args)

    print("Converting the layer diffuse decoder ...")
    convert_unet1024(args)

    print("You're good to go Marty !")


if __name__ == "__main__":
    main()
