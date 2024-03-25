from typing import Any
from torch import Tensor, float16, cat, flip, from_numpy, rot90, stack, median, device as Device, dtype as DType #type: ignore
from tqdm import tqdm
import refiners.fluxion.layers as fl
from refiners.fluxion.context import Contexts
from refiners.fluxion.layers import SelfAttention2d
#from refiners.foundationals.latent_diffusion.auto_encoder import Resnet
from utils.utils import checkerboard #type: ignore
import numpy as np


import cv2

def zero_module(module: fl.Module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class LatentTransparencyOffsetEncoderRefiners(fl.Chain):
    """
    Predict the Latent Transparency Offset Encoder to hide transparency informations.
    """

    def __init__(self):
        super().__init__(
            fl.Conv2d(4, 32, kernel_size=3, padding=1, stride=1),
            fl.SiLU(),
            fl.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            fl.SiLU(),
            fl.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            fl.SiLU(),
            fl.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            fl.SiLU(),
            fl.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            fl.SiLU(),
            fl.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            fl.SiLU(),
            fl.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            fl.SiLU(),
            fl.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            fl.SiLU(),
            fl.Conv2d(256, 4, kernel_size=3, padding=1, stride=1),
        )

class Resnet(fl.Sum):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 32,
        device: Device | str | None = None,
        dtype: DType | None = None ,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        shortcut = (
            fl.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device, dtype=dtype)
            if in_channels != out_channels
            else fl.Identity()
        )
        super().__init__(
            fl.Chain(
                fl.GroupNorm(channels=in_channels, num_groups=num_groups, device=device, dtype=dtype),
                fl.SiLU(),
                fl.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
                fl.GroupNorm(channels=out_channels, num_groups=num_groups, device=device, dtype=dtype),
                fl.SiLU(),
                fl.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            ),
            shortcut,
        )
class DownBlock2D(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        super().__init__(
            fl.SetContext("unet1024", "residuals", callback=lambda l, x: l.append(x)),
            fl.Chain(
                Resnet(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    num_groups=self.num_groups,
                    device=device,
                    dtype=dtype,
                ),
                fl.SetContext("unet1024", "residuals", callback=lambda l, x: l.append(x)),
            ),
            fl.Chain(
                Resnet(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    num_groups=self.num_groups,
                    device=device,
                    dtype=dtype,
                ),
                fl.SetContext("unet1024", "residuals", callback=lambda l, x: l.append(x)),
            ),
            fl.Chain(
                fl.Downsample(channels=self.out_channels, scale_factor=2, padding=1, device=device, dtype=dtype),
            ),
        )

    def monitor(self, x: Tensor) -> Tensor:
        print("DownBlock Input : ", x.size())
        return x


class AttnDownBlock2D(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        add_downsample: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups

        super().__init__(
            fl.Chain(
                fl.SetContext("unet1024", "residuals", callback=lambda l, x: l.append(x)),
                Resnet(
                    in_channels=self.in_channels,
                    num_groups=self.num_groups,
                    out_channels=self.out_channels,
                    device=device,
                    dtype=dtype,
                ),
                fl.Residual(
                    fl.GroupNorm(channels=self.out_channels, num_groups=self.num_groups),
                    SelfAttention2d(
                        channels=self.out_channels,
                        num_heads=self.out_channels // 8,
                        device=device,
                        dtype=dtype,
                    ),
                ),
                fl.SetContext("unet1024", "residuals", callback=lambda l, x: l.append(x)),
            ),
            fl.Chain(
                Resnet(
                    in_channels=self.out_channels,
                    out_channels=self.out_channels,
                    num_groups=self.num_groups,
                    device=device,
                    dtype=dtype,
                ),
                fl.Residual(
                    fl.GroupNorm(channels=self.out_channels, num_groups=self.num_groups),
                    SelfAttention2d(
                        channels=self.out_channels,
                        num_heads=self.out_channels // 8,
                        device=device,
                        dtype=dtype,
                    ),
                ),
                fl.SetContext("unet1024", "residuals", callback=lambda l, x: l.append(x)),
            ),
            fl.Chain(
                fl.Downsample(
                    channels=self.out_channels,
                    scale_factor=2,
                    padding=1,
                    device=device,
                    dtype=dtype,
                ),
            )
            if add_downsample
            else fl.Identity(),
        )


class AttnUpBlock2D(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_out_channels: int,
        num_groups: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.prev_out_channels = prev_out_channels

        super().__init__(
            fl.Concatenate(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext("unet1024", "residuals").compose(lambda x: x.pop()),
                ),
                dim=1,
            ),
            fl.Chain(
                Resnet(
                    in_channels=self.prev_out_channels + self.out_channels,
                    out_channels=self.out_channels,
                    num_groups=self.num_groups,
                    device=device,
                    dtype=dtype,
                ),
                fl.Residual(
                    fl.GroupNorm(channels=self.out_channels, num_groups=self.num_groups),
                    SelfAttention2d(
                        channels=self.out_channels,
                        num_heads=self.out_channels // 8,
                        device=device,
                        dtype=dtype,
                    ),
                ),
            ),
            fl.Concatenate(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext("unet1024", "residuals").compose(lambda x: x.pop()),
                ),
                dim=1,
            ),
            fl.Chain(
                Resnet(
                    in_channels=self.out_channels * 2,
                    out_channels=self.out_channels,
                    num_groups=self.num_groups,
                    device=device,
                    dtype=dtype,
                ),
                fl.Residual(
                    fl.GroupNorm(channels=self.out_channels, num_groups=self.num_groups),
                    SelfAttention2d(
                        channels=self.out_channels,
                        num_heads=self.out_channels // 8,
                        device=device,
                        dtype=dtype,
                    ),
                ),
            ),
            fl.Concatenate(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext("unet1024", "residuals").compose(lambda x: x.pop()),
                ),
                dim=1,
            ),
            fl.Chain(
                Resnet(
                    in_channels=self.in_channels + self.out_channels,
                    out_channels=self.out_channels,
                    num_groups=self.num_groups,
                    device=device,
                    dtype=dtype,
                ),
                fl.Residual(
                    fl.GroupNorm(channels=self.out_channels, num_groups=self.num_groups),
                    SelfAttention2d(
                        channels=self.out_channels,
                        num_heads=self.out_channels // 8,
                        device=device,
                        dtype=dtype,
                    ),
                ),
            ),
            fl.Upsample(channels=self.out_channels, device=device, dtype=dtype),
        )

    def monitor(self, x: Tensor) -> Tensor:
        print("x_up : ", x.size())
        return x


class UpBlock2D(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_out_channels: int,
        num_groups: int,
        add_upsample: bool = True,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        self.prev_out_channels = prev_out_channels

        super().__init__(
            fl.Concatenate(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext("unet1024", "residuals").compose(lambda x: x.pop()),
                ),
                dim=1,
            ),
            fl.Chain(
                Resnet(
                    in_channels=self.prev_out_channels + self.out_channels,
                    out_channels=self.out_channels,
                    num_groups=self.num_groups,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Concatenate(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext("unet1024", "residuals").compose(lambda x: x.pop()),
                ),
                dim=1,
            ),
            fl.Chain(
                Resnet(
                    in_channels=self.out_channels + self.out_channels,
                    out_channels=self.out_channels,
                    num_groups=self.num_groups,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Concatenate(
                fl.Identity(),
                fl.Chain(
                    fl.UseContext("unet1024", "residuals").compose(lambda x: x.pop()),
                ),
                dim=1,
            ),
            fl.Chain(
                Resnet(
                    in_channels=self.in_channels + self.out_channels,
                    out_channels=self.out_channels,
                    num_groups=self.num_groups,
                    device=device,
                    dtype=dtype,
                ),
            ),
            fl.Chain(
                fl.Upsample(channels=self.out_channels, device=device, dtype=dtype),
            )
            if add_upsample
            else fl.Identity(),
        )

    def monitor(self, x: Tensor) -> Tensor:
        print("x_up2 : ", x.size())
        return x


class MiddleBlock(fl.Chain):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        device: Device | str | None = None,
        dtype: DType | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_groups = num_groups
        super().__init__(
            Resnet(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                num_groups=self.num_groups,
                device=device,
                dtype=dtype,
            ),
            fl.Residual(
                fl.GroupNorm(channels=self.out_channels, num_groups=self.num_groups),
                SelfAttention2d(
                    channels=self.out_channels,
                    num_heads=self.out_channels // 8,
                    device=device,
                    dtype=dtype,
                ),
            ),
            Resnet(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                num_groups=self.num_groups,
                device=device,
                dtype=dtype,
            ),
        )


# 1024 * 1024 * 3 -> 16 * 16 * 512 -> 1024 * 1024 * 3
class UNet1024Refiners(fl.Chain):
    def __init__(self, out_channels: int = 3, device: Device | str | None = "cuda", dtype: DType | None = None) -> None:
        self.out_channels = out_channels
        super().__init__(
            fl.Passthrough(
                fl.UseContext("unet1024", "latent"),
                fl.Conv2d(4, 64, kernel_size=1),
                fl.SetContext("unet1024", "latent_repr"),
            ),
            fl.Conv2d(3, 32, kernel_size=3, padding=(1, 1)),
            fl.Sum(
                fl.Chain(
                    DownBlock2D(32, 32, num_groups=4, device=device, dtype=dtype),
                    DownBlock2D(32, 32, num_groups=4, device=device, dtype=dtype),
                    DownBlock2D(32, 64, num_groups=4, device=device, dtype=dtype),
                ),
                fl.UseContext("unet1024", "latent_repr"),
            ),
            fl.Chain(
                DownBlock2D(64, 128, num_groups=4, device=device, dtype=dtype),
                AttnDownBlock2D(128, 256, num_groups=4, add_downsample=True, device=device, dtype=dtype),
                AttnDownBlock2D(256, 512, num_groups=4, add_downsample=True, device=device, dtype=dtype),
                AttnDownBlock2D(512, 512, num_groups=4, add_downsample=False, device=device, dtype=dtype),
            ),
            MiddleBlock(512, 512, num_groups=4, device=device, dtype=dtype),
            fl.Chain(
                AttnUpBlock2D(512, 512, prev_out_channels=512, num_groups=4, device=device, dtype=dtype),
                AttnUpBlock2D(256, 512, prev_out_channels=512, num_groups=4, device=device, dtype=dtype),
                AttnUpBlock2D(128, 256, prev_out_channels=512, num_groups=4, device=device, dtype=dtype),
                UpBlock2D(64, 128, prev_out_channels=256, num_groups=4, device=device, dtype=dtype),
                UpBlock2D(32, 64, prev_out_channels=128, num_groups=4, device=device, dtype=dtype),
                UpBlock2D(32, 32, prev_out_channels=64, num_groups=4, device=device, dtype=dtype),
                UpBlock2D(32, 32, prev_out_channels=32, num_groups=4, add_upsample=False, device=device, dtype=dtype),
            ),
            fl.GroupNorm(channels=32, num_groups=4),
            fl.SiLU(),
            fl.Conv2d(32, self.out_channels, kernel_size=3, padding=1),
        )

    def monitor(self, x: Tensor) -> Tensor:
        print("DownBlockOutput : ", x.size())
        return x

    def nextstep(self, x: Tensor) -> Tensor:
        print("Finished a block")
        return x

    def init_context(self) -> Contexts:
        return {
            "unet1024": {"residuals": []},
            "sampling": {"shapes": []},
        }


class TransparentVAEDecoder:
    def __init__(self, state_dict: str):
        self.model = UNet1024Refiners(out_channels=4)
        self.model.load_from_safetensors(state_dict)
        self.model.to("cuda", float16)
    
    def postprocess(self, y: Tensor) -> tuple[Tensor, Any]:
        y = y.clip(0, 1).movedim(1, -1)
        alpha = y[..., :1]
        fg = y[..., 1:]

        _, H, W, _ = fg.shape
        cb = checkerboard(shape=(H // 64, W // 64)) #type: ignore
        cb = cv2.resize(cb, (W, H), interpolation=cv2.INTER_NEAREST) #type: ignore
        cb = (0.5 + (cb - 0.5) * 0.1)[None, ..., None]#type: ignore
        cb = from_numpy(cb).to(fg) #type: ignore

        vis = fg * alpha + cb * (1 - alpha)

        png = cat([fg, alpha], dim=3)[0]
        png = (png * 255.0).detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)

        return vis.movedim(-1,1), png

    def run(self, pixel: Tensor, latent: Tensor) -> tuple[Tensor, Any]:
        args = [
            [False, 0], [False, 1], [False, 2], [False, 3], [True, 0], [True, 1], [True, 2], [True, 3],
        ]

        result: list[Tensor] = []

        for doflip, rok in tqdm(args):
            feed_pixel = pixel.clone()
            feed_latent = latent.clone()

            if doflip:
                feed_pixel = flip(feed_pixel, dims=(3,))
                feed_latent = flip(feed_latent, dims=(3,))

            feed_pixel = rot90(feed_pixel, k=rok, dims=(2, 3))
            feed_latent = rot90(feed_latent, k=rok, dims=(2, 3))

            self.model.set_context("unet1024", {"latent": feed_latent})
            eps = self.model(feed_pixel).clip(0,1)
            eps = rot90(eps, k=-rok, dims=(2, 3))
            if doflip:
                eps = flip(eps, dims=(3,))

            result += [eps]

        stacked_result = stack(result, dim=0)
        _median = median(stacked_result, dim=0).values
        return self.postprocess(_median)

