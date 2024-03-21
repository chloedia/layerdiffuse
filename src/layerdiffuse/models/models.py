from torch import Tensor, device as Device, dtype as DType

import refiners.fluxion.layers as fl
from refiners.fluxion.context import Contexts
from refiners.fluxion.layers import SelfAttention2d
from refiners.foundationals.latent_diffusion.auto_encoder import Resnet


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
    def __init__(self, out_channels: int = 3, device: Device | str | None = None, dtype: DType | None = None) -> None:
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
