import pytest
import torch
import torch.nn as nn
from diffusers.models.attention_processor import Attention
from diffusers.models.resnet import ResnetBlock2D

import refiners.fluxion.layers as fl
from refiners.fluxion.layers import SelfAttention2d
from refiners.fluxion.model_converter import ModelConverter
from refiners.fluxion.utils import norm
from refiners.foundationals.latent_diffusion.auto_encoder import Resnet
from refiners.foundationals.source_layerdiffuse.lib_layerdiffusion.unet_2d_blocks import get_down_block, UNetMidBlock2D
from refiners.foundationals.layer_diffuse.models import AttnDownBlock2D


@pytest.fixture(scope="module")
def attention_refiners_fixture() -> fl.Module:
    model = fl.Chain(
        fl.Residual(
            fl.GroupNorm(channels=256, num_groups=4),
            SelfAttention2d(
                channels=256,
                num_heads=256 // 8,
            ),
        ),
    )
    return model


@pytest.fixture(scope="module")
def attention_diffusers_fixture() -> nn.Module:
    model = get_down_block(
        "AttnDownBlock2D",
        num_layers=1,
        in_channels=128,
        out_channels=256,
        temb_channels=None,
        add_downsample=False,
        resnet_eps=1e-5,
        resnet_act_fn="silu",
        resnet_groups=4,
        attention_head_dim=8,
        downsample_padding=1,
        resnet_time_scale_shift="default",
        downsample_type=None,
        dropout=0.0,
    )

    return model.attentions[0]


def test_attention_layer(attention_refiners_fixture: fl.Module, attention_diffusers_fixture: nn.Module) -> None:
    x = torch.randn(1, 256, 64, 64)

    converter = ModelConverter(
        source_model=attention_refiners_fixture, target_model=attention_diffusers_fixture, skip_output_check=False
    )

    assert converter.run(source_args=(x,), target_args=(x,))
    assert norm(converter.source_model(x) - converter.target_model(x, None)).item() == 0.0


@pytest.fixture(scope="module")
def res_refiners_fixture() -> fl.Module:
    model = fl.Chain(
        Resnet(
            in_channels=128,
            num_groups=4,
            out_channels=256,
        )
    )
    return model


@pytest.fixture(scope="module")
def res_diffusers_fixture() -> nn.Module:
    model = get_down_block(
        "AttnDownBlock2D",
        num_layers=1,
        in_channels=128,
        out_channels=256,
        temb_channels=None,
        add_downsample=False,
        resnet_eps=1e-5,
        resnet_act_fn="silu",
        resnet_groups=4,
        attention_head_dim=8,
        downsample_padding=1,
        resnet_time_scale_shift="default",
        downsample_type=None,
        dropout=0.0,
    )

    return model.resnets[0]


def test_res_layer(res_refiners_fixture: fl.Module, res_diffusers_fixture: nn.Module) -> None:
    x = torch.randn(1, 128, 64, 64)

    converter = ModelConverter(
        source_model=res_refiners_fixture, target_model=res_diffusers_fixture, skip_output_check=False
    )

    assert converter.run(source_args=(x,), target_args=(x, None))
    assert norm(converter.source_model(x) - converter.target_model(x, None)).item() == 0.0


@pytest.fixture(scope="module")
def downsample_refiners_fixture() -> fl.Module:
    model = fl.Downsample(channels=256, scale_factor=2, padding=1)
    model.set_context("sampling", {"shapes": []})
    return model


@pytest.fixture(scope="module")
def downsample_diffusers_fixture() -> nn.Module:
    model = get_down_block(
        "AttnDownBlock2D",
        num_layers=2,
        in_channels=128,
        out_channels=256,
        temb_channels=None,
        add_downsample=True,
        resnet_eps=1e-5,
        resnet_act_fn="silu",
        resnet_groups=4,
        attention_head_dim=8,
        downsample_padding=1,
        resnet_time_scale_shift="default",
        downsample_type=None,
        dropout=0.0,
    )

    return model.downsamplers[0]


def test_downsample_layer(downsample_refiners_fixture: fl.Module, downsample_diffusers_fixture: nn.Module) -> None:
    x = torch.randn(1, 256, 64, 64)

    converter = ModelConverter(
        source_model=downsample_refiners_fixture, target_model=downsample_diffusers_fixture, skip_output_check=False
    )

    assert converter.run(source_args=(x,), target_args=(x, None))
    assert norm(converter.source_model(x) - converter.target_model(x, None)).item() == 0.0


@pytest.fixture(scope="module")
def attn_down_block_refiners() -> fl.Module:
    model = fl.Chain(
        Resnet(
            in_channels=128,
            num_groups=4,
            out_channels=256,
        ),
        fl.Residual(
            fl.GroupNorm(channels=256, num_groups=4),
            SelfAttention2d(
                channels=256,
                num_heads=256 // 8,
            ),
        ),
        Resnet(
            in_channels=256,
            num_groups=4,
            out_channels=256,
        ),
        fl.Residual(
            fl.GroupNorm(channels=256, num_groups=4),
            SelfAttention2d(
                channels=256,
                num_heads=256 // 8,
            ),
        ),
        fl.Downsample(
            channels=256,
            scale_factor=2,
            padding=1,
        ),
    )
    model.set_context("sampling", {"shapes": []})
    return model


@pytest.fixture(scope="module")
def attn_down_block_diffuser() -> nn.Module:
    model = get_down_block(
        "AttnDownBlock2D",
        num_layers=2,
        in_channels=128,
        out_channels=256,
        temb_channels=None,
        add_downsample=True,
        resnet_eps=1e-5,
        resnet_act_fn="silu",
        resnet_groups=4,
        attention_head_dim=8,
        downsample_padding=1,
        resnet_time_scale_shift="default",
        downsample_type=None,
        dropout=0.0,
    )

    return model


def test_attndown_layer(attn_down_block_refiners: fl.Module, attn_down_block_diffuser: nn.Module) -> None:
    x = torch.randn(1, 128, 64, 64)

    converter = ModelConverter(
        source_model=attn_down_block_refiners,
        target_model=attn_down_block_diffuser,
        skip_output_check=False,
        threshold=0.0005,
    )

    assert converter.run(source_args=(x,), target_args=(x,))
    print("ok")
    assert norm(converter.source_model(x) - converter.target_model(x, None)[0]).item() < 0.0005


@pytest.fixture(scope="module")
def attn_middle_block_refiners() -> fl.Module:
    model = fl.Chain(
        Resnet(
            in_channels=512,
            out_channels=512,
            num_groups=4,
        ),
        fl.Residual(
            fl.GroupNorm(channels=512, num_groups=4),
            SelfAttention2d(
                channels=512,
                num_heads=512 // 8,
            ),
        ),
        Resnet(
            in_channels=512,
            out_channels=512,
            num_groups=4,
        ),
    )
    return model


@pytest.fixture(scope="module")
def attn_middle_block_diffuser() -> nn.Module:
    model = UNetMidBlock2D(
        in_channels=512,
        temb_channels=None,
        resnet_act_fn="silu",
        resnet_time_scale_shift="default",
        attention_head_dim=8,
        resnet_groups=4,
        attn_groups=None,
        add_attention=True,
    )

    return model


def test_middle_layer(attn_middle_block_refiners: fl.Module, attn_middle_block_diffuser: nn.Module) -> None:
    x = torch.randn(1, 512, 32, 32)

    converter = ModelConverter(
        source_model=attn_middle_block_refiners,
        target_model=attn_middle_block_diffuser,
        skip_output_check=False,
        threshold=0.1,
    )

    assert converter.run(source_args=(x,), target_args=(x,))
    print("ok")
    assert norm(converter.source_model(x) - converter.target_model(x, None)[0]).item() < 0.1
