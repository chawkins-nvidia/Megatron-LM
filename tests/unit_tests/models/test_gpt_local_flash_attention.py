# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.core.models.gpt import gpt_layer_specs
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnBackend
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.argument_utils import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml


class _FakeTEDotProductAttention:
    pass


def _config_args(*, use_flash_attn: bool) -> SimpleNamespace:
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
    args = vars(config).copy()
    args.update(
        multi_latent_attention=False,
        heterogeneous_layers_config_path=None,
        no_persist_layer_norm=not config.persist_layer_norm,
        num_experts=None,
        rotary_interleaved=False,
        decoder_first_pipeline_num_layers=None,
        decoder_last_pipeline_num_layers=None,
        fp8_param_gather=False,
        fp4_param_gather=False,
        swiglu=False,
        bias_gelu_fusion=False,
        squared_relu=False,
        quick_geglu=False,
        init_method_xavier_uniform=False,
        group_query_attention=False,
        config_logger_dir=None,
        rope_type=None,
        cp_comm_type=["p2p"],
        hybrid_layer_pattern=None,
        seed=123,
        kitchen_config_file=None,
        kitchen_recipe_number=None,
        moe_latent_size=None,
        te_precision_config_file=None,
        use_flash_attn=use_flash_attn,
    )
    return SimpleNamespace(**args)


def test_local_spec_defaults_to_local_attention():
    submodules = gpt_layer_specs.get_gpt_layer_local_submodules()

    assert submodules.self_attention.submodules.core_attention is DotProductAttention


def test_local_spec_flash_attention_changes_only_core_attention(monkeypatch):
    monkeypatch.setattr(gpt_layer_specs, "HAVE_TE", True)
    monkeypatch.setattr(
        gpt_layer_specs, "TEDotProductAttention", _FakeTEDotProductAttention
    )

    local = gpt_layer_specs.get_gpt_layer_local_submodules()
    flash = gpt_layer_specs.get_gpt_layer_local_submodules(
        use_flash_attn=True, attention_backend=AttnBackend.flash
    )

    assert flash.self_attention.submodules.core_attention is _FakeTEDotProductAttention
    assert flash.self_attention.module is local.self_attention.module
    assert flash.self_attention.submodules.linear_qkv is ColumnParallelLinear
    assert flash.self_attention.submodules.linear_proj is RowParallelLinear
    assert flash.input_layernorm is local.input_layernorm
    assert flash.pre_mlp_layernorm is local.pre_mlp_layernorm
    assert get_submodules(flash.mlp).linear_fc1 is ColumnParallelLinear
    assert get_submodules(flash.mlp).linear_fc2 is RowParallelLinear


def test_local_spec_flash_attention_fails_closed(monkeypatch):
    monkeypatch.setattr(gpt_layer_specs, "HAVE_TE", False)
    monkeypatch.setattr(gpt_layer_specs, "TEDotProductAttention", None)
    with pytest.raises(ImportError, match="requires Transformer Engine"):
        gpt_layer_specs.get_gpt_layer_local_submodules(
            use_flash_attn=True, attention_backend=AttnBackend.flash
        )

    monkeypatch.setattr(gpt_layer_specs, "HAVE_TE", True)
    monkeypatch.setattr(
        gpt_layer_specs, "TEDotProductAttention", _FakeTEDotProductAttention
    )
    with pytest.raises(ValueError, match="attention-backend flash or auto"):
        gpt_layer_specs.get_gpt_layer_local_submodules(
            use_flash_attn=True, attention_backend=AttnBackend.unfused
        )


def test_use_flash_attn_propagates_from_args_to_config():
    config = core_transformer_config_from_args(_config_args(use_flash_attn=True))

    assert config.use_flash_attn is True


def test_use_flash_attn_propagates_from_top_level_yaml():
    language_model = _config_args(use_flash_attn=False)
    del language_model.use_flash_attn
    language_model.activation_func = "gelu"
    root = SimpleNamespace(
        language_model=language_model,
        model_parallel=SimpleNamespace(),
        use_flash_attn=True,
    )

    config = core_transformer_config_from_yaml(root)

    assert config.use_flash_attn is True
