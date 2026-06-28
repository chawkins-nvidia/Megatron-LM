# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_inference_spec,
    get_gpt_mtp_block_spec,
    get_gpt_decoder_layer_specs,
)
from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml


def maybe_apply_parametrization_init(args, config):
    """Apply the unified Parametrization object's per-module-type init rules to ``config``."""
    inline_block = getattr(args, 'parametrization', None)
    has_inline = inline_block is not None
    has_file = getattr(args, 'parametrization_config', None) is not None
    if not has_inline and not has_file:
        return config
    cand = getattr(args, 'parametrization_candidate', None)
    inline_candidate = (
        inline_block.get('candidate_name')
        if isinstance(inline_block, dict)
        else getattr(inline_block, 'candidate_name', None)
    )
    if has_file and not cand:
        raise ValueError("--parametrization-config requires --parametrization-candidate.")

    from megatron.training.parametrization import load_parametrization, load_parametrization_block

    depth_base = getattr(args, 'parametrization_depth_base', None)
    m_L = (config.num_layers / depth_base) if depth_base else None
    kwargs = dict(
        m_N=getattr(args, 'parametrization_m_n', 1.0),
        m_L=m_L,
        alpha=getattr(args, 'parametrization_alpha', None),
        residual_const=getattr(args, 'parametrization_residual_const', None),
        residual_attention_const=getattr(args, 'parametrization_residual_attention_const', None),
        residual_mlp_const=getattr(args, 'parametrization_residual_mlp_const', None),
        depth_base=depth_base,
    )
    if has_file:
        par = load_parametrization(args.parametrization_config, cand, **kwargs)
        source = "file"
        name = cand
    else:
        par = load_parametrization_block(inline_block, **kwargs)
        source = "inline"
        name = inline_candidate
    residual_mult = 1.0 if getattr(config, 'is_hybrid_model', False) else 2.0
    par.apply_init(config, residual_depth_multiplier=residual_mult)
    print_rank_0(
        f'[#118 param] applied init: source={source}, candidate={name}, '
        f'm_N={getattr(args, "parametrization_m_n", 1.0)}, '
        f'm_L={m_L}, residual_depth_multiplier={residual_mult}, '
        f'enabled={par.cfg.enabled}, alpha={par.cfg.alpha}, '
        f'depth_base={par.cfg.depth_base}, '
        f'residual_attention_mult={getattr(config, "residual_attention_mult", 1.0)}, '
        f'residual_mlp_mult={getattr(config, "residual_mlp_mult", 1.0)}'
    )
    return config


def gpt_builder(args, pre_process, post_process, vp_stage=None, config=None, pg_collection=None):
    print_rank_0('building GPT model ...')
    if config is None:
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)
        maybe_apply_parametrization_init(args, config)
    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        use_te = args.transformer_impl == "transformer_engine"

        if args.experimental_attention_variant is not None:
            transformer_layer_spec = (
                get_transformer_block_with_experimental_attention_variant_spec(
                    config=config, vp_stage=vp_stage
                )
            )
        elif args.num_experts:
            # Define the decoder block spec
            transformer_layer_spec = get_gpt_decoder_block_spec(
                config,
                use_transformer_engine=use_te,
                normalization=args.normalization,
                qk_l2_norm=args.qk_l2_norm,
                vp_stage=vp_stage,
            )
        elif args.heterogeneous_layers_config_path is not None:
            assert not (config.transformer_impl == "inference_optimized")
            transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
        else:
            # Define the decoder layer spec
            transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
    mtp_block_spec = None
    if args.mtp_num_layers is not None:
        assert not (config.transformer_impl == "inference_optimized")
        if (
            hasattr(transformer_layer_spec, 'layer_specs')
            and len(transformer_layer_spec.layer_specs) == 0
        ):
            # Get the decoder layer spec explicitly if no decoder layer in the last stage,
            # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
            transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
        else:
            # Define the decoder block spec
            decoder_layer_specs = get_gpt_decoder_layer_specs(
                config, use_transformer_engine=use_te, normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, vp_stage=vp_stage
            )
            transformer_layer_spec_for_mtp = decoder_layer_specs[-1]
        # Use spec of the last layer in decoder block as spec of the transformer layer in MTP
        mtp_block_spec = get_gpt_mtp_block_spec(
            config,
            transformer_layer_spec_for_mtp,
            use_transformer_engine=use_te,
            vp_stage=vp_stage,
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        rope_scaling=args.use_rope_scaling,
        mtp_block_spec=mtp_block_spec,
        vp_stage=vp_stage,
        pg_collection=pg_collection,
    )

    return model


def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.

    Args:
        use_te (bool): Whether to use Transformer Engine
        config: Model configuration

    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            config.num_moe_experts,
            config.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            config.experimental_attention_variant,
            qk_l2_norm=config.qk_l2_norm,
            use_kitchen=config.use_kitchen,
            use_te_activation_func=config.use_te_activation_func,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
            mla_down_proj_fusion=getattr(config, "mla_down_proj_fusion", False),
        )
    elif config.transformer_impl == "inference_optimized":
        return get_gpt_layer_with_inference_spec(
            config.qk_layernorm,
            config.multi_latent_attention,
            qk_l2_norm=config.qk_l2_norm,
        )
    else:
        return get_gpt_layer_local_spec(
            config.num_moe_experts,
            config.moe_grouped_gemm,
            config.qk_layernorm,
            config.multi_latent_attention,
            config.experimental_attention_variant,
            normalization=config.normalization,
            use_kitchen=config.use_kitchen,
            use_kitchen_attention=config.use_kitchen_attention,
            kitchen_attention_backend=config.kitchen_attention_backend,
        )
