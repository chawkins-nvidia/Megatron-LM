# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy

import pytest
import torch
from emerging_optimizers.shampoo import Shampoo
from emerging_optimizers.soap import SOAP

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import ShardedTensor, load, save
from megatron.core.dist_checkpointing.mapping import LocalNonpersistentObject, ShardedObject
from megatron.core.dist_checkpointing.strategies import filesystem_async
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer import emerging_optimizers as emerging
from megatron.core.optimizer.optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)


def _param_group(param: torch.nn.Parameter) -> dict:
    return {
        "params": [param],
        "wd_mult": 1.0,
        "lr_mult": 1.0,
        "is_expert_parallel": False,
        "is_decoupled_lr": False,
    }


def _raw_optimizer(
    optimizer_name: str, param: torch.nn.Parameter
) -> tuple[torch.optim.Optimizer, emerging.LocalAuxStateCheckpointAdapter]:
    if optimizer_name == "soap":
        optimizer_cls = SOAP
        kwargs = {"lr": 1.0e-3, "weight_decay": 0.0, "use_eigh": True}
    elif optimizer_name == "shampoo":
        optimizer_cls = Shampoo
        kwargs = {"lr": 1.0e-3, "weight_decay": 0.0, "block_size": 2, "graft": True}
    else:
        raise AssertionError(optimizer_name)

    optimizer = optimizer_cls([_param_group(param)], **kwargs)
    adapter = emerging._local_aux_checkpoint_adapter_factory(
        optimizer_name, optimizer_cls, emerging._effective_constructor_config(optimizer_cls, kwargs)
    )
    setattr(optimizer, emerging._CHECKPOINT_ADAPTER_ATTR, adapter)
    optimizer._init_group(optimizer.param_groups[0], skip_non_grad_params=False)
    return optimizer, adapter


def _wrapper(
    wrapper_kind: str, optimizer_name: str, shape: tuple[int, ...] = (2, 3)
) -> tuple[MegatronOptimizer, torch.nn.Parameter]:
    config = OptimizerConfig()
    if wrapper_kind == "fp32":
        model_param = torch.nn.Parameter(torch.ones(shape, dtype=torch.float32))
        optimizer, _ = _raw_optimizer(optimizer_name, model_param)
        wrapper = object.__new__(FP32Optimizer)
        MegatronOptimizer.__init__(wrapper, optimizer, config, lambda *_: None)
        wrapper.is_stub_optimizer = False
    elif wrapper_kind == "bf16":
        model_param = torch.nn.Parameter(torch.ones(shape, dtype=torch.bfloat16))
        main_param = torch.nn.Parameter(model_param.detach().float())
        optimizer, _ = _raw_optimizer(optimizer_name, main_param)
        wrapper = object.__new__(Float16OptimizerWithFloat16Params)
        MegatronOptimizer.__init__(wrapper, optimizer, config, lambda *_: None)
        wrapper.grad_scaler = None
        wrapper.float16_groups = [[model_param]]
        wrapper.fp32_from_float16_groups = [[main_param]]
        wrapper.fp32_from_fp32_groups = []
    else:
        raise AssertionError(wrapper_kind)
    return wrapper, model_param


def _fallback_adam_wrapper() -> Float16OptimizerWithFloat16Params:
    config = OptimizerConfig()
    model_groups = []
    main_groups = []
    optimizer_groups = []
    group_specs = ((1, 1.0, 4.4e-3), (1, 1.0 / 3.0, 1.7e-3), (8, 0.0, 2.4e-3), (9, 0.0, 6.1e-3))
    for size, wd_mult, max_lr in group_specs:
        model_group = [torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16)) for _ in range(size)]
        main_group = [torch.nn.Parameter(param.detach().float()) for param in model_group]
        optimizer_groups.append(
            {
                "params": main_group,
                "wd_mult": wd_mult,
                "lr_mult": 1.0,
                "is_expert_parallel": False,
                "is_decoupled_lr": False,
                "max_lr": max_lr,
                "min_lr": max_lr / 100.0,
                "optimizer": "adam",
            }
        )
        model_groups.append(model_group)
        main_groups.append(main_group)

    optimizer = torch.optim.AdamW(optimizer_groups, lr=1.0e-3)

    def init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for param in group["params"]:
                opt.state[param]["step"] = torch.tensor(0.0)
                opt.state[param]["exp_avg"] = torch.zeros_like(param)
                opt.state[param]["exp_avg_sq"] = torch.zeros_like(param)

    init_state_fn(optimizer)
    wrapper = object.__new__(Float16OptimizerWithFloat16Params)
    MegatronOptimizer.__init__(wrapper, optimizer, config, init_state_fn)
    wrapper.grad_scaler = None
    wrapper.float16_groups = model_groups
    wrapper.fp32_from_float16_groups = main_groups
    wrapper.fp32_from_fp32_groups = []
    return wrapper


@pytest.mark.parametrize("optimizer_name", ("soap", "shampoo"))
@pytest.mark.parametrize("wrapper_kind", ("fp32", "bf16"))
@pytest.mark.parametrize("shape", ((2, 3), (2, 2)))
def test_auxiliary_state_uses_independent_tensors_in_both_wrapper_paths(
    optimizer_name: str, wrapper_kind: str, shape: tuple[int, ...]
) -> None:
    wrapper, model_param = _wrapper(wrapper_kind, optimizer_name, shape)
    model_shard = ShardedTensor.from_rank_offsets("model.weight", model_param)

    state_dict = wrapper.sharded_state_dict({"weight": model_shard})

    optimizer_state = state_dict["optimizer"] if wrapper_kind == "bf16" else state_dict
    param_state = optimizer_state["state"][0]
    assert isinstance(state_dict["optimizer_checkpoint_adapter"], ShardedObject)
    assert (
        state_dict["optimizer_checkpoint_adapter"].data["payload"]["optimizer_identity"]
        == optimizer_name
    )
    assert isinstance(param_state["exp_avg"], ShardedTensor)
    assert param_state["exp_avg"].key == "optimizer.state.exp_avg.model.weight"
    assert isinstance(param_state["exp_avg_sq"], ShardedTensor)

    if optimizer_name == "soap":
        auxiliary_keys = {"L", "R", "Q_L", "Q_R"}
    else:
        assert isinstance(param_state["block_ranges"], LocalNonpersistentObject)
        auxiliary_keys = {
            key
            for key in param_state
            if key.startswith(
                ("left_factor_", "right_factor_", "left_inverse_root_", "right_inverse_root_")
            )
        }
        assert auxiliary_keys
    for state_key in auxiliary_keys:
        sharded = param_state[state_key]
        assert isinstance(sharded, ShardedTensor)
        assert sharded.key.startswith(
            f"optimizer.state.{optimizer_name}.{state_key}.param_0.model.weight"
        )
        assert sharded.global_shape == tuple(sharded.data.shape)


@pytest.mark.parametrize("wrapper_kind", ("fp32", "bf16"))
def test_torch_dist_load_requires_fingerprint_in_both_wrapper_paths(wrapper_kind: str) -> None:
    wrapper, model_param = _wrapper(wrapper_kind, "soap")
    model_shard = ShardedTensor.from_rank_offsets("model.weight", model_param)
    wrapper.sharded_state_dict({"weight": model_shard}, is_loading=True)

    state_dict = wrapper.state_dict()

    with pytest.raises(RuntimeError, match="missing required adapter fingerprint"):
        wrapper.load_state_dict(state_dict)


@pytest.mark.parametrize("optimizer_name", ("soap", "shampoo"))
def test_raw_torch_state_dict_round_trip_preserves_next_update(
    monkeypatch: pytest.MonkeyPatch, optimizer_name: str
) -> None:
    # SOAP's CPU math is valid, but its optional NVTX annotations require a CUDA build.
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda *_: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda *_: None)

    first_param = torch.nn.Parameter(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    first_optimizer, _ = _raw_optimizer(optimizer_name, first_param)
    first_param.grad = torch.tensor([[0.1, -0.2, 0.3], [0.4, 0.1, -0.1]])
    first_optimizer.step()

    second_param = torch.nn.Parameter(first_param.detach().clone())
    second_optimizer, _ = _raw_optimizer(optimizer_name, second_param)
    second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))

    next_grad = torch.tensor([[-0.3, 0.2, 0.1], [0.2, -0.4, 0.5]])
    first_param.grad = next_grad.clone()
    second_param.grad = next_grad.clone()
    first_optimizer.step()
    second_optimizer.step()

    torch.testing.assert_close(second_param, first_param, rtol=0.0, atol=0.0)
    for key, first_value in first_optimizer.state[first_param].items():
        second_value = second_optimizer.state[second_param][key]
        if torch.is_tensor(first_value):
            torch.testing.assert_close(second_value, first_value, rtol=0.0, atol=0.0)
        else:
            assert second_value == first_value


def test_fingerprint_rejects_optimizer_layout_change() -> None:
    first_param = torch.nn.Parameter(torch.ones(2, 3))
    first_optimizer, first_adapter = _raw_optimizer("shampoo", first_param)
    loaded = first_adapter.fingerprint(first_optimizer, first_optimizer.state_dict())

    second_param = torch.nn.Parameter(torch.ones(2, 3))
    second_optimizer = Shampoo(
        [_param_group(second_param)], lr=1.0e-3, weight_decay=0.0, block_size=1, graft=True
    )
    second_kwargs = {"lr": 1.0e-3, "weight_decay": 0.0, "block_size": 1, "graft": True}
    second_adapter = emerging._local_aux_checkpoint_adapter_factory(
        "shampoo", Shampoo, emerging._effective_constructor_config(Shampoo, second_kwargs)
    )
    second_optimizer._init_group(second_optimizer.param_groups[0], skip_non_grad_params=False)

    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        second_adapter.validate_fingerprint(loaded, second_optimizer, second_optimizer.state_dict())


def test_fingerprint_ignores_scheduler_owned_live_values() -> None:
    saved_param = torch.nn.Parameter(torch.ones(2, 3))
    saved_optimizer, saved_adapter = _raw_optimizer("soap", saved_param)
    saved_optimizer.param_groups[0]["lr"] = 2.5e-5
    saved_optimizer.param_groups[0]["weight_decay"] = 0.04
    loaded = saved_adapter.fingerprint(saved_optimizer, saved_optimizer.state_dict())

    fresh_param = torch.nn.Parameter(torch.ones(2, 3))
    fresh_optimizer, fresh_adapter = _raw_optimizer("soap", fresh_param)

    fresh_adapter.validate_fingerprint(loaded, fresh_optimizer, fresh_optimizer.state_dict())


def test_chained_fallback_adam_preserves_duplicate_param_groups_on_load() -> None:
    source_fallback = _fallback_adam_wrapper()
    source_soap, _ = _wrapper("bf16", "soap")
    source = ChainedOptimizer([source_fallback, source_soap])

    destination_fallback = _fallback_adam_wrapper()
    destination_soap, _ = _wrapper("bf16", "soap")
    destination = ChainedOptimizer([destination_fallback, destination_soap])

    destination.load_state_dict(copy.deepcopy(source.state_dict()))

    assert [len(group["params"]) for group in destination_fallback.optimizer.param_groups] == [
        1,
        1,
        8,
        9,
    ]
    assert [group["max_lr"] for group in destination_fallback.optimizer.param_groups] == [
        4.4e-3,
        1.7e-3,
        2.4e-3,
        6.1e-3,
    ]
    source_state = source_fallback.optimizer.state_dict()["state"]
    destination_state = destination_fallback.optimizer.state_dict()["state"]
    assert source_state.keys() == destination_state.keys()
    for param_id, param_state in source_state.items():
        for state_key, value in param_state.items():
            torch.testing.assert_close(destination_state[param_id][state_key], value)


@pytest.mark.parametrize("optimizer_name", ("soap", "shampoo"))
def test_one_rank_torch_dist_round_trip(
    tmp_path, monkeypatch: pytest.MonkeyPatch, optimizer_name: str
) -> None:
    if torch.distributed.is_initialized():
        pytest.skip("this CPU checkpoint test owns its world-size-one process group")

    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(filesystem_async, "_process_memory", lambda: 0)
    rendezvous_path = tmp_path / "torch_dist_rendezvous"
    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{rendezvous_path}", rank=0, world_size=1
    )
    parallel_state.initialize_model_parallel(1, 1, create_gloo_process_groups=False)
    try:
        first_wrapper, first_model_param = _wrapper("fp32", optimizer_name)
        first_state = next(iter(first_wrapper.optimizer.state.values()))
        first_state["step"] = 7
        for index, value in enumerate(first_state.values()):
            if torch.is_tensor(value):
                value.fill_(index + 0.25)

        first_model_shard = ShardedTensor.from_rank_offsets(
            "model.weight", first_model_param, replica_id=0
        )
        checkpoint_dir = tmp_path / f"{optimizer_name}_checkpoint"
        checkpoint_dir.mkdir()
        save(
            first_wrapper.sharded_state_dict({"weight": first_model_shard}),
            checkpoint_dir,
            async_sharded_save=False,
        )

        second_wrapper, second_model_param = _wrapper("fp32", optimizer_name)
        second_model_shard = ShardedTensor.from_rank_offsets(
            "model.weight", second_model_param, replica_id=0
        )
        load_template = second_wrapper.sharded_state_dict(
            {"weight": second_model_shard}, is_loading=True
        )
        second_wrapper.load_state_dict(load(load_template, checkpoint_dir))

        first_loaded_state = first_wrapper.optimizer.state_dict()["state"][0]
        second_loaded_state = second_wrapper.optimizer.state_dict()["state"][0]
        assert first_loaded_state.keys() == second_loaded_state.keys()
        for state_key, first_value in first_loaded_state.items():
            second_value = second_loaded_state[state_key]
            if torch.is_tensor(first_value):
                torch.testing.assert_close(second_value, first_value, rtol=0.0, atol=0.0)
            else:
                assert second_value == first_value
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


def test_unsupported_model_parallel_topology_fails_closed() -> None:
    topology = {"world_size": 2, "dp": 1, "tp": 2, "pp": 1, "cp": 1, "ep": 1, "etp": 1}

    with pytest.raises(RuntimeError, match="TP=PP=CP=EP=ETP=1"):
        emerging._validate_local_aux_checkpoint_topology(topology)
