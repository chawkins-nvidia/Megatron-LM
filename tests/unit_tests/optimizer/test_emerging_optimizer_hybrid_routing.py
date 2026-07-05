# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from unittest.mock import Mock

import pytest
import torch
from torch import nn

import megatron.core.optimizer as optimizer_module
from megatron.core.optimizer import (
    OptimizerConfig,
    ParamKey,
    ParamPredicate,
    _get_param_groups,
    check_config_overrides_consistency,
)
from megatron.core.optimizer import emerging_optimizers as emerging
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.training.parametrization import Parametrization, ParametrizationConfig


class HybridRoleModel(nn.Module):
    """Small model covering supported matrices and every Adam fallback category."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Parameter(torch.ones(4, 4))
        self.embedding = nn.Parameter(torch.ones(8, 4))
        self.embedding.is_embedding_or_output_parameter = True
        self.readout = nn.Parameter(torch.ones(4, 8))
        self.readout.is_embedding_or_output_parameter = True
        self.norm = nn.Parameter(torch.ones(4))
        self.bias = nn.Parameter(torch.zeros(4))
        self.scalar = nn.Parameter(torch.tensor(1.0))
        self.tensor3 = nn.Parameter(torch.ones(2, 2, 2))


class CompiledRoleModel(nn.Module):
    """Model with a 2-D router and tied/readout parameters requiring explicit Adam roles."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Parameter(torch.ones(4, 4))
        self.router = nn.Parameter(torch.ones(8, 4))
        self.tied = nn.Parameter(torch.ones(8, 4))
        self.tied.is_embedding_or_output_parameter = True
        self.norm = nn.Parameter(torch.ones(4))


class RankLocalRoleModel(nn.Module):
    """Model whose only parameter routes differently on each test rank."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        shape = (4, 4) if rank == 0 else (4,)
        self.local = nn.Parameter(torch.ones(shape))


class RankLocalExpertModel(nn.Module):
    """Model whose identical parameter is dense on one rank and expert on the other."""

    def __init__(self, rank: int) -> None:
        super().__init__()
        self.local = nn.Parameter(torch.ones(4, 4))
        self.local.allreduce = rank == 0


@pytest.fixture
def single_rank_param_group_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the production synchronization path in single-process unit tests."""

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)

    def all_gather_object(output: list, local: object) -> None:
        output[0] = local

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)


def _groups_by_parameter(param_groups: list[dict]) -> dict[nn.Parameter, dict]:
    result = {}
    for group in param_groups:
        for param in group["params"]:
            assert param not in result
            result[param] = group
    return result


def _distributed_param_group_sync_worker(
    rank: int, world_size: int, rendezvous_path: str, selected_optimizer: str
) -> None:
    """Prove WORLD synchronization retains rank-local empty optimizer groups."""

    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{rendezvous_path}", rank=rank, world_size=world_size
    )
    try:
        groups = _get_param_groups(
            [RankLocalRoleModel(rank)],
            OptimizerConfig(optimizer=selected_optimizer, lr=0.1, min_lr=0.01),
            {},
            hybrid_optimizer=selected_optimizer,
        )
        schema = [
            {key: value for key, value in group.items() if key != "params"} for group in groups
        ]
        local_counts = [len(group["params"]) for group in groups]

        gathered_schemas = [None] * world_size
        gathered_counts = [None] * world_size
        torch.distributed.all_gather_object(gathered_schemas, schema)
        torch.distributed.all_gather_object(gathered_counts, local_counts)

        assert gathered_schemas == [schema] * world_size
        assert [group["optimizer"] for group in schema] == ["adam", selected_optimizer]
        assert gathered_counts == [[0, 1], [1, 0]]
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("selected_optimizer", ["muon", "soap", "shampoo"])
def test_hybrid_group_schemas_are_synchronized_with_empty_local_groups(
    tmp_path, selected_optimizer: str
) -> None:
    """Every rank creates the same ordered hybrid buckets even when one is empty."""

    rendezvous_path = tmp_path / f"{selected_optimizer}_param_group_sync"
    torch.multiprocessing.spawn(
        _distributed_param_group_sync_worker,
        args=(2, str(rendezvous_path), selected_optimizer),
        nprocs=2,
        join=True,
    )


def _distributed_expert_group_order_worker(
    rank: int, world_size: int, rendezvous_path: str
) -> None:
    """Prove identical overrides are ordered globally across dense and expert groups."""

    torch.distributed.init_process_group(
        backend="gloo", init_method=f"file://{rendezvous_path}", rank=rank, world_size=world_size
    )
    try:
        groups = _get_param_groups(
            [RankLocalExpertModel(rank)],
            OptimizerConfig(optimizer="soap", lr=0.1, min_lr=0.01),
            {},
            hybrid_optimizer="soap",
        )
        schema = [
            {key: value for key, value in group.items() if key != "params"} for group in groups
        ]
        local_counts = [len(group["params"]) for group in groups]

        gathered_schemas = [None] * world_size
        gathered_counts = [None] * world_size
        torch.distributed.all_gather_object(gathered_schemas, schema)
        torch.distributed.all_gather_object(gathered_counts, local_counts)

        assert gathered_schemas == [schema] * world_size
        assert [group["optimizer"] for group in schema] == ["soap", "soap"]
        assert [group["is_expert_parallel"] for group in schema] == [False, True]
        assert gathered_counts == [[1, 0], [0, 1]]
    finally:
        torch.distributed.destroy_process_group()


def test_identical_override_groups_have_global_dense_expert_order(tmp_path) -> None:
    """Dense/expert keys with identical overrides have a deterministic total order."""

    rendezvous_path = tmp_path / "dense_expert_param_group_order"
    torch.multiprocessing.spawn(
        _distributed_expert_group_order_worker,
        args=(2, str(rendezvous_path)),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("selected_optimizer", ["muon", "soap", "shampoo"])
def test_hybrid_group_routing_is_exhaustive_and_preserves_completep_overrides(
    monkeypatch: pytest.MonkeyPatch, single_rank_param_group_sync: None, selected_optimizer: str
) -> None:
    """Matrix roles and numerical CompleteP overrides compose without losing fallback safety."""
    monkeypatch.setenv("MEGATRON_LOG_OPTIMIZER_PARAM_GROUPS", "0")
    model = HybridRoleModel()
    config = OptimizerConfig(optimizer=selected_optimizer, lr=0.1, min_lr=0.01)
    overrides = {
        ParamKey(name="hidden"): ParamGroupOverride(
            optimizer=selected_optimizer, max_lr=0.2, min_lr=0.02, wd_mult=0.5, eps=1e-12
        ),
        ParamKey(name="embedding"): ParamGroupOverride(
            optimizer="adam", max_lr=0.3, min_lr=0.03, wd_mult=0.0, eps=2e-12
        ),
    }

    groups = _get_param_groups([model], config, overrides, hybrid_optimizer=selected_optimizer)
    by_param = _groups_by_parameter(groups)

    assert set(by_param) == set(model.parameters())
    assert by_param[model.hidden]["optimizer"] == selected_optimizer
    assert by_param[model.hidden]["max_lr"] == pytest.approx(0.2)
    assert by_param[model.hidden]["min_lr"] == pytest.approx(0.02)
    assert by_param[model.hidden]["wd_mult"] == pytest.approx(0.5)
    assert by_param[model.hidden]["eps"] == pytest.approx(1e-12)

    assert by_param[model.embedding]["optimizer"] == "adam"
    assert by_param[model.embedding]["max_lr"] == pytest.approx(0.3)
    assert by_param[model.embedding]["min_lr"] == pytest.approx(0.03)
    assert by_param[model.embedding]["wd_mult"] == pytest.approx(0.0)
    assert by_param[model.embedding]["eps"] == pytest.approx(2e-12)

    fallback_params = [
        model.embedding,
        model.readout,
        model.norm,
        model.bias,
        model.scalar,
        model.tensor3,
    ]
    assert all(by_param[param]["optimizer"] == "adam" for param in fallback_params)


@pytest.mark.parametrize("selected_optimizer", ["muon", "soap", "shampoo"])
def test_parametrization_compiles_closed_world_optimizer_roles(
    monkeypatch: pytest.MonkeyPatch, single_rank_param_group_sync: None, selected_optimizer: str
) -> None:
    """A 2-D router stays on Adam because the trusted rule role is compiled explicitly."""
    monkeypatch.setenv("MEGATRON_LOG_OPTIMIZER_PARAM_GROUPS", "0")
    model = CompiledRoleModel()
    parametrization = Parametrization(
        ParametrizationConfig.from_dict(
            {
                "enabled": True,
                "expected_types": ["hidden", "router", "tied", "norm"],
                "type_registry": {
                    "hidden": {"name_globs": ["hidden"], "min_dim": 2},
                    "router": {"name_globs": ["router"], "min_dim": 2},
                    "tied": {"name_globs": ["tied"], "min_dim": 2},
                    "norm": {"name_globs": ["norm"], "max_dim": 1},
                },
                "rules": [
                    {
                        "name": "hidden",
                        "types": ["hidden"],
                        "optimizer_role": "matrix",
                        "lr_mult": {"const": 2.0},
                        "eps_mult": {"const": 3.0},
                        "wd_mult": {"const": 4.0},
                    },
                    {"name": "router", "types": ["router"], "optimizer_role": "router_adam"},
                    {"name": "tied", "types": ["tied"], "optimizer_role": "tied_adam"},
                    {"name": "norm", "types": ["norm"], "optimizer_role": "fallback_adam"},
                ],
            }
        )
    )
    overrides = parametrization.build_config_overrides(
        base_lr=0.1, base_min_lr=0.01, base_eps=1e-8, selected_optimizer=selected_optimizer
    )

    groups = _get_param_groups(
        [model],
        OptimizerConfig(optimizer=selected_optimizer, lr=0.1, min_lr=0.01),
        overrides,
        hybrid_optimizer=selected_optimizer,
    )
    by_param = _groups_by_parameter(groups)

    assert by_param[model.hidden]["optimizer"] == selected_optimizer
    assert by_param[model.hidden]["max_lr"] == pytest.approx(0.2)
    assert by_param[model.hidden]["min_lr"] == pytest.approx(0.02)
    assert by_param[model.hidden]["eps"] == pytest.approx(3e-8)
    assert by_param[model.hidden]["wd_mult"] == pytest.approx(4.0)
    assert by_param[model.router]["optimizer"] == "adam"
    assert by_param[model.tied]["optimizer"] == "adam"
    assert by_param[model.norm]["optimizer"] == "adam"


def test_hybrid_supported_matrix_optimizer_conflicts_remain_strict(
    monkeypatch: pytest.MonkeyPatch, single_rank_param_group_sync: None
) -> None:
    """Two contradictory explicit roles on a supported matrix are rejected."""
    monkeypatch.setenv("MEGATRON_LOG_OPTIMIZER_PARAM_GROUPS", "0")
    model = HybridRoleModel()
    overrides = {
        ParamKey(name="hidden"): ParamGroupOverride(optimizer="muon"),
        ParamKey(
            predicate=ParamPredicate(name="same_hidden", fn=lambda param: param is model.hidden)
        ): ParamGroupOverride(optimizer="adam"),
    }

    with pytest.raises(ValueError, match="Conflicting overrides for optimizer"):
        _get_param_groups(
            [model], OptimizerConfig(optimizer="muon", lr=0.1), overrides, hybrid_optimizer="muon"
        )


def test_hybrid_unsupported_shape_rejects_contradictory_compiled_role(
    monkeypatch: pytest.MonkeyPatch, single_rank_param_group_sync: None
) -> None:
    """An explicit matrix role on a fallback-only parameter is a startup error."""
    monkeypatch.setenv("MEGATRON_LOG_OPTIMIZER_PARAM_GROUPS", "0")
    model = HybridRoleModel()

    with pytest.raises(ValueError, match="requires optimizer='adam'"):
        _get_param_groups(
            [model],
            OptimizerConfig(optimizer="muon", lr=0.1),
            {ParamKey(name="norm"): ParamGroupOverride(optimizer="muon")},
            hybrid_optimizer="muon",
        )


def test_hybrid_compiled_role_must_agree_with_package_fallback(
    monkeypatch: pytest.MonkeyPatch, single_rank_param_group_sync: None
) -> None:
    """Package fallback declarations cannot silently replace a trusted compiled role."""
    monkeypatch.setenv("MEGATRON_LOG_OPTIMIZER_PARAM_GROUPS", "0")
    model = HybridRoleModel()

    with pytest.raises(ValueError, match="conflicts with package fallback"):
        _get_param_groups(
            [model],
            OptimizerConfig(optimizer="muon", lr=0.1),
            {ParamKey(name="hidden"): ParamGroupOverride(optimizer="muon")},
            hybrid_optimizer="muon",
            hybrid_default_overrides={
                ParamKey(name="hidden"): ParamGroupOverride(optimizer="adam")
            },
        )


def test_optimizer_override_allowlist_is_hybrid_only() -> None:
    """Standard optimizers reject role changes and hybrid paths accept only selected+Adam."""
    overrides = {ParamKey(name="hidden"): ParamGroupOverride(optimizer="muon")}

    with pytest.raises(ValueError, match="Field optimizer should not be overriden"):
        check_config_overrides_consistency(OptimizerConfig(optimizer="adam"), overrides)

    check_config_overrides_consistency(
        OptimizerConfig(optimizer="muon"), overrides, allowed_optimizer_overrides={"muon", "adam"}
    )

    with pytest.raises(ValueError, match="Unsupported optimizer override 'soap'"):
        check_config_overrides_consistency(
            OptimizerConfig(optimizer="muon"),
            {ParamKey(name="hidden"): ParamGroupOverride(optimizer="soap")},
            allowed_optimizer_overrides={"muon", "adam"},
        )


def test_standalone_shampoo_constructor_uses_prefixed_config_and_global_adam_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Standalone Shampoo construction is distinct from SOAP and follows its registry API."""

    class MockShampoo:
        def __init__(
            self,
            params,
            lr=3e-4,
            betas=(0.9, 0.95),
            eps=1e-12,
            weight_decay=0.01,
            *,
            block_size=256,
            precondition_frequency=10,
            graft=False,
            graft_beta2=0.999,
            graft_eps=1e-8,
            start_preconditioning_step=1,
            rank_deficient_stability="perturbation",
            rank_atol=0.0,
            rank_rtol=0.0,
        ) -> None:
            self.params = params
            self.kwargs = {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
                "block_size": block_size,
                "precondition_frequency": precondition_frequency,
                "graft": graft,
                "graft_beta2": graft_beta2,
                "graft_eps": graft_eps,
                "start_preconditioning_step": start_preconditioning_step,
                "rank_deficient_stability": rank_deficient_stability,
                "rank_atol": rank_atol,
                "rank_rtol": rank_rtol,
            }

    registry = Mock()
    registry.get_optimizer_cls.return_value = MockShampoo
    monkeypatch.setattr(emerging, "registry", registry, raising=False)
    config = OptimizerConfig(
        optimizer="shampoo",
        lr=0.004,
        weight_decay=0.02,
        adam_beta1=0.8,
        adam_beta2=0.97,
        shampoo_eps=3e-11,
        shampoo_block_size=128,
        shampoo_precondition_frequency=7,
        shampoo_graft=True,
        shampoo_graft_beta2=0.96,
        shampoo_graft_eps=4e-9,
        shampoo_start_preconditioning_step=11,
        shampoo_rank_deficient_stability="pseudoinverse",
        shampoo_rank_atol=5e-12,
        shampoo_rank_rtol=None,
    )

    kwargs = emerging._shampoo_config_to_kwargs(config, model_chunks=[], pg_collection=None)

    registry.get_optimizer_cls.assert_called_once_with("shampoo")
    assert kwargs == {
        "lr": pytest.approx(0.004),
        "betas": (pytest.approx(0.8), pytest.approx(0.97)),
        "eps": pytest.approx(3e-11),
        "weight_decay": pytest.approx(0.02),
        "block_size": 128,
        "precondition_frequency": 7,
        "graft": True,
        "graft_beta2": pytest.approx(0.96),
        "graft_eps": pytest.approx(4e-9),
        "start_preconditioning_step": 11,
        "rank_deficient_stability": "pseudoinverse",
        "rank_atol": pytest.approx(5e-12),
        "rank_rtol": None,
    }

    registry.reset_mock()
    monkeypatch.setitem(
        emerging._EMERGING_OPTIMIZERS,
        "shampoo",
        emerging.EmergingOptimizerEntry(
            optimizer_cls=MockShampoo, config_to_kwargs=emerging._shampoo_config_to_kwargs
        ),
    )
    param_groups = [{"params": [nn.Parameter(torch.ones(2, 2))]}]
    optimizer, init_state_fn = emerging._create_emerging_optimizer(
        config, param_groups, "shampoo", model_chunks=[], pg_collection=None
    )

    assert isinstance(optimizer, MockShampoo)
    assert optimizer.params is param_groups
    assert optimizer.kwargs == {
        "lr": pytest.approx(0.004),
        "betas": (pytest.approx(0.8), pytest.approx(0.97)),
        "eps": pytest.approx(3e-11),
        "weight_decay": pytest.approx(0.02),
        "block_size": 128,
        "precondition_frequency": 7,
        "graft": True,
        "graft_beta2": pytest.approx(0.96),
        "graft_eps": pytest.approx(4e-9),
        "start_preconditioning_step": 11,
        "rank_deficient_stability": "pseudoinverse",
        "rank_atol": pytest.approx(5e-12),
        "rank_rtol": None,
    }
    assert init_state_fn is emerging._eopt_init_state_fn


def test_standalone_shampoo_reports_missing_package_registry_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stock package without standalone Shampoo fails before process-group setup."""
    monkeypatch.setattr(optimizer_module, "HAVE_EMERGING_OPTIMIZERS", True)
    monkeypatch.delitem(optimizer_module._EMERGING_OPTIMIZERS, "shampoo", raising=False)

    with pytest.raises(ValueError, match="'shampoo' is not registered"):
        optimizer_module._get_megatron_emerging_optimizer(
            OptimizerConfig(optimizer="shampoo", lr=0.004), model_chunks=[], config_overrides={}
        )
