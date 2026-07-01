# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tier-1/2 runtime bound to the successful-update diagnostic heartbeat.

The first production slice is intentionally narrow and fail closed: dense local
MCore GPT, TP=CP=1, PP<=2, BF16 distributed Adam, and arbitrary DP.  Tier 0
remains available outside that surface.  The replay and secant payloads still
use rank-independent global registries and one shared packed reduction sequence.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.tensor_parallel import get_cuda_rng_tracker
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import unwrap_model

from .accumulator import PackedSufficientStatistics, ReductionBinding
from .diagnostic_replay import (
    CollectiveBinding,
    NonInterleavedReplaySchedule,
    PopulationCollectiveWorkspace,
    ProductionFatalAbort,
    ReadinessConsensus,
    ReplayBatchRecorder,
    ReplayMemoryPolicy,
    StableSampleDataset,
    Tier1ReplayEngine,
    Tier1ReplayTransaction,
    build_distributed_source_plan,
)
from .distributed_optimizer import (
    Bf16DistributedOptimizerDiagnosticAdapter,
    DistributedOptimizerEventStatus,
)
from .function_response import (
    RESPONSE_FAMILIES,
    TIER1_KEYS,
    FunctionResponseProbe,
    ResponseAccumulator,
    ResponseFamily,
    derive_tier1_summaries,
    discover_response_hooks,
)
from .registry import MetricFamily
from .secant import (
    TIER2_OUTPUT_KEYS,
    SecantCellDescriptor,
    SecantObservation,
    SecantStatistics,
    SecantSufficientStatisticsView,
    build_secant_registry,
    derive_secant_cell,
    derive_tier2_outputs,
)

_FAMILY_MAP = {
    ResponseFamily.RESIDUAL: MetricFamily.RESIDUAL,
    ResponseFamily.QKV: MetricFamily.QKV,
    ResponseFamily.ATTN_OUT: MetricFamily.ATTN_OUT,
    ResponseFamily.FC1: MetricFamily.FC1,
    ResponseFamily.FC2: MetricFamily.FC2,
}


def diagnostics_requested_tier(args: Any) -> int:
    """Return the requested in-process tier while preserving old Tier-0 flags."""

    if not bool(getattr(args, "diagnostic_heartbeat", False)):
        return -1
    if not bool(getattr(args, "diag_enabled", False)):
        return 0
    value = int(getattr(args, "diag_max_tier", 0))
    if value not in (0, 1, 2):
        raise ValueError("diag_max_tier must be one of 0, 1, or 2")
    return value


def wrap_stable_training_dataset(dataset: Any, args: Any) -> Any:
    """Attach sampler-issued identities only for active Tier-1/2 training."""

    if dataset is None or diagnostics_requested_tier(args) < 1:
        return dataset
    if isinstance(dataset, StableSampleDataset):
        return dataset
    return StableSampleDataset(dataset)


class TieredDiagnosticRuntime:
    """Own selected-token replay and full-step secant state for one trainer."""

    def __init__(
        self,
        args: Any,
        model: Sequence[torch.nn.Module],
        optimizer: object,
        forward_step_func: Callable[..., Any],
        forward_backward_func: Callable[..., Any],
        *,
        reduction_binding: ReductionBinding,
        num_microbatches: int,
        fatal_abort: Callable[[BaseException], None] | None = None,
    ) -> None:
        self.args = args
        self.wrapped_models = tuple(model)
        unwrapped = unwrap_model(list(model))
        self.models = tuple(unwrapped if isinstance(unwrapped, list) else (unwrapped,))
        self.optimizer = optimizer
        self.forward_step_func = forward_step_func
        self.forward_backward_func = forward_backward_func
        self.reduction_binding = reduction_binding
        self.tier = diagnostics_requested_tier(args)
        self.required_tier = int(getattr(args, "diag_require_tier", 0))
        self.num_microbatches = int(num_microbatches)
        self.fatal_abort = fatal_abort or ProductionFatalAbort()
        self.device = self._model_device()
        self.recorder: ReplayBatchRecorder | None = None
        self.transaction: Tier1ReplayTransaction | None = None
        self.response: ResponseAccumulator | None = None
        self._attempt_due = False
        self._schedule_started = False
        self._full_displacement_sq = torch.zeros(
            (), dtype=torch.float64, device=self.device
        )
        self._midpoint_displacement_sq = torch.zeros_like(
            self._full_displacement_sq
        )
        self._restore_verified = torch.zeros(
            (), dtype=torch.bool, device=self.device
        )

        if self.tier < 1:
            self.probe = None
            self.secant = None
            self.secant_binding = None
            return
        self._validate_first_backend()
        if self.tier < 1:
            self.probe = None
            self.secant = None
            self.secant_binding = None
            return
        if self.required_tier > self.tier:
            raise RuntimeError(
                f"required diagnostic tier {self.required_tier} exceeds effective tier {self.tier}"
            )

        local_layers = tuple(
            sorted(
                module.layer_number - 1
                for model_chunk in self.models
                for module in model_chunk.modules()
                if type(module) is TransformerLayer
            )
        )
        descriptors = discover_response_hooks(
            self.models,
            global_layers=int(args.num_layers),
            expected_local_layers=local_layers,
            tensor_parallel_rank=self._tp_rank(),
            sequence_parallel=bool(getattr(args, "sequence_parallel", False)),
        )
        self.probe = FunctionResponseProbe(
            descriptors,
            global_layers=int(args.num_layers),
            device=self.device,
            expected_hook_calls=self.num_microbatches,
            sequence_parallel=bool(getattr(args, "sequence_parallel", False)),
            attention_owner=True,
            attention_required=True,
            retain_secant_endpoints=self.tier >= 2,
            reduction_binding=reduction_binding,
        )
        self.secant_binding = None
        self.secant = None
        if self.tier >= 2:
            local_owners = {descriptor.key: descriptor.owner for descriptor in descriptors}
            cells = tuple(
                SecantCellDescriptor(
                    logical_name=f"layer_{layer}/{family.value}",
                    family=_FAMILY_MAP[family],
                    global_layer=layer,
                    local_owner=bool(local_owners.get((layer, family), False)),
                )
                for layer in range(int(args.num_layers))
                for family in RESPONSE_FAMILIES
            )
            self.secant_binding = build_secant_registry(
                cells, reduction_binding=reduction_binding
            )
            self.secant = SecantStatistics(self.secant_binding, self.device)

        self._dp_binding, self._dp_readiness = self._data_parallel_control()
        self._mp_binding, self._mp_readiness = self._cuda_control(
            "model_parallel", self._model_parallel_group()
        )
        self._world_binding, self._world_readiness = self._cuda_control("world", None)
        maximum_local_samples = max(
            1,
            self.num_microbatches * int(getattr(args, "micro_batch_size", 1)),
        )
        self.population_workspace = PopulationCollectiveWorkspace(
            self._dp_binding,
            maximum_local_samples=maximum_local_samples,
            device="cpu",
        )

    @property
    def active(self) -> bool:
        """Whether selected-token replay is enabled for this run."""

        return self.tier >= 1

    @property
    def output_keys(self) -> tuple[str, ...]:
        """Return exact ordered scalar additions for the effective tier."""

        if self.tier < 1:
            return ()
        return (*TIER1_KEYS, *(TIER2_OUTPUT_KEYS if self.tier >= 2 else ()))

    @property
    def accumulators(self) -> tuple[PackedSufficientStatistics, ...]:
        """Return persistent packs to append to the heartbeat reduction arenas."""

        if self.probe is None:
            return ()
        packed = [self.probe.accumulator.statistics]
        if self.secant is not None:
            packed.append(self.secant.accumulator)
        return tuple(packed)

    def prepare_attempt(self, *, due: bool, num_microbatches: int) -> None:
        """Reset event-local replay state for one rerun attempt."""

        self.abort_attempt()
        self._attempt_due = bool(due and self.active)
        if not self._attempt_due:
            return
        if num_microbatches != self.num_microbatches:
            raise RuntimeError("diagnostic replay microbatch count changed after startup")

    def wrap_data_iterator(self, data_iterator: Any) -> Any:
        """Record raw CPU batches while transparently advancing the real iterator."""

        if not self._attempt_due:
            return data_iterator
        if isinstance(data_iterator, (list, tuple)):
            raise RuntimeError("Tier-1/2 replay rejects virtual-pipeline iterators")
        if data_iterator is None:
            raise RuntimeError("Tier-1/2 replay requires a training data iterator")
        self.recorder = ReplayBatchRecorder(
            data_iterator,
            maximum_batches=self.num_microbatches,
            maximum_host_bytes=int(
                getattr(
                    self.args,
                    "diag_replay_input_bytes_per_rank",
                    256 * 1024**2,
                )
            ),
        )
        return self.recorder

    def run_pre(self, *, event_id: int) -> None:
        """Build one immutable replay plan and execute PRE before the real update."""

        if not self._attempt_due:
            return
        error: BaseException | None = None
        try:
            if self.recorder is None:
                raise RuntimeError("diagnostic replay did not record the training schedule")
            plan = build_distributed_source_plan(
                self.recorder.recorded,
                workspace=self.population_workspace,
                readiness=self._dp_readiness,
                probe_tokens=min(
                    int(getattr(self.args, "diag_max_valid_positions_global", 4096)),
                    self.num_microbatches
                    * int(getattr(self.args, "micro_batch_size", 1)),
                ),
                run_seed=int(getattr(self.args, "diag_sample_seed", self.args.seed)),
                event_id=event_id,
                micro_batch_size=int(self.args.micro_batch_size),
                maximum_microbatches=self.num_microbatches,
            )
            assert self.probe is not None
            self.probe.reset_event(expected_hook_calls=plan.num_microbatches)
            if self.secant is not None:
                self.secant.accumulator.reset_()
            schedule = NonInterleavedReplaySchedule(
                forward_backward_func=self.forward_backward_func,
                forward_step_func=self.forward_step_func,
                model=self.models,
                sequence_length=int(self.args.seq_length),
                micro_batch_size=int(self.args.micro_batch_size),
                probe_device=self.device,
                tensor_parallel_rank=self._tp_rank(),
                tensor_parallel_size=self._tp_size(),
                context_parallel_rank=0,
                context_parallel_size=1,
                sequence_parallel=bool(getattr(self.args, "sequence_parallel", False)),
                pipeline_data_owner=True,
                virtual_pipeline_size=getattr(
                    self.args, "virtual_pipeline_model_parallel_size", None
                ),
                decoder_sequence_length=getattr(self.args, "decoder_seq_length", None),
            )
            engine = Tier1ReplayEngine(
                self.models,
                readiness=self._mp_readiness,
                mutable_buffer_names=self._mutable_buffer_names(),
                tracker_getter=get_cuda_rng_tracker,
                cuda_device=self.device if self.device.type == "cuda" else None,
                fatal_abort=self.fatal_abort,
            )
            maximum_extra = getattr(
                self.args,
                "diag_max_extra_allocated_bytes_per_rank",
                getattr(self.args, "diagnostic_max_extra_bytes", 8 * 1024**3),
            )
            if maximum_extra is None:
                maximum_extra = 8 * 1024**3
            currently_reserved = (
                torch.cuda.memory_reserved(self.device)
                if self.device.type == "cuda"
                else 0
            )
            total_device = (
                torch.cuda.get_device_properties(self.device).total_memory
                if self.device.type == "cuda"
                else 1 << 50
            )
            self.transaction = engine.prepare(
                plan=plan,
                probe=self.probe,
                schedule=schedule,
                memory_policy=ReplayMemoryPolicy(
                    headroom_fraction=float(
                        getattr(self.args, "diag_max_extra_allocated_fraction", 0.08)
                    )
                ),
                maximum_extra_bytes=int(maximum_extra),
                currently_reserved_bytes=int(currently_reserved),
                total_device_bytes=int(total_device),
            )
            self.transaction.run_pre()
            self._schedule_started = True
        except BaseException as caught:
            error = caught
        self._settle_world(error, "pre endpoint", fatal=False)

    def complete_optimizer_event(
        self,
        adapter: Bf16DistributedOptimizerDiagnosticAdapter,
        update_accumulator: PackedSufficientStatistics,
        *,
        update_successful: bool,
    ) -> None:
        """Finish Tier 1 or the four-endpoint Tier-2 secant transaction."""

        if not self._attempt_due:
            return
        if self.transaction is None:
            self._fatal(RuntimeError("diagnostic replay transaction is absent"))
        assert self.transaction is not None
        if update_successful:
            try:
                self._require_adapter_ok(adapter, "pre-update snapshot")
            except BaseException as error:
                self._fatal(error)
            if not adapter.armed:
                self._fatal(RuntimeError("diagnostic optimizer snapshot was not armed"))
        if not update_successful:
            adapter.abort_event()
            self.transaction.finish(update_succeeded=False)
            self.transaction = None
            return
        if self.tier == 1:
            adapter.finish_event(update_accumulator, update_successful=True)
            self.response = self.transaction.finish(update_succeeded=True)
            self.transaction = None
            return
        self._complete_secant(adapter, update_accumulator)

    def _complete_secant(
        self,
        adapter: Bf16DistributedOptimizerDiagnosticAdapter,
        update_accumulator: PackedSufficientStatistics,
    ) -> None:
        assert self.transaction is not None
        assert self.probe is not None
        assert self.secant is not None
        failure: BaseException | None = None
        try:
            if adapter.commit_secant_delta(
                update_accumulator, update_successful=True
            ) is None:
                raise RuntimeError("optimizer secant delta commit failed")
            self._require_adapter_ok(adapter, "post-update delta commit")
            delta = adapter.secant_delta_buffer
            if delta is None:
                raise RuntimeError("optimizer secant delta buffer is unavailable")
            self._full_displacement_sq.zero_()
            for chunk in delta.split(65_536):
                self._full_displacement_sq.add_(
                    torch.sum(chunk.to(dtype=torch.float64).square())
                )
            self._all_reduce_sum(self._full_displacement_sq)
            self._midpoint_displacement_sq.copy_(self._full_displacement_sq)
            self._midpoint_displacement_sq.mul_(0.25)

            self.transaction.run_endpoint("post")
            self.transaction.run_endpoint("post_repeat")
            adapter.install_secant_midpoint()
            self._require_adapter_ok(adapter, "midpoint install")
            self._materialize_parameters()
            self.transaction.run_endpoint("midpoint")
            self._accumulate_secant_rows()
        except BaseException as caught:
            failure = caught
        finally:
            restore_error: BaseException | None = None
            try:
                adapter.restore_secant_post()
                self._require_adapter_ok(adapter, "post restore")
                self._materialize_parameters()
                self._restore_verified.fill_(True)
            except BaseException as caught:
                self._restore_verified.fill_(False)
                restore_error = caught
            if restore_error is not None:
                failure = (
                    restore_error
                    if failure is None
                    else BaseExceptionGroup(
                        "secant endpoint and restoration failed", (failure, restore_error)
                    )
                )
        if failure is not None:
            self._fatal(failure)
        self.response = self.transaction.finalize_endpoints()
        self.transaction = None
        adapter.release_secant_event()

    def derive_outputs(self) -> dict[str, torch.Tensor]:
        """Derive exact ordered Tier-1/2 scalars after the shared reduction."""

        if self.tier < 1:
            return {}
        if self.response is None:
            raise RuntimeError("diagnostic response accumulator is unavailable")
        payload = derive_tier1_summaries(self.response)
        if tuple(payload) != TIER1_KEYS:
            raise RuntimeError("Tier-1 runtime payload order changed")
        if self.tier >= 2:
            assert self.secant_binding is not None and self.secant is not None
            metrics = tuple(
                derive_secant_cell(
                    SecantSufficientStatisticsView.from_accumulator(
                        self.secant_binding, self.secant.accumulator, cell
                    ),
                    midpoint_displacement_sq=self._midpoint_displacement_sq,
                    full_displacement_sq=self._full_displacement_sq,
                    restore_verified=self._restore_verified,
                    replay_floor_multiplier=float(
                        getattr(
                            self.args,
                            "diag_tier2_min_response_over_replay_floor",
                            10.0,
                        )
                    ),
                    midpoint_min=0.5
                    - float(getattr(self.args, "diag_tier2_midpoint_tolerance", 0.05)),
                    midpoint_max=0.5
                    + float(getattr(self.args, "diag_tier2_midpoint_tolerance", 0.05)),
                )
                for cell in self.secant_binding.cells
            )
            tier2 = derive_tier2_outputs(metrics)
            if tuple(tier2) != TIER2_OUTPUT_KEYS:
                raise RuntimeError("Tier-2 runtime payload order changed")
            payload.update(tier2)
        if tuple(payload) != self.output_keys:
            raise RuntimeError("tiered diagnostic payload does not match exact key order")
        return payload

    def abort_attempt(self) -> None:
        """Best-effort release without changing heartbeat cadence counters."""

        if self.transaction is not None:
            try:
                self.transaction.release()
            except BaseException:
                pass
        if self.recorder is not None:
            self.recorder.clear()
        if self.probe is not None:
            self.probe.release()
        self.recorder = None
        self.transaction = None
        self.response = None
        self._attempt_due = False
        self._schedule_started = False

    def _accumulate_secant_rows(self) -> None:
        assert self.probe is not None
        assert self.secant_binding is not None and self.secant is not None
        descriptors = {descriptor.key: descriptor for descriptor in self.probe.descriptors}
        empty = torch.empty(
            0,
            dtype=self.models[0].config.params_dtype,
            device=self.device,
        )
        observations = []
        for cell in self.secant_binding.cells:
            family = next(
                family
                for family in RESPONSE_FAMILIES
                if _FAMILY_MAP[family] == cell.family
                and cell.logical_name.endswith(f"/{family.value}")
            )
            key = (cell.global_layer, family)
            descriptor = descriptors.get(key)
            endpoints = []
            for phase in ("pre", "post", "post_repeat", "midpoint"):
                rows = self.probe.secant_endpoint_rows(phase, key)
                if descriptor is None or not descriptor.owner:
                    endpoints.append(empty)
                elif not rows:
                    raise RuntimeError(f"missing {phase} rows for {cell.logical_name}")
                else:
                    endpoints.append(torch.cat(rows, dim=0).contiguous())
            observations.append(
                SecantObservation(cell.logical_name, *endpoints)
            )
        self.secant.add_observations(tuple(observations))

    def _materialize_parameters(self) -> None:
        for model_chunk in self.wrapped_models:
            start = getattr(model_chunk, "start_param_sync", None)
            if not callable(start):
                raise RuntimeError("distributed model chunk lacks parameter materialization")
            start(force_sync=True, force_dispatch=True)

    def _require_adapter_ok(
        self,
        adapter: Bf16DistributedOptimizerDiagnosticAdapter,
        phase: str,
    ) -> None:
        status = adapter.status_for_event_consensus().detach().clone()
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
        if int(status.reshape(-1)[0]) != int(DistributedOptimizerEventStatus.OK):
            raise RuntimeError(f"{phase} failed with adapter status {int(status.reshape(-1)[0])}")

    def _all_reduce_sum(self, tensor: torch.Tensor) -> None:
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def _settle_world(
        self, error: BaseException | None, phase: str, *, fatal: bool
    ) -> None:
        try:
            self._world_readiness.settle(error, phase)
        except BaseException as settled:
            if fatal or self._schedule_started:
                self._fatal(settled)
            raise

    def _fatal(self, error: BaseException) -> None:
        self.fatal_abort(error)
        raise RuntimeError("diagnostic fatal-abort protocol returned") from error

    def _validate_first_backend(self) -> None:
        reasons = []
        if self._tp_size() != 1:
            reasons.append("tensor_parallel_size_must_be_1")
        if int(getattr(self.args, "context_parallel_size", 1)) != 1:
            reasons.append("context_parallel_size_must_be_1")
        if int(getattr(self.args, "pipeline_model_parallel_size", 1)) > 2:
            reasons.append("pipeline_parallel_size_must_be_at_most_2")
        if len(self.models) != 1:
            reasons.append("one_noninterleaved_model_chunk_required")
        if getattr(self.args, "transformer_impl", "transformer_engine") != "local":
            reasons.append("local_transformer_required")
        if reasons:
            if self.required_tier >= 1 or getattr(
                self.args, "diagnostic_unsupported_policy", "error"
            ) == "error":
                raise RuntimeError(
                    "Tier-1/2 diagnostic backend is unsupported: " + ", ".join(reasons)
                )
            self.tier = 0

    def _mutable_buffer_names(self) -> tuple[str, ...]:
        return tuple(
            f"{model_index}:{module_name + '.' if module_name else ''}{buffer_name}"
            for model_index, model in enumerate(self.models)
            for module_name, module in model.named_modules()
            for buffer_name in module._buffers
        )

    def _data_parallel_control(
        self,
    ) -> tuple[CollectiveBinding, ReadinessConsensus]:
        if not dist.is_available() or not dist.is_initialized():
            binding = CollectiveBinding("data_parallel", None, 1)
        else:
            group = parallel_state.get_data_parallel_group_gloo(
                with_context_parallel=False
            )
            binding = CollectiveBinding(
                "data_parallel", group, dist.get_world_size(group)
            )
        return binding, ReadinessConsensus(binding, "cpu")

    def _cuda_control(
        self, identity: str, group: object | None
    ) -> tuple[CollectiveBinding, ReadinessConsensus]:
        size = dist.get_world_size(group) if dist.is_available() and dist.is_initialized() else 1
        binding = CollectiveBinding(identity, group, size)
        device: torch.device | str = self.device if size > 1 else "cpu"
        if size == 1 and dist.is_available() and dist.is_initialized():
            backend = str(dist.get_backend(group)).lower()
            device = self.device if backend == "nccl" else "cpu"
        return binding, ReadinessConsensus(binding, device)

    def _model_parallel_group(self) -> object | None:
        if not dist.is_available() or not dist.is_initialized():
            return None
        return parallel_state.get_model_parallel_group()

    def _model_device(self) -> torch.device:
        for model in self.models:
            parameter = next(model.parameters(), None)
            if parameter is not None:
                return parameter.device
        return torch.device("cpu")

    @staticmethod
    def _tp_rank() -> int:
        return (
            parallel_state.get_tensor_model_parallel_rank()
            if parallel_state.model_parallel_is_initialized()
            else 0
        )

    @staticmethod
    def _tp_size() -> int:
        return (
            parallel_state.get_tensor_model_parallel_world_size()
            if parallel_state.model_parallel_is_initialized()
            else 1
        )


__all__ = [
    "TieredDiagnosticRuntime",
    "diagnostics_requested_tier",
    "wrap_stable_training_dataset",
]
