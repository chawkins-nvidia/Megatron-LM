# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import json
import math
from pathlib import Path

import pytest
import torch

from megatron.training.diagnostics.accumulator import ReductionBinding
from megatron.training.diagnostics.function_response import (
    RESPONSE_FAMILIES,
    TIER1_KEYS,
    FunctionResponseProbe,
    FunctionResponseResult,
    ResponseFamily,
    ResponseHookDescriptor,
    ResponseHookRegistration,
    _selected_rows,
    derive_tier1_summaries,
    descriptor_fingerprint,
)


def _descriptors(layers: int, *, owner: bool = True):
    descriptors = []
    modules = []
    for layer in range(layers):
        for family in RESPONSE_FAMILIES:
            module = torch.nn.Identity()
            modules.append(module)
            descriptors.append(
                ResponseHookDescriptor(
                    global_layer=layer,
                    family=family,
                    module=module,
                    owner=owner,
                    sequence_sharded=False,
                    affine_bias_output=family != ResponseFamily.RESIDUAL,
                )
            )
    return tuple(descriptors), tuple(modules)


def _probe(layers: int, *, owner: bool = True, expected_hook_calls: int = 0):
    descriptors, modules = _descriptors(layers, owner=owner)
    probe = FunctionResponseProbe(
        descriptors,
        global_layers=layers,
        device="cpu",
        expected_hook_calls=expected_hook_calls,
        reduction_binding=ReductionBinding.flat_world(None),
        scratch_element_capacity=2,
    )
    probe.bind_preflight(
        selected_row_capacity=max(expected_hook_calls, 1),
        response_widths={family: 1 for family in RESPONSE_FAMILIES},
        response_dtype=torch.float32,
        attention_heads=1,
        attention_key_length=2,
    )
    return probe, modules


def _add_relative_response(probe: FunctionResponseProbe, slot: int, response: float) -> None:
    before = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
    after = before.float() * (1 + response)
    probe.registry.add_update(
        probe.accumulator.statistics, probe.registry.slot_names[slot], before, after
    )


def test_capture_returns_exact_immutable_probe_hook_registrations_and_cleans_up() -> None:
    probe, modules = _probe(1)

    with probe.capture_pre() as registrations:
        assert type(registrations) is tuple
        assert len(registrations) == len(probe.descriptors)
        for descriptor, module, registration in zip(
            probe.descriptors, modules, registrations, strict=True
        ):
            assert type(registration) is ResponseHookRegistration
            assert registration.descriptor is descriptor
            assert registration.module is module
            assert registration.registry_name == "_forward_hooks"
            assert type(registration.handle_id) is int
            assert module._forward_hooks[registration.handle_id] is registration.hook

    assert probe._handles == []
    assert all(not module._forward_hooks for module in modules)


def test_tier2_probe_retains_exact_four_endpoints_and_reuses_packs() -> None:
    descriptors, modules = _descriptors(1)
    probe = FunctionResponseProbe(
        descriptors,
        global_layers=1,
        device="cpu",
        expected_hook_calls=1,
        attention_owner=False,
        attention_required=False,
        retain_secant_endpoints=True,
        reduction_binding=ReductionBinding.flat_world(None),
        scratch_element_capacity=2,
    )
    probe.bind_preflight(
        selected_row_capacity=1,
        response_widths={family: 1 for family in RESPONSE_FAMILIES},
        response_dtype=torch.float32,
        attention_heads=1,
        attention_key_length=1,
    )

    for phase, value in (
        ("pre", 1.0),
        ("post", 3.0),
        ("post_repeat", 3.0),
        ("midpoint", 2.0),
    ):
        probe.set_masks(torch.ones((1, 1), dtype=torch.bool))
        with probe.capture_endpoint(phase):
            for module in modules:
                module(torch.full((1, 1, 1), value))

    accumulator = probe.finalize()
    key = descriptors[0].key
    assert tuple(
        float(probe.secant_endpoint_rows(phase, key)[0].item())
        for phase in ("pre", "post", "post_repeat", "midpoint")
    ) == (1.0, 3.0, 3.0, 2.0)
    accumulator.finalize_local_()
    assert derive_tier1_summaries(accumulator)[
        "diag/v2/t1/response/residual/dy_rel/first"
    ] == pytest.approx(2.0)

    sum_storage = accumulator.statistics.sum_pack.untyped_storage().data_ptr()
    probe.reset_event(expected_hook_calls=1)
    assert accumulator.statistics.sum_pack.untyped_storage().data_ptr() == sum_storage
    assert all(not probe.secant_endpoint_rows(phase, key) for phase in probe._secant_rows)


def test_additional_response_endpoints_are_fail_closed_by_default() -> None:
    probe, _modules = _probe(1, expected_hook_calls=1)

    with pytest.raises(RuntimeError, match="Tier-2 retention"):
        with probe.capture_endpoint("midpoint"):
            pass


def test_tier1_schema_is_exactly_30_unique_canonical_keys() -> None:
    fixture_path = Path(
        "/home/chawkins/src/scaling-worktrees/issue-209-launch-review/"
        "tests/fixtures/diag_v2_canonical_keys.json"
    )
    fixture = json.loads(fixture_path.read_bytes())
    approved = tuple(fixture["tier1_core"] + fixture["tier1_attention"])

    assert len(TIER1_KEYS) == 30
    assert len(set(TIER1_KEYS)) == 30
    assert b"\0".join(key.encode() for key in TIER1_KEYS) == b"\0".join(
        key.encode() for key in approved
    )


def test_exact_pooled_formulas_produce_all_30_outputs() -> None:
    layers = 5
    probe, _modules = _probe(layers)
    for layer in range(layers):
        for family in range(len(RESPONSE_FAMILIES)):
            _add_relative_response(
                probe, layer * len(RESPONSE_FAMILIES) + family, response=(layer + 1) / 16
            )
    probe.accumulator.finalize_local_()

    result = FunctionResponseResult.from_reduced(probe.accumulator)
    payload = derive_tier1_summaries(probe.accumulator)

    assert tuple(payload) == TIER1_KEYS
    assert result.valid.all()
    torch.testing.assert_close(
        result.dy_rel[:, 0], torch.arange(1, layers + 1, dtype=torch.float64) / 16
    )
    assert payload["diag/v2/t1/response/residual/dy_rel/first"] == pytest.approx(1 / 16)
    assert payload["diag/v2/t1/response/residual/dy_rel/last"] == pytest.approx(5 / 16)
    assert all(
        math.isnan(payload[key]) for key in TIER1_KEYS if key.startswith("diag/v2/t1/attention/")
    )


def test_pooled_sums_are_not_an_average_of_rank_or_microbatch_ratios() -> None:
    probe, _modules = _probe(1)
    slot = probe.registry.slot_names[0]
    probe.registry.add_update(
        probe.accumulator.statistics, slot, torch.tensor([1.0]), torch.tensor([2.0])
    )
    probe.registry.add_update(
        probe.accumulator.statistics, slot, torch.tensor([100.0]), torch.tensor([110.0])
    )
    probe.accumulator.finalize_local_()

    result = FunctionResponseResult.from_reduced(probe.accumulator)

    expected = math.sqrt((1.0**2 + 10.0**2) / (1.0**2 + 100.0**2))
    assert result.dy_rel[0, 0] == pytest.approx(expected)
    assert result.dy_rel[0, 0] != pytest.approx((1.0 + 0.1) / 2)


def test_no_population_and_observation_errors_are_invalid_nan() -> None:
    probe, _modules = _probe(2, expected_hook_calls=1)
    accumulator = probe.finalize()
    accumulator.finalize_local_()

    result = FunctionResponseResult.from_reduced(accumulator)
    payload = derive_tier1_summaries(accumulator)

    assert not result.valid.any()
    assert torch.isnan(result.dy_rel).all()
    assert all(
        math.isnan(payload[f"diag/v2/t1/response/{family.value}/starved_fraction"])
        for family in RESPONSE_FAMILIES
    )


def test_attention_keys_derive_from_attention_logits_and_probabilities() -> None:
    probe, _modules = _probe(3, expected_hook_calls=1)
    probe._phase = "post"
    probe.set_masks(torch.tensor([[True, False]]))
    probabilities = torch.tensor([[[[0.5, 0.5], [0.5, 0.5]]]])
    for layer in range(3):
        logits = torch.full_like(probabilities, float(layer + 1))
        probe.observe_attention(layer, logits, probabilities)
    probe._phase = None
    accumulator = probe.finalize()
    accumulator.finalize_local_()

    payload = derive_tier1_summaries(accumulator)

    assert payload["diag/v2/t1/attention/logit_abs_p50"] == pytest.approx(2)
    assert payload["diag/v2/t1/attention/logit_abs_p90"] == pytest.approx(2.8)
    assert payload["diag/v2/t1/attention/entropy_p10"] == pytest.approx(math.log(2))
    assert payload["diag/v2/t1/attention/entropy_p50"] == pytest.approx(math.log(2))
    assert payload["diag/v2/t1/attention/collapse_fraction"] == 0


def test_nonowner_slots_remain_exactly_neutral() -> None:
    probe, _modules = _probe(1, owner=False)
    _add_relative_response(probe, 0, response=0.5)

    assert probe.accumulator.statistics.sum_pack.count_nonzero() == 0
    assert torch.isneginf(probe.accumulator.statistics.max_pack).all()
    assert torch.isposinf(probe.accumulator.statistics.min_pack).all()


def test_rows_are_selected_before_bias_and_unselected_nonfinite_values_are_ignored() -> None:
    activation = torch.tensor(
        [[[1.0, 2.0]], [[torch.inf, torch.inf]], [[5.0, 6.0]], [[torch.nan, torch.nan]]]
    )
    mask = torch.tensor([[True], [False], [True], [False]])
    bias = torch.tensor([0.25, 0.5])

    rows = _selected_rows(activation, mask, bias)

    torch.testing.assert_close(rows, torch.tensor([[1.25, 2.5], [5.25, 6.5]]))
    assert torch.isfinite(rows).all()


def test_selected_pre_rows_are_released_after_each_post_observation() -> None:
    descriptors, _modules = _descriptors(1)
    descriptor = descriptors[0]
    probe = FunctionResponseProbe(
        (descriptor,), global_layers=1, device="cpu", expected_hook_calls=2
    )
    probe.bind_preflight(
        selected_row_capacity=2,
        response_widths={family: 1 for family in RESPONSE_FAMILIES},
        response_dtype=torch.float32,
        attention_heads=1,
        attention_key_length=1,
    )
    mask = torch.tensor([[True], [False]])
    probe.set_masks(mask)
    probe._phase = "pre"
    probe._observe(descriptor, torch.tensor([[[2.0]], [[9.0]]]))
    probe._observe(descriptor, torch.tensor([[[4.0]], [[9.0]]]))
    assert probe.retained_pre_bytes == 8

    probe._phase = "post"
    probe._observe(descriptor, torch.tensor([[[3.0]], [[9.0]]]))
    assert probe.retained_pre_bytes == 4
    probe._observe(descriptor, torch.tensor([[[6.0]], [[9.0]]]))
    assert probe.retained_pre_bytes == 0
    probe._phase = None


def test_extra_pre_hook_observation_rejects_before_cloning() -> None:
    descriptors, _modules = _descriptors(1)
    descriptor = descriptors[0]
    probe = FunctionResponseProbe(
        (descriptor,), global_layers=1, device="cpu", expected_hook_calls=1
    )
    probe.bind_preflight(
        selected_row_capacity=1,
        response_widths={family: 1 for family in RESPONSE_FAMILIES},
        response_dtype=torch.float32,
        attention_heads=1,
        attention_key_length=1,
    )
    probe.set_masks(torch.tensor([[True]]))
    probe._phase = "pre"
    probe._observe(descriptor, torch.ones(1, 1, 1))
    retained = probe.retained_pre_bytes

    with pytest.raises(RuntimeError, match="preflight cardinality"):
        probe._observe(descriptor, torch.ones(1, 1, 1))

    assert probe.retained_pre_bytes == retained
    assert len(probe._pre_rows[descriptor.key]) == 1
    probe._phase = None


@pytest.mark.parametrize(
    ("activation", "message"),
    ((torch.ones(1, 1, 2), "activation"), (torch.ones(1, 1, 1, dtype=torch.float64), "activation")),
)
def test_live_response_shape_and_dtype_must_match_preflight(
    activation: torch.Tensor, message: str
) -> None:
    descriptors, _modules = _descriptors(1)
    descriptor = descriptors[0]
    probe = FunctionResponseProbe(
        (descriptor,), global_layers=1, device="cpu", expected_hook_calls=1
    )
    probe.bind_preflight(
        selected_row_capacity=1,
        response_widths={family: 1 for family in RESPONSE_FAMILIES},
        response_dtype=torch.float32,
        attention_heads=1,
        attention_key_length=1,
    )
    probe.set_masks(torch.tensor([[True]]))
    probe._phase = "pre"

    with pytest.raises(ValueError, match=message):
        probe._observe(descriptor, activation)

    assert probe.retained_pre_bytes == 0
    probe._phase = None


def test_descriptor_order_and_missing_slots_fail_closed() -> None:
    descriptors, _modules = _descriptors(2)

    with pytest.raises(ValueError, match="canonical global slot order"):
        descriptor_fingerprint(tuple(reversed(descriptors)))
    with pytest.raises(ValueError, match="duplicate"):
        descriptor_fingerprint((descriptors[0], descriptors[0]))


def test_descriptor_hash_is_global_and_independent_of_local_ownership() -> None:
    owner, _owner_modules = _probe(3, owner=True)
    nonowner, _nonowner_modules = _probe(3, owner=False)

    assert owner.descriptor_hash == nonowner.descriptor_hash
    assert owner.registry.slot_names == nonowner.registry.slot_names
    assert len(owner.registry.slot_names) == 3 * (len(RESPONSE_FAMILIES) + 3)


def test_canonical_accumulator_scratch_bound_is_used() -> None:
    probe, _modules = _probe(2)
    assert probe.maximum_accumulator_scratch_bytes == 2 * 96
