# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.training.diagnostic_layer_selection import (
    DEFAULT_LAYER_PATTERN,
    DiagnosticLayerSelector,
    selected_global_layer_ids,
    selected_layers,
)


def test_log4pluslast_selects_power_of_four_and_predecessor_pairs():
    # 1-indexed internally: 0/1, 3/4, 15/16, last=20 in 0-indexed layer names.
    assert selected_layers(21, "log4pluslast") == {1, 2, 4, 5, 16, 17, 21}


def test_log4firstlast_selects_first_powers_of_four_and_last():
    assert selected_layers(16, "log4firstlast") == {1, 4, 16}
    assert selected_layers(17, "log4firstlast") == {1, 4, 16, 17}
    assert selected_global_layer_ids(12, "log4firstlast") == (0, 3, 11)


def test_log4plusonelast_alias_matches_log4pluslast():
    assert selected_layers(21, "log4plusonelast") == selected_layers(21, "log4pluslast")


def test_log4pluslast_is_default():
    assert DEFAULT_LAYER_PATTERN == "log4pluslast"
    assert selected_layers(21, None) == {1, 2, 4, 5, 16, 17, 21}


def test_selector_allows_only_requested_layers_and_special_anchors():
    selector = DiagnosticLayerSelector(
        pattern="log4pluslast",
        num_layers=21,
        include_special=True,
    )

    assert selector.allows("decoder.layers.0.mlp.linear_fc1")
    assert selector.allows("decoder.layers.1.mlp.linear_fc1")
    assert not selector.allows("decoder.layers.2.mlp.linear_fc1")
    assert selector.allows("decoder.layers.3.mlp.linear_fc1")
    assert selector.allows("decoder.layers.4.mlp.linear_fc1")
    assert not selector.allows("decoder.layers.5.mlp.linear_fc1")
    assert selector.allows("decoder.layers.15.mlp.linear_fc1")
    assert selector.allows("decoder.layers.16.mlp.linear_fc1")
    assert selector.allows("decoder.layers.20.mlp.linear_fc1")
    assert selector.allows("embedding.word_embeddings")
    assert selector.allows("output_layer")


def test_selector_can_exclude_special_anchors():
    selector = DiagnosticLayerSelector(
        pattern="log4pluslast",
        num_layers=21,
        include_special=False,
    )

    assert not selector.allows("embedding.word_embeddings")
    assert not selector.allows("output_layer")
