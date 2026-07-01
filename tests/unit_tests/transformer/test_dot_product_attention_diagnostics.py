# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest import mock

import torch

from megatron.core.transformer import dot_product_attention as attention_module
from megatron.core.transformer.dot_product_attention import DotProductAttention


class _CpuGlobalMemoryBuffer:
    def get_tensor(
        self, shape: tuple[int, ...], dtype: torch.dtype, _name: str
    ) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype)


def test_local_dot_product_attention_observes_pre_dropout_softmax_boundary() -> None:
    attention = DotProductAttention.__new__(DotProductAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(sequence_parallel=True)
    attention.layer_number = 7
    attention.num_attention_heads_per_partition = 1
    attention.num_query_groups_per_partition = 1
    attention.hidden_size_per_partition = 2
    attention.softmax_scale = 1.0
    attention.softmax_offset = None
    attention.scale_mask_softmax = lambda scores, _mask, _offset: torch.softmax(
        scores, dim=-1
    )
    attention.attention_dropout = torch.nn.Identity()

    query = torch.tensor([[[[1.0, 0.0]]], [[[0.0, 1.0]]]])
    key = query.clone()
    value = query.clone()
    with (
        mock.patch.object(
            attention_module.parallel_state,
            "get_global_memory_buffer",
            return_value=_CpuGlobalMemoryBuffer(),
        ),
        mock.patch.object(attention_module, "observe_diagnostic_attention") as observer,
    ):
        output = attention(query, key, value, attention_mask=None)

    assert output.shape == (2, 1, 2)
    observer.assert_called_once()
    global_layer, logits, probabilities = observer.call_args.args
    assert global_layer == 6
    torch.testing.assert_close(logits, torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]))
    torch.testing.assert_close(probabilities, torch.softmax(logits, dim=-1))
