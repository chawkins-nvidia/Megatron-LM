# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

"""Shared layer-selection helpers for high-cardinality diagnostics.

Layer selectors use 1-indexed layer numbers at the config boundary. Megatron
module names remain 0-indexed internally, so ``decoder.layers.0`` is layer 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


DEFAULT_LAYER_PATTERN = "log2pluslast"

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_TRUE_VALUES = {"1", "true", "yes", "on"}
_SPECIAL_NAME_PARTS = (
    "embedding",
    "embeddings",
    "word_embeddings",
    "position_embeddings",
    "output_layer",
    "lm_head",
    "final_layernorm",
)


def layer_number_from_name(name: str) -> int | None:
    """Return the 1-indexed transformer layer number encoded in ``name``."""
    match = _LAYER_RE.search(name)
    if match is None:
        return None
    return int(match.group(1)) + 1


def is_special_diagnostic_name(name: str) -> bool:
    """Return True for non-layer anchors such as embedding and unembedding."""
    lowered = name.lower()
    return any(part in lowered for part in _SPECIAL_NAME_PARTS)


def infer_num_layers_from_names(names: Iterable[str]) -> int:
    max_layer = 0
    for name in names:
        layer = layer_number_from_name(name)
        if layer is not None:
            max_layer = max(max_layer, layer)
    return max_layer


def selected_layers(num_layers: int, pattern: str | None = None) -> set[int]:
    """Return selected 1-indexed layer numbers for ``pattern``.

    Supported patterns:

    * ``log2pluslast``: layers 1,2,4,... plus the final layer.
    * ``log4plusonelast``: layers 1,2,4,5,16,17,... plus the final layer.
    * ``all``: every layer.
    * ``none``: no transformer layers.
    * ``every:N`` or ``stride:N``: layers 1,1+N,1+2N,... plus the final layer.
    * ``every:N:K``: same, but start at 1-indexed layer K.
    * comma/list patterns such as ``1,2,4,last`` or ``list:1,3,last``.
    """
    if num_layers <= 0:
        return set()
    raw = (pattern or DEFAULT_LAYER_PATTERN).strip().lower()
    if raw in {"", "default"}:
        raw = DEFAULT_LAYER_PATTERN
    if raw in {"all", "*"}:
        return set(range(1, num_layers + 1))
    if raw in {"none", "off"}:
        return set()
    if raw == "log2pluslast":
        layers: set[int] = set()
        layer = 1
        while layer <= num_layers:
            layers.add(layer)
            layer *= 2
        layers.add(num_layers)
        return layers
    if raw == "log4plusonelast":
        layers: set[int] = set()
        layer = 1
        while layer <= num_layers:
            layers.add(layer)
            if layer + 1 <= num_layers:
                layers.add(layer + 1)
            layer *= 4
        layers.add(num_layers)
        return layers
    if raw.startswith(("every:", "stride:")):
        parts = raw.split(":")
        if len(parts) not in {2, 3}:
            raise ValueError(f"invalid diagnostic layer pattern {pattern!r}")
        stride = int(parts[1])
        first = int(parts[2]) if len(parts) == 3 else 1
        if stride <= 0 or first <= 0:
            raise ValueError(f"invalid diagnostic layer pattern {pattern!r}")
        layers = set(range(first, num_layers + 1, stride))
        layers.add(num_layers)
        return {layer for layer in layers if 1 <= layer <= num_layers}
    if raw.startswith("list:"):
        raw = raw[len("list:") :]
    if "," in raw or raw.isdigit() or raw == "last":
        layers = set()
        for piece in raw.split(","):
            token = piece.strip()
            if not token:
                continue
            if token == "last":
                layers.add(num_layers)
            else:
                layer = int(token)
                if 1 <= layer <= num_layers:
                    layers.add(layer)
        return layers
    raise ValueError(f"invalid diagnostic layer pattern {pattern!r}")


def _stream_attr(stream: str) -> str:
    return f"{stream}_layer_pattern"


def _bool_arg(args, name: str, default: bool) -> bool:
    value = getattr(args, name, default) if args is not None else default
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_VALUES
    return bool(value)


def pattern_for_stream(args, stream: str) -> str:
    """Resolve a per-stream layer pattern, falling back to the global default."""
    if args is None:
        return DEFAULT_LAYER_PATTERN
    stream_value = getattr(args, _stream_attr(stream), None)
    if stream_value not in (None, ""):
        return str(stream_value)
    global_value = getattr(args, "diagnostic_layer_pattern", None)
    if global_value not in (None, ""):
        return str(global_value)
    return DEFAULT_LAYER_PATTERN


@dataclass(frozen=True)
class DiagnosticLayerSelector:
    pattern: str
    num_layers: int
    include_special: bool = True

    @property
    def layers(self) -> set[int]:
        return selected_layers(self.num_layers, self.pattern)

    def allows(self, name: str) -> bool:
        layer = layer_number_from_name(name)
        if layer is not None:
            return layer in self.layers
        return self.include_special and is_special_diagnostic_name(name)


def make_layer_selector(
    model,
    args=None,
    *,
    stream: str,
    include_special: bool | None = None,
) -> DiagnosticLayerSelector:
    names: list[str] = []
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        named_modules = getattr(chunk, "named_modules", None)
        if named_modules is not None:
            names.extend(name for name, _module in named_modules())
        named_parameters = getattr(chunk, "named_parameters", None)
        if named_parameters is not None:
            names.extend(name for name, _param in named_parameters())

    num_layers = int(getattr(args, "num_layers", 0) or 0) if args is not None else 0
    if num_layers <= 0:
        num_layers = infer_num_layers_from_names(names)
    if include_special is None:
        include_special = _bool_arg(args, "diagnostic_include_special_layers", True)
    return DiagnosticLayerSelector(
        pattern=pattern_for_stream(args, stream),
        num_layers=num_layers,
        include_special=include_special,
    )


def make_layer_name_filter(
    model,
    args=None,
    *,
    stream: str,
    include_special: bool | None = None,
):
    selector = make_layer_selector(
        model,
        args,
        stream=stream,
        include_special=include_special,
    )
    return selector.allows
