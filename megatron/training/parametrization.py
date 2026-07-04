"""#118 unified, config-driven Parametrization object (init + optimizer).

ONE declarative config drives every parametrization experiment (C0-C5, per-depth R4,
Muon R7). Candidates are *config*, not code. Two stages:

  * a closed-world TYPE REGISTRY classifies each trainable parameter into exactly one
    module type (attr-first, then anchored name globs; similar names are DISTINCT ids), and
  * an ordered RULE TABLE attaches per-(type[, global-depth-range]) init/LR/eps/WD
    multipliers, each a numeric exponent vector over the scale ratios (m_N, m_L, m_B, m_D).

Design + review synthesis:
  investigations/20260605-stability-hpo-parametrization/design/60-review-synthesis.md

Optimizer groups route through the existing, upstream-aligned ParamKey/config_overrides/
_get_param_groups path (deterministic sort + cross-rank key sync). This module adds the
three things that path lacks:
  (1) closed-world classification with STRICT coverage -- every grad param must map to
      exactly one rule; unmatched / unknown-type / multi-match is a hard error (no silent
      default group);
  (2) PP/VPP-aware GLOBAL layer-depth matching, via a `param_global_layer_number` attribute
      stamped at build time from `get_transformer_layer_offset` (local decoder.layers.<i> is
      the wrong, PP-local index);
  (3) a checkpoint MANIFEST recomputed and asserted on resume to catch optimizer-partition
      drift before optimizer state is loaded.

`enabled: false` (C0 negative control) makes this a no-op: no overrides, no init change,
baseline byte-identical.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# NOTE: torch and the megatron optimizer types (ParamKey / ParamWithNamePredicate /
# ParamGroupOverride) are imported LAZILY inside build_config_overrides so the pure-Python
# logic (classification, coverage, depth, manifest, multipliers) can be unit-tested locally
# without a torch/megatron install. Annotations are strings (`from __future__ import annotations`).

logger = logging.getLogger(__name__)

# Attribute stamped onto each layer-owned parameter at build time (see stamp_global_layer_numbers).
# None / absent => the parameter is not inside a transformer layer (embedding, final norm, unembed).
GLOBAL_LAYER_ATTR = "param_global_layer_number"

RATIO_VARS = ("m_N", "m_L", "m_B", "m_D")
OPTIMIZER_ROLES = ("matrix", "fallback_adam", "router_adam", "tied_adam")


# --------------------------------------------------------------------------------------
# Config dataclasses (parsed from the Hydra `parametrization:` block)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TypeSpec:
    """Closed-world module-type recognizer.

    A parameter matches this type iff it passes the optional shape gate AND
    (matches any `attr` OR any `name_glob`). `attr` is preferred (survives name churn);
    globs must be ANCHORED so similar names do not collide (`*.linear_fc1.weight` does NOT
    match `*.linear_fc1_experts.weight`). `fixture` is a representative name for tests/docs.
    """

    id: str
    attr: Tuple[str, ...] = ()
    name_globs: Tuple[str, ...] = ()
    exclude_globs: Tuple[str, ...] = ()  # negative match: if any matches, this is NOT the type
    min_dim: Optional[int] = None
    max_dim: Optional[int] = None
    fixture: str = ""

    def matches(self, param: "torch.nn.Parameter", name: str) -> bool:
        dim = param.dim()
        if self.min_dim is not None and dim < self.min_dim:
            return False
        if self.max_dim is not None and dim > self.max_dim:
            return False
        for g in self.exclude_globs:
            if fnmatch.fnmatch(name, g):
                return False
        for a in self.attr:
            if getattr(param, a, False):
                return True
        for g in self.name_globs:
            if fnmatch.fnmatch(name, g):
                return True
        return False


@dataclass(frozen=True)
class Rule:
    """One row of the rule table: attach multipliers to a set of types and an optional
    GLOBAL depth range [start, end] (inclusive). Multipliers are exponent vectors over the
    ratio variables, e.g. ``{"m_N": -1, "m_L": -0.5}`` -> m_N**-1 * m_L**-0.5. An optional
    ``const`` key multiplies a scalar. ``optimizer_role`` is one closed-world semantic role
    compiled to either the selected matrix optimizer or Adam fallback. An empty multiplier
    dict means 1.0 while an optimizer role still emits a routing override."""

    name: str
    types: Tuple[str, ...]
    optimizer_role: Optional[str] = None
    depth_start: Optional[int] = None
    depth_end: Optional[int] = None
    init_std: Dict[str, float] = field(default_factory=dict)
    lr: Dict[str, float] = field(default_factory=dict)
    eps: Dict[str, float] = field(default_factory=dict)
    wd: Dict[str, float] = field(default_factory=dict)

    def depth_ok(self, param: "torch.nn.Parameter") -> bool:
        if self.depth_start is None and self.depth_end is None:
            return True
        d = getattr(param, GLOBAL_LAYER_ATTR, None)
        if d is None:
            return False
        if self.depth_start is not None and d < self.depth_start:
            return False
        if self.depth_end is not None and d > self.depth_end:
            return False
        return True


def _as_plain_config(value: Any) -> Any:
    """Convert YAML SimpleNamespace/list trees back into plain dict/list trees."""
    if isinstance(value, dict):
        return {k: _as_plain_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain_config(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _as_plain_config(v) for k, v in vars(value).items()}
    return value


def _mult_block(rule: dict, old_key: str, new_key: str) -> Dict[str, float]:
    """Read a multiplier block, accepting old internal names and new explicit names.

    Candidate files historically used ``init_std``, ``lr``, ``eps``, and ``wd``
    for multiplier blocks. Hydra-facing configs should use the explicit
    ``*_mult`` names so the values cannot be mistaken for resolved optimizer or
    initialization values.
    """
    old_value = rule.get(old_key)
    new_value = rule.get(new_key)
    if old_value is not None and new_value is not None and old_value != new_value:
        raise ValueError(
            f"[#118 param] rule {rule.get('name')!r} sets both {old_key} and {new_key}; "
            "use only the explicit *_mult key for inline Hydra config."
        )
    return dict((new_value if new_value is not None else old_value) or {})


@dataclass
class ParametrizationConfig:
    enabled: bool = False
    ratios: Dict[str, float] = field(default_factory=lambda: {v: 1.0 for v in RATIO_VARS})
    alpha: float = 0.5  # load-bearing depth exponent: residual branch mult = C * m_L ** (-alpha)
    width_base: Optional[int] = None  # base hidden width N_0; used to derive m_N
    depth_base: Optional[int] = None  # base depth L_0; None => depth scaling OFF (mult 1.0, inert)
    residual_const: float = 1.0  # backward-compatible default C for both residual branch types
    residual_attention_const: Optional[float] = None  # attention branch C; None => residual_const
    residual_mlp_const: Optional[float] = None  # MLP branch C; None => residual_const
    on_unmatched: str = "error"  # "error" is the only supported strict mode
    require_unique: bool = True  # each param must match exactly one rule
    allow_shadowing: bool = False  # if True, first matching rule wins instead of erroring
    types: Tuple[TypeSpec, ...] = ()
    expected_types: Tuple[str, ...] = ()
    rules: Tuple[Rule, ...] = ()

    @staticmethod
    def from_dict(d: Optional[dict]) -> "ParametrizationConfig":
        d = _as_plain_config(d)
        if not d or not d.get("enabled", False):
            return ParametrizationConfig(enabled=False)
        ratios = {v: float(d.get("ratios", {}).get(v, 1.0)) for v in RATIO_VARS}
        types = tuple(
            TypeSpec(
                id=tid,
                attr=tuple(spec.get("attr", []) or []),
                name_globs=tuple(spec.get("name_globs", []) or []),
                exclude_globs=tuple(spec.get("exclude_globs", []) or []),
                min_dim=spec.get("min_dim"),
                max_dim=spec.get("max_dim"),
                fixture=spec.get("fixture", ""),
            )
            for tid, spec in (d.get("type_registry", {}) or {}).items()
        )
        rules = tuple(
            Rule(
                name=r["name"],
                types=tuple(r["types"]),
                optimizer_role=r.get("optimizer_role"),
                depth_start=(r.get("depth") or {}).get("start"),
                depth_end=(r.get("depth") or {}).get("end"),
                init_std=_mult_block(r, "init_std", "init_std_mult"),
                lr=_mult_block(r, "lr", "lr_mult"),
                eps=_mult_block(r, "eps", "eps_mult"),
                wd=_mult_block(r, "wd", "wd_mult"),
            )
            for r in (d.get("rules", []) or [])
        )
        width_base = d.get("width_base")
        depth_base = d.get("depth_base")
        return ParametrizationConfig(
            enabled=True,
            ratios=ratios,
            alpha=float(d.get("alpha", 0.5)),
            width_base=int(width_base) if width_base is not None else None,
            depth_base=int(depth_base) if depth_base is not None else None,
            residual_const=float(d.get("residual_const", 1.0)),
            residual_attention_const=(
                float(d["residual_attention_const"])
                if d.get("residual_attention_const") is not None
                else None
            ),
            residual_mlp_const=(
                float(d["residual_mlp_const"]) if d.get("residual_mlp_const") is not None else None
            ),
            on_unmatched=d.get("on_unmatched", "error"),
            require_unique=bool(d.get("require_unique", True)),
            allow_shadowing=bool(d.get("allow_shadowing", False)),
            types=types,
            expected_types=tuple(d.get("expected_types", []) or []),
            rules=rules,
        )


# --------------------------------------------------------------------------------------
# Multiplier math
# --------------------------------------------------------------------------------------
def _mult(exps: Dict[str, float], ratios: Dict[str, float]) -> float:
    m = float(exps.get("const", 1.0))
    for v in RATIO_VARS:
        e = exps.get(v, 0.0)
        if e:
            m *= ratios[v] ** float(e)
    return m


# --------------------------------------------------------------------------------------
# The compiled Parametrization
# --------------------------------------------------------------------------------------
class Parametrization:
    """Compiled parametrization: classify params, build optimizer overrides + init, validate."""

    def __init__(self, cfg: ParametrizationConfig):
        self.cfg = cfg
        if cfg.enabled:
            self._validate_static()

    # ---- static config sanity ------------------------------------------------------
    def _validate_static(self) -> None:
        if self.cfg.width_base is not None and self.cfg.width_base <= 0:
            raise ValueError("[#118 param] width_base must be positive")
        if self.cfg.depth_base is not None and self.cfg.depth_base <= 0:
            raise ValueError("[#118 param] depth_base must be positive")
        for ratio_name, ratio in self.cfg.ratios.items():
            if ratio <= 0:
                raise ValueError(f"[#118 param] ratio {ratio_name} must be positive")
        type_ids = {t.id for t in self.cfg.types}
        if self.cfg.expected_types and set(self.cfg.expected_types) - type_ids:
            missing = set(self.cfg.expected_types) - type_ids
            raise ValueError(f"[#118 param] expected_types not in type_registry: {sorted(missing)}")
        rules_with_optimizer_roles = [r.name for r in self.cfg.rules if r.optimizer_role is not None]
        if rules_with_optimizer_roles and len(rules_with_optimizer_roles) != len(self.cfg.rules):
            rules_without_optimizer_roles = [
                r.name for r in self.cfg.rules if r.optimizer_role is None
            ]
            raise ValueError(
                "[#118 param] optimizer_role is closed-world: once any rule declares one, "
                f"every rule must declare one; missing={rules_without_optimizer_roles}"
            )
        for r in self.cfg.rules:
            unknown = set(r.types) - type_ids
            if unknown:
                raise ValueError(f"[#118 param] rule '{r.name}' references unknown types {sorted(unknown)}")
            if r.optimizer_role is not None and r.optimizer_role not in OPTIMIZER_ROLES:
                raise ValueError(
                    f"[#118 param] rule '{r.name}' has unknown optimizer_role "
                    f"'{r.optimizer_role}'; expected one of {list(OPTIMIZER_ROLES)}"
                )

    # ---- classification -------------------------------------------------------------
    def classify(self, param: "torch.nn.Parameter", name: str) -> str:
        hits = [t.id for t in self.cfg.types if t.matches(param, name)]
        if len(hits) == 1:
            return hits[0]
        if not hits:
            raise ValueError(
                f"[#118 param] STRICT COVERAGE: parameter '{name}' (dim={param.dim()}) matched no "
                f"type in the registry. Onboard it: add a type_registry entry + expected_types + a rule."
            )
        raise ValueError(
            f"[#118 param] AMBIGUOUS TYPE: parameter '{name}' matched multiple types {hits}. "
            f"Type recognizers must be mutually exclusive (use anchored globs / distinct attrs)."
        )

    def rule_for(self, param: "torch.nn.Parameter", name: str, type_id: str) -> Rule:
        cands = [r for r in self.cfg.rules if type_id in r.types and r.depth_ok(param)]
        if len(cands) == 1:
            return cands[0]
        if not cands:
            d = getattr(param, GLOBAL_LAYER_ATTR, None)
            raise ValueError(
                f"[#118 param] STRICT COVERAGE: parameter '{name}' (type={type_id}, depth={d}) "
                f"matched no rule. Every type/depth must have exactly one rule."
            )
        if self.cfg.allow_shadowing:
            return cands[0]
        raise ValueError(
            f"[#118 param] OVERLAPPING RULES for '{name}' (type={type_id}): {[r.name for r in cands]}. "
            f"Make depth ranges disjoint, or set allow_shadowing=true (first match wins)."
        )

    # ---- optimizer overrides --------------------------------------------------------
    def _override_for_rule(
        self, rule: Rule, base_lr, base_min_lr, base_eps, selected_optimizer: Optional[str] = None
    ) -> ParamGroupOverride:
        ov: ParamGroupOverride = {}
        if rule.optimizer_role is not None:
            if selected_optimizer is None:
                raise ValueError(
                    f"[#118 param] rule '{rule.name}' declares optimizer_role="
                    f"'{rule.optimizer_role}', but no selected optimizer was provided"
                )
            if selected_optimizer.startswith("dist_"):
                selected_optimizer = selected_optimizer[len("dist_") :]
            ov["optimizer"] = selected_optimizer if rule.optimizer_role == "matrix" else "adam"
        if rule.lr:
            lm = _mult(rule.lr, self.cfg.ratios)
            if lm != 1.0:
                if base_lr is not None:
                    ov["max_lr"] = base_lr * lm
                if base_min_lr is not None:
                    ov["min_lr"] = base_min_lr * lm
        if rule.eps and base_eps is not None:
            em = _mult(rule.eps, self.cfg.ratios)
            if em != 1.0:
                ov["eps"] = base_eps * em
        if rule.wd:
            wm = _mult(rule.wd, self.cfg.ratios)
            if wm != 1.0:
                ov["wd_mult"] = wm
        return ov

    def build_config_overrides(
        self, base_lr, base_min_lr, base_eps, selected_optimizer: Optional[str] = None
    ) -> "Dict[ParamKey, ParamGroupOverride]":
        """Build one optimizer override per non-identity rule.

        ``optimizer_role`` is compiled here, alongside CompleteP's numerical overrides, so
        routing and LR/min-LR/WD/epsilon scaling share the same closed-world classifier.
        """
        if not self.cfg.enabled:
            return {}
        normalized_optimizer = selected_optimizer
        if normalized_optimizer is not None and normalized_optimizer.startswith("dist_"):
            normalized_optimizer = normalized_optimizer[len("dist_") :]
        if normalized_optimizer not in (None, "adam", "sgd"):
            rules_without_optimizer_roles = [
                rule.name for rule in self.cfg.rules if rule.optimizer_role is None
            ]
            if rules_without_optimizer_roles:
                raise ValueError(
                    f"[#118 param] emerging optimizer '{normalized_optimizer}' requires an "
                    "explicit optimizer_role on every rule; "
                    f"missing={rules_without_optimizer_roles}"
                )
        from megatron.core.optimizer.optimizer_config import ParamKey, ParamWithNamePredicate

        out: "Dict[ParamKey, ParamGroupOverride]" = {}
        for rule in self.cfg.rules:
            ov = self._override_for_rule(
                rule, base_lr, base_min_lr, base_eps, selected_optimizer=selected_optimizer
            )
            if not ov:
                continue
            # Capture rule by value in the closure.
            def _fn(param, name, _rule=rule):
                try:
                    tid = self.classify(param, name)
                except ValueError:
                    return False
                return tid in _rule.types and _rule.depth_ok(param)

            key = ParamKey(with_name_predicate=ParamWithNamePredicate(name=f"#118:{rule.name}", fn=_fn))
            out[key] = ov
        return out

    # ---- strict coverage validation -------------------------------------------------
    def validate_coverage(self, model_chunks: Sequence[Any]) -> Dict[str, Any]:
        """Classify every trainable param; assert exactly-one-rule coverage + closed world.
        Returns the manifest (also used for resume assertion)."""
        if not self.cfg.enabled:
            return {"enabled": False}
        realized_types: Dict[str, int] = {}
        per_rule: Dict[str, List[str]] = {r.name: [] for r in self.cfg.rules}
        param_meta: Dict[str, List[Any]] = {}
        for chunk in model_chunks:
            for name, param in chunk.named_parameters():
                if not param.requires_grad:
                    continue
                tid = self.classify(param, name)  # raises on 0 / >1
                rule = self.rule_for(param, name, tid)  # raises on 0 / >1
                realized_types[tid] = realized_types.get(tid, 0) + 1
                per_rule[rule.name].append(name)
                d = getattr(param, GLOBAL_LAYER_ATTR, None)
                param_meta[name] = [
                    tid,
                    rule.name,
                    rule.optimizer_role,
                    d,
                    list(param.shape),
                    str(param.dtype),
                ]
        # Closed world: no realized type outside expected_types.
        if self.cfg.expected_types:
            unknown = set(realized_types) - set(self.cfg.expected_types)
            if unknown:
                raise ValueError(
                    f"[#118 param] CLOSED-WORLD VIOLATION: realized types {sorted(unknown)} not in "
                    f"expected_types {sorted(self.cfg.expected_types)}. Onboard them explicitly."
                )
        for name in per_rule:
            per_rule[name].sort()
        return {
            "enabled": True,
            "config_hash": self.config_hash(),
            "ratios": self.cfg.ratios,
            "realized_types": dict(sorted(realized_types.items())),
            "per_rule": {k: per_rule[k] for k in sorted(per_rule)},
            "param_meta": dict(sorted(param_meta.items())),
        }

    # ---- init resolver --------------------------------------------------------------
    def init_std_mult(self, type_id: str) -> float:
        """Init-std multiplier for a module type (depth-independent rules only).

        This helper only drives the legacy config-time init handles in ``apply_init``.
        Fine-grained per-rule init is applied by ``reinitialize_rule_inits`` after model
        construction, which lets HPO candidates tune qkv/proj/fc1/fc2/readout init
        separately instead of being limited to Megatron's shared init handles."""
        cands = [r for r in self.cfg.rules if type_id in r.types and r.init_std]
        depth_specific = [r for r in cands if r.depth_start is not None or r.depth_end is not None]
        flat = [r for r in cands if r.depth_start is None and r.depth_end is None]
        if depth_specific:
            raise ValueError(
                f"[#118 param] config-time init_std cannot consume per-depth rule(s) "
                f"{[r.name for r in depth_specific]}; use reinitialize_rule_inits for per-rule init."
            )
        if not flat:
            return 1.0
        if len(flat) > 1:
            raise ValueError(f"[#118 param] multiple flat init_std rules for type '{type_id}'")
        return _mult(flat[0].init_std, self.cfg.ratios)

    def reinitialize_rule_inits(self, model_chunks: Sequence[Any], base_init_std: float) -> Dict[str, Any]:
        """Apply rule-table init_std values directly to matching trainable parameters.

        Megatron's config object has only a small number of init handles, which is too coarse
        for per-parameter-group HPO. This post-build pass keeps optimizer/LR routing and init
        routing in the same declarative rule table: each rule with ``init_std`` reinitializes
        exactly the parameters it covers to N(0, base_init_std * multiplier). Rules without
        ``init_std`` are left untouched. One-dimensional norm/bias params are rejected if a
        candidate tries to initialize them this way.
        """
        if not self.cfg.enabled:
            return {"enabled": False, "reinitialized": {}}
        import torch

        counts: Dict[str, int] = {}
        stds: Dict[str, float] = {}
        for chunk in model_chunks:
            for name, param in chunk.named_parameters():
                if not param.requires_grad:
                    continue
                tid = self.classify(param, name)
                rule = self.rule_for(param, name, tid)
                if not rule.init_std:
                    continue
                if param.dim() < 2:
                    raise ValueError(
                        f"[#118 param] init_std rule '{rule.name}' matched non-matrix "
                        f"parameter '{name}' with shape {list(param.shape)}"
                    )
                std = float(base_init_std) * _mult(rule.init_std, self.cfg.ratios)
                with torch.no_grad():
                    param.data.normal_(mean=0.0, std=std)
                counts[rule.name] = counts.get(rule.name, 0) + 1
                stds[rule.name] = std
        return {
            "enabled": True,
            "base_init_std": float(base_init_std),
            "reinitialized": dict(sorted(counts.items())),
            "stds": dict(sorted(stds.items())),
        }

    # ---- depth (CompleteP residual) -------------------------------------------------
    def residual_branch_mult(self, num_layers) -> float:
        """CompleteP forward residual multiplier C * (L/L_0) ** (-alpha) (#118).

        Backward-compatible scalar applied to both attention/mlp residual-branch output:
        ``h^{l+1} = h^l + residual_branch_mult * F_l(h^l)``. Returns 1.0 (inert) when the
        parametrization is disabled or depth scaling is OFF (``depth_base is None``); also
        1.0 at L == L_0 or alpha == 0."""
        return self._residual_mult(self.cfg.residual_const, num_layers)

    def _residual_mult(self, residual_const: float, num_layers) -> float:
        if not self.cfg.enabled or self.cfg.depth_base is None:
            return 1.0
        m_L = num_layers / self.cfg.depth_base
        return residual_const * (m_L ** (-self.cfg.alpha))

    def residual_branch_mults(self, num_layers) -> tuple[float, float]:
        """Attention/MLP residual multipliers for ``x + C_type * f_l(x)`` (#118).

        ``residual_const`` remains the backward-compatible default for both branch types.
        Candidate YAML may override only one branch by setting ``residual_attention_const``
        or ``residual_mlp_const``.
        """
        attention_const = (
            self.cfg.residual_attention_const
            if self.cfg.residual_attention_const is not None
            else self.cfg.residual_const
        )
        mlp_const = (
            self.cfg.residual_mlp_const
            if self.cfg.residual_mlp_const is not None
            else self.cfg.residual_const
        )
        return (
            self._residual_mult(attention_const, num_layers),
            self._residual_mult(mlp_const, num_layers),
        )

    # ---- init application ------------------------------------------------------------
    def apply_init(self, config, *, residual_depth_multiplier: float = 2.0,
                   embed_type: str = "embedding", hidden_type: str = "hidden",
                   unembed_type: str = "unembedding"):
        """Wire the per-type init-std multipliers onto a TransformerConfig's init handles.

        Must be called BEFORE model construction (the init partials are consumed in each
        layer's ``__init__``). No-op when the parametrization is disabled (SP control) ->
        byte-identical. Sets, with ``base = config.init_method_std`` and the depth-flat
        ``init_std`` rules:

          - ``config.embedding_init_method``    = N(0, base * init_std_mult(embed_type))
          - ``config.init_method``              = N(0, base * init_std_mult(hidden_type))
          - ``config.output_layer_init_method`` = N(0, sigma_hidden) when CompleteP depth
            scaling is active, otherwise the legacy Megatron residual-output scaling
          - ``config.unembedding_init_method``  = N(0, base * init_std_mult(unembed_type))

        Forces ``config.mup_output_mult = 1.0`` (the table encoding carries the readout scale
        in the unembed INIT, not a logit multiplier). Requires UNTIED embeddings -- a tied
        weight cannot carry distinct embed vs unembed init stds (design risk-3).

        The three ``*_type`` args name the registry type ids the init handles map onto; the
        validated candidate configs use the defaults ("embedding"/"hidden"/"unembedding").
        """
        if not self.cfg.enabled:
            return config
        import math
        from megatron.core.utils import init_method_normal

        if getattr(config, "share_embeddings_and_output_weights", False):
            raise ValueError(
                "[#118 param] parametrization init requires UNTIED embeddings, but "
                "config.share_embeddings_and_output_weights is True. A tied embed/unembed "
                "weight cannot carry distinct embed vs unembed init stds. Pass "
                "--untie-embeddings-and-output-weights."
            )

        base = config.init_method_std
        sigma_embed = base * self.init_std_mult(embed_type)
        sigma_hidden = base * self.init_std_mult(hidden_type)
        sigma_unembed = base * self.init_std_mult(unembed_type)

        num_layers = getattr(config, "num_layers")
        if self.cfg.depth_base is not None:
            # CompleteP depth scaling is owned by the forward residual multiplier below.
            # Do not also apply Megatron's residual-output init factor 1/sqrt(2L).
            out_std = sigma_hidden
        else:
            # Depth scaling OFF: preserve Megatron's existing output-projection
            # residual-depth factor std / sqrt(mult * L).
            out_std = sigma_hidden / math.sqrt(residual_depth_multiplier * num_layers)

        config.embedding_init_method = init_method_normal(sigma_embed)
        config.init_method = init_method_normal(sigma_hidden)
        config.output_layer_init_method = init_method_normal(out_std)
        config.unembedding_init_method = init_method_normal(sigma_unembed)
        config.mup_output_mult = 1.0
        # CompleteP forward residual multipliers C_type * (L/L_0)^(-alpha); 1.0 (inert) when
        # depth scaling is OFF. Consumed in each TransformerLayer's attention/mlp residual branch.
        attention_mult, mlp_mult = self.residual_branch_mults(num_layers)
        config.residual_branch_mult = self.residual_branch_mult(num_layers)
        config.residual_attention_mult = attention_mult
        config.residual_mlp_mult = mlp_mult
        return config

    # ---- manifest / resume ----------------------------------------------------------
    def config_hash(self) -> str:
        blob = json.dumps(
            {
                "ratios": self.cfg.ratios,
                "alpha": self.cfg.alpha,
                "width_base": self.cfg.width_base,
                "depth_base": self.cfg.depth_base,
                "residual_const": self.cfg.residual_const,
                "residual_attention_const": self.cfg.residual_attention_const,
                "residual_mlp_const": self.cfg.residual_mlp_const,
                "types": [
                    [
                        t.id,
                        list(t.attr),
                        list(t.name_globs),
                        list(t.exclude_globs),
                        t.min_dim,
                        t.max_dim,
                    ]
                    for t in self.cfg.types
                ],
                "expected_types": list(self.cfg.expected_types),
                "rules": [
                    [
                        r.name,
                        list(r.types),
                        r.optimizer_role,
                        r.depth_start,
                        r.depth_end,
                        r.init_std,
                        r.lr,
                        r.eps,
                        r.wd,
                    ]
                    for r in self.cfg.rules
                ],
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @staticmethod
    def assert_manifest_matches(saved: dict, current: dict) -> None:
        """Hard-fail (before optimizer-state load) on any param-group partition drift."""
        for k in ("config_hash", "per_rule", "realized_types"):
            if saved.get(k) != current.get(k):
                raise RuntimeError(
                    f"[#118 param] RESUME DRIFT: parametrization manifest field '{k}' changed across "
                    f"resume. Optimizer param-group partition would be misaligned. saved != current."
                )


# --------------------------------------------------------------------------------------
# Loader: build a compiled Parametrization from the candidates YAML
# --------------------------------------------------------------------------------------
def _block_with_runtime_overrides(
    block: dict,
    *,
    ratios: Optional[Dict[str, float]] = None,
    m_N: Optional[float] = None,
    m_L: Optional[float] = None,
    model_width: Optional[int] = None,
    model_depth: Optional[int] = None,
    width_base: Optional[int] = None,
    alpha: Optional[float] = None,
    residual_const: Optional[float] = None,
    residual_attention_const: Optional[float] = None,
    residual_mlp_const: Optional[float] = None,
    depth_base: Optional[int] = None,
) -> dict:
    block = dict(block)  # shallow copy so we never mutate the loaded doc
    if width_base is not None:
        block["width_base"] = int(width_base)
    if depth_base is not None:
        block["depth_base"] = int(depth_base)
    resolved_width_base = block.get("width_base")
    resolved_depth_base = block.get("depth_base")
    if m_N is None and model_width is not None and resolved_width_base is not None:
        m_N = float(model_width) / float(resolved_width_base)
    if m_L is None and model_depth is not None and resolved_depth_base is not None:
        m_L = float(model_depth) / float(resolved_depth_base)
    if ratios or m_N is not None or m_L is not None:
        merged = dict(block.get("ratios", {}) or {})
        if ratios:
            merged.update(ratios)
        if m_N is not None:
            merged["m_N"] = float(m_N)
        if m_L is not None:
            merged["m_L"] = float(m_L)
        block["ratios"] = merged
    # CompleteP depth knobs: launch-arg overrides win over the YAML block when provided.
    if alpha is not None:
        block["alpha"] = float(alpha)
    if residual_const is not None:
        block["residual_const"] = float(residual_const)
    if residual_attention_const is not None:
        block["residual_attention_const"] = float(residual_attention_const)
    if residual_mlp_const is not None:
        block["residual_mlp_const"] = float(residual_mlp_const)
    return block


def load_parametrization_block(
    block: dict,
    *,
    ratios: Optional[Dict[str, float]] = None,
    m_N: Optional[float] = None,
    m_L: Optional[float] = None,
    model_width: Optional[int] = None,
    model_depth: Optional[int] = None,
    width_base: Optional[int] = None,
    alpha: Optional[float] = None,
    residual_const: Optional[float] = None,
    residual_attention_const: Optional[float] = None,
    residual_mlp_const: Optional[float] = None,
    depth_base: Optional[int] = None,
) -> "Parametrization":
    """Compile an inline Hydra ``megatron.parametrization`` block."""
    block = _as_plain_config(block)
    if block and "parametrization" in block:
        block = block["parametrization"]
    block = _block_with_runtime_overrides(
        block,
        ratios=ratios,
        m_N=m_N,
        m_L=m_L,
        model_width=model_width,
        model_depth=model_depth,
        width_base=width_base,
        alpha=alpha,
        residual_const=residual_const,
        residual_attention_const=residual_attention_const,
        residual_mlp_const=residual_mlp_const,
        depth_base=depth_base,
    )
    return Parametrization(ParametrizationConfig.from_dict(block))


def load_parametrization(
    path: str,
    candidate: str,
    *,
    ratios: Optional[Dict[str, float]] = None,
    m_N: Optional[float] = None,
    m_L: Optional[float] = None,
    model_width: Optional[int] = None,
    model_depth: Optional[int] = None,
    width_base: Optional[int] = None,
    alpha: Optional[float] = None,
    residual_const: Optional[float] = None,
    residual_attention_const: Optional[float] = None,
    residual_mlp_const: Optional[float] = None,
    depth_base: Optional[int] = None,
) -> "Parametrization":
    """Load ``candidate`` from a candidates YAML and compile it into a ``Parametrization``.

    The YAML maps candidate name -> ``{parametrization: {...}}`` (see
    ``conf/parametrization_candidates.yaml``: SP / muP / everett_sp). ``ratios`` (or the
    convenience scalars ``m_N`` / ``m_L``) override the per-cell width/depth ratios at
    render/launch time -- the only things that vary across a width/depth sweep. The
    CompleteP depth knobs ``alpha`` / residual C constants / ``depth_base`` override the YAML
    block when not None (so a depth sweep can drive them from the launch args). YAML
    anchors/aliases are resolved by the loader, so the shared ``_type_registry`` is expanded
    per candidate. Underscore-prefixed top-level keys (anchors) are not selectable candidates.
    """
    import yaml

    with open(path) as f:
        doc = yaml.safe_load(f) or {}
    if candidate not in doc:
        avail = sorted(k for k in doc if not k.startswith("_"))
        raise KeyError(f"[#118 param] candidate '{candidate}' not in {path}; available: {avail}")
    block = (doc[candidate] or {}).get("parametrization")
    if block is None:
        raise KeyError(
            f"[#118 param] candidate '{candidate}' has no 'parametrization' block in {path}"
        )
    return load_parametrization_block(
        block,
        ratios=ratios,
        m_N=m_N,
        m_L=m_L,
        model_width=model_width,
        model_depth=model_depth,
        width_base=width_base,
        alpha=alpha,
        residual_const=residual_const,
        residual_attention_const=residual_attention_const,
        residual_mlp_const=residual_mlp_const,
        depth_base=depth_base,
    )
