"""Local CPU tests for the #118 Parametrization core logic (no torch/megatron needed).

Run: python3 megatron/training/test_parametrization_local.py
Covers: closed-world classification + mutual exclusivity, strict coverage (unmatched/unknown
raise), PP-depth rule routing, exponent-vector multipliers, reduce-to-baseline at all-ones,
per-depth-init guard, and the resume manifest. build_config_overrides() needs megatron
(ParamKey) and is validated on-cluster, not here.
"""
import copy
import functools
import math
import os
import sys
import types

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
import parametrization as P  # noqa: E402


# --- fakes (duck-typed; no torch) -----------------------------------------------------
class FakeParam:
    def __init__(self, shape, requires_grad=True, depth=None, **attrs):
        self._shape = tuple(shape)
        self.requires_grad = requires_grad
        self.dtype = "torch.bfloat16"
        if depth is not None:
            setattr(self, P.GLOBAL_LAYER_ATTR, depth)
        for k, v in attrs.items():
            setattr(self, k, v)

    def dim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape


class FakeChunk:
    def __init__(self, named):
        self._named = named

    def named_parameters(self):
        return iter(self._named)


class FakeConfig:
    def __init__(self, num_layers=16, init_method_std=0.02):
        self.num_layers = num_layers
        self.init_method_std = init_method_std
        self.share_embeddings_and_output_weights = False


# --- the C1 (Complete(d)P alpha=0.5) config, as Hydra would deliver it ----------------
def c1_config(m_N=1.0, m_L=1.0, m_B=1.0, m_D=1.0):
    return {
        "enabled": True,
        "ratios": {"m_N": m_N, "m_L": m_L, "m_B": m_B, "m_D": m_D},
        "alpha": 0.5,
        "expected_types": ["embedding", "unembedding", "qk_norm", "norm_bias", "hidden"],
        "type_registry": {
            "embedding": {"attr": ["is_embedding_parameter", "shared_embedding"],
                          "name_globs": ["*word_embeddings.weight"]},
            "unembedding": {"name_globs": ["*output_layer.weight"]},
            "qk_norm": {"name_globs": ["*.q_layernorm.weight", "*.k_layernorm.weight",
                                       "*.q_layernorm.bias", "*.k_layernorm.bias"]},
            "norm_bias": {"name_globs": ["*layernorm.weight", "*.bias", "*final_layernorm.weight",
                                          "*.layer_norm_weight", "*.layer_norm_bias"],
                          "exclude_globs": ["*.q_layernorm.*", "*.k_layernorm.*"], "max_dim": 1},
            "hidden": {"name_globs": ["*.linear_qkv.weight", "*.linear_proj.weight",
                                      "*.linear_fc1.weight", "*.linear_fc2.weight"],
                       "exclude_globs": ["*experts*"], "min_dim": 2},
        },
        "rules": [
            {"name": "hidden", "types": ["hidden"], "init_std": {"m_N": -0.5},
             "lr": {"m_N": -1, "m_L": -0.5, "m_B": 0.5, "m_D": -0.5},
             "eps": {"m_N": -1, "m_L": -0.5, "m_B": -0.5, "m_D": 0.5}, "wd": {"m_N": 1}},
            {"name": "embedding", "types": ["embedding"],
             "lr": {"m_B": 0.5, "m_D": -0.5}, "eps": {"m_N": -1}},
            {"name": "unembedding", "types": ["unembedding"], "init_std": {"m_N": -1},
             "lr": {"m_N": -1, "m_B": 0.5, "m_D": -0.5}, "wd": {"m_N": 1}},
            {"name": "norm_bias", "types": ["norm_bias"], "lr": {"m_L": -0.5, "m_B": 0.5, "m_D": -0.5}},
            {"name": "qk_norm", "types": ["qk_norm"]},
        ],
    }


def gpt_params(n_layers=2):
    """A representative 2-layer mcore-GPT trainable param set."""
    out = [("embedding.word_embeddings.weight", FakeParam((50304, 512), is_embedding_parameter=True,
                                                           is_embedding_or_output_parameter=True)),
           ("output_layer.weight", FakeParam((50304, 512), is_embedding_or_output_parameter=True)),
           ("decoder.final_layernorm.weight", FakeParam((512,)))]
    for i in range(n_layers):
        p = f"decoder.layers.{i}"
        out += [
            (f"{p}.input_layernorm.weight", FakeParam((512,), depth=i)),
            (f"{p}.self_attention.linear_qkv.layer_norm_weight", FakeParam((512,), depth=i)),
            (f"{p}.self_attention.linear_qkv.layer_norm_bias", FakeParam((512,), depth=i)),
            (f"{p}.self_attention.linear_qkv.weight", FakeParam((1536, 512), depth=i)),
            (f"{p}.self_attention.q_layernorm.weight", FakeParam((128,), depth=i)),
            (f"{p}.self_attention.k_layernorm.weight", FakeParam((128,), depth=i)),
            (f"{p}.self_attention.linear_proj.weight", FakeParam((512, 512), depth=i)),
            (f"{p}.pre_mlp_layernorm.weight", FakeParam((512,), depth=i)),
            (f"{p}.mlp.linear_fc1.layer_norm_weight", FakeParam((512,), depth=i)),
            (f"{p}.mlp.linear_fc1.layer_norm_bias", FakeParam((512,), depth=i)),
            (f"{p}.mlp.linear_fc1.weight", FakeParam((2048, 512), depth=i)),
            (f"{p}.mlp.linear_fc2.weight", FakeParam((512, 2048), depth=i)),
        ]
    return out


# --- tests ----------------------------------------------------------------------------
def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def install_fake_megatron_utils():
    utils = types.ModuleType("megatron.core.utils")
    utils.init_method_normal = lambda sigma: functools.partial(
        torch.nn.init.normal_, mean=0.0, std=sigma
    )
    sys.modules.setdefault("megatron", types.ModuleType("megatron"))
    sys.modules.setdefault("megatron.core", types.ModuleType("megatron.core"))
    sys.modules["megatron.core.utils"] = utils


def test_classification_exclusive():
    par = P.Parametrization(P.ParametrizationConfig.from_dict(c1_config()))
    cases = {
        "embedding.word_embeddings.weight": ("embedding", dict(is_embedding_parameter=True, is_embedding_or_output_parameter=True), (50304, 512)),
        "output_layer.weight": ("unembedding", dict(is_embedding_or_output_parameter=True), (50304, 512)),
        "decoder.layers.0.self_attention.linear_qkv.weight": ("hidden", {}, (1536, 512)),
        "decoder.layers.3.mlp.linear_fc2.weight": ("hidden", {}, (512, 2048)),
        "decoder.layers.0.self_attention.q_layernorm.weight": ("qk_norm", {}, (128,)),
        "decoder.layers.0.self_attention.k_layernorm.weight": ("qk_norm", {}, (128,)),
        "decoder.layers.0.self_attention.linear_qkv.layer_norm_weight": ("norm_bias", {}, (512,)),
        "decoder.layers.0.self_attention.linear_qkv.layer_norm_bias": ("norm_bias", {}, (512,)),
        "decoder.layers.0.mlp.linear_fc1.layer_norm_weight": ("norm_bias", {}, (512,)),
        "decoder.layers.0.mlp.linear_fc1.layer_norm_bias": ("norm_bias", {}, (512,)),
        "decoder.layers.0.input_layernorm.weight": ("norm_bias", {}, (512,)),
        "decoder.final_layernorm.weight": ("norm_bias", {}, (512,)),
    }
    for name, (want, attrs, shp) in cases.items():
        got = par.classify(FakeParam(shp, **attrs), name)
        assert got == want, f"classify({name}) = {got}, want {want}"


def test_closed_world_unknown_raises():
    par = P.Parametrization(P.ParametrizationConfig.from_dict(c1_config()))
    # MoE expert linear_fc1 must NOT silently classify as dense hidden (excluded) -> error.
    for name, shp in [("decoder.layers.0.mlp.experts.linear_fc1.weight", (2048, 512)),
                      ("decoder.layers.0.some_new_module.weight", (512, 512))]:
        try:
            par.classify(FakeParam(shp), name)
            raise AssertionError(f"expected closed-world error for {name}")
        except ValueError as e:
            assert "STRICT COVERAGE" in str(e) or "no type" in str(e)


def test_multipliers_and_reduce_to_baseline():
    base_lr, base_min, base_eps = 3e-3, 3e-5, 1e-15
    # All-ones -> every override empty (reduce-to-baseline).
    par1 = P.Parametrization(P.ParametrizationConfig.from_dict(c1_config()))
    for r in par1.cfg.rules:
        assert par1._override_for_rule(r, base_lr, base_min, base_eps) == {}, f"{r.name} not identity at all-ones"
    # m_N=2: hidden lr x0.5, eps x0.5, wd x2; init x 2^-0.5; unembed init x0.5.
    par2 = P.Parametrization(P.ParametrizationConfig.from_dict(c1_config(m_N=2.0)))
    hid = next(r for r in par2.cfg.rules if r.name == "hidden")
    ov = par2._override_for_rule(hid, base_lr, base_min, base_eps)
    assert approx(ov["max_lr"], base_lr * 0.5), ov
    assert approx(ov["min_lr"], base_min * 0.5), ov
    assert approx(ov["eps"], base_eps * 0.5), ov
    assert approx(ov["wd_mult"], 2.0), ov
    assert approx(par2.init_std_mult("hidden"), 2.0 ** -0.5)
    assert approx(par2.init_std_mult("unembedding"), 0.5)
    assert approx(par2.init_std_mult("embedding"), 1.0)
    assert approx(par2.init_std_mult("norm_bias"), 1.0)


def test_explicit_multiplier_aliases_and_inline_loader():
    cfg = c1_config(m_N=2.0)
    hidden = next(r for r in cfg["rules"] if r["name"] == "hidden")
    hidden["init_std_mult"] = hidden.pop("init_std")
    hidden["lr_mult"] = hidden.pop("lr")
    par = P.load_parametrization_block(types.SimpleNamespace(**cfg))
    hid = next(r for r in par.cfg.rules if r.name == "hidden")
    ov = par._override_for_rule(hid, 3e-3, 3e-5, 1e-15)
    assert approx(ov["max_lr"], 3e-3 * 0.5), ov
    assert approx(par.init_std_mult("hidden"), 2.0 ** -0.5)


def test_inline_loader_derives_width_and_depth_ratios():
    cfg = c1_config()
    cfg["width_base"] = 1536
    cfg["depth_base"] = 8
    par = P.load_parametrization_block(cfg, model_width=768, model_depth=4)
    assert approx(par.cfg.ratios["m_N"], 0.5)
    assert approx(par.cfg.ratios["m_L"], 0.5)
    assert par.cfg.width_base == 1536
    assert par.cfg.depth_base == 8

    hidden = next(rule for rule in par.cfg.rules if rule.name == "hidden")
    override = par._override_for_rule(hidden, 3e-3, 3e-5, 1e-15)
    expected_lr_mult = 0.5**-1 * 0.5**-0.5
    assert approx(override["max_lr"], 3e-3 * expected_lr_mult)
    assert approx(par.init_std_mult("hidden"), 0.5**-0.5)

    explicit = P.load_parametrization_block(
        cfg,
        model_width=768,
        model_depth=4,
        m_N=2.0,
        m_L=4.0,
    )
    assert explicit.cfg.ratios["m_N"] == 2.0
    assert explicit.cfg.ratios["m_L"] == 4.0


def test_completep_depth_uses_unscaled_output_init_std():
    install_fake_megatron_utils()
    cfg = c1_config()
    cfg["depth_base"] = 8
    par = P.Parametrization(P.ParametrizationConfig.from_dict(cfg))
    config = par.apply_init(FakeConfig(num_layers=16, init_method_std=0.02))
    assert approx(config.output_layer_init_method.keywords["std"], 0.02)


def test_depth_off_preserves_legacy_output_init_std():
    install_fake_megatron_utils()
    cfg = c1_config()
    cfg["depth_base"] = None
    par = P.Parametrization(P.ParametrizationConfig.from_dict(cfg))
    config = par.apply_init(FakeConfig(num_layers=16, init_method_std=0.02))
    assert approx(config.output_layer_init_method.keywords["std"], 0.02 / math.sqrt(2.0 * 16))


def test_coverage_and_manifest():
    par = P.Parametrization(P.ParametrizationConfig.from_dict(c1_config(m_N=2.0)))
    man = par.validate_coverage([FakeChunk(gpt_params(2))])
    assert man["realized_types"] == {"embedding": 1, "hidden": 8, "norm_bias": 13, "qk_norm": 4, "unembedding": 1}, man["realized_types"]
    assert set(man["realized_types"]) <= set(par.cfg.expected_types)
    # Unknown param -> coverage raises.
    bad = gpt_params(1) + [("decoder.layers.0.mlp.experts.linear_fc1.weight", FakeParam((2048, 512), depth=0))]
    try:
        par.validate_coverage([FakeChunk(bad)])
        raise AssertionError("expected coverage error on expert param")
    except ValueError:
        pass
    # Resume manifest: identical OK, drift raises.
    man2 = par.validate_coverage([FakeChunk(gpt_params(2))])
    P.Parametrization.assert_manifest_matches(man, man2)
    drift = copy.deepcopy(man2)
    drift["per_rule"]["hidden"].append("ghost.weight")
    try:
        P.Parametrization.assert_manifest_matches(man, drift)
        raise AssertionError("expected resume-drift error")
    except RuntimeError:
        pass


def test_depth_rules():
    cfg = c1_config(m_L=2.0)
    # Split hidden into disjoint depth ranges (R4 shape).
    cfg["rules"] = [r for r in cfg["rules"] if r["name"] != "hidden"] + [
        {"name": "hidden_early", "types": ["hidden"], "depth": {"start": 0, "end": 0}, "lr": {"m_N": -1}},
        {"name": "hidden_late", "types": ["hidden"], "depth": {"start": 1, "end": 1}, "lr": {"m_N": -1, "m_L": -0.5}},
    ]
    par = P.Parametrization(P.ParametrizationConfig.from_dict(cfg))
    p0 = FakeParam((1536, 512), depth=0)
    p1 = FakeParam((1536, 512), depth=1)
    assert par.rule_for(p0, "decoder.layers.0.self_attention.linear_qkv.weight", "hidden").name == "hidden_early"
    assert par.rule_for(p1, "decoder.layers.1.self_attention.linear_qkv.weight", "hidden").name == "hidden_late"
    # Overlapping rules (both depth=None for hidden) -> error.
    cfg2 = c1_config()
    cfg2["rules"].append({"name": "hidden_dup", "types": ["hidden"], "lr": {"m_N": -1}})
    par2 = P.Parametrization(P.ParametrizationConfig.from_dict(cfg2))
    try:
        par2.rule_for(FakeParam((1536, 512), depth=0), "decoder.layers.0.self_attention.linear_qkv.weight", "hidden")
        raise AssertionError("expected overlapping-rule error")
    except ValueError as e:
        assert "OVERLAPPING" in str(e)


def test_per_depth_init_guard():
    cfg = c1_config()
    cfg["rules"].append({"name": "hidden_deep_init", "types": ["hidden"], "depth": {"start": 5, "end": 9},
                         "init_std": {"m_N": -0.5}})
    par = P.Parametrization(P.ParametrizationConfig.from_dict(cfg))
    try:
        par.init_std_mult("hidden")
        raise AssertionError("expected config-time per-depth-init ValueError")
    except ValueError:
        pass


def test_post_build_rule_init_reinitializes_only_matching_rule():
    cfg = {
        "enabled": True,
        "expected_types": ["linear_qkv", "linear_proj"],
        "type_registry": {
            "linear_qkv": {"name_globs": ["*.linear_qkv.weight"], "min_dim": 2},
            "linear_proj": {"name_globs": ["*.linear_proj.weight"], "min_dim": 2},
        },
        "rules": [
            {"name": "qkv_tuned", "types": ["linear_qkv"], "init_std": {"const": 0.25}},
            {"name": "proj_default", "types": ["linear_proj"]},
        ],
    }
    qkv = torch.nn.Parameter(torch.ones(512, 512))
    proj = torch.nn.Parameter(torch.ones(512, 512))
    par = P.Parametrization(P.ParametrizationConfig.from_dict(cfg))
    torch.manual_seed(1234)
    manifest = par.reinitialize_rule_inits(
        [
            FakeChunk(
                [
                    ("decoder.layers.0.self_attention.linear_qkv.weight", qkv),
                    ("decoder.layers.0.self_attention.linear_proj.weight", proj),
                ]
            )
        ],
        base_init_std=0.028,
    )
    assert manifest["reinitialized"] == {"qkv_tuned": 1}, manifest
    assert approx(manifest["stds"]["qkv_tuned"], 0.007)
    assert approx(float(qkv.detach().std()), 0.007, tol=0.05)
    assert torch.equal(proj.detach(), torch.ones_like(proj))


def test_disabled_is_noop():
    par = P.Parametrization(P.ParametrizationConfig.from_dict({"enabled": False}))
    assert par.build_config_overrides(3e-3, 3e-5, 1e-15) == {}
    assert par.validate_coverage([FakeChunk(gpt_params(1))]) == {"enabled": False}


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
