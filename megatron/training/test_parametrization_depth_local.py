"""Local CPU tests for the #118 CompleteP DEPTH scaling (no torch/megatron needed).

Run: python3 megatron/training/test_parametrization_depth_local.py
Covers: the CompleteP forward residual multiplier C * (L/L_0) ** (-alpha), its OFF cases
(alpha==0 / depth_base==None), the m_L exponent in _mult, and reduce-to-baseline at m_L==1
(no extra optimizer group vs width-only). The forward-pass tensor scaling lives in
transformer_layer.py (needs torch) and is verified on-cluster; the guard is grep-checked.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parametrization as P  # noqa: E402


# A minimal depth-aware config: depth_base=2, alpha=1.0, residual_const=1.0, plus a hidden
# rule whose lr/eps carry m_L terms (the Table-1 depth terms on the optimizer rules).
def depth_config(
    m_N=1.0,
    m_L=1.0,
    alpha=1.0,
    residual_const=1.0,
    residual_attention_const=None,
    residual_mlp_const=None,
    depth_base=2,
):
    cfg = {
        "enabled": True,
        "ratios": {"m_N": m_N, "m_L": m_L, "m_B": 1.0, "m_D": 1.0},
        "alpha": alpha,
        "residual_const": residual_const,
        "depth_base": depth_base,
        "expected_types": ["hidden"],
        "type_registry": {
            "hidden": {"name_globs": ["*.linear_qkv.weight", "*.linear_proj.weight",
                                      "*.linear_fc1.weight", "*.linear_fc2.weight"],
                       "exclude_globs": ["*experts*"], "min_dim": 2},
        },
        "rules": [
            {"name": "hidden", "types": ["hidden"],
             "lr": {"m_N": -1.0, "m_L": 0.0}, "eps": {"m_N": -1.0, "m_L": -1.0}},
        ],
    }
    if residual_attention_const is not None:
        cfg["residual_attention_const"] = residual_attention_const
    if residual_mlp_const is not None:
        cfg["residual_mlp_const"] = residual_mlp_const
    return cfg


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(b))


def test_residual_branch_mult_power_law():
    # depth_base=2, alpha=1.0, C=1.0 -> mult = (L/2) ** -1.
    par = P.Parametrization(P.ParametrizationConfig.from_dict(depth_config()))
    assert approx(par.residual_branch_mult(2), 1.0), par.residual_branch_mult(2)
    assert approx(par.residual_branch_mult(4), 0.5), par.residual_branch_mult(4)
    assert approx(par.residual_branch_mult(8), 0.25), par.residual_branch_mult(8)
    # residual_const multiplies the whole branch.
    par_c = P.Parametrization(P.ParametrizationConfig.from_dict(depth_config(residual_const=2.0)))
    assert approx(par_c.residual_branch_mult(8), 0.5), par_c.residual_branch_mult(8)


def test_split_residual_branch_mult_power_law():
    # depth_base=2, alpha=1.0, L=8 -> m_L=4, so multipliers are C_type / 4.
    par = P.Parametrization(P.ParametrizationConfig.from_dict(
        depth_config(residual_attention_const=2.0, residual_mlp_const=0.5)
    ))
    attn_mult, mlp_mult = par.residual_branch_mults(8)
    assert approx(attn_mult, 0.5), attn_mult
    assert approx(mlp_mult, 0.125), mlp_mult
    # The legacy scalar remains tied to residual_const for backward compatibility.
    assert approx(par.residual_branch_mult(8), 0.25), par.residual_branch_mult(8)


def test_split_residual_branch_mult_falls_back_to_scalar_const():
    par = P.Parametrization(P.ParametrizationConfig.from_dict(depth_config(residual_const=3.0)))
    attn_mult, mlp_mult = par.residual_branch_mults(8)
    assert approx(attn_mult, 0.75), attn_mult
    assert approx(mlp_mult, 0.75), mlp_mult


def test_residual_branch_mult_off_cases():
    # alpha == 0 -> mult == 1.0 for any L.
    par_a0 = P.Parametrization(P.ParametrizationConfig.from_dict(depth_config(alpha=0.0)))
    for L in (2, 4, 8, 64):
        assert approx(par_a0.residual_branch_mult(L), 1.0), (L, par_a0.residual_branch_mult(L))
    # depth_base None -> depth scaling OFF -> mult == 1.0.
    cfg = depth_config()
    cfg["depth_base"] = None
    par_none = P.Parametrization(P.ParametrizationConfig.from_dict(cfg))
    assert par_none.cfg.depth_base is None
    for L in (2, 4, 8, 64):
        assert approx(par_none.residual_branch_mult(L), 1.0), (L, par_none.residual_branch_mult(L))
    # disabled parametrization -> 1.0.
    par_off = P.Parametrization(P.ParametrizationConfig.from_dict({"enabled": False}))
    assert approx(par_off.residual_branch_mult(8), 1.0)
    assert par_off.residual_branch_mults(8) == (1.0, 1.0)


def test_m_L_exponent_in_lr_and_eps():
    # m_L=4 with hidden lr {m_N:-1, m_L:0} -> lr mult = m_N^-1 * m_L^0 = m_N^-1 (m_L inert).
    base_lr, base_min, base_eps = 3e-3, 3e-5, 1e-15
    par = P.Parametrization(P.ParametrizationConfig.from_dict(
        depth_config(m_N=2.0, m_L=4.0)))
    hid = next(r for r in par.cfg.rules if r.name == "hidden")
    ov = par._override_for_rule(hid, base_lr, base_min, base_eps)
    assert approx(ov["max_lr"], base_lr * (2.0 ** -1.0)), ov   # m_L^0 = 1
    assert approx(ov["min_lr"], base_min * (2.0 ** -1.0)), ov
    # eps {m_N:-1, m_L:-1} -> m_N^-1 * m_L^-1 = 0.5 * 0.25 = 0.125.
    assert approx(ov["eps"], base_eps * (2.0 ** -1.0) * (4.0 ** -1.0)), ov
    # Spot-check _mult directly on the eps exponent vector.
    assert approx(P._mult({"m_N": -1.0, "m_L": -1.0}, par.cfg.ratios),
                  (2.0 ** -1.0) * (4.0 ** -1.0))


def test_reduce_to_baseline_at_m_L_one():
    # At m_N=m_L=1 every depth (and width) term -> 1.0 -> no override emitted (reduce-to-baseline).
    base_lr, base_min, base_eps = 3e-3, 3e-5, 1e-15
    par = P.Parametrization(P.ParametrizationConfig.from_dict(depth_config(m_N=1.0, m_L=1.0)))
    for r in par.cfg.rules:
        assert par._override_for_rule(r, base_lr, base_min, base_eps) == {}, \
            f"{r.name} not identity at m_N=m_L=1"
    # And the residual forward multiplier is 1.0 at L == L_0 (depth_base=2).
    assert approx(par.residual_branch_mult(2), 1.0)


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
