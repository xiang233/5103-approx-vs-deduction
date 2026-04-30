"""Smoke tests. Run with: python -m pytest tests/  (or:  python tests/test_smoke.py)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classical import classical_accept
from src.conditional import delta_gamma
from src.elimination import disjunctive_abduce
from src.rules import Conjunction, Literal
from src.synthetic import WorldSpec, sample_world
from src.validity import empirical_validity, hoeffding_sample_bound, wilson_ci


def test_validity_basic():
    a = np.array([1, 1, 1, 0, 0]).astype(bool)
    b = np.array([1, 0, 1, 1, 0]).astype(bool)
    rep = empirical_validity(a, b)
    assert rep.n_alpha == 3
    assert rep.n_alpha_beta == 2
    assert math.isclose(rep.val, 2 / 3, rel_tol=1e-9)
    assert math.isclose(rep.cov, 3 / 5, rel_tol=1e-9)
    assert 0 <= rep.ci_low <= rep.val <= rep.ci_high <= 1


def test_wilson_endpoints():
    low, high = wilson_ci(0, 10)
    assert low == 0.0 and high < 0.4
    low, high = wilson_ci(10, 10)
    assert math.isclose(high, 1.0, abs_tol=1e-9) and low > 0.6


def test_hoeffding_grows_with_eps():
    m_loose = hoeffding_sample_bound(eps=0.1, delta=0.05, log2_hyp_class_size=10)
    m_tight = hoeffding_sample_bound(eps=0.01, delta=0.05, log2_hyp_class_size=10)
    assert m_tight > 50 * m_loose


def test_classical_rejects_with_one_counterexample():
    a = np.array([1, 1, 1, 0]).astype(bool)
    b = np.array([1, 0, 1, 1]).astype(bool)
    rep = classical_accept(a, b)
    assert rep.accepted is False
    assert rep.n_counterexamples == 1


def test_classical_accepts_when_no_counterexample():
    a = np.array([1, 1, 0, 0]).astype(bool)
    b = np.array([1, 1, 0, 1]).astype(bool)
    rep = classical_accept(a, b)
    assert rep.accepted is True
    assert rep.n_counterexamples == 0


def test_synthetic_planted_rule_holds_approximately():
    spec = WorldSpec(seed=0, eps=0.05, gamma_var=None)
    wd = sample_world(spec, m=5000)
    X = wd.X.astype(bool)
    alpha = Conjunction(tuple(Literal(v) for v in spec.alpha_vars))
    a = alpha(X); b = X[:, spec.beta_var]
    rep = empirical_validity(a, b)
    # eps=0.05 means we expect val around 0.95
    assert rep.val > 0.9
    assert rep.val < 1.0   # almost surely the classical baseline rejects


def test_classical_likely_rejects_under_exceptions():
    spec = WorldSpec(seed=42, eps=0.05, gamma_var=None)
    wd = sample_world(spec, m=2000)
    X = wd.X.astype(bool)
    alpha = Conjunction(tuple(Literal(v) for v in spec.alpha_vars))
    a = alpha(X); b = X[:, spec.beta_var]
    cls = classical_accept(a, b)
    # Vanishingly unlikely a 5%-exception rule survives 2000 examples without
    # a single counterexample.
    assert cls.accepted is False
    assert cls.n_counterexamples > 0


def test_delta_gamma_reverses_under_strong_flip():
    spec = WorldSpec(seed=1, eps=0.02, flip_in_context=0.9, p_gamma=0.2)
    wd = sample_world(spec, m=10000)
    X = wd.X.astype(bool)
    alpha = Conjunction(tuple(Literal(v) for v in spec.alpha_vars))
    a = alpha(X); b = X[:, spec.beta_var]; g = X[:, spec.gamma_var]
    rep = delta_gamma(a, b, g)
    assert rep.base.val > 0.8
    assert rep.refined.val < 0.5
    assert rep.delta < -0.4
    assert rep.reversed is True


def test_elimination_recovers_planted_disjunction():
    rng = np.random.default_rng(0)
    n, m = 20, 4000
    X = (rng.random((n, m)) < 0.3).T.astype(bool)
    cause_vars = [3, 11]
    for v in cause_vars:
        X[:, v] = rng.random(m) < 0.2
    cause = X[:, cause_vars[0]] | X[:, cause_vars[1]]
    q = cause.copy()
    res = disjunctive_abduce(X, q)
    recovered = {l.var for l in res.hypothesis.lits if not l.negated}
    # The two planted causes must survive Elimination.
    for v in cause_vars:
        assert v in recovered
    # And the empirical validity is exactly 1 because q == cause.
    assert math.isclose(res.val, 1.0)


if __name__ == "__main__":
    fns = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
