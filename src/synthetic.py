"""Synthetic Boolean world generator.

We produce examples x in {0,1}^n with:
  * a planted base rule  alpha -> beta  that holds with probability 1 - eps
    (eps is the exception rate),
  * an optional context predicate gamma whose presence flips beta to NOT beta
    with probability flip_in_context (the non-monotonic ingredient),
  * iid uniform "noise" attributes that have nothing to do with the rule.

The ground truth (which variables are alpha, beta, gamma, exception flags) is
returned alongside the data so experiments can audit recovery.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class WorldSpec:
    n_vars: int = 12
    alpha_vars: tuple[int, ...] = (0, 1)        # antecedent: AND of these
    beta_var: int = 2                           # consequent (single literal)
    gamma_var: int | None = 3                   # context variable
    eps: float = 0.05                           # base exception rate
    flip_in_context: float = 0.85               # P(NOT beta | alpha & gamma)
    p_alpha: float = 0.4                        # P(alpha holds) -- target marginal
    p_gamma: float = 0.15                       # P(gamma) within alpha-firing rows
    seed: int = 0


@dataclass(frozen=True)
class WorldData:
    X: np.ndarray                               # (m, n_vars) Boolean
    spec: WorldSpec
    is_exception: np.ndarray                    # length-m boolean


def sample_world(spec: WorldSpec, m: int) -> WorldData:
    rng = np.random.default_rng(spec.seed)
    n = spec.n_vars
    X = rng.integers(0, 2, size=(m, n)).astype(bool)

    # Force the marginal Pr[alpha] roughly to spec.p_alpha by resampling the
    # alpha vars jointly. With independent uniform vars Pr[alpha] = 2^{-|A|};
    # if the user wants something other than that, adjust per-var bias.
    bias = spec.p_alpha ** (1 / max(len(spec.alpha_vars), 1))
    for v in spec.alpha_vars:
        X[:, v] = rng.random(m) < bias

    if spec.gamma_var is not None:
        X[:, spec.gamma_var] = rng.random(m) < spec.p_gamma

    alpha_mask = np.logical_and.reduce([X[:, v] for v in spec.alpha_vars])
    if spec.gamma_var is not None:
        gamma_mask = X[:, spec.gamma_var]
    else:
        gamma_mask = np.zeros(m, dtype=bool)

    # Default beta wherever alpha holds: True. Then introduce two kinds of
    # exceptions:
    #   1. iid base exceptions on alpha-firing rows at rate eps (regardless of gamma)
    #   2. context-driven flips on (alpha & gamma) rows at rate flip_in_context
    beta = X[:, spec.beta_var].copy()  # start from whatever was sampled
    beta[alpha_mask] = True            # rule says: alpha -> beta in the no-exception case

    base_exc = (rng.random(m) < spec.eps) & alpha_mask
    beta[base_exc] = ~beta[base_exc]

    if spec.gamma_var is not None:
        ctx_exc = (rng.random(m) < spec.flip_in_context) & alpha_mask & gamma_mask
        beta[ctx_exc] = ~beta[ctx_exc]
    else:
        ctx_exc = np.zeros(m, dtype=bool)

    X[:, spec.beta_var] = beta
    return WorldData(X=X.astype(np.uint8), spec=spec,
                     is_exception=(base_exc | ctx_exc))
