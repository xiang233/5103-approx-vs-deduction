"""Approximate-reasoning metrics from textbook ch.6.

We compute three quantities for a candidate rule alpha -> beta on a Boolean
matrix X of shape (m, n):

  empirical_validity = #{x: alpha(x)=1 and beta(x)=1} / #{x: alpha(x)=1}
  coverage           = #{x: alpha(x)=1} / m
  delta_gamma        = val((alpha & gamma) -> beta) - val(alpha -> beta)

We also expose Wilson confidence intervals on the validity proportion (so the
proposal's "address sample size and variance" comment becomes one line of code
at every call site) and the Hoeffding sample-size bound from Theorem 6.6.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidityReport:
    val: float          # empirical Pr[beta | alpha]
    cov: float          # empirical Pr[alpha]
    n_alpha: int        # number of examples with alpha=1
    n_alpha_beta: int   # number of examples with alpha=1 AND beta=1
    ci_low: float       # Wilson 95% lower bound on val
    ci_high: float      # Wilson 95% upper bound on val
    n_total: int


def empirical_validity(alpha_mask: np.ndarray, beta_mask: np.ndarray) -> ValidityReport:
    """Both arguments are 1-D boolean arrays evaluated on the same dataset."""
    if alpha_mask.shape != beta_mask.shape:
        raise ValueError("alpha and beta masks must have identical shapes")
    n_total = int(alpha_mask.shape[0])
    n_alpha = int(alpha_mask.sum())
    n_alpha_beta = int(np.logical_and(alpha_mask, beta_mask).sum())
    val = (n_alpha_beta / n_alpha) if n_alpha > 0 else float("nan")
    cov = n_alpha / n_total if n_total > 0 else 0.0
    low, high = wilson_ci(n_alpha_beta, n_alpha) if n_alpha > 0 else (float("nan"), float("nan"))
    return ValidityReport(val, cov, n_alpha, n_alpha_beta, low, high, n_total)


def wilson_ci(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided Wilson interval for a binomial proportion.

    Far better than the Wald interval at small `trials` or proportions near 0/1,
    which is exactly the regime the evaluation flagged.
    """
    if trials == 0:
        return (float("nan"), float("nan"))
    p_hat = successes / trials
    denom = 1 + z * z / trials
    center = (p_hat + z * z / (2 * trials)) / denom
    half = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * trials)) / trials) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def hoeffding_sample_bound(eps: float, delta: float, log2_hyp_class_size: float) -> int:
    """Sample size m such that all queries q in a class of size 2^B are
    eps-validity-estimated with prob >= 1-delta (Theorem 6.6).

    m >= (1 / (2 eps^2)) * (B ln 2 + ln(2/delta))
    """
    if not 0 < eps < 1 or not 0 < delta < 1:
        raise ValueError("eps and delta must be in (0,1)")
    return math.ceil((1 / (2 * eps * eps)) * (log2_hyp_class_size * math.log(2) + math.log(2 / delta)))


def min_examples_for_antecedent(coverage: float, eps: float, delta: float) -> int:
    """Heuristic: we need the antecedent to fire enough times to estimate
    Pr[beta | alpha] to within eps with prob 1-delta. Apply Hoeffding to the
    conditional sample.
    """
    inner = math.ceil((1 / (2 * eps * eps)) * math.log(2 / delta))
    return math.ceil(inner / max(coverage, 1e-12))
