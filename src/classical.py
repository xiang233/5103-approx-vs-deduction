"""Classical-deduction baseline.

Theorem 6.2 in the textbook gives the natural classical baseline against
which we compare approximate reasoning: "accept q iff q is true on every
sample example, otherwise reject." Equivalently, an implication alpha -> beta
is classically accepted iff there is no counterexample on the sample.

This module makes that algorithm explicit so the comparison is not rhetorical.
The two views can then be reported side-by-side on the same data and the same
candidate rule.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClassicalReport:
    accepted: bool
    n_counterexamples: int
    n_alpha: int
    n_total: int


def classical_accept(alpha_mask: np.ndarray, beta_mask: np.ndarray) -> ClassicalReport:
    """Classical perfect-on-sample acceptor for alpha -> beta.

    Accept iff every example satisfying alpha also satisfies beta. Equivalent
    to: "there exists no counterexample" -- this is what classical entailment
    requires, restricted to the observed sample.
    """
    n_total = int(alpha_mask.shape[0])
    n_alpha = int(alpha_mask.sum())
    counter = np.logical_and(alpha_mask, ~beta_mask)
    n_counter = int(counter.sum())
    return ClassicalReport(
        accepted=(n_counter == 0 and n_alpha > 0),
        n_counterexamples=n_counter,
        n_alpha=n_alpha,
        n_total=n_total,
    )
