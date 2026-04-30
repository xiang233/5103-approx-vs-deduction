"""Abduction wrapper: scoring + k-DNF reduction on top of `elimination.py`.

The textbook (Theorem 6.14 + remark on the reduction of Theorem 4.14) extends
disjunctive abduction to k-DNF abduction by lifting each conjunction of up to
k literals to a fresh "macro-variable", running the disjunctive Elimination
algorithm in the lifted space, and reading the result back as a k-DNF.

We also score each candidate hypothesis with the proposal's two criteria:

  plausibility       = empirical Pr[h]            ("how often H occurs")
  explanatory_power  = empirical Pr[q | h]        ("how often Y occurs when H holds")
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from .elimination import disjunctive_abduce
from .rules import Conjunction, KDNF, Literal


@dataclass(frozen=True)
class AbductionScore:
    plausibility: float       # Pr[h]
    explanatory_power: float  # Pr[q | h]
    f_score: float            # harmonic mean of the two -- a single comparable number


def score_hypothesis(h_mask: np.ndarray, q_mask: np.ndarray) -> AbductionScore:
    n = h_mask.shape[0]
    n_h = int(h_mask.sum())
    n_h_q = int(np.logical_and(h_mask, q_mask).sum())
    plaus = n_h / n if n > 0 else 0.0
    expl = (n_h_q / n_h) if n_h > 0 else 0.0
    f = (2 * plaus * expl / (plaus + expl)) if (plaus + expl) > 0 else 0.0
    return AbductionScore(plausibility=plaus, explanatory_power=expl, f_score=f)


@dataclass(frozen=True)
class KDNFAbductionResult:
    hypothesis: KDNF
    score: AbductionScore
    n_terms: int


def kdnf_abduce(X: np.ndarray, q_mask: np.ndarray, k: int = 2,
                exclude_vars: set[int] | None = None) -> KDNFAbductionResult:
    """k-DNF abduction by lifting to macro-variables, then Elimination.

    Caveat: the macro-variable space has size 2 * sum_{i=1..k} C(n,i) * 2^i.
    For k=1 this is just disjunctive abduction with negations; for k=2 on
    n=20 vars the lifted dimension is ~800. Stay in the small-k regime.
    """
    n_vars = X.shape[1]
    Xb = X.astype(bool)
    exclude_vars = exclude_vars or set()

    macros: list[tuple[Conjunction, np.ndarray]] = []
    for size in range(1, k + 1):
        for vs in combinations(range(n_vars), size):
            if any(v in exclude_vars for v in vs):
                continue
            for signs in product([False, True], repeat=size):
                conj = Conjunction(tuple(Literal(v, s) for v, s in zip(vs, signs)))
                macros.append((conj, conj(Xb)))

    if not macros:
        empty = KDNF(terms=tuple(), k=k)
        return KDNFAbductionResult(empty, AbductionScore(0.0, 0.0, 0.0), 0)

    Xmacro = np.column_stack([m for _, m in macros]).astype(bool)
    # Forbid negated macros (they would mean "this conjunction is false"); we
    # only want positive disjuncts, matching the textbook's k-DNF representation.
    forbid_neg = set(range(Xmacro.shape[1]))  # block all negations
    res = disjunctive_abduce(Xmacro, q_mask, exclude_vars=set())
    # Filter the resulting disjunction to positive literals only.
    pos_indices = [lit.var for lit in res.hypothesis.lits if not lit.negated]
    terms = tuple(macros[i][0] for i in pos_indices)
    h = KDNF(terms=terms, k=k)
    h_mask = h(Xb)
    return KDNFAbductionResult(h, score_hypothesis(h_mask, q_mask), len(terms))
