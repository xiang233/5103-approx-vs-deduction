"""Abduction wrapper: scoring + k-DNF reduction on top of `elimination.py`.

The textbook (Theorem 6.14 + remark on the reduction of Theorem 4.14) extends
disjunctive abduction to k-DNF abduction by lifting each conjunction of up to
k literals to a fresh "macro-variable", running the disjunctive Elimination
algorithm in the lifted space, and reading the result back as a k-DNF.

After Elimination, every surviving positive macro-literal has empirical
Pr[q | macro] = 1.0 on the training sample (that is the algorithm's
guarantee). The remaining choice is *plausibility*: how often the macro fires.
Rare macros (say, firing on 2 out of 8000 examples) are technically valid
explanations but carry no predictive weight. `min_plausibility` prunes them;
`top_k` caps the final list. The result is a human-readable k-DNF.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Optional

import numpy as np

from .elimination import disjunctive_abduce
from .rules import Conjunction, KDNF, Literal


@dataclass(frozen=True)
class AbductionScore:
    plausibility: float       # Pr[h]  ("how often H occurs")
    explanatory_power: float  # Pr[q | h]  ("how often Y occurs when H holds")
    f_score: float            # harmonic mean -- single comparable number


def score_hypothesis(h_mask: np.ndarray, q_mask: np.ndarray) -> AbductionScore:
    n = h_mask.shape[0]
    n_h = int(h_mask.sum())
    n_h_q = int(np.logical_and(h_mask, q_mask).sum())
    plaus = n_h / n if n > 0 else 0.0
    expl  = (n_h_q / n_h) if n_h > 0 else 0.0
    f     = (2 * plaus * expl / (plaus + expl)) if (plaus + expl) > 0 else 0.0
    return AbductionScore(plausibility=plaus, explanatory_power=expl, f_score=f)


@dataclass(frozen=True)
class KDNFAbductionResult:
    hypothesis: KDNF
    score: AbductionScore              # score of the full disjunction on training data
    n_terms: int
    per_term_scores: tuple             # tuple of (str_label, AbductionScore) sorted by plausibility


def _term_label(conj: Conjunction, feature_names: Optional[list] = None) -> str:
    if feature_names is None or not conj.lits:
        return str(conj)
    parts = []
    for lit in conj.lits:
        name = feature_names[lit.var]
        parts.append(f"~{name}" if lit.negated else name)
    return " & ".join(parts)


def kdnf_abduce(X: np.ndarray,
                q_mask: np.ndarray,
                k: int = 2,
                exclude_vars: Optional[set] = None,
                min_plausibility: float = 0.01,
                top_k: Optional[int] = 15,
                feature_names: Optional[list] = None) -> KDNFAbductionResult:
    """k-DNF abduction by lifting to macro-variables, then Elimination.

    Parameters
    ----------
    X                 : (m, n) bool/int training matrix.
    q_mask            : length-m bool query to be explained.
    k                 : max conjunction length; k=1 is plain disjunctive abduction.
    exclude_vars      : column indices to omit from the hypothesis vocabulary.
    min_plausibility  : drop surviving terms with Pr[macro] < this threshold.
                        Defaults to 0.01 (fire on ≥1% of training examples).
    top_k             : keep at most this many terms, ranked by plausibility.
                        None = no cap.

    Caveat: the macro-variable space has size sum_{i=1..k} C(n,i) * 2^i.
    For k=2, n=116 (mushroom) that is ~26,000 macros. Elimination is linear
    in that, so it completes in seconds, but raising k to 3 would be slow.
    """
    n_vars = X.shape[1]
    Xb = X.astype(bool)
    qb = q_mask.astype(bool)
    exclude_vars = exclude_vars or set()

    # Build macro table: (Conjunction, evaluated mask on X)
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
        return KDNFAbductionResult(empty, AbductionScore(0.0, 0.0, 0.0), 0, tuple())

    Xmacro = np.column_stack([m for _, m in macros]).astype(bool)
    res = disjunctive_abduce(Xmacro, qb, exclude_vars=set())

    # Only positive literals survive usefully (negated literals mean the
    # conjunction is *absent*, which is not a useful explanation).
    pos_indices = [lit.var for lit in res.hypothesis.lits if not lit.negated]

    # Score each surviving term individually, then filter + rank.
    scored: list[tuple[float, int]] = []   # (plausibility, macro_index)
    m = Xb.shape[0]
    for i in pos_indices:
        macro_mask = macros[i][1]
        plaus = float(macro_mask.sum()) / m
        if plaus >= min_plausibility:
            scored.append((plaus, i))

    # Sort descending by plausibility; apply top_k cap.
    scored.sort(key=lambda t: -t[0])
    if top_k is not None:
        scored = scored[:top_k]

    # Build output structures.
    kept_terms = tuple(macros[i][0] for _, i in scored)
    h = KDNF(terms=kept_terms, k=k)
    h_mask = h(Xb)
    overall_score = score_hypothesis(h_mask, qb)

    per_term_scores = tuple(
        (_term_label(macros[i][0], feature_names), score_hypothesis(macros[i][1], qb))
        for _, i in scored
    )
    return KDNFAbductionResult(h, overall_score, len(kept_terms), per_term_scores)
