"""Contextual validity and the non-monotonicity score Delta_gamma.

Implements Definition 6.8 in the textbook: filter examples by a context
predicate and recompute empirical validity. The reversal between
val(alpha -> beta) and val((alpha & gamma) -> beta) is the empirical signature
of non-monotonic reasoning (Sections 6.3-6.4, the bird/penguin example).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .validity import ValidityReport, empirical_validity


@dataclass(frozen=True)
class DeltaGammaReport:
    base: ValidityReport
    refined: ValidityReport
    delta: float       # refined.val - base.val (NaN if either side undefined)
    reversed: bool     # True iff base.val > 0.5 and refined.val < 0.5 (or vice versa)


def delta_gamma(alpha_mask: np.ndarray, beta_mask: np.ndarray,
                gamma_mask: np.ndarray) -> DeltaGammaReport:
    """Compute the change in empirical validity when conditioning on gamma."""
    base = empirical_validity(alpha_mask, beta_mask)
    refined_alpha = np.logical_and(alpha_mask, gamma_mask)
    refined = empirical_validity(refined_alpha, beta_mask)
    if np.isnan(base.val) or np.isnan(refined.val):
        delta = float("nan")
        reversed_ = False
    else:
        delta = refined.val - base.val
        reversed_ = (base.val > 0.5 and refined.val < 0.5) or \
                    (base.val < 0.5 and refined.val > 0.5)
    return DeltaGammaReport(base=base, refined=refined, delta=delta, reversed=reversed_)
