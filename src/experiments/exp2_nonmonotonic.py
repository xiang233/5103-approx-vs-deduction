"""Experiment 2: non-monotonic reversal.

H2: When a planted rule alpha -> beta has empirical validity > 0.85
unconditionally, there exists a context predicate gamma in our generator's
specification such that val((alpha & gamma) -> beta) drops below 0.5
(Pearl's bird/penguin reversal). Vanilla classical deduction has no
mechanism to express this -- a single counterexample already disqualifies
the rule.

Operational falsification: across 20 seeds at m=5000, the mean Delta_gamma
is < -0.4 (a strong reversal) and the reversed-flag fires in >= 80% of seeds.

Output: results/exp2_nonmonotonic.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..conditional import delta_gamma
from ..rules import Conjunction, Literal
from ..synthetic import WorldSpec, sample_world


CONFIG = {
    "n_seeds": 20,
    "m": 5000,
    "flip_grid": [0.0, 0.5, 0.85, 1.0],   # P(flip beta | alpha & gamma)
    "out": "results/exp2_nonmonotonic.csv",
}


def run() -> None:
    out_path = Path(CONFIG["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for flip in CONFIG["flip_grid"]:
        for seed in range(CONFIG["n_seeds"]):
            spec = WorldSpec(seed=seed, eps=0.05, flip_in_context=flip)
            wd = sample_world(spec, CONFIG["m"])
            X = wd.X.astype(bool)
            alpha = Conjunction(tuple(Literal(v) for v in spec.alpha_vars))
            beta = Literal(spec.beta_var)
            gamma = Literal(spec.gamma_var)
            a, b, g = alpha(X), beta(X), gamma(X)
            rep = delta_gamma(a, b, g)
            rows.append({
                "flip_prob": flip, "seed": seed,
                "base_val": rep.base.val, "base_cov": rep.base.cov,
                "refined_val": rep.refined.val, "refined_cov": rep.refined.cov,
                "delta": rep.delta, "reversed": int(rep.reversed),
                "n_alpha": rep.base.n_alpha,
                "n_alpha_gamma": rep.refined.n_alpha,
            })

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    summarise(rows)


def summarise(rows: list[dict]) -> None:
    from collections import defaultdict
    bucket: dict[float, list[dict]] = defaultdict(list)
    for r in rows:
        bucket[r["flip_prob"]].append(r)
    print("\n  flip   mean_base   mean_refined   mean_delta   reversal_rate")
    for flip, rs in sorted(bucket.items()):
        bv = np.array([r["base_val"] for r in rs])
        rv = np.array([r["refined_val"] for r in rs])
        d = np.array([r["delta"] for r in rs])
        rev = sum(r["reversed"] for r in rs) / len(rs)
        print(f"  {flip:>4.2f}   {bv.mean():>9.4f}   {rv.mean():>12.4f}   {d.mean():>+10.4f}   {rev:>13.2f}")


if __name__ == "__main__":
    run()
