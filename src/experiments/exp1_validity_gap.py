"""Experiment 1: the falsifiable hypothesis.

H1: Fix a planted rule alpha -> beta with exception rate eps > 0. Then for
any sample size m large enough that the conditional Hoeffding bound holds,

   * the CLASSICAL acceptor (Theorem 6.2) rejects alpha -> beta on the sample
     with probability approaching 1 as m grows;
   * the APPROXIMATE estimator (Theorem 6.6) reports val close to 1 - eps
     with shrinking variance.

Operational falsification: at m = 5000, eps = 0.05 we expect classical
acceptance rate < 5% across seeds and approximate val within +/-0.01 of 0.95.
If either fails, the hypothesis is wrong and the experiment says so.

Output: results/exp1_validity_gap.csv with one row per (eps, m, seed).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..classical import classical_accept
from ..rules import Conjunction, Literal
from ..synthetic import WorldSpec, sample_world
from ..validity import empirical_validity


CONFIG = {
    "eps_grid": [0.0, 0.01, 0.05, 0.1, 0.2],
    "m_grid": [200, 1000, 5000],
    "n_seeds": 10,
    "out": "results/exp1_validity_gap.csv",
}


def run() -> None:
    out_path = Path(CONFIG["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for eps in CONFIG["eps_grid"]:
        for m in CONFIG["m_grid"]:
            for seed in range(CONFIG["n_seeds"]):
                spec = WorldSpec(seed=seed, eps=eps, gamma_var=None)  # disable context
                wd = sample_world(spec, m)
                X = wd.X
                alpha = Conjunction(tuple(Literal(v) for v in spec.alpha_vars))
                beta = Literal(spec.beta_var)
                a_mask = alpha(X.astype(bool))
                b_mask = beta(X.astype(bool))

                cls = classical_accept(a_mask, b_mask)
                emp = empirical_validity(a_mask, b_mask)
                rows.append({
                    "eps": eps, "m": m, "seed": seed,
                    "classical_accepted": int(cls.accepted),
                    "n_counterexamples": cls.n_counterexamples,
                    "empirical_val": emp.val,
                    "empirical_cov": emp.cov,
                    "wilson_low": emp.ci_low,
                    "wilson_high": emp.ci_high,
                    "n_alpha": emp.n_alpha,
                })

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    summarise(rows)


def summarise(rows: list[dict]) -> None:
    """Print one-line falsification report per (eps, m)."""
    from collections import defaultdict
    bucket: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for r in rows:
        bucket[(r["eps"], r["m"])].append(r)
    print("\n  eps    m   classical_accept_rate   mean_val   sd_val")
    for (eps, m), rs in sorted(bucket.items()):
        acc_rate = sum(r["classical_accepted"] for r in rs) / len(rs)
        vals = np.array([r["empirical_val"] for r in rs])
        print(f"  {eps:>5.2f}  {m:>5}   {acc_rate:>20.2f}   {vals.mean():>7.4f}   {vals.std():>5.4f}")


if __name__ == "__main__":
    run()
