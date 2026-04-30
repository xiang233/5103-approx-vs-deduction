"""Experiment 3: Elimination Algorithm recovers a planted disjunctive cause.

H3 (textbook Theorem 6.14): plant a target query q := c(x) where c is a
disjunction of `r` literals over n vars, with Pr[c] >= mu and the relation
Pr[q | c] = 1 by construction. With m = O((1/(mu*eps)) (n + log(1/delta)))
samples, the Elimination algorithm returns a hypothesis h whose empirical
val Pr[q | h] >= 1 - eps and whose coverage Pr[h] >= (1-gamma)*mu.

Operational falsification: across 10 seeds, n in {16, 32}, r in {2, 4},
mu = 0.3, eps = 0.05, delta = 0.05, the recovered hypothesis includes all
true literals in 90%+ of seeds and val >= 0.95 always.

Output: results/exp3_abduction.csv
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from ..elimination import disjunctive_abduce


CONFIG = {
    "n_grid": [16, 32],
    "r_grid": [2, 4],
    "n_seeds": 10,
    "mu_target": 0.3,
    "eps": 0.05,
    "delta": 0.05,
    "out": "results/exp3_abduction.csv",
}


def sample_size(n: int, mu: float, eps: float, delta: float) -> int:
    return math.ceil((1 / (mu * eps)) * (n + math.log2(1 / delta)))


def plant_world(n: int, r: int, mu: float, m: int, seed: int) -> tuple[np.ndarray, np.ndarray, set[int]]:
    rng = np.random.default_rng(seed)
    X = (rng.random((m, n)) < 0.3).astype(bool)
    cause_vars = sorted(rng.choice(n, size=r, replace=False).tolist())
    # Bias each cause var so that the union has marginal ~mu.
    p_each = 1 - (1 - mu) ** (1 / r)
    for v in cause_vars:
        X[:, v] = rng.random(m) < p_each
    cause_mask = np.logical_or.reduce([X[:, v] for v in cause_vars])
    # Query is fully entailed by the cause; add a small fraction of "noise"
    # firings of the query unrelated to the cause to make the task realistic.
    q = cause_mask.copy()
    extra = rng.random(m) < 0.02
    q = q | extra
    return X, q, set(cause_vars)


def run() -> None:
    out_path = Path(CONFIG["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for n in CONFIG["n_grid"]:
        m = sample_size(n, CONFIG["mu_target"], CONFIG["eps"], CONFIG["delta"])
        for r in CONFIG["r_grid"]:
            for seed in range(CONFIG["n_seeds"]):
                X, q, cause_vars = plant_world(n, r, CONFIG["mu_target"], m, seed)
                res = disjunctive_abduce(X, q, exclude_vars=set())
                # which planted literals survived elimination?
                recovered_pos = {lit.var for lit in res.hypothesis.lits if not lit.negated}
                hits = recovered_pos & cause_vars
                rows.append({
                    "n": n, "r": r, "m": m, "seed": seed,
                    "n_lits_in_h": len(res.hypothesis.lits),
                    "n_planted_recovered": len(hits),
                    "n_planted_total": len(cause_vars),
                    "exact_recovery": int(hits == cause_vars),
                    "h_val": res.val, "h_cov": res.coverage,
                })

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_path}")
    summarise(rows)


def summarise(rows: list[dict]) -> None:
    from collections import defaultdict
    bucket: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for r in rows:
        bucket[(r["n"], r["r"])].append(r)
    print("\n   n   r        m    mean_val   mean_cov   exact_recovery_rate   mean_extra_lits")
    for (n, rr), rs in sorted(bucket.items()):
        m = rs[0]["m"]
        v = np.mean([r["h_val"] for r in rs])
        c = np.mean([r["h_cov"] for r in rs])
        ex = np.mean([r["exact_recovery"] for r in rs])
        # extra literals = (lits in h) - (planted positive literals recovered)
        extra = np.mean([r["n_lits_in_h"] - r["n_planted_recovered"] for r in rs])
        print(f"  {n:>3} {rr:>3}   {m:>6}    {v:>7.4f}   {c:>7.4f}   {ex:>19.2f}   {extra:>14.2f}")


if __name__ == "__main__":
    run()
