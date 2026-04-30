"""Experiment 4: mushroom case study.

This is the real-data demonstration that approximate validity finds rules
that classical deduction would reject. We do four things:

  1. Reproduce the Duch et al. rules (odor not in {a,l,n} -> poisonous) and
     show their classical-vs-approximate gap on the full dataset.
  2. Exhaustively search short conjunctions of named predicates and rank by
     empirical validity for the consequent `poisonous`.
  3. Test non-monotonicity: take a rule with high validity, add each other
     predicate as a context, find the largest negative Delta_gamma.
  4. Run k-DNF abduction (k=2) to discover an explanation for `poisonous`,
     scored by plausibility & explanatory power.

Output: results/exp4_mushroom_*.csv
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from ..abduction import kdnf_abduce
from ..booleanize import BooleanizedDataset, load_mushroom, predicate_mask
from ..classical import classical_accept
from ..conditional import delta_gamma
from ..validity import empirical_validity


def part1_duch_rules(ds: BooleanizedDataset, writer: csv.DictWriter) -> None:
    print("\n[Part 1] Duch et al. benchmark rules vs. classical baseline")
    odor_almond = predicate_mask(ds, "odor=a")
    odor_anise  = predicate_mask(ds, "odor=l")
    odor_none   = predicate_mask(ds, "odor=n")
    edible_per_duch = odor_almond | odor_anise | odor_none

    # Rule P_1: NOT(odor in {a,l,n})  ->  poisonous
    alpha = ~edible_per_duch
    beta = ds.y
    cls = classical_accept(alpha, beta)
    emp = empirical_validity(alpha, beta)
    rec = {
        "rule": "NOT(odor in {a,l,n}) -> poisonous (Duch P1)",
        "classical_accepted": int(cls.accepted),
        "n_counterexamples": cls.n_counterexamples,
        "val": emp.val, "cov": emp.cov,
        "wilson_low": emp.ci_low, "wilson_high": emp.ci_high,
        "n_alpha": emp.n_alpha,
    }
    print(f"  {rec}")
    writer.writerow(rec)

    # Rule P_2: spore-print-color=green -> poisonous
    alpha = predicate_mask(ds, "spore-print-color=r")  # r = green per the .names file
    cls = classical_accept(alpha, beta)
    emp = empirical_validity(alpha, beta)
    rec = {
        "rule": "spore-print-color=green -> poisonous (Duch P2)",
        "classical_accepted": int(cls.accepted),
        "n_counterexamples": cls.n_counterexamples,
        "val": emp.val, "cov": emp.cov,
        "wilson_low": emp.ci_low, "wilson_high": emp.ci_high,
        "n_alpha": emp.n_alpha,
    }
    print(f"  {rec}")
    writer.writerow(rec)


def part2_top_conjunctions(ds: BooleanizedDataset,
                           max_pairs: int = 200, max_size: int = 2) -> list[dict]:
    """Search over conjunctions of size 1..max_size and report top ones by val."""
    print("\n[Part 2] Top conjunctions for `poisonous` (single + pair)")
    n = ds.X.shape[1]
    beta = ds.y
    rows: list[dict] = []

    # singletons
    for j, name in enumerate(ds.feature_names):
        a = ds.X[:, j]
        if a.sum() < 30:                # require coverage to avoid noise
            continue
        emp = empirical_validity(a, beta)
        cls = classical_accept(a, beta)
        rows.append({
            "antecedent": name, "size": 1,
            "val": emp.val, "cov": emp.cov,
            "wilson_low": emp.ci_low, "wilson_high": emp.ci_high,
            "n_alpha": emp.n_alpha,
            "classical_accepted": int(cls.accepted),
            "n_counterexamples": cls.n_counterexamples,
        })

    # pairs (capped to keep this experiment runnable)
    if max_size >= 2:
        # To keep it tractable, only pair predicates from different attributes.
        attr_of = [name.split("=", 1)[0] for name in ds.feature_names]
        for i, j in combinations(range(n), 2):
            if attr_of[i] == attr_of[j]:
                continue
            a = ds.X[:, i] & ds.X[:, j]
            if a.sum() < 30:
                continue
            emp = empirical_validity(a, beta)
            if emp.val < 0.95 and emp.val > 0.05:
                continue
            cls = classical_accept(a, beta)
            rows.append({
                "antecedent": f"{ds.feature_names[i]} & {ds.feature_names[j]}",
                "size": 2,
                "val": emp.val, "cov": emp.cov,
                "wilson_low": emp.ci_low, "wilson_high": emp.ci_high,
                "n_alpha": emp.n_alpha,
                "classical_accepted": int(cls.accepted),
                "n_counterexamples": cls.n_counterexamples,
            })

    rows.sort(key=lambda r: (-r["val"], -r["cov"]))
    print(f"  evaluated {len(rows)} candidate rules; top 5 by val:")
    for r in rows[:5]:
        print(f"    val={r['val']:.4f} cov={r['cov']:.4f} cls={r['classical_accepted']}  {r['antecedent']}")
    return rows


def part3_nonmonotonicity(ds: BooleanizedDataset, base_rule: str) -> list[dict]:
    """For a base rule with high val, try every other predicate as gamma and
    report the most context-sensitive (largest |delta_gamma|) ones."""
    print(f"\n[Part 3] Context sensitivity around base rule: {base_rule}")
    base_alpha = predicate_mask(ds, base_rule)
    beta = ds.y
    base = empirical_validity(base_alpha, beta)
    rows: list[dict] = []
    for name in ds.feature_names:
        if name == base_rule:
            continue
        gamma = predicate_mask(ds, name)
        rep = delta_gamma(base_alpha, beta, gamma)
        if rep.refined.n_alpha < 30:
            continue
        rows.append({
            "base_rule": base_rule,
            "gamma": name,
            "base_val": rep.base.val,
            "refined_val": rep.refined.val,
            "delta": rep.delta,
            "reversed": int(rep.reversed),
            "n_alpha_gamma": rep.refined.n_alpha,
        })
    rows.sort(key=lambda r: r["delta"])
    print(f"  base val = {base.val:.4f} (n_alpha = {base.n_alpha})")
    print("  most negative Delta_gamma:")
    for r in rows[:5]:
        print(f"    delta={r['delta']:+.4f}  refined={r['refined_val']:.4f}  gamma={r['gamma']}")
    return rows


def part4_kdnf_abduction(ds: BooleanizedDataset, k: int = 2) -> dict:
    """Run k-DNF abduction with `poisonous` as the query."""
    print(f"\n[Part 4] k-DNF abduction (k={k}) for `poisonous`")
    res = kdnf_abduce(ds.X, ds.y, k=k)
    print(f"  found {res.n_terms} terms; "
          f"plausibility={res.score.plausibility:.4f}  "
          f"explanatory_power={res.score.explanatory_power:.4f}  "
          f"f_score={res.score.f_score:.4f}")
    if res.n_terms <= 10:
        for t in res.hypothesis.terms:
            print(f"    term: {t}")
    return {
        "k": k,
        "n_terms": res.n_terms,
        "plausibility": res.score.plausibility,
        "explanatory_power": res.score.explanatory_power,
        "f_score": res.score.f_score,
    }


def run(data_path: str = "mushroom/agaricus-lepiota.data") -> None:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_mushroom(data_path)
    print(f"loaded mushroom: m={ds.X.shape[0]} samples, n={ds.X.shape[1]} predicates")

    # Part 1: Duch rules
    with (out_dir / "exp4_mushroom_duch.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "rule", "classical_accepted", "n_counterexamples",
            "val", "cov", "wilson_low", "wilson_high", "n_alpha",
        ])
        w.writeheader()
        part1_duch_rules(ds, w)

    # Part 2: top conjunctions
    rows2 = part2_top_conjunctions(ds)
    with (out_dir / "exp4_mushroom_top_conjs.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows2[0].keys()))
        w.writeheader()
        w.writerows(rows2)

    # Part 3: non-monotonicity around the strongest single antecedent
    if rows2:
        strongest = next(r for r in rows2 if r["size"] == 1 and r["val"] >= 0.85
                         and r["cov"] >= 0.05 and not r["classical_accepted"])
        rows3 = part3_nonmonotonicity(ds, strongest["antecedent"])
        with (out_dir / "exp4_mushroom_delta_gamma.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows3[0].keys()))
            w.writeheader()
            w.writerows(rows3)

    # Part 4: k-DNF abduction
    res4 = part4_kdnf_abduction(ds, k=2)
    with (out_dir / "exp4_mushroom_abduction.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res4.keys()))
        w.writeheader()
        w.writerow(res4)


if __name__ == "__main__":
    run()
