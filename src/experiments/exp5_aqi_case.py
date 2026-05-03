"""Experiment 5: AQI case study on elevated-AQI days.

The project proposal framed AQI as a context-sensitive real-data case study:
among higher-AQI events, which pollutant is usually dominant, and how do those
rules change across seasons and broad geographic subgroups?

We therefore:
  1. evaluate a few proposal-inspired rules on train and held-out test;
  2. discover strong singleton/pair context rules for Ozone and PM2.5;
  3. measure context sensitivity (Delta_gamma) for one anchor rule per target.

Output: results/exp5_aqi_*.csv
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from ..aqi_booleanize import (
    AQIDataset,
    load_aqi_context_dataset,
    predicate_mask,
    temporal_train_test_split,
)
from ..classical import classical_accept
from ..conditional import delta_gamma
from ..validity import empirical_validity

MIN_VAL = 0.60
MIN_COV_ABS = 2000
MIN_YEAR = 2016
MAX_YEAR = 2021
TEST_START_YEAR = 2020


def _eval_rule(alpha: np.ndarray, beta: np.ndarray, label: str) -> dict:
    cls = classical_accept(alpha, beta)
    emp = empirical_validity(alpha, beta)
    return {
        "rule": label,
        "classical_accepted": int(cls.accepted),
        "n_counterexamples": cls.n_counterexamples,
        "val": round(emp.val, 6),
        "cov": round(emp.cov, 6),
        "wilson_low": round(emp.ci_low, 6),
        "wilson_high": round(emp.ci_high, 6),
        "n_alpha": emp.n_alpha,
    }


def _mask_for_names(ds: AQIDataset, names: list[str]) -> np.ndarray:
    if not names:
        return np.ones(ds.X.shape[0], dtype=bool)
    masks = [predicate_mask(ds, name) for name in names]
    return np.logical_and.reduce(masks)


def part1_proposal_rules(train_sets: dict[str, AQIDataset],
                         test_sets: dict[str, AQIDataset],
                         writer: csv.DictWriter) -> None:
    print("\n[Part 1] Proposal-inspired AQI rules")
    rules = [
        ("Ozone", ["season=summer"], "summer -> Ozone [AQI>=51]"),
        ("Ozone", ["region=West", "season=summer"], "West & summer -> Ozone [AQI>=51]"),
        ("PM2.5", ["season=winter"], "winter -> PM2.5 [AQI>=51]"),
        ("PM2.5", ["coastal=False", "season=winter"], "inland & winter -> PM2.5 [AQI>=51]"),
    ]

    for split_name, ds_map in [("train", train_sets), ("test", test_sets)]:
        for target, names, label in rules:
            ds = ds_map[target]
            rec = _eval_rule(_mask_for_names(ds, names), ds.y, label)
            rec.update({"target_parameter": target, "split": split_name})
            print(f"  {split_name:>5}  {target:>5}  {rec}")
            writer.writerow(rec)


def part2_top_conjunctions(train: AQIDataset,
                           test: AQIDataset) -> list[dict]:
    target = train.target_parameter
    print(f"\n[Part 2] Top conjunctions for {target}")
    n = train.X.shape[1]
    rows: list[dict] = []
    attr_of = [name.split("=", 1)[0] for name in train.feature_names]

    def _row(antecedent: str,
             size: int,
             a_tr: np.ndarray,
             a_te: np.ndarray) -> dict | None:
        if int(a_tr.sum()) < MIN_COV_ABS or int(a_te.sum()) < MIN_COV_ABS:
            return None
        tr_emp = empirical_validity(a_tr, train.y)
        tr_cls = classical_accept(a_tr, train.y)
        te_emp = empirical_validity(a_te, test.y)
        te_cls = classical_accept(a_te, test.y)
        if tr_cls.accepted and not te_cls.accepted:
            gap = "classical_train_only"
        elif tr_cls.accepted and te_cls.accepted:
            gap = "both_classical"
        elif not tr_cls.accepted and te_emp.val >= MIN_VAL:
            gap = "approx_generalizes"
        else:
            gap = "low_validity"
        return {
            "target_parameter": target,
            "antecedent": antecedent,
            "size": size,
            "train_val": round(tr_emp.val, 6),
            "train_cov": round(tr_emp.cov, 6),
            "train_classical": int(tr_cls.accepted),
            "train_counterexamples": tr_cls.n_counterexamples,
            "train_n_alpha": tr_emp.n_alpha,
            "test_val": round(te_emp.val, 6),
            "test_cov": round(te_emp.cov, 6),
            "test_classical": int(te_cls.accepted),
            "test_counterexamples": te_cls.n_counterexamples,
            "test_wilson_low": round(te_emp.ci_low, 6),
            "test_wilson_high": round(te_emp.ci_high, 6),
            "test_n_alpha": te_emp.n_alpha,
            "gap_type": gap,
        }

    for j, name in enumerate(train.feature_names):
        rec = _row(name, 1, train.X[:, j], test.X[:, j])
        if rec:
            rows.append(rec)

    for i, j in combinations(range(n), 2):
        if attr_of[i] == attr_of[j]:
            continue
        a_tr = train.X[:, i] & train.X[:, j]
        a_te = test.X[:, i] & test.X[:, j]
        rec = _row(f"{train.feature_names[i]} & {train.feature_names[j]}", 2, a_tr, a_te)
        if rec and rec["train_val"] >= 0.40:
            rows.append(rec)

    rows.sort(key=lambda r: (-r["test_val"], -r["test_cov"]))
    by_gap: dict[str, int] = {}
    for row in rows:
        by_gap[row["gap_type"]] = by_gap.get(row["gap_type"], 0) + 1
    print(f"  {len(rows)} candidate rules evaluated; gap breakdown: {by_gap}")
    print("  top 5 approx_generalizes rules:")
    shown = 0
    for row in rows:
        if row["gap_type"] == "approx_generalizes":
            print(
                f"    train_val={row['train_val']:.4f}  "
                f"test_val={row['test_val']:.4f} [{row['test_wilson_low']:.3f},{row['test_wilson_high']:.3f}]  "
                f"{row['antecedent']}"
            )
            shown += 1
            if shown >= 5:
                break
    return rows


def part3_context_sensitivity(train: AQIDataset,
                              test: AQIDataset,
                              base_rule: str) -> list[dict]:
    target = train.target_parameter
    print(f"\n[Part 3] Context sensitivity for {target}: '{base_rule}'")
    tr_alpha = predicate_mask(train, base_rule)
    te_alpha = predicate_mask(test, base_rule)
    base_attr = base_rule.split("=", 1)[0]

    rows: list[dict] = []
    for name in train.feature_names:
        if name == base_rule or name.split("=", 1)[0] == base_attr:
            continue
        tr_rep = delta_gamma(tr_alpha, train.y, predicate_mask(train, name))
        if tr_rep.refined.n_alpha < MIN_COV_ABS:
            continue
        te_rep = delta_gamma(te_alpha, test.y, predicate_mask(test, name))
        rows.append({
            "target_parameter": target,
            "base_rule": base_rule,
            "gamma": name,
            "train_base_val": round(tr_rep.base.val, 6),
            "train_refined_val": round(tr_rep.refined.val, 6),
            "train_delta": round(tr_rep.delta, 6),
            "train_reversed": int(tr_rep.reversed),
            "test_base_val": round(te_rep.base.val, 6),
            "test_refined_val": round(te_rep.refined.val, 6),
            "test_delta": round(te_rep.delta, 6),
            "test_reversed": int(te_rep.reversed),
            "test_n_alpha_gamma": te_rep.refined.n_alpha,
        })

    rows.sort(key=lambda r: -abs(r["train_delta"]))
    print("  top 5 contexts by |train Delta_gamma|:")
    for row in rows[:5]:
        print(
            f"    gamma={row['gamma']:<18} train Delta={row['train_delta']:+.4f}  "
            f"test Delta={row['test_delta']:+.4f}"
        )
    return rows


def run(data_path: str = "AQI/aqi_daily_1980_to_2021.csv") -> None:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        target: load_aqi_context_dataset(
            data_path,
            target_parameter=target,
            min_year=MIN_YEAR,
            max_year=MAX_YEAR,
            min_aqi=51,
        )
        for target in ("Ozone", "PM2.5")
    }
    train_sets = {target: temporal_train_test_split(ds, TEST_START_YEAR)[0]
                  for target, ds in datasets.items()}
    test_sets = {target: temporal_train_test_split(ds, TEST_START_YEAR)[1]
                 for target, ds in datasets.items()}

    for target in ("Ozone", "PM2.5"):
        full = datasets[target]
        train = train_sets[target]
        test = test_sets[target]
        print(
            f"loaded AQI {target}: {full.X.shape[0]} elevated-AQI rows, "
            f"{train.X.shape[0]} train / {test.X.shape[0]} test, "
            f"positive rate train={train.y.mean():.4f} test={test.y.mean():.4f}"
        )

    with (out_dir / "exp5_aqi_proposal_rules.csv").open("w", newline="") as f:
        fieldnames = [
            "target_parameter", "split", "rule", "classical_accepted",
            "n_counterexamples", "val", "cov", "wilson_low",
            "wilson_high", "n_alpha",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        part1_proposal_rules(train_sets, test_sets, writer)

    for target in ("Ozone", "PM2.5"):
        rows2 = part2_top_conjunctions(train_sets[target], test_sets[target])
        if rows2:
            out_path = out_dir / f"exp5_aqi_{target.lower().replace('.', '').replace('2', '2')}_top_conjs.csv"
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows2[0].keys()))
                writer.writeheader()
                writer.writerows(rows2)

    anchors = {
        "Ozone": "region=West",
        "PM2.5": "coastal=True",
    }
    for target, base_rule in anchors.items():
        rows3 = part3_context_sensitivity(train_sets[target], test_sets[target], base_rule)
        if rows3:
            slug = target.lower().replace(".", "").replace("2", "2")
            out_path = out_dir / f"exp5_aqi_{slug}_delta_gamma.csv"
            with out_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows3[0].keys()))
                writer.writeheader()
                writer.writerows(rows3)


if __name__ == "__main__":
    run()
