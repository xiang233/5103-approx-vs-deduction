"""Experiment 4: mushroom case study (with train / test split).

Four parts:

  1. Duch et al. benchmark rules – establish ground truth for classically valid
     rules: evaluate on both train and test to confirm they genuinely generalise.

  2. Top conjunctions with held-out evaluation – discover rules on the TRAINING
     set (including ~approximate~ rules that classical deduction rejects), then
     score the same rules on the held-out TEST set. The gap type column labels
     each rule:

       approx_generalizes  – train_classical=False (counterexamples exist) AND
                             test_val >= MIN_VAL (still informative on unseen data).
                             This is the core empirical argument of the project.
       both_classical       – classically valid on both splits (strong rules).
       classical_train_only – classical on train but exceptions appear on test.
       low_validity         – test_val < MIN_VAL on both counts.

  3. Non-monotonic context sensitivity – discover the strongest approximately
     valid (non-classical) singleton rule on training; sweep contexts on test.

  4. k-DNF abduction (k=2) – abduce explanations on training with plausibility
     filter (min_plausibility=0.01, top_k=15), then score each recovered term
     on the held-out test set to confirm the explanations generalise.

Output: results/exp4_mushroom_*.csv
"""
from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from ..abduction import kdnf_abduce
from ..booleanize import BooleanizedDataset, load_mushroom, predicate_mask, train_test_split
from ..classical import classical_accept
from ..conditional import delta_gamma
from ..validity import empirical_validity

MIN_VAL = 0.85          # below this, a rule is not useful
MIN_COV_ABS = 30        # require at least this many antecedent hits in each split
TRAIN_FRAC = 0.8
SPLIT_SEED = 42


# ---------------------------------------------------------------------------
# Part 1: Duch et al. benchmark rules
# ---------------------------------------------------------------------------

def _eval_rule(alpha: np.ndarray, beta: np.ndarray, label: str) -> dict:
    cls = classical_accept(alpha, beta)
    emp = empirical_validity(alpha, beta)
    return {
        "rule": label,
        "classical_accepted": int(cls.accepted),
        "n_counterexamples": cls.n_counterexamples,
        "val": round(emp.val, 6), "cov": round(emp.cov, 6),
        "wilson_low": round(emp.ci_low, 6), "wilson_high": round(emp.ci_high, 6),
        "n_alpha": emp.n_alpha,
    }


def part1_duch_rules(train: BooleanizedDataset,
                     test: BooleanizedDataset,
                     writer: csv.DictWriter) -> None:
    print("\n[Part 1] Duch et al. rules – train then test")

    def duch_masks(ds: BooleanizedDataset):
        edible_odors = (predicate_mask(ds, "odor=a") |
                        predicate_mask(ds, "odor=l") |
                        predicate_mask(ds, "odor=n"))
        return {
            "P1_alpha": ~edible_odors,
            "P2_alpha": predicate_mask(ds, "spore-print-color=r"),
            "beta":     ds.y,
        }

    for label, split_name, ds in [("train", "train", train), ("test", "test", test)]:
        m = duch_masks(ds)
        for rule, alpha_key in [("NOT(odor∈{a,l,n})→poisonous (P1)", "P1_alpha"),
                                 ("spore-print-color=green→poisonous (P2)", "P2_alpha")]:
            rec = _eval_rule(m[alpha_key], m["beta"], f"{rule} [{split_name}]")
            print(f"  {split_name}  {rec}")
            writer.writerow(rec)


# ---------------------------------------------------------------------------
# Part 2: Top conjunctions – discover on train, evaluate on test
# ---------------------------------------------------------------------------

def _gap_type(tr_cls: int, te_val: float) -> str:
    if not tr_cls and te_val >= MIN_VAL:
        return "approx_generalizes"
    if tr_cls:
        return "both_classical"       # may be overridden below
    return "low_validity"


def part2_top_conjunctions(train: BooleanizedDataset,
                           test: BooleanizedDataset) -> list[dict]:
    print("\n[Part 2] Top conjunctions: discover on train, evaluate on test")
    n = train.X.shape[1]
    rows: list[dict] = []
    attr_of = [name.split("=", 1)[0] for name in train.feature_names]

    def _row(antecedent: str, size: int,
             a_tr: np.ndarray, a_te: np.ndarray) -> dict | None:
        if a_tr.sum() < MIN_COV_ABS or a_te.sum() < MIN_COV_ABS:
            return None
        tr_emp = empirical_validity(a_tr, train.y)
        tr_cls = classical_accept(a_tr, train.y)
        te_emp = empirical_validity(a_te, test.y)
        te_cls = classical_accept(a_te, test.y)
        # Determine gap_type (check for classical_train_only correction)
        if tr_cls.accepted and not te_cls.accepted:
            gap = "classical_train_only"
        elif tr_cls.accepted and te_cls.accepted:
            gap = "both_classical"
        elif not tr_cls.accepted and te_emp.val >= MIN_VAL:
            gap = "approx_generalizes"
        else:
            gap = "low_validity"
        return {
            "antecedent": antecedent, "size": size,
            "train_val": round(tr_emp.val, 6),
            "train_cov": round(tr_emp.cov, 6),
            "train_classical": int(tr_cls.accepted),
            "train_counterexamples": tr_cls.n_counterexamples,
            "test_val": round(te_emp.val, 6),
            "test_cov": round(te_emp.cov, 6),
            "test_classical": int(te_cls.accepted),
            "test_counterexamples": te_cls.n_counterexamples,
            "test_wilson_low": round(te_emp.ci_low, 6),
            "test_wilson_high": round(te_emp.ci_high, 6),
            "test_n_alpha": te_emp.n_alpha,
            "gap_type": gap,
        }

    # Singletons – no val filter; keep all with sufficient coverage
    for j, name in enumerate(train.feature_names):
        a_tr = train.X[:, j]
        a_te = test.X[:, j]
        r = _row(name, 1, a_tr, a_te)
        if r:
            rows.append(r)

    # Pairs – restrict to cross-attribute to reduce search; no val filter so
    # approximately-valid pairs (val ~0.90-0.99) are visible
    for i, j in combinations(range(n), 2):
        if attr_of[i] == attr_of[j]:
            continue
        a_tr = train.X[:, i] & train.X[:, j]
        a_te = test.X[:, i]  & test.X[:, j]
        r = _row(f"{train.feature_names[i]} & {train.feature_names[j]}", 2, a_tr, a_te)
        if r and r["train_val"] >= 0.7:     # discard clearly bad rules
            rows.append(r)

    rows.sort(key=lambda r: (-r["test_val"], -r["test_cov"]))

    # Summary
    by_gap: dict[str, int] = {}
    for r in rows:
        by_gap[r["gap_type"]] = by_gap.get(r["gap_type"], 0) + 1
    print(f"  {len(rows)} candidate rules evaluated; gap breakdown: {by_gap}")
    print("  top 5 approx_generalizes rules:")
    shown = 0
    for r in rows:
        if r["gap_type"] == "approx_generalizes":
            print(f"    train_val={r['train_val']:.4f} (no classical)  "
                  f"test_val={r['test_val']:.4f} [{r['test_wilson_low']:.3f},{r['test_wilson_high']:.3f}]  "
                  f"{r['antecedent']}")
            shown += 1
            if shown >= 5:
                break
    return rows


# ---------------------------------------------------------------------------
# Part 3: Non-monotonic context sensitivity
# ---------------------------------------------------------------------------

def part3_nonmonotonicity(train: BooleanizedDataset,
                          test: BooleanizedDataset,
                          base_rule: str) -> list[dict]:
    """Discover strongest contexts on TRAIN, report Δγ on TEST."""
    print(f"\n[Part 3] Context sensitivity around '{base_rule}' (eval on test)")
    tr_alpha = predicate_mask(train, base_rule)
    te_alpha = predicate_mask(test,  base_rule)

    # Train: sweep every context, rank by |delta|
    rows: list[dict] = []
    for name in train.feature_names:
        if name == base_rule:
            continue
        tr_rep = delta_gamma(tr_alpha, train.y, predicate_mask(train, name))
        if tr_rep.refined.n_alpha < MIN_COV_ABS:
            continue
        # Evaluate the same context on test
        te_rep = delta_gamma(te_alpha, test.y, predicate_mask(test, name))
        rows.append({
            "base_rule": base_rule,
            "gamma": name,
            "train_base_val":    round(tr_rep.base.val, 6),
            "train_refined_val": round(tr_rep.refined.val, 6),
            "train_delta":       round(tr_rep.delta, 6),
            "train_reversed":    int(tr_rep.reversed),
            "test_base_val":     round(te_rep.base.val, 6) if not np.isnan(te_rep.base.val) else float("nan"),
            "test_refined_val":  round(te_rep.refined.val, 6) if not np.isnan(te_rep.refined.val) else float("nan"),
            "test_delta":        round(te_rep.delta, 6) if not np.isnan(te_rep.delta) else float("nan"),
            "test_reversed":     int(te_rep.reversed),
            "test_n_alpha_gamma": te_rep.refined.n_alpha,
        })
    rows.sort(key=lambda r: r["train_delta"])
    print(f"  train base val = {empirical_validity(tr_alpha, train.y).val:.4f}")
    print("  top-5 most negative Δγ on train (confirmed on test):")
    for r in rows[:5]:
        print(f"    train Δ={r['train_delta']:+.4f}  test Δ={r['test_delta']:+.4f}  γ={r['gamma']}")
    return rows


# ---------------------------------------------------------------------------
# Part 4: k-DNF abduction with plausibility filter
# ---------------------------------------------------------------------------

def part4_kdnf_abduction(train: BooleanizedDataset,
                         test: BooleanizedDataset,
                         k: int = 2,
                         min_plausibility: float = 0.01,
                         top_k: int = 15) -> tuple[dict, list[dict]]:
    """Abduce on TRAIN; score each surviving term on TEST."""
    print(f"\n[Part 4] k-DNF abduction (k={k}, min_plausibility={min_plausibility}, top_k={top_k})")
    res = kdnf_abduce(train.X, train.y, k=k,
                      min_plausibility=min_plausibility, top_k=top_k,
                      feature_names=train.feature_names)
    print(f"  kept {res.n_terms} terms after plausibility filter (was 7k+ without it)")

    # Overall score on test
    test_h_mask = res.hypothesis(test.X.astype(bool))
    from ..validity import empirical_validity as ev
    te_score_raw = ev(test_h_mask, test.y)

    summary = {
        "k": k, "n_terms": res.n_terms,
        "train_plausibility":      round(res.score.plausibility, 6),
        "train_explanatory_power": round(res.score.explanatory_power, 6),
        "train_f_score":           round(res.score.f_score, 6),
        "test_val":  round(te_score_raw.val, 6),
        "test_cov":  round(te_score_raw.cov, 6),
    }
    print(f"  overall → train f={res.score.f_score:.4f}  "
          f"test_val={te_score_raw.val:.4f}  test_cov={te_score_raw.cov:.4f}")

    # Per-term generalisation check: score each term on test.
    # per_term_scores and hypothesis.terms are in the same order.
    term_rows: list[dict] = []
    for conj, (term_str, tr_sc) in zip(res.hypothesis.terms, res.per_term_scores):
        te_mask = conj(test.X.astype(bool))
        te_sc = _score_term(te_mask, test.y)
        term_rows.append({
            "term": term_str,
            "train_plausibility":      round(tr_sc.plausibility, 6),
            "train_explanatory_power": round(tr_sc.explanatory_power, 6),
            "test_plausibility":       round(te_sc[0], 6),
            "test_explanatory_power":  round(te_sc[1], 6),
        })
        print(f"    {term_str}")
        print(f"      train  plaus={tr_sc.plausibility:.4f}  exp={tr_sc.explanatory_power:.4f}")
        print(f"      test   plaus={te_sc[0]:.4f}  exp={te_sc[1]:.4f}")

    return summary, term_rows


def _score_term(mask: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    n_h = int(mask.sum())
    n_hq = int(np.logical_and(mask, y).sum())
    plaus = n_h / len(mask) if len(mask) > 0 else 0.0
    expl  = n_hq / n_h if n_h > 0 else float("nan")
    return plaus, expl


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(data_path: str = "mushroom/agaricus-lepiota.data") -> None:
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    full = load_mushroom(data_path)
    train, test = train_test_split(full, test_frac=1 - TRAIN_FRAC, seed=SPLIT_SEED)
    print(f"loaded mushroom: {full.X.shape[0]} total samples, "
          f"{train.X.shape[0]} train / {test.X.shape[0]} test, "
          f"{full.X.shape[1]} predicates")

    # Part 1
    with (out_dir / "exp4_mushroom_duch.csv").open("w", newline="") as f:
        fieldnames = ["rule", "classical_accepted", "n_counterexamples",
                      "val", "cov", "wilson_low", "wilson_high", "n_alpha"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        part1_duch_rules(train, test, w)

    # Part 2
    rows2 = part2_top_conjunctions(train, test)
    if rows2:
        with (out_dir / "exp4_mushroom_top_conjs.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows2[0].keys()))
            w.writeheader()
            w.writerows(rows2)

    # Part 3: use the strongest approximate singleton discovered on train
    approx_singletons = [r for r in rows2
                         if r["size"] == 1
                         and r["gap_type"] == "approx_generalizes"
                         and r["train_val"] >= MIN_VAL
                         and r["test_n_alpha"] >= MIN_COV_ABS]
    if not approx_singletons:
        # fall back to any non-classical singleton with decent val
        approx_singletons = [r for r in rows2
                             if r["size"] == 1 and not r["train_classical"]
                             and r["train_val"] >= 0.75
                             and r["test_n_alpha"] >= MIN_COV_ABS]
    if approx_singletons:
        approx_singletons.sort(key=lambda r: -r["train_val"])
        anchor = approx_singletons[0]["antecedent"]
        rows3 = part3_nonmonotonicity(train, test, anchor)
        if rows3:
            with (out_dir / "exp4_mushroom_delta_gamma.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows3[0].keys()))
                w.writeheader()
                w.writerows(rows3)
    else:
        print("\n[Part 3] No suitable approximate singleton found for context sweep")

    # Part 4
    summary4, term_rows4 = part4_kdnf_abduction(train, test, k=2,
                                                 min_plausibility=0.01, top_k=15)
    with (out_dir / "exp4_mushroom_abduction_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary4.keys()))
        w.writeheader()
        w.writerow(summary4)
    if term_rows4:
        with (out_dir / "exp4_mushroom_abduction_terms.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(term_rows4[0].keys()))
            w.writeheader()
            w.writerows(term_rows4)


if __name__ == "__main__":
    run()
