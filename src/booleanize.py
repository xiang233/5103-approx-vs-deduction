"""Convert the UCI mushroom dataset into a Boolean predicate matrix.

Each categorical attribute v with possible values {a, b, c, ...} becomes a
group of one-hot Boolean predicates ("v=a", "v=b", ...). Missing values in
stalk-root (encoded as "?") are treated as their own predicate "stalk-root=?".

The 22 raw attributes (after dropping veil-type, which is constant in this
release of the dataset) yield ~117 Boolean predicates. The class column
becomes a single predicate `poisonous`.

Returns a NamedColumns object so experiments can refer to predicates by name
("odor=n") rather than column index.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


ATTRIBUTE_NAMES = [
    "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root",
    "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring",
    "veil-type", "veil-color", "ring-number", "ring-type",
    "spore-print-color", "population", "habitat",
]


@dataclass(frozen=True)
class BooleanizedDataset:
    X: np.ndarray              # (m, n_predicates) bool, one-hot encoded features
    y: np.ndarray              # (m,) bool, True = poisonous
    feature_names: list[str]   # length n_predicates
    name_to_col: dict[str, int]


def load_mushroom(path: str | Path = "mushroom/agaricus-lepiota.data",
                  drop_constant: bool = True) -> BooleanizedDataset:
    rows: list[list[str]] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(line.split(","))
    arr = np.array(rows)  # shape (m, 23): col 0 = class, cols 1..22 = attrs
    y = (arr[:, 0] == "p")

    feature_names: list[str] = []
    cols: list[np.ndarray] = []
    for i, attr in enumerate(ATTRIBUTE_NAMES):
        col_vals = arr[:, i + 1]
        unique = sorted(set(col_vals))
        if drop_constant and len(unique) <= 1:
            continue
        for val in unique:
            feature_names.append(f"{attr}={val}")
            cols.append(col_vals == val)
    X = np.column_stack(cols).astype(bool)
    name_to_col = {n: i for i, n in enumerate(feature_names)}
    return BooleanizedDataset(X=X, y=y, feature_names=feature_names,
                              name_to_col=name_to_col)


def predicate_mask(ds: BooleanizedDataset, name: str) -> np.ndarray:
    """Convenience: 1-D mask for a named predicate."""
    return ds.X[:, ds.name_to_col[name]]
