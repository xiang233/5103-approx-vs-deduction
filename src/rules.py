"""Boolean rule language: literals, conjunctions, disjunctions, k-DNF, implications.

A `Predicate` is anything callable on a Boolean matrix `X` of shape (m, n) that
returns a length-m boolean vector. We keep the representation transparent so
that classical-validity checks and approximate-validity estimates evaluate the
exact same object.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Literal:
    var: int
    negated: bool = False

    def __call__(self, X: np.ndarray) -> np.ndarray:
        col = X[:, self.var].astype(bool)
        return ~col if self.negated else col

    def vars(self) -> set[int]:
        return {self.var}

    def __str__(self) -> str:
        return f"~x{self.var}" if self.negated else f"x{self.var}"


@dataclass(frozen=True)
class Conjunction:
    """Conjunction of literals. An empty conjunction is the constant True."""
    lits: tuple[Literal, ...] = field(default_factory=tuple)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if not self.lits:
            return np.ones(X.shape[0], dtype=bool)
        return np.logical_and.reduce([lit(X) for lit in self.lits])

    def vars(self) -> set[int]:
        return {lit.var for lit in self.lits}

    def with_extra(self, extra: "Conjunction") -> "Conjunction":
        return Conjunction(tuple(self.lits) + tuple(extra.lits))

    def __str__(self) -> str:
        return "T" if not self.lits else " & ".join(str(l) for l in self.lits)


@dataclass(frozen=True)
class Disjunction:
    """Disjunction of literals. An empty disjunction is the constant False."""
    lits: tuple[Literal, ...] = field(default_factory=tuple)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if not self.lits:
            return np.zeros(X.shape[0], dtype=bool)
        return np.logical_or.reduce([lit(X) for lit in self.lits])

    def vars(self) -> set[int]:
        return {lit.var for lit in self.lits}

    def __str__(self) -> str:
        return "F" if not self.lits else " | ".join(str(l) for l in self.lits)


@dataclass(frozen=True)
class KDNF:
    """k-DNF: disjunction of conjunctions of length <= k."""
    terms: tuple[Conjunction, ...]
    k: int

    def __call__(self, X: np.ndarray) -> np.ndarray:
        if not self.terms:
            return np.zeros(X.shape[0], dtype=bool)
        return np.logical_or.reduce([t(X) for t in self.terms])

    def __str__(self) -> str:
        return " | ".join(f"({t})" for t in self.terms)


@dataclass(frozen=True)
class Implication:
    """alpha -> beta. The classical and empirical evaluations live in
    `classical.py` and `validity.py`; here we just store the pair."""
    antecedent: Conjunction
    consequent: Literal | Conjunction


def all_literals(n: int, include_negations: bool = True) -> list[Literal]:
    if include_negations:
        return [Literal(i, neg) for i in range(n) for neg in (False, True)]
    return [Literal(i, False) for i in range(n)]


def enumerate_conjunctions(n: int, k: int, include_negations: bool = True) -> Iterable[Conjunction]:
    """All conjunctions of distinct variables of length 1..k."""
    from itertools import combinations, product
    lits_per_var = [(Literal(i, False), Literal(i, True))] if False else None
    for size in range(1, k + 1):
        for var_combo in combinations(range(n), size):
            if include_negations:
                for signs in product([False, True], repeat=size):
                    yield Conjunction(tuple(Literal(v, s) for v, s in zip(var_combo, signs)))
            else:
                yield Conjunction(tuple(Literal(v, False) for v in var_combo))
