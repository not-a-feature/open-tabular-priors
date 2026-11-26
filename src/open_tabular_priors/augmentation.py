from dataclasses import dataclass, replace

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .core import Dataset, Prior, Task, _quantize, _readout
from .graph import _mechanism, _standardize


def _correlate_blocks(X: NDArray[np.float64], categorical: NDArray[np.bool_], rng: Generator) -> None:
    rows, width = X.shape
    selected = rng.choice(width, rng.integers(2, width + 1), replace=False)
    for block in np.array_split(selected, rng.integers(1, min(4, selected.size // 2) + 1)):
        factor = rng.standard_normal(rows)
        strength = rng.uniform(0.2, 0.7)
        numeric = block[~categorical[block]]
        X[:, numeric] = (1 - strength) * X[:, numeric] + strength * factor[:, None]
        order = np.argsort(factor)
        for j in block[categorical[block]]:
            X[order, j] = np.sort(X[:, j])


def _coarsen(X: NDArray[np.float64], candidates: NDArray[np.intp], rng: Generator) -> None:
    for j in rng.permutation(candidates)[: max(1, len(candidates) // 3)]:
        count = min(np.unique(X[:, j]).size, rng.integers(2, 9))
        if count >= 2:
            X[:, j] = _quantize(X[:, j], count)


def _recode(X: NDArray[np.float64], categorical: NDArray[np.bool_], rng: Generator) -> None:
    numerical = np.flatnonzero(~categorical)
    for j in rng.permutation(numerical)[: rng.integers(1, 4)]:
        cap = min(np.unique(X[:, j]).size, 512)
        if cap >= 201:
            count = rng.integers(201, cap + 1)
            weights = rng.permutation(np.arange(1.0, count + 1) ** -rng.uniform(0.5, 1.5))
            codes = _quantize(X[:, j], count, tuple(weights))
            X[:, j] = rng.permutation(count)[codes]
            categorical[j] = True


@dataclass(frozen=True)
class AugmentedPrior:
    """Apply configurable table-level transforms to a complete-data prior."""

    base: Prior
    block_probability: float = 0.0
    relabel_probability: float = 0.0
    coarsen_probability: float = 0.0
    recode_probability: float = 0.0
    heavy_tail_probability: float = 0.0
    structured_mask_probability: float = 0.0
    cell_mask_probability: float = 0.0
    near_empty_mask_probability: float = 0.0
    latent_mask_probability: float = 0.0
    shuffle_columns: bool = False

    def __post_init__(self) -> None:
        assert callable(self.base), "The base prior must be callable"
        for name, probability in (
            ("block_probability", self.block_probability),
            ("relabel_probability", self.relabel_probability),
            ("coarsen_probability", self.coarsen_probability),
            ("recode_probability", self.recode_probability),
            ("heavy_tail_probability", self.heavy_tail_probability),
            ("structured_mask_probability", self.structured_mask_probability),
            ("cell_mask_probability", self.cell_mask_probability),
            ("near_empty_mask_probability", self.near_empty_mask_probability),
            ("latent_mask_probability", self.latent_mask_probability),
        ):
            assert 0 <= probability <= 1, f"{name} must be in [0, 1]"

    def __call__(self, task: Task, rng: Generator) -> Dataset:
        data = self.base(task, rng)
        assert np.isfinite(data.X).all(), "Apply table augmentations before missingness"
        X, y, categorical = data.X.copy(), data.y, data.categorical.copy()
        if task.n_features > 1 and self.block_probability and rng.random() < self.block_probability:
            _correlate_blocks(X, categorical, rng)

        if self.relabel_probability and rng.random() < self.relabel_probability:
            columns = rng.choice(task.n_features, rng.integers(1, min(4, task.n_features) + 1), replace=False)
            kind = ("linear", "neural", "fourier")[rng.choice(3, p=(0.5, 0.3, 0.2))]
            score = _mechanism(_standardize(X[:, columns]), 1, kind, rng).ravel()
            if np.unique(score).size >= max(2, task.n_classes):
                data = _readout(X, score, task, rng, categorical=categorical)
                y = data.y
        if self.coarsen_probability and rng.random() < self.coarsen_probability:
            _coarsen(X, np.flatnonzero(~categorical), rng)
        if (not task.numeric_features_only and task.n_rows >= 201
                and self.recode_probability and rng.random() < self.recode_probability):
            _recode(X, categorical, rng)
        if task.n_classes == 0 and self.heavy_tail_probability and rng.random() < self.heavy_tail_probability:
            scaled = _standardize(y)
            positive, negative = np.exp(rng.uniform(-1.5, 1.5, 2))
            power = np.where(scaled > 0, positive, negative)
            y = np.clip(np.sign(scaled) * np.abs(scaled) ** power, -100, 100)

        if self.shuffle_columns:
            order = rng.permutation(task.n_features)
            X, categorical = X[:, order], categorical[order]
        if (self.structured_mask_probability or self.cell_mask_probability
                or self.near_empty_mask_probability or self.latent_mask_probability):
            self._mask_features(X, rng)
        return replace(data, X=X, y=y, categorical=categorical)


    def _mask_features(self, X: NDArray[np.float64], rng: Generator) -> None:
        rows, width = X.shape
        mask = np.zeros(X.shape, dtype=bool)
        if self.near_empty_mask_probability and rng.random() < self.near_empty_mask_probability:
            columns = rng.permutation(width)[: max(1, int(width * rng.uniform(0.5, 0.8)))]
            for family in np.array_split(columns, rng.integers(1, min(4, columns.size) + 1)):
                observed = rng.choice(rows, max(1, int(rows * rng.uniform(0.01, 0.10))), replace=False)
                mask[:, family] = True
                mask[np.ix_(observed, family)] = False
        elif self.structured_mask_probability and rng.random() < self.structured_mask_probability:
            columns = rng.permutation(width)[: rng.integers(1, width + 1)]
            if rng.random() < 0.5:
                affected = rng.random(rows) < rng.uniform(0.05, 0.30)
                mask[np.ix_(affected, columns)] = True
            else:
                tails = np.quantile(X[:, columns], rng.uniform(0.7, 0.95), axis=0)
                mask[:, columns] = X[:, columns] > tails
        if self.cell_mask_probability and rng.random() < self.cell_mask_probability:
            mask |= rng.random(X.shape) < rng.uniform(0.01, 0.08)
        if width > 1 and self.latent_mask_probability and rng.random() < self.latent_mask_probability:
            order = rng.permutation(width)
            n_drivers = rng.integers(1, min(3, width - 1) + 1)
            drivers = _standardize(X[:, order[:n_drivers]]) @ rng.normal(size=n_drivers)
            drivers = _standardize(drivers) + rng.normal(0, 0.25, rows)
            high = drivers > np.quantile(drivers, rng.uniform(0.55, 0.85))
            rate = np.where(high, rng.uniform(0.30, 0.90), rng.uniform(0, 0.10))
            columns = order[n_drivers:]
            mask[:, columns] |= rng.random((rows, columns.size)) < rate[:, None]
        X[mask] = np.nan
