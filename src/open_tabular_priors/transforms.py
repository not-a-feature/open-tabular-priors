from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .core import Dataset, Prior, Task, _quantize

WideNumericalMode = Literal["dense", "binary", "zero_inflated"]


@dataclass(frozen=True)
class Observation:
    """Add noisy proxies, categorical codes and missing features to a sampled table."""

    base: Prior
    categorical_fraction: float = 0.0
    max_categories: int = 10
    missing_rate: float = 0.0
    missing_pattern: Literal["mcar", "row"] = "mcar"
    proxy_fraction: float = 0.0
    proxy_noise: float = 0.2
    measurement_retention: float | tuple[float, float] = 1.0

    def __post_init__(self) -> None:
        assert 0 <= self.categorical_fraction <= 1, "Invalid categorical fraction"
        assert self.max_categories >= 2, "At least two categories are required"
        assert 0 <= self.missing_rate <= 1, "Invalid missing rate"
        assert self.missing_pattern in ("mcar", "row"), "Unknown missingness pattern"
        assert 0 <= self.proxy_fraction < 1, "Proxy fraction must be in [0, 1)"
        assert np.isfinite(self.proxy_noise) and self.proxy_noise >= 0, "Invalid proxy noise"
        retention = self.measurement_retention
        if isinstance(retention, tuple):
            low, high = retention
        else:
            low = high = retention
        assert 0 <= low <= high <= 1, "Invalid measurement retention"

    def __call__(self, task: Task, rng: Generator) -> Dataset:
        data = self.base(task, rng)
        X = data.X.copy()
        categorical = data.categorical.copy()
        numerical = np.flatnonzero(~categorical)
        n_proxies = int(self.proxy_fraction * numerical.size)
        if n_proxies:
            shuffled = rng.permutation(numerical)
            proxies = shuffled[:n_proxies]
            donors = rng.choice(shuffled[n_proxies:], n_proxies)
            measurements = X[:, donors]
            assert np.isfinite(measurements).all(), "Proxy donors must be finite"
            noise = rng.standard_normal(measurements.shape) * measurements.std(axis=0)
            X[:, proxies] = measurements + self.proxy_noise * noise

        retention = self.measurement_retention
        if isinstance(retention, tuple):
            low, high = retention
            retention = low if low == high else rng.uniform(low, high)
        if retention < 1:
            for start in range(0, numerical.size, 256):
                columns = numerical[start:start + 256]
                values = X[:, columns]
                assert np.isfinite(values).all(), "Apply measurement noise before missingness"
                center = values.mean(axis=0)
                scale = values.std(axis=0) * np.sqrt(1 - retention**2)
                X[:, columns] = center + retention * (values - center) + scale * rng.standard_normal(values.shape)

        n_categories = 0 if task.numeric_features_only else int(self.categorical_fraction * numerical.size)
        for j in rng.choice(numerical, n_categories, replace=False):
            observed = ~np.isnan(X[:, j])
            values = X[observed, j]
            cardinality = min(self.max_categories, np.unique(values).size)
            if cardinality >= 2:
                bins = rng.integers(2, cardinality + 1)
                X[observed, j] = rng.permutation(bins)[_quantize(values, bins)]
                categorical[j] = True

        if self.missing_rate:
            rate = self.missing_rate
            if self.missing_pattern == "row":
                spread = min(rate, 1 - rate)
                rate = rng.uniform(rate - spread, rate + spread, (task.n_rows, 1))
            X[rng.random(X.shape) < rate] = np.nan
        return replace(data, X=X, categorical=categorical)


def _categorical_column(
    X: NDArray[np.floating], density: float, cap: int, rng: Generator
) -> NDArray[np.int64]:
    n_rows, n_features = X.shape
    donors = rng.choice(n_features, max(1, int(density * n_features)), replace=False)
    _, codes, counts = np.unique(
        X[np.arange(n_rows), rng.choice(donors, n_rows)],
        return_inverse=True,
        return_counts=True,
    )
    if counts.size <= cap:
        return codes
    common = np.argsort(-counts, kind="stable")[:cap]
    mapping = np.full(counts.size, -1)
    mapping[common] = np.arange(cap)
    merged = mapping[codes]
    rare = merged < 0
    merged[rare] = rng.choice(cap, rare.sum(), p=counts[common] / counts[common].sum())
    return merged


@dataclass(frozen=True)
class WidePrior:
    """Widen a narrow prior's features while preserving its target."""

    base: Prior
    base_features: int = 16
    density: float = 0.05
    noise: float = 1.0
    include_original: bool = True
    max_categories: int = 20
    numerical_mode: WideNumericalMode = "dense"
    zero_fraction: float = 0.85

    def __post_init__(self) -> None:
        assert self.base_features >= 1, "At least one donor column is required"
        assert 0 <= self.density <= 1, "Projection density must be in [0, 1]"
        assert np.isfinite(self.noise) and self.noise >= 0, "Invalid projection noise"
        assert self.max_categories >= 2, "At least two categories are required"
        assert self.numerical_mode in ("dense", "binary", "zero_inflated"), "Invalid numerical widening mode"
        assert 0 < self.zero_fraction < 1, "zero_fraction must be inside (0, 1)"

    def __call__(self, task: Task, rng: Generator) -> Dataset:
        n_original = self.base_features if self.include_original else 0
        assert task.n_features >= self.base_features, "Requested width is smaller than the donor width"
        n_added = task.n_features - n_original
        data = self.base(replace(task, n_features=self.base_features), rng)
        if n_added == 0:
            return data
        assert np.isfinite(data.X).all(), "Add missingness after widening"

        n_cat = n_added * np.count_nonzero(data.categorical) // self.base_features
        n_num = n_added - n_cat
        X = np.empty((task.n_rows, task.n_features))
        categorical = np.zeros(task.n_features, dtype=bool)
        X[:, :n_original] = data.X[:, :n_original]
        categorical[:n_original] = data.categorical[:n_original]

        numerical = data.X[:, ~data.categorical]
        if n_num:
            for start in range(n_original, n_original + n_num, 256):
                stop = min(start + 256, n_original + n_num)
                shape = (numerical.shape[1], stop - start)
                weights = rng.standard_normal(shape) * (rng.random(shape) < self.density)
                projected = numerical @ weights
                scale = projected.std(axis=0)
                scale[scale == 0] = 1
                projected += rng.standard_normal(projected.shape) * (self.noise * scale)
                if self.numerical_mode == "binary":
                    projected = projected > np.quantile(projected, self.zero_fraction, axis=0)
                elif self.numerical_mode == "zero_inflated":
                    cutoff = np.quantile(projected, self.zero_fraction, axis=0)
                    projected = np.log1p(np.maximum(projected - cutoff, 0))
                X[:, start:stop] = projected

        donors = data.X[:, data.categorical]
        for j in range(n_original + n_num, task.n_features):
            X[:, j] = _categorical_column(donors, self.density, self.max_categories, rng)
            categorical[j] = True

        order = rng.permutation(task.n_features)
        return replace(data, X=X[:, order], categorical=categorical[order])
