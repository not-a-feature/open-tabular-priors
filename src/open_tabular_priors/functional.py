from dataclasses import dataclass, replace
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .core import Dataset, Size, Task, _bounds, _readout, _size
from .generators import _roots, _tree
from .graph import _standardize


TargetMechanism = Literal["linear", "additive", "tree", "interaction", "local", "gated"]
FeatureGeometry = Literal["independent", "blocks", "heavy_tail", "subgroups"]
_TARGETS: tuple[TargetMechanism, ...] = ("linear", "additive", "tree", "interaction", "local", "gated")
_GEOMETRIES: tuple[FeatureGeometry, ...] = ("independent", "blocks", "heavy_tail", "subgroups")


def _observed_features(
    task: Task, geometry: FeatureGeometry, rng: Generator
) -> NDArray[np.float64]:
    n, d = task.n_rows, task.n_features
    if geometry == "heavy_tail":
        return rng.standard_t(rng.uniform(2.5, 8), (n, d))
    if geometry == "blocks":
        k = min(d, int(rng.integers(2, 9)))
        factors = rng.standard_normal((n, k))
        group = rng.integers(k, size=d)
        loading = rng.uniform(0.5, 1.5, d) * rng.choice((-1, 1), d)
        return factors[:, group] * loading + rng.standard_normal((n, d)) * rng.uniform(0.15, 0.65)
    X = _roots(n, d, rng)
    if geometry == "subgroups":
        groups = int(rng.integers(2, 6))
        offsets = rng.normal(0, rng.uniform(0.5, 2), (groups, d))
        X += offsets[rng.integers(groups, size=n)]
    return X


def _response(
    X: NDArray[np.float64], mechanism: TargetMechanism, rng: Generator, tree_count: Size,
) -> NDArray[np.float64]:
    n, d = X.shape
    weights = rng.normal(0, d**-0.5, d)
    if mechanism == "linear":
        return X @ weights
    if mechanism == "additive":
        values = np.stack((np.tanh(X), np.sin(2 * X), np.maximum(0, X)), axis=-1)
        return values[:, np.arange(d), rng.integers(3, size=d)] @ weights
    if mechanism == "tree":
        score = _tree(X, int(rng.integers(2, 5)), rng)
        low, high = _bounds(tree_count)
        if high > 1:
            ensemble_rng, = rng.spawn(1)
            count = low if low == high else _size(tree_count, ensemble_rng)
            for _ in range(count - 1):
                score += _tree(X, int(ensemble_rng.integers(2, 5)), ensemble_rng)
        return score
    if mechanism == "interaction":
        order = min(d, int(rng.integers(2, 5)))
        return np.prod(np.where(X[:, :order] > rng.uniform(-0.3, 0.3, order), 1.0, -1.0), axis=1)
    if mechanism == "local":
        centers = X[rng.choice(n, min(n, 6), replace=False)]
        distance = np.square(X[:, None, :] - centers).mean(axis=-1)
        return np.exp(-distance / rng.uniform(0.15, 1.5)) @ rng.standard_normal(len(centers))
    assert mechanism == "gated", f"Unknown functional target mechanism {mechanism!r}"
    boundary = X[:, 0] > rng.uniform(-0.5, 0.5)
    return np.where(boundary, X @ weights, X @ rng.normal(0, d**-0.5, d))


@dataclass(frozen=True)
class FunctionalSCMPrior:
    """Sample observed causes, nuisance features, and a task-specific target mechanism."""

    mechanisms: tuple[TargetMechanism, ...] = _TARGETS
    geometries: tuple[FeatureGeometry, ...] = _GEOMETRIES
    active_features: tuple[int, int] = (1, 12)
    noise: tuple[float, float] = (0.03, 0.8)
    tree_count: Size = 1
    distractor_weights: tuple[float, float, float] | None = None
    null_block_size: int = 8
    null_correlation: float = 0.8
    proxy_noise: float = 0.3
    shuffle_columns: bool = True
    covariate_shift: float = 0.0
    target_drift: float = 0.0

    def __post_init__(self) -> None:
        assert isinstance(self.mechanisms, tuple) and self.mechanisms and set(self.mechanisms) <= set(_TARGETS), (
            "Provide a nonempty tuple of target mechanisms"
        )
        assert isinstance(self.geometries, tuple) and self.geometries and set(self.geometries) <= set(_GEOMETRIES), (
            "Provide a nonempty tuple of feature geometries"
        )
        low, high = self.active_features
        assert all(isinstance(n, Integral) and not isinstance(n, bool) for n in (low, high)), (
            "Active feature counts must be integers"
        )
        assert 1 <= low <= high, "Invalid active feature bounds"
        low, high = self.noise
        assert 0 < low <= high and np.isfinite(high), "Noise must have finite positive bounds"
        low, high = _bounds(self.tree_count)
        assert all(isinstance(n, Integral) and not isinstance(n, bool) for n in (low, high)), (
            "Tree counts must be integers"
        )
        assert 1 <= low <= high, "Invalid tree count bounds"
        if self.distractor_weights is not None:
            weights = self.distractor_weights
            assert len(weights) == 3 and min(weights) >= 0 and 0 < sum(weights) < np.inf, (
                "Provide three nonnegative distractor weights with a finite positive sum"
            )
        assert isinstance(self.null_block_size, Integral) and self.null_block_size >= 2, (
            "Null block size must be an integer of at least two"
        )
        assert 0 <= self.null_correlation <= 1, "Null correlation must be in [0, 1]"
        for scale in (self.proxy_noise, self.covariate_shift, self.target_drift):
            assert np.isfinite(scale) and scale >= 0, "Noise and shift scales must be finite and nonnegative"

    def __call__(self, task: Task, rng: Generator) -> Dataset:
        assert task.split_mode != "random" or (self.covariate_shift == self.target_drift == 0), (
            "Covariate shift and target drift require a grouped or temporal split"
        )
        geometry = self.geometries[rng.integers(len(self.geometries))]
        if self.distractor_weights is None:
            X = _observed_features(task, geometry, rng)
        mechanism = self.mechanisms[rng.integers(len(self.mechanisms))]
        low, high = (min(task.n_features, size) for size in self.active_features)
        if mechanism == "interaction" and high >= 2:
            low = max(low, 2)
        count = int(rng.integers(low, high + 1))
        if self.distractor_weights is not None:
            X = _observed_features(replace(task, n_features=count), geometry, rng)
            distractor_rng, = rng.spawn(1)
            active = slice(None)
        else:
            active = rng.choice(task.n_features, size=count, replace=False)

        if task.split_mode != "random":
            if task.split_mode == "grouped":
                position = (np.arange(task.n_rows) >= task.context_rows).astype(float) - 0.5
            else:
                position = np.linspace(-0.5, 0.5, task.n_rows)
            X[:, active] += (self.covariate_shift * position[:, None]) * rng.standard_normal(count)
        values = _standardize(X[:, active])
        score = _standardize(_response(values, mechanism, rng, self.tree_count))
        if task.split_mode != "random":
            score += (self.target_drift * position) * (values @ rng.normal(0, count**-0.5, count))
        scale = np.exp(rng.uniform(*np.log(self.noise)))
        score += rng.normal(0, scale, task.n_rows)
        data = _readout(X, score, task, rng)
        if self.distractor_weights is None:
            return data
        X = self._add_distractors(X, task.n_features, self.distractor_weights, distractor_rng)
        return replace(data, X=X, categorical=np.zeros(task.n_features, dtype=bool))

    def _add_distractors(
        self, active: NDArray[np.float64], width: int, weights: tuple[float, float, float], rng: Generator,
    ) -> NDArray[np.float64]:
        rows, count = active.shape
        X = np.empty((rows, width))
        X[:, :count] = active
        independent_end, block_end = count + np.rint(
            (width - count) * np.cumsum(np.asarray(weights, dtype=float) / sum(weights))[:-1]
        ).astype(int)
        X[:, count:independent_end] = rng.standard_normal((rows, independent_end - count))
        strength = np.sqrt(self.null_correlation)
        noise = np.sqrt(1 - self.null_correlation)
        for start in range(independent_end, block_end, self.null_block_size):
            stop = min(start + self.null_block_size, block_end)
            factor = rng.standard_normal((rows, 1))
            X[:, start:stop] = strength * factor + noise * rng.standard_normal((rows, stop - start))
        if block_end < width:
            donors = rng.integers(count, size=width - block_end)
            for start in range(block_end, width, 256):
                stop = min(start + 256, width)
                columns = active[:, donors[start - block_end:stop - block_end]]
                X[:, start:stop] = columns + self.proxy_noise * columns.std(axis=0) * rng.standard_normal(columns.shape)
        return X[:, rng.permutation(width)] if self.shuffle_columns else X
