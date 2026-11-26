from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Protocol, TypeVar

import numpy as np
from numpy.random import Generator, SeedSequence, default_rng
from numpy.typing import NDArray


Size = int | tuple[int, int]
SplitMode = Literal["random", "grouped", "temporal"]
Number = TypeVar("Number", int, float)


def _bounds(value: Number | tuple[Number, Number]) -> tuple[Number, Number]:
    if isinstance(value, tuple):
        assert len(value) == 2, "Use a scalar or two bounds"
        return value
    return value, value


def _size(value: Size, rng: Generator, *, log: bool = False) -> int:
    low, high = _bounds(value)
    if log and low < high:
        return min(high, max(low, int(np.exp(rng.uniform(np.log(low), np.log(high + 1))))))
    return int(rng.integers(low, high + 1))


@dataclass(frozen=True)
class Task:
    n_rows: int
    n_features: int
    n_classes: int = 0
    class_weights: tuple[float, ...] = ()
    binary_label_strength: float | None = None
    numeric_features_only: bool = False
    multiclass_label_strength: float | None = None
    split_mode: SplitMode = "random"
    context_rows: int | None = None

    def __post_init__(self) -> None:
        assert all(
            isinstance(value, Integral) and not isinstance(value, bool)
            for value in (self.n_rows, self.n_features, self.n_classes)
        ), "Task dimensions and class count must be integers"
        assert self.n_rows >= 2, "At least two rows are required"
        assert self.n_features >= 1, "At least one feature is required"
        assert self.n_classes == 0 or 2 <= self.n_classes <= self.n_rows, (
            "Use zero classes for regression or 2 to n_rows for classification"
        )
        if self.class_weights:
            assert len(self.class_weights) == self.n_classes, "Provide one weight per class"
            weights = np.asarray(self.class_weights)
            assert np.isfinite(weights.sum()) and (weights > 0).all(), "Invalid class weights"
        if self.binary_label_strength is not None:
            assert self.n_classes == 2, "Stochastic binary labels require exactly two classes"
            assert np.isfinite(self.binary_label_strength) and 0 <= self.binary_label_strength <= 20, (
                "Binary label strength must be in [0, 20]"
            )
        if self.multiclass_label_strength is not None:
            assert self.n_classes >= 3, "Stochastic multiclass labels require at least three classes"
            assert 0 <= self.multiclass_label_strength < 1, "Multiclass label strength must be in [0, 1)"
        assert self.split_mode in ("random", "grouped", "temporal"), "Invalid split mode"
        if self.context_rows is not None:
            assert isinstance(self.context_rows, Integral) and not isinstance(self.context_rows, bool), (
                "Context row count must be an integer"
            )
            assert max(1, self.n_classes) <= self.context_rows < self.n_rows, "Invalid context row count"
        assert self.split_mode == "random" or self.context_rows is not None, "Ordered splits require context_rows"


@dataclass(frozen=True)
class Dataset:
    X: NDArray[np.floating]
    y: NDArray[np.floating] | NDArray[np.integer]
    categorical: NDArray[np.bool_]
    class_probabilities: NDArray[np.floating] | None = None


class Prior(Protocol):
    def __call__(self, task: Task, rng: Generator, /) -> Dataset: ...


def _seed_sequence(seed: int, index: int) -> SeedSequence:
    for name, value in (("seed", seed), ("index", index)):
        assert isinstance(value, Integral) and not isinstance(value, bool) and value >= 0, (
            f"{name} must be a nonnegative integer"
        )
    return SeedSequence(int(seed), spawn_key=(int(index),))


def sample(prior: Prior, task: Task, *, seed: int, index: int = 0) -> Dataset:
    """Sample one dataset using the independent stream for this index."""
    return _sample(prior, task, default_rng(_seed_sequence(seed, index)))


def _sample(prior: Prior, task: Task, rng: Generator) -> Dataset:
    data = prior(task, rng)
    assert data.X.shape == (task.n_rows, task.n_features), "Unexpected feature shape"
    assert data.y.shape == (task.n_rows,), "Unexpected target shape"
    assert data.categorical.shape == (task.n_features,), "Unexpected category mask shape"
    assert data.categorical.dtype == bool, "Category mask must be boolean"
    assert not task.numeric_features_only or not data.categorical.any(), (
        "This task requires numeric features. Custom priors must respect Task.numeric_features_only."
    )
    assert np.issubdtype(data.X.dtype, np.floating), "Features must be floating point"
    assert not np.isinf(data.X).any(), "Features contain infinities"
    assert np.isfinite(data.y).all(), "Targets must be finite"

    categories = data.X[:, data.categorical]
    observed = categories[~np.isnan(categories)]
    assert ((observed >= 0) & (observed == np.floor(observed))).all(), (
        "Observed categorical values must be nonnegative integer codes"
    )
    if task.n_classes:
        assert np.issubdtype(data.y.dtype, np.integer), "Class labels must be integers"
        assert np.array_equal(np.unique(data.y), np.arange(task.n_classes)), (
            "Targets must contain every requested class"
        )
    else:
        assert np.issubdtype(data.y.dtype, np.floating), "Regression targets must be floating point"
    assert task.multiclass_label_strength is None or data.class_probabilities is not None, (
        "The prior must return class probabilities for stochastic multiclass tasks"
    )
    if data.class_probabilities is not None:
        probs = data.class_probabilities
        assert task.n_classes and probs.shape == (task.n_rows, task.n_classes), "Unexpected probability shape"
        assert (probs >= 0).all() and np.allclose(probs.sum(axis=1), 1), "Invalid class probabilities"
    return data


def _quantize(
    values: NDArray[np.floating], bins: int, weights: tuple[float, ...] = (),
    *, tie_break_rng: Generator | None = None,
) -> NDArray[np.int64]:
    """Split at empirical quantiles, breaking ties only when needed and enabled."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    if counts.size < bins and tie_break_rng:
        order = np.lexsort((tie_break_rng.permutation(values.size), inverse))
        inverse = np.argsort(order)
        counts = np.ones(values.size, dtype=np.int64)
    assert counts.size >= bins, f"Need {bins} distinct values, found {counts.size}"
    cumulative = counts.cumsum()
    fractions = np.arange(1, bins) / bins
    if weights:
        fractions = np.cumsum(weights)[:-1] / np.sum(weights)
    cuts = []
    lower = 1
    for i, fraction in enumerate(fractions, start=1):
        cut = int(np.searchsorted(cumulative, fraction * values.size) + 1)
        cut = min(max(cut, lower), counts.size - bins + i)
        cuts.append(cut)
        lower = cut + 1
    return np.searchsorted(cuts, inverse, side="right")


def _target(
    score: NDArray[np.float64], task: Task, rng: Generator
) -> NDArray[np.float64] | NDArray[np.int64]:
    assert np.isfinite(score).all(), "Prior produced nonfinite scores"
    if task.n_classes == 0:
        return score
    if task.binary_label_strength is not None:
        return _binary_target(score, task, task.binary_label_strength, rng)
    classes = rng.permutation(task.n_classes)
    weights = tuple(task.class_weights[j] for j in classes) if task.class_weights else ()
    return classes[_quantize(score, task.n_classes, weights, tie_break_rng=rng)]


def _binary_target(
    score: NDArray[np.float64], task: Task, strength: float, rng: Generator
) -> NDArray[np.int64]:
    prevalence = task.class_weights[1] / sum(task.class_weights) if task.class_weights else 0.5
    floor = 0.5 / score.size
    prevalence = float(np.clip(prevalence, floor, 1 - floor))
    spread = score.std()
    if strength == 0 or spread == 0:
        probability = np.full(score.shape, prevalence)
    else:
        direction = 1 if rng.integers(2) else -1
        logits = (score - score.mean()) / spread * (strength * direction)
        low, high = -40 - logits.max(), 40 - logits.min()
        for _ in range(24):
            intercept = (low + high) / 2
            probability = 1 / (1 + np.exp(-np.clip(intercept + logits, -40, 40)))
            if probability.mean() < prevalence:
                low = intercept
            else:
                high = intercept
    for _ in range(64):
        labels = rng.random(score.size) < probability
        if labels.any() and not labels.all():
            return labels.astype(np.int64)
    raise AssertionError("Unable to draw both classes with the requested binary label law")


def _multiclass_target(
    score: NDArray[np.float64], task: Task, strength: float, rng: Generator,
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    assert np.isfinite(score).all(), "Prior produced nonfinite scores"
    classes = rng.permutation(task.n_classes)
    weights = np.asarray(task.class_weights) if task.class_weights else np.ones(task.n_classes)
    prevalence = np.maximum(weights / weights.sum(), np.finfo(float).tiny)
    prevalence /= prevalence.sum()

    # Intersect each tied score group's quantile interval with the class intervals.
    _, inverse, group_counts = np.unique(score, return_inverse=True, return_counts=True)
    row_edges = np.r_[0, group_counts.cumsum() / score.size]
    class_edges = np.r_[0, np.minimum(prevalence[classes].cumsum(), 1)]
    class_edges[-1] = 1
    overlap = np.minimum(row_edges[1:, None], class_edges[None, 1:])
    overlap -= np.maximum(row_edges[:-1, None], class_edges[None, :-1])
    probs = np.empty((score.size, task.n_classes))
    probs[:, classes] = np.maximum(overlap, 0)[inverse] / (group_counts[inverse, None] / score.size)
    probs *= strength
    probs += (1 - strength) * prevalence
    probs /= probs.sum(axis=1, keepdims=True)

    thresholds = probs.cumsum(axis=1)[:, :-1]
    labels = (rng.random((score.size, 1)) >= thresholds).sum(axis=1, dtype=np.int64)
    context = labels[:task.context_rows]
    context_counts = np.bincount(context, minlength=task.n_classes)
    # Supply missing context classes without conditioning any query draws.
    for label in rng.permutation(np.flatnonzero(context_counts == 0)):
        candidates = np.flatnonzero(context_counts[context] > 1)
        row_weights = np.maximum(probs[candidates, label], np.finfo(float).tiny)
        row = rng.choice(candidates, p=row_weights / row_weights.sum())
        context_counts[context[row]] -= 1
        context[row] = label
        context_counts[label] += 1
    return labels, probs


def _readout(
    X: NDArray[np.float64], score: NDArray[np.float64], task: Task, rng: Generator,
    *, categorical: NDArray[np.bool_] | None = None,
) -> Dataset:
    if categorical is None:
        categorical = np.zeros(X.shape[1], dtype=bool)
    strength = task.multiclass_label_strength
    if strength is not None:
        labels, probs = _multiclass_target(score, task, strength, rng)
        return Dataset(X, labels, categorical, probs)
    return Dataset(X, _target(score, task, rng), categorical)
