from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from .core import Prior, Size, SplitMode, Task, _bounds, _sample, _seed_sequence, _size
from .generators import MLPPrior
from .graph import GraphSCMPrior
from .transforms import Observation, WideNumericalMode, WidePrior

Family = Literal["scm", "neural"]
FeatureType = Literal["continuous", "categorical", "mixed"]
WidthMode = Literal["direct", "wide"]
Rate = float | tuple[float, float]


@dataclass(frozen=True)
class TrainingConfig:
    """Independent choices for synthetic dataset size, feature schema and generating family."""

    n_rows: Size = (10, 1_000)
    n_features: Size = (10, 10_000)
    n_classes: Size = (2, 5)
    families: tuple[Family, ...] = ("scm", "neural")
    feature_types: tuple[FeatureType, ...] = ("continuous", "categorical", "mixed")
    width_modes: tuple[WidthMode, ...] = ("direct", "wide")
    size_sampling: Literal["log_uniform", "uniform"] = "log_uniform"
    context_fraction: Rate = (0.5, 0.8)
    categorical_fraction: Rate = (0.2, 0.8)
    max_categories: int = 10
    missing_rate: Rate = 0.0
    missing_pattern: Literal["mcar", "row"] = "mcar"
    wide_base_features: int = 32
    wide_density: float = 0.05
    wide_noise: float = 1.0
    wide_include_original: bool = True
    scm: Prior = GraphSCMPrior(categorical_fraction=(0.0, 0.0), measurement_probability=0.0)
    neural: Prior = MLPPrior(max_active_features=64)
    class_logit_scale: float | tuple[float, float] = 0.0
    binary_label_strength: float | tuple[float, float] | None = None
    multiclass_label_strength: Rate | None = None
    split_mode: SplitMode = "random"
    wide_numerical_mode: WideNumericalMode = "dense"
    wide_zero_fraction: float = 0.85

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("n_rows", self.n_rows, 2),
            ("n_features", self.n_features, 1),
            ("n_classes", self.n_classes, 0),
        ):
            low, high = _bounds(value)
            assert all(isinstance(x, Integral) and not isinstance(x, bool) for x in (low, high)), (
                f"{name} requires integer bounds"
            )
            assert minimum <= low <= high, f"Invalid {name} bounds"
        assert isinstance(self.max_categories, Integral) and self.max_categories >= 2, (
            "max_categories must be an integer of at least 2"
        )
        assert isinstance(self.wide_base_features, Integral) and self.wide_base_features >= 1, (
            "wide_base_features must be a positive integer"
        )
        class_low, class_high = _bounds(self.n_classes)
        assert class_high == 0 or class_low >= 2, "Use zero classes for regression or at least two for classification"
        scale_low, scale_high = _bounds(self.class_logit_scale)
        assert 0 <= scale_low <= scale_high and np.isfinite(scale_high), "Invalid class logit scale"
        if self.binary_label_strength is not None:
            assert class_low == class_high == 2, "Stochastic binary labels require n_classes=2"
            low, high = _bounds(self.binary_label_strength)
            assert 0 <= low <= high <= 20, "Invalid binary label strength bounds"
        if self.multiclass_label_strength is not None:
            assert class_low >= 3, "Stochastic multiclass labels require n_classes >= 3"
            low, high = _bounds(self.multiclass_label_strength)
            assert 0 <= low <= high < 1, "Multiclass label strength must be in [0, 1)"
        assert self.split_mode in ("random", "grouped", "temporal"), "Invalid split mode"
        for name, value in (
            ("context_fraction", self.context_fraction),
            ("categorical_fraction", self.categorical_fraction),
            ("missing_rate", self.missing_rate),
        ):
            low, high = _bounds(value)
            assert 0 <= low <= high <= 1, f"Invalid {name} bounds"
        lower_context, upper_context = _bounds(self.context_fraction)
        assert 0 < lower_context <= upper_context < 1, "context_fraction must be inside (0, 1)"
        min_context = max(1, int(_bounds(self.n_rows)[0] * lower_context))
        assert class_low <= min_context, "The smallest context must fit the minimum number of classes"
        assert self.families and set(self.families) <= {"scm", "neural"}, "Invalid families"
        assert self.feature_types and set(self.feature_types) <= {"continuous", "categorical", "mixed"}, (
            "Invalid feature types"
        )
        assert self.width_modes and set(self.width_modes) <= {"direct", "wide"}, "Invalid width modes"
        min_features = _bounds(self.n_features)[0]
        assert "wide" not in self.width_modes or min_features >= 2, "Wide tasks need at least two features"
        if "mixed" in self.feature_types:
            if "wide" in self.width_modes:
                min_features = min(min_features // 2, self.wide_base_features)
            assert min_features >= 2, "Mixed tasks need at least two donor features"
        assert callable(self.scm) and callable(self.neural), "Training priors must be callable"
        assert self.size_sampling in ("log_uniform", "uniform"), "Invalid size sampling"
        assert self.missing_pattern in ("mcar", "row"), "Invalid missingness pattern"
        assert 0 <= self.wide_density <= 1 and np.isfinite(self.wide_noise) and self.wide_noise >= 0, (
            "Invalid widening parameters"
        )
        assert self.wide_numerical_mode in ("dense", "binary", "zero_inflated"), "Invalid numerical widening mode"
        assert 0 < self.wide_zero_fraction < 1, "wide_zero_fraction must be inside (0, 1)"


@dataclass(frozen=True)
class TrainingMixture:
    """Select a complete episode recipe by relative weight."""

    recipes: tuple[TrainingConfig, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        assert len(self.recipes) == len(self.weights) > 0, "Provide one weight per recipe"
        assert all(isinstance(recipe, TrainingConfig) for recipe in self.recipes), "Expected training configs"
        assert np.isfinite(self.weights).all() and min(self.weights) >= 0, "Invalid recipe weights"
        assert np.isfinite(sum(self.weights)) and sum(self.weights) > 0, "Invalid recipe weight sum"

    def select_index(self, *, seed: int, index: int = 0) -> int:
        """Choose a recipe independently of the random draws used by that recipe."""
        rng = default_rng(_seed_sequence(seed, index).spawn(4)[3])
        return int(rng.choice(len(self.recipes), p=np.asarray(self.weights) / sum(self.weights)))


@dataclass(frozen=True)
class Episode:
    X_context: NDArray[np.floating]
    y_context: NDArray[np.floating] | NDArray[np.integer]
    X_query: NDArray[np.floating]
    y_query: NDArray[np.floating] | NDArray[np.integer]
    categorical: NDArray[np.bool_]
    task: Task
    family: Family
    feature_type: FeatureType
    width_mode: WidthMode
    missing_rate: float
    query_probabilities: NDArray[np.floating] | None = None


def _context_first(
    y: NDArray, n_context: int, n_classes: int, rng: Generator, *, keep_random_split: bool = False
) -> NDArray[np.intp]:
    order = rng.permutation(len(y))
    if n_classes == 0:
        return order
    if keep_random_split and np.unique(y[order[:n_context]]).size == n_classes:
        return order
    _, first = np.unique(y[order], return_index=True)
    reserved = np.zeros(len(y), dtype=bool)
    reserved[first] = True
    remaining = order[~reserved]
    extra = n_context - n_classes
    context = rng.permutation(np.concatenate((order[first], remaining[:extra])))
    return np.concatenate((context, remaining[extra:]))


def sample_episode(config: TrainingConfig | TrainingMixture, *, seed: int, index: int = 0) -> Episode:
    """Sample a table and a context/query split from the stream for this index."""
    if isinstance(config, TrainingMixture):
        config = config.recipes[config.select_index(seed=seed, index=index)]
    spec_rng, data_rng, split_rng = map(default_rng, _seed_sequence(seed, index).spawn(3))
    log = config.size_sampling == "log_uniform"
    rows = _size(config.n_rows, spec_rng, log=log)
    features = _size(config.n_features, spec_rng, log=log)
    n_context = min(rows - 1, max(1, int(rows * spec_rng.uniform(*_bounds(config.context_fraction)))))
    class_low, class_high = _bounds(config.n_classes)
    classes = _size((class_low, min(class_high, n_context)), spec_rng)
    family = config.families[spec_rng.integers(len(config.families))]
    feature_type = config.feature_types[spec_rng.integers(len(config.feature_types))]
    width_mode = config.width_modes[spec_rng.integers(len(config.width_modes))]
    missing_rate = float(spec_rng.uniform(*_bounds(config.missing_rate)))

    prior = config.scm if family == "scm" else config.neural
    donors = min(config.wide_base_features, features // 2) if width_mode == "wide" else features
    if feature_type != "continuous":
        fraction = 1.0
        if feature_type == "mixed":
            ratio = spec_rng.uniform(*_bounds(config.categorical_fraction))
            count = min(donors - 1, max(1, int(ratio * donors)))
            fraction = (count + 0.5) / donors
        prior = Observation(prior, categorical_fraction=fraction, max_categories=config.max_categories)
    if width_mode == "wide":
        prior = WidePrior(
            prior,
            base_features=donors,
            density=config.wide_density,
            noise=config.wide_noise,
            include_original=config.wide_include_original,
            max_categories=config.max_categories,
            numerical_mode=config.wide_numerical_mode,
            zero_fraction=config.wide_zero_fraction,
        )
    if missing_rate:
        prior = Observation(prior, missing_rate=missing_rate, missing_pattern=config.missing_pattern)

    weights = ()
    if classes and _bounds(config.class_logit_scale)[1] > 0:
        scale = spec_rng.uniform(*_bounds(config.class_logit_scale))
        logits = spec_rng.standard_normal(classes)
        weights = tuple(np.exp(scale * np.maximum(logits - logits.max(), -50 / max(1.0, scale))))
    strength = config.binary_label_strength
    if strength is not None:
        strength = float(spec_rng.uniform(*_bounds(strength)))
    multiclass_strength = config.multiclass_label_strength
    if multiclass_strength is not None:
        multiclass_strength = float(spec_rng.uniform(*_bounds(multiclass_strength)))
    prefix = config.split_mode != "random" or multiclass_strength is not None
    task = Task(
        rows, features, classes, weights, strength, numeric_features_only=(feature_type == "continuous"),
        multiclass_label_strength=multiclass_strength, split_mode=config.split_mode,
        context_rows=n_context if prefix else None,
    )
    data = _sample(prior, task, data_rng)
    if prefix:
        if classes and multiclass_strength is None:
            for _ in range(63):
                if np.unique(data.y[:n_context]).size == classes:
                    break
                data = _sample(prior, task, data_rng)
        assert not classes or np.unique(data.y[:n_context]).size == classes, (
            "The context is missing classes. Use more context rows, a milder shift, or stochastic multiclass labels."
        )
        order = slice(None)
    else:
        order = _context_first(
            data.y, n_context, classes, split_rng, keep_random_split=bool(weights) or strength is not None
        )
    if feature_type == "categorical":
        assert data.categorical.all(), "Categorical episodes require nonconstant donor columns"
    if feature_type == "mixed":
        assert 0 < data.categorical.sum() < features, "Mixed episodes require both feature types"
    X, y = data.X[order], data.y[order]
    probs = data.class_probabilities
    return Episode(
        X[:n_context], y[:n_context], X[n_context:], y[n_context:], data.categorical,
        task, family, feature_type, width_mode, missing_rate,
        query_probabilities=probs[order][n_context:] if probs is not None else None,
    )
