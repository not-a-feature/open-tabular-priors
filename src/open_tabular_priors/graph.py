from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .core import Dataset, Size, Task, _bounds, _readout, _size
from .generators import _roots, _tree

GraphMechanism = Literal["linear", "neural", "tree", "quadratic", "fourier", "cluster", "discrete", "product"]
_MECHANISMS: tuple[GraphMechanism, ...] = (
    "linear", "neural", "tree", "quadratic", "fourier", "cluster", "discrete", "product",
)


def _standardize(X: NDArray[np.float64]) -> NDArray[np.float64]:
    scale = X.std(axis=0)
    return (X - X.mean(axis=0)) / np.where(scale > 0, scale, 1)


def _project(X: NDArray[np.float64], width: int, rng: Generator) -> NDArray[np.float64]:
    return X @ rng.normal(0, X.shape[1] ** -0.5, (X.shape[1], width))


def _mechanism(
    X: NDArray[np.float64], width: int, kind: GraphMechanism, rng: Generator
) -> NDArray[np.float64]:
    if kind == "linear":
        return _project(X, width, rng)
    if kind == "neural":
        hidden = _project(X, 2 * width, rng) + rng.normal(size=2 * width)
        activation = (np.tanh, np.sin, np.abs)[rng.integers(3)]
        return _project(activation(hidden), width, rng)
    if kind == "tree":
        depth = rng.integers(1, 4)
        return np.column_stack([_tree(X, depth, rng) for _ in range(width)])
    if kind == "quadratic":
        return _project(X, width, rng) * _project(X, width, rng)
    if kind == "fourier":
        features = 32
        scale = np.exp(rng.uniform(-1.5, 1.5))
        phase = rng.uniform(0, 2 * np.pi, features)
        return _project(np.cos(scale * _project(X, features, rng) + phase), width, rng)
    if kind in ("cluster", "discrete"):
        centers = rng.normal(size=(rng.integers(3, 9), X.shape[1]))
        distance = (
            np.square(X).sum(axis=1, keepdims=True)
            + np.square(centers).sum(axis=1)
            - 2 * X @ centers.T
        )
        values = rng.normal(size=(len(centers), width))
        if kind == "discrete":
            return values[distance.argmin(axis=1)]
        logits = -distance / X.shape[1] * rng.uniform(0.5, 4.0)
        weights = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (weights @ values) / weights.sum(axis=1, keepdims=True)

    assert kind == "product", f"Unknown graph mechanism {kind!r}"
    first = _MECHANISMS[rng.integers(len(_MECHANISMS) - 1)]
    second = _MECHANISMS[rng.integers(len(_MECHANISMS) - 1)]
    return _mechanism(X, width, first, rng) * _mechanism(X, width, second, rng)


def _measurement(channels: NDArray[np.float64], rng: Generator) -> NDArray[np.float64]:
    location, scale, group = (5 * np.tanh(channels / 5)).T
    subgroup = rng.random(location.size) < 1 / (1 + np.exp(-group))
    signal = rng.uniform(0.25, 0.85)
    slope = rng.uniform(0.2, 1.0)
    heteroscedastic = np.exp(slope * scale - slope**2 / 2)
    value = (
        np.sqrt(signal) * (location + subgroup * rng.normal())
        + np.sqrt(1 - signal) * heteroscedastic * rng.standard_normal(location.size)
    )
    if rng.random() < 0.5:
        value = np.exp(rng.uniform(0.0, 4.0)) / (1 + np.exp(-np.clip(value, -30, 30)))
    if rng.random() < 0.3:
        rounded = np.rint(value)
        if rounded.min() < rounded.max():
            return rounded
    return value


@dataclass(frozen=True)
class GraphSCMPrior:
    """Sample multichannel causal graphs with mixed mechanisms and observation models."""

    n_nodes: Size = 12
    channels: Size = 8
    max_parents: int = 4
    mechanisms: tuple[GraphMechanism, ...] = _MECHANISMS
    categorical_fraction: tuple[float, float] = (0.0, 0.6)
    max_categories: int = 20
    measurement_probability: float = 0.1
    node_noise: float = 0.05
    target_node_features: int | tuple[int, int] = 0

    def __post_init__(self) -> None:
        node_low, node_high = _bounds(self.n_nodes)
        channel_low, channel_high = _bounds(self.channels)
        counts = (node_low, node_high, channel_low, channel_high, self.max_parents, self.max_categories)
        assert all(isinstance(value, Integral) and not isinstance(value, bool) for value in counts), (
            "Graph dimensions and category cap must be integers"
        )
        assert 2 <= node_low <= node_high, "Invalid graph node bounds"
        assert 1 <= channel_low <= channel_high, "Invalid graph channel bounds"
        assert self.max_parents >= 1, "At least one parent must be allowed"
        assert self.mechanisms and all(kind in _MECHANISMS for kind in self.mechanisms), "Invalid graph mechanisms"
        low, high = self.categorical_fraction
        assert 0 <= low <= high <= 1, "Invalid categorical range"
        assert self.max_categories >= 2, "Category cap must be at least two"
        assert 0 <= self.measurement_probability <= 1, "Invalid measurement probability"
        assert np.isfinite(self.node_noise) and self.node_noise >= 0, "Invalid node noise"
        count = self.target_node_features
        assert isinstance(count, Integral) or (isinstance(count, tuple) and len(count) == 2), (
            "Specify a target node feature count or two bounds"
        )
        low, high = count if isinstance(count, tuple) else (count, count)
        assert all(isinstance(n, Integral) and not isinstance(n, bool) for n in (low, high)), (
            "Target node feature counts must be integers"
        )
        assert 0 <= low <= high, "Invalid target node feature bounds"

    def __call__(self, task: Task, rng: Generator) -> Dataset:
        node_low, node_high = _bounds(self.n_nodes)
        channel_low, channel_high = _bounds(self.channels)
        n_nodes = node_low if node_low == node_high else _size(self.n_nodes, rng, log=True)
        channels = channel_low if channel_low == channel_high else _size(self.channels, rng, log=True)
        bias = rng.standard_cauchy()
        outgoing = rng.standard_cauchy(n_nodes)
        incoming = rng.standard_cauchy(n_nodes)
        concentration = np.exp(rng.uniform(-2.5, 1.0))
        preferences = rng.dirichlet(np.full(len(self.mechanisms), concentration))
        nodes = np.empty((n_nodes, task.n_rows, channels))
        for child in range(n_nodes):
            logits = bias + outgoing[:child] + incoming[child]
            parents = np.flatnonzero(rng.logistic(size=child) < logits)
            if parents.size > self.max_parents:
                parents = rng.choice(parents, self.max_parents, replace=False)
            if parents.size == 0:
                value = _roots(task.n_rows, channels, rng)
            else:
                parent_values = nodes[parents]
                kind = self.mechanisms[rng.choice(len(self.mechanisms), p=preferences)]
                if parents.size > 1 and rng.random() < 0.5:
                    transformed = [
                        _standardize(_mechanism(parent, channels, kind, rng))
                        for parent in parent_values
                    ]
                    reducer = (np.sum, np.prod, np.max, np.logaddexp.reduce)[rng.integers(4)]
                    value = reducer(transformed, axis=0)
                else:
                    inputs = np.concatenate(parent_values, axis=1)
                    value = _mechanism(inputs, channels, kind, rng)
            noise = rng.normal(0, self.node_noise, value.shape)
            nodes[child] = _standardize(value + noise)

        target_node = rng.integers(n_nodes)
        feature_nodes = rng.integers(n_nodes, size=task.n_features)
        count = self.target_node_features
        if isinstance(count, tuple):
            low, high = (min(n, task.n_features) for n in count)
            count = rng.integers(low, high + 1)
        if count:
            requested = min(count, task.n_features)
            other_nodes = np.flatnonzero(feature_nodes != target_node)
            extra = max(0, requested - (task.n_features - other_nodes.size))
            feature_nodes[rng.choice(other_nodes, size=extra, replace=False)] = target_node
        score = _project(nodes[target_node], 1, rng).ravel()
        X = np.empty((task.n_rows, task.n_features))
        categorical = np.zeros(task.n_features, dtype=bool)
        fraction = rng.uniform(*self.categorical_fraction)
        n_cat = 0 if task.numeric_features_only else int(fraction * task.n_features)
        categorical[rng.choice(task.n_features, n_cat, replace=False)] = True
        measurement_prob = self.measurement_probability
        if n_cat == 0 and measurement_prob == 0:
            weights = rng.normal(0, channels**-0.5, (task.n_features, channels))
            for node in np.unique(feature_nodes):
                columns = np.flatnonzero(feature_nodes == node)
                X[:, columns] = nodes[node] @ weights[columns].T
            return _readout(X, score, task, rng, categorical=categorical)

        if 0 < measurement_prob < 1:
            measurement_prob = rng.beta(4 * measurement_prob, 4 * (1 - measurement_prob))
        nearest = rng.beta(1, 1)
        for j, node in enumerate(feature_nodes):
            values = nodes[node]
            if categorical[j]:
                cap = min(self.max_categories, task.n_rows)
                log_size = rng.uniform(0, np.log(cap / 2 + 0.5))
                size = min(cap, int(2 * np.exp(log_size)))
                if rng.random() < nearest:
                    centers = rng.normal(size=(size, channels))
                    logits = 2 * values @ centers.T - np.square(centers).sum(axis=1)
                else:
                    logits = _project(values, size, rng) + rng.gumbel(size=(task.n_rows, size))
                X[:, j] = np.unique(logits.argmax(axis=1), return_inverse=True)[1]
            elif rng.random() < measurement_prob:
                X[:, j] = _measurement(_standardize(_project(values, 3, rng)), rng)
            else:
                X[:, j] = _project(values, 1, rng).ravel()
        return _readout(X, score, task, rng, categorical=categorical)
