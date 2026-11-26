from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

from .core import Dataset, Prior, Task, _readout


def _roots(rows: int, columns: int, rng: Generator) -> NDArray[np.float64]:
    distribution = rng.integers(3)
    if distribution == 0:
        return rng.standard_normal((rows, columns))
    if distribution == 1:
        return rng.uniform(-1.73, 1.73, (rows, columns))
    centers = rng.standard_normal((3, columns))
    modes = rng.integers(len(centers), size=(rows, columns))
    return centers[modes, np.arange(columns)] + rng.normal(0, 0.4, (rows, columns))


def _tree(X: NDArray[np.float64], depth: int, rng: Generator) -> NDArray[np.float64]:
    node = np.zeros(X.shape[0], dtype=int)
    rows = np.arange(X.shape[0])
    for level in range(depth):
        features = rng.integers(X.shape[1], size=2**level)
        quantiles = rng.uniform(0.15, 0.85, size=features.size)
        cuts = np.array([np.quantile(X[:, j], q) for j, q in zip(features, quantiles)])
        node = 2 * node + (X[rows, features[node]] > cuts[node])
    return rng.standard_normal(2**depth)[node]


@dataclass(frozen=True)
class MLPPrior:
    """Draw a random neural prediction function of the observed features."""

    width: int = 24
    depth: int = 2
    noise: float = 0.1
    max_active_features: int | None = None

    def __post_init__(self) -> None:
        assert self.width > 0 and self.depth > 0, "MLP width and depth must be positive"
        assert np.isfinite(self.noise) and self.noise >= 0, "Invalid MLP noise"
        assert self.max_active_features is None or (
            isinstance(self.max_active_features, Integral) and self.max_active_features >= 1
        ), "max_active_features must be a positive integer"

    def __call__(self, task: Task, rng: Generator) -> Dataset:
        X = _roots(task.n_rows, task.n_features, rng)
        hidden = X
        if self.max_active_features is not None and task.n_features > self.max_active_features:
            active = rng.choice(task.n_features, self.max_active_features, replace=False)
            hidden = X[:, active]
        activation = (np.tanh, np.sin, np.abs)[rng.integers(3)]
        for _ in range(self.depth):
            weights = rng.normal(0, hidden.shape[1] ** -0.5, (hidden.shape[1], self.width))
            hidden = activation(hidden @ weights + rng.normal(0, 0.5, self.width))
        score = hidden @ rng.normal(0, self.width**-0.5, self.width)
        score += rng.normal(0, self.noise, task.n_rows)
        return _readout(X, score, task, rng)


Mechanism = Literal["linear", "neural", "tree", "mixed"]


@dataclass(frozen=True)
class SCMPrior:
    """Generate a joint table by reading separate feature and target nodes of a random DAG."""

    mechanism: Mechanism = "mixed"
    latent_nodes: int = 8
    max_parents: int = 4
    noise: float = 0.15
    tree_depth: int = 3

    def __post_init__(self) -> None:
        assert self.mechanism in ("linear", "neural", "tree", "mixed"), "Unknown SCM mechanism"
        assert self.latent_nodes >= 0 and self.max_parents > 0, "Invalid SCM graph size"
        assert np.isfinite(self.noise) and self.noise >= 0, "Invalid SCM noise"
        assert 1 <= self.tree_depth <= 8, "Tree depth must be in [1, 8]"

    def __call__(self, task: Task, rng: Generator) -> Dataset:
        n_nodes = task.n_features + self.latent_nodes + 1
        n_roots = max(1, n_nodes // 4)
        nodes = np.empty((n_nodes, task.n_rows))
        nodes[:n_roots] = _roots(task.n_rows, n_roots, rng).T
        degrees = np.ones(n_nodes)
        activation = (np.tanh, np.sin, np.abs)[rng.integers(3)]
        for child in range(n_roots, n_nodes):
            count = rng.integers(1, min(child, self.max_parents) + 1)
            weights = degrees[:child]
            parents = rng.choice(child, size=count, replace=False, p=weights / weights.sum())
            degrees[parents] += 1
            X_parent = nodes[parents].T
            mechanism = self.mechanism
            if mechanism == "mixed":
                mechanism = ("linear", "neural", "tree")[rng.integers(3)]
            if mechanism == "tree":
                value = _tree(X_parent, self.tree_depth, rng)
            else:
                value = X_parent @ rng.normal(0, count**-0.5, count)
                if mechanism == "neural":
                    value = activation(value + rng.normal())
            nodes[child] = value + rng.normal(0, self.noise, task.n_rows)

        target = rng.integers(n_roots, n_nodes)
        feature_nodes = rng.choice(np.delete(np.arange(n_nodes), target), task.n_features, replace=False)
        return _readout(nodes[feature_nodes].T, nodes[target], task, rng)


@dataclass(frozen=True)
class Mixture:
    """Draw a generator once per dataset with the supplied relative weights."""

    priors: tuple[Prior, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        assert len(self.priors) == len(self.weights) > 0, "Provide one weight per prior"
        assert all(callable(prior) for prior in self.priors), "Priors must be callable"
        assert np.isfinite(self.weights).all() and min(self.weights) >= 0, "Invalid mixture weights"
        assert np.isfinite(sum(self.weights)) and sum(self.weights) > 0, "Invalid weight sum"

    def __call__(self, task: Task, rng: Generator) -> Dataset:
        index = rng.choice(len(self.priors), p=np.asarray(self.weights) / sum(self.weights))
        return self.priors[index](task, rng)
