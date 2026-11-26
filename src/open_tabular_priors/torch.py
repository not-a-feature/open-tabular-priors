from operator import index
from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .core import Prior, Task, _seed_sequence, sample
from .training import TrainingConfig, TrainingMixture, sample_episode

__all__ = [
    "EpisodeDataset", "PaddedEpisodes", "PriorDataset", "TensorEpisode", "TensorTable",
    "collate_episodes", "make_dataloader", "pad_episodes",
]


class TensorTable(NamedTuple):
    X: torch.Tensor
    y: torch.Tensor
    categorical: torch.Tensor


def _tensor(values: np.ndarray, dtype: type) -> torch.Tensor:
    with np.errstate(over="raise"):
        return torch.from_numpy(np.array(values, dtype=dtype, order="C", copy=True))


class PriorDataset(Dataset[TensorTable]):
    """A finite, lazily generated range of synthetic tables for a PyTorch DataLoader."""

    def __init__(
        self,
        prior: Prior,
        task: Task,
        *,
        seed: int,
        count: int,
        start_index: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.prior = prior
        self.task = task
        _seed_sequence(seed, start_index)
        assert not isinstance(count, bool), "count must be a positive integer"
        self.seed = index(seed)
        self.count = index(count)
        self.start_index = index(start_index)
        assert self.count > 0, "count must be positive"
        assert self.seed >= 0 and self.start_index >= 0, "seed and start_index must be nonnegative"
        assert dtype in (torch.float32, torch.float64), "dtype must be torch.float32 or torch.float64"
        self._feature_dtype = np.float32 if dtype == torch.float32 else np.float64
        self._target_dtype = np.int64 if task.n_classes else self._feature_dtype

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, position: int) -> TensorTable:
        position = index(position)
        if not 0 <= position < self.count:
            raise IndexError(position)

        table = sample(self.prior, self.task, seed=self.seed, index=self.start_index + position)
        return TensorTable(
            _tensor(table.X, self._feature_dtype),
            _tensor(table.y, self._target_dtype),
            _tensor(table.categorical, np.bool_),
        )


class TensorEpisode(NamedTuple):
    X_context: torch.Tensor
    y_context: torch.Tensor
    X_query: torch.Tensor
    y_query: torch.Tensor
    categorical: torch.Tensor
    n_classes: int


class EpisodeDataset(Dataset[TensorEpisode]):
    """Generate context/query episodes on demand for a range of stream indices."""

    def __init__(
        self,
        config: TrainingConfig | TrainingMixture,
        *,
        seed: int,
        count: int,
        start_index: int = 0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        _seed_sequence(seed, start_index)
        assert not isinstance(count, bool), "count must be a positive integer"
        self.seed = index(seed)
        self.count = index(count)
        self.start_index = index(start_index)
        assert self.count > 0, "count must be positive"
        assert self.seed >= 0 and self.start_index >= 0, "seed and start_index must be nonnegative"
        assert dtype in (torch.float32, torch.float64), "dtype must be torch.float32 or torch.float64"
        self._feature_dtype = np.float32 if dtype == torch.float32 else np.float64

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, position: int) -> TensorEpisode:
        position = index(position)
        if not 0 <= position < self.count:
            raise IndexError(position)
        episode = sample_episode(self.config, seed=self.seed, index=self.start_index + position)
        y_dtype = np.int64 if episode.task.n_classes else self._feature_dtype
        return TensorEpisode(
            _tensor(episode.X_context, self._feature_dtype),
            _tensor(episode.y_context, y_dtype),
            _tensor(episode.X_query, self._feature_dtype),
            _tensor(episode.y_query, y_dtype),
            _tensor(episode.categorical, np.bool_),
            episode.task.n_classes,
        )


def collate_episodes(episodes: list[TensorEpisode]) -> list[TensorEpisode]:
    """Keep episode sizes independent when batching."""
    return episodes


class PaddedEpisodes(NamedTuple):
    X_context: torch.Tensor
    y_context: torch.Tensor
    X_query: torch.Tensor
    y_query: torch.Tensor
    categorical: torch.Tensor
    n_classes: torch.Tensor
    context_mask: torch.Tensor
    query_mask: torch.Tensor
    feature_mask: torch.Tensor


def pad_episodes(episodes: list[TensorEpisode]) -> PaddedEpisodes:
    """Pad a batch, with masks that mark its real rows and columns."""
    assert episodes, "At least one episode is required"
    first = episodes[0]
    device = first.X_context.device
    for episode in episodes:
        assert all(actual.dtype == template.dtype and actual.device == device
                   for actual, template in zip(episode[:-1], first[:-1], strict=True)), (
            "Episodes in a padded batch must share tensor types and device"
        )
        assert episode.X_context.shape[1] == episode.X_query.shape[1] == episode.categorical.numel(), (
            "Episode context, query, and category mask must agree on feature count"
        )
    batch = len(episodes)
    context_rows = max(episode.X_context.shape[0] for episode in episodes)
    query_rows = max(episode.X_query.shape[0] for episode in episodes)
    features = max(episode.X_context.shape[1] for episode in episodes)
    X_context = first.X_context.new_zeros((batch, context_rows, features))
    X_query = first.X_query.new_zeros((batch, query_rows, features))
    y_context = first.y_context.new_zeros((batch, context_rows))
    y_query = first.y_query.new_zeros((batch, query_rows))
    categorical = torch.zeros((batch, features), dtype=torch.bool, device=device)
    context_mask = torch.zeros((batch, context_rows), dtype=torch.bool, device=device)
    query_mask = torch.zeros((batch, query_rows), dtype=torch.bool, device=device)
    feature_mask = torch.zeros((batch, features), dtype=torch.bool, device=device)
    for i, episode in enumerate(episodes):
        n_context, n_features = episode.X_context.shape
        n_query = episode.X_query.shape[0]
        X_context[i, :n_context, :n_features] = episode.X_context
        X_query[i, :n_query, :n_features] = episode.X_query
        y_context[i, :n_context] = episode.y_context
        y_query[i, :n_query] = episode.y_query
        categorical[i, :n_features] = episode.categorical
        context_mask[i, :n_context] = True
        query_mask[i, :n_query] = True
        feature_mask[i, :n_features] = True
    n_classes = torch.tensor([episode.n_classes for episode in episodes], dtype=torch.int64, device=device)
    return PaddedEpisodes(
        X_context, y_context, X_query, y_query, categorical, n_classes,
        context_mask, query_mask, feature_mask,
    )


def make_dataloader(
    config: TrainingConfig | TrainingMixture,
    *,
    seed: int,
    count: int,
    batch_size: int = 1,
    start_index: int = 0,
    num_workers: int = 0,
    padded: bool = False,
    shuffle: bool = False,
    pin_memory: bool = False,
    dtype: torch.dtype = torch.float32,
) -> DataLoader:
    """Batch training episodes while preserving their deterministic stream indices."""
    dataset = EpisodeDataset(config, seed=seed, count=count, start_index=start_index, dtype=dtype)
    shuffle_seed = int(_seed_sequence(seed, start_index).generate_state(1, dtype=np.uint64)[0])
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=pad_episodes if padded else collate_episodes,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(shuffle_seed),
        pin_memory=pin_memory,
    )
