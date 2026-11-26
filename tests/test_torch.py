import numpy as np
import pytest

from open_tabular_priors import FunctionalSCMPrior, MLPPrior, Observation, Task, TrainingConfig, WidePrior, sample

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader

from open_tabular_priors.torch import EpisodeDataset, PriorDataset, collate_episodes, make_dataloader, pad_episodes


def test_loader_batches_whole_tables_and_preserves_indexed_samples():
    base = Observation(MLPPrior(), categorical_fraction=0.5)
    prior = Observation(WidePrior(base, base_features=4), missing_rate=0.1)
    task = Task(32, 16, 3)
    dataset = PriorDataset(prior, task, seed=17, count=3, start_index=100)
    loader = DataLoader(dataset, batch_size=2)

    first, last = list(loader)
    assert first.X.shape == (2, 32, 16)
    assert first.y.shape == (2, 32)
    assert first.categorical.shape == (2, 16)
    assert last.X.shape == (1, 32, 16)
    assert (first.X.dtype, first.y.dtype, first.categorical.dtype) == (
        torch.float32, torch.int64, torch.bool
    )
    expected = sample(prior, task, seed=17, index=101)
    np.testing.assert_array_equal(first.X[1].numpy(), expected.X.astype(np.float32))
    np.testing.assert_array_equal(first.y[1].numpy(), expected.y)
    np.testing.assert_array_equal(first.categorical[1].numpy(), expected.categorical)


def test_regression_and_bounds():
    dataset = PriorDataset(MLPPrior(), Task(8, 4), seed=1, count=2, dtype=torch.float64)
    table = next(iter(DataLoader(dataset, batch_size=None)))
    assert table.X.shape == (8, 4)
    assert table.X.dtype == table.y.dtype == torch.float64
    with pytest.raises(IndexError):
        dataset[len(dataset)]


def test_spawn_workers_match_single_process():
    dataset = PriorDataset(MLPPrior(), Task(12, 4, 3), seed=2, count=5)
    single = DataLoader(dataset, batch_size=2)
    workers = DataLoader(dataset, batch_size=2, num_workers=2, multiprocessing_context="spawn")

    for expected, actual in zip(single, workers, strict=True):
        torch.testing.assert_close(actual.X, expected.X, rtol=0, atol=0)
        torch.testing.assert_close(actual.y, expected.y)
        torch.testing.assert_close(actual.categorical, expected.categorical)


def test_episode_collators_preserve_data_and_padding_masks():
    config = TrainingConfig(n_rows=(10, 22), n_features=(10, 30), missing_rate=0.15)
    dataset = EpisodeDataset(config, seed=1, count=3, start_index=11)
    ragged = next(iter(DataLoader(dataset, batch_size=3, collate_fn=collate_episodes)))
    padded = next(iter(DataLoader(dataset, batch_size=3, collate_fn=pad_episodes)))
    assert len({episode.X_context.shape for episode in ragged}) > 1
    for i, episode in enumerate(ragged):
        context, query, features = padded.context_mask[i], padded.query_mask[i], padded.feature_mask[i]
        torch.testing.assert_close(padded.X_context[i][context][:, features], episode.X_context, equal_nan=True)
        torch.testing.assert_close(padded.X_query[i][query][:, features], episode.X_query, equal_nan=True)
        torch.testing.assert_close(padded.y_query[i][query], episode.y_query)
        torch.testing.assert_close(padded.categorical[i][features], episode.categorical)
        assert padded.n_classes[i] == episode.n_classes
        assert (padded.X_query[i][~query] == 0).all()
        assert (padded.X_query[i][:, ~features] == 0).all()


def test_variable_episodes_support_contextual_training():
    config = TrainingConfig(n_rows=(20, 45), n_features=(10, 45), n_classes=(2, 5), missing_rate=0.1)
    dataset = EpisodeDataset(config, seed=4, count=4)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_episodes)
    temperature = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([temperature], lr=0.1)
    for batch in loader:
        losses = []
        for episode in batch:
            context = torch.nan_to_num(episode.X_context)
            query = torch.nan_to_num(episode.X_query)
            centers = torch.stack([
                context[episode.y_context == label].mean(dim=0) for label in range(episode.n_classes)
            ])
            logits = -torch.cdist(query, centers).square() * temperature / context.shape[1]
            losses.append(torch.nn.functional.cross_entropy(logits, episode.y_query))
        loss = torch.stack(losses).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert abs(temperature.item() - 1) > 1e-6


def test_episode_loader_spawn_workers_replay():
    config = TrainingConfig(n_rows=(10, 22), n_features=(10, 25), missing_rate=0.2)
    dataset = EpisodeDataset(config, seed=5, count=2, start_index=9)
    loader = DataLoader(dataset, batch_size=2, num_workers=2, collate_fn=collate_episodes,
                        multiprocessing_context="spawn")

    actual, = list(loader)
    for i, episode in enumerate(actual):
        expected = dataset[i]
        torch.testing.assert_close(episode.X_context, expected.X_context, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(episode.X_query, expected.X_query, rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(episode.y_context, expected.y_context)
        torch.testing.assert_close(episode.y_query, expected.y_query)


def test_make_dataloader_replays_shuffled_padded_batches():
    config = TrainingConfig(n_rows=(10, 25), n_features=(10, 40), n_classes=2, binary_label_strength=1)
    options = dict(seed=16, count=6, batch_size=2, start_index=20, padded=True, shuffle=True)
    for left, right in zip(make_dataloader(config, **options), make_dataloader(config, **options), strict=True):
        torch.testing.assert_close(left.X_context, right.X_context, equal_nan=True)
        torch.testing.assert_close(left.y_query, right.y_query)
        torch.testing.assert_close(left.query_mask, right.query_mask)


@pytest.mark.parametrize("seed", [np.int64(7), 2**64])
def test_dataloader_installs_numpy_stream_in_torch(seed):
    config = TrainingConfig(n_rows=10, n_features=10, n_classes=0)
    batch = next(iter(make_dataloader(
        config, seed=seed, count=2, batch_size=2, shuffle=True, padded=True, dtype=torch.float64,
    )))
    assert batch.X_context.dtype == batch.y_query.dtype == torch.float64


def test_multiclass_tree_ensemble_trains_on_wide_grouped_episodes():
    config = TrainingConfig(
        n_rows=(18, 28), n_features=(64, 96), n_classes=4, families=("scm",),
        feature_types=("mixed",), width_modes=("direct",), split_mode="grouped", missing_rate=0.1,
        multiclass_label_strength=(0.5, 0.9),
        scm=FunctionalSCMPrior(mechanisms=("tree",), tree_count=(4, 8), active_features=(3, 3),
                              distractor_weights=(2, 2, 1), covariate_shift=0.8),
    )
    batch = next(iter(make_dataloader(config, seed=421, count=2, batch_size=2, padded=True)))
    for labels, mask in zip(batch.y_context, batch.context_mask, strict=True):
        torch.testing.assert_close(labels[mask].unique(), torch.arange(4))
    model = torch.nn.Linear(batch.X_query.shape[-1], 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    predictions = model(batch.X_query[batch.query_mask].nan_to_num())
    loss = torch.nn.functional.cross_entropy(predictions, batch.y_query[batch.query_mask])
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
