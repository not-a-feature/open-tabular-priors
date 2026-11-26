from dataclasses import replace

import numpy as np
import pytest

from open_tabular_priors import (
    AugmentedPrior,
    GraphSCMPrior,
    Task,
    TrainingConfig,
    WidePrior,
    sample,
    sample_episode,
)
from open_tabular_priors.augmentation import _correlate_blocks, _recode


@pytest.mark.parametrize(
    "mechanism",
    ["linear", "neural", "tree", "quadratic", "fourier", "cluster", "discrete", "product"],
)
def test_graph_mechanisms_replay(mechanism):
    prior = GraphSCMPrior(n_nodes=8, channels=4, mechanisms=(mechanism,))
    task = Task(48, 12, 3)
    data = sample(prior, task, seed=55, index=3)
    replay = sample(prior, task, seed=55, index=3)
    np.testing.assert_array_equal(data.X, replay.X)
    np.testing.assert_array_equal(data.y, replay.y)
    np.testing.assert_array_equal(data.categorical, replay.categorical)


def test_categorical_graph_respects_shape_and_cap():
    prior = GraphSCMPrior(categorical_fraction=(1, 1), max_categories=3)
    data = sample(prior, Task(2, 4, 2), seed=3)
    assert data.categorical.all()
    assert (data.X < 3).all()


def test_graph_size_ranges_replay_and_preserve_fixed_stream():
    task = Task(32, 6, 2)
    fixed = sample(GraphSCMPrior(n_nodes=8, channels=3), task, seed=340)
    singleton = sample(GraphSCMPrior(n_nodes=(8, 8), channels=(3, 3)), task, seed=340)
    variable = GraphSCMPrior(n_nodes=(2, 32), channels=(1, 32))
    produced = sample(variable, task, seed=340)
    replay = sample(variable, task, seed=340)
    np.testing.assert_array_equal(fixed.X, singleton.X)
    np.testing.assert_array_equal(fixed.y, singleton.y)
    np.testing.assert_array_equal(produced.X, replay.X)
    np.testing.assert_array_equal(produced.y, replay.y)
    assert not np.array_equal(fixed.X, produced.X)


@pytest.mark.parametrize("extra_masks", [False, True])
def test_missingness_only_masks_features(extra_masks):
    complete = AugmentedPrior(GraphSCMPrior(), block_probability=0.3, shuffle_columns=True)
    incomplete = replace(
        complete, structured_mask_probability=0.2, cell_mask_probability=0.01,
        near_empty_mask_probability=0.05 if extra_masks else 0.0,
        latent_mask_probability=0.03 if extra_masks else 0.0,
    )
    task = Task(192, 6, 4, (2, 1, 4, 1))
    masked_cells = 0
    for index in range(16):
        clean = sample(complete, task, seed=3, index=index)
        masked = sample(incomplete, task, seed=3, index=index)
        observed = ~np.isnan(masked.X)
        masked_cells += np.count_nonzero(~observed)
        np.testing.assert_array_equal(masked.X[observed], clean.X[observed])
        np.testing.assert_array_equal(masked.y, clean.y)
        np.testing.assert_array_equal(masked.categorical, clean.categorical)
        np.testing.assert_array_equal(np.bincount(masked.y), [48, 24, 96, 24])
    assert masked_cells > 0


def test_augmented_graph_can_be_widened():
    base = AugmentedPrior(
        GraphSCMPrior(), block_probability=0.3, relabel_probability=0.1,
        coarsen_probability=0.1, heavy_tail_probability=0.2, shuffle_columns=True,
    )
    narrow = sample(base, Task(128, 8), seed=7)
    wide = sample(WidePrior(base, base_features=8), Task(128, 128), seed=7)
    np.testing.assert_array_equal(wide.y, narrow.y)
    assert np.isfinite(wide.X).all()


def test_high_cardinality_recoding_is_optional():
    graph = GraphSCMPrior(categorical_fraction=(0, 0))
    raw = AugmentedPrior(graph, recode_probability=0.05)
    numeric = AugmentedPrior(graph)
    task = Task(512, 6)
    affected = [i for i in range(64) if sample(raw, task, seed=43, index=i).categorical.any()]
    assert affected
    for i in affected:
        assert not sample(numeric, task, seed=43, index=i).categorical.any()


@pytest.mark.parametrize("feature_type", ["continuous", "categorical", "mixed"])
@pytest.mark.parametrize("width_mode", ["direct", "wide"])
def test_augmented_graph_can_supply_training_episodes(feature_type, width_mode):
    scm = AugmentedPrior(
        GraphSCMPrior(n_nodes=(2, 32), channels=(1, 32),
                      categorical_fraction=(0, 0), target_node_features=(0, 3)),
        block_probability=0.3, relabel_probability=0.1, coarsen_probability=0.1,
    )
    config = TrainingConfig(
        n_rows=256, n_features=24, feature_types=(feature_type,), width_modes=(width_mode,),
        n_classes=2, families=("scm",), missing_rate=0.1, scm=scm,
    )
    episode = sample_episode(config, seed=18)
    X = np.concatenate((episode.X_context, episode.X_query))
    assert X.shape == (256, 24)
    assert np.isnan(X).any()
    assert episode.categorical.any() == (feature_type != "continuous")
    assert episode.categorical.all() == (feature_type == "categorical")


def test_invalid_augmentation_probability():
    with pytest.raises(AssertionError, match="relabel_probability"):
        AugmentedPrior(GraphSCMPrior(), relabel_probability=1.5)


def test_correlated_blocks_preserve_categorical_counts():
    rng = np.random.default_rng(2)
    X = rng.integers(0, 5, size=(100, 8)).astype(float)
    before = np.sort(X, axis=0)
    _correlate_blocks(X, np.ones(8, dtype=bool), rng)
    assert not np.array_equal(X, before)
    np.testing.assert_array_equal(np.sort(X, axis=0), before)


def test_recoding_marks_high_cardinality_columns():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, 5))
    categorical = np.zeros(5, dtype=bool)
    _recode(X, categorical, rng)
    assert 1 <= categorical.sum() <= 3
    assert all(np.unique(column).size >= 201 for column in X[:, categorical].T)
