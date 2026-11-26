from dataclasses import replace
from itertools import product

import numpy as np
import pytest

from open_tabular_priors import (
    AugmentedPrior, Dataset, GraphSCMPrior, Task, TrainingConfig, TrainingMixture, sample, sample_episode,
)
from open_tabular_priors.training import _size


def test_weighted_training_recipes_replay():
    narrow = TrainingConfig(n_rows=16, n_features=3, n_classes=0, feature_types=("continuous",),
                            width_modes=("direct",))
    wide = replace(narrow, n_rows=24, n_features=16, n_classes=3, width_modes=("wide",))
    mixture = TrainingMixture((narrow, wide), (0, 5))
    episode = sample_episode(mixture, seed=18, index=7)
    expected = sample_episode(wide, seed=18, index=7)
    assert episode.task == expected.task and episode.task.n_classes == 3
    np.testing.assert_array_equal(episode.X_query, expected.X_query)
    np.testing.assert_array_equal(episode.y_context, expected.y_context)

    mixed = TrainingMixture((narrow, wide), (1, 3))
    choices = [mixed.select_index(seed=18, index=i) for i in range(512)]
    assert 0.68 < np.mean(choices) < 0.82
    with pytest.raises(AssertionError, match="weight sum"):
        TrainingMixture((narrow, wide), (0, 0))


@pytest.mark.parametrize("mode", ("binary", "zero_inflated"))
def test_sparse_widening(mode):
    config = TrainingConfig(n_rows=100, n_features=32, n_classes=0, families=("neural",),
                            feature_types=("continuous",), width_modes=("wide",),
                            wide_include_original=False, wide_numerical_mode=mode, wide_zero_fraction=0.8)
    episode = sample_episode(config, seed=21)
    features = np.concatenate((episode.X_context, episode.X_query))
    assert np.all(features >= 0)
    assert np.all((features == 0).mean(axis=0) == 0.8)
    if mode == "binary":
        np.testing.assert_array_equal(np.unique(features), (0, 1))


@pytest.mark.parametrize(
    "family,feature_type,width_mode",
    tuple(product(("scm", "neural"), ("continuous", "categorical", "mixed"), ("direct", "wide"))),
)
def test_episode_combinations(family, feature_type, width_mode):
    config = TrainingConfig(
        n_rows=10, n_features=30, n_classes=3, context_fraction=0.5,
        families=(family,), feature_types=(feature_type,), width_modes=(width_mode,),
        categorical_fraction=0.5, missing_rate=0.2,
    )
    episode = sample_episode(config, seed=11)
    assert episode.X_context.shape == episode.X_query.shape == (5, 30)
    assert (episode.family, episode.feature_type, episode.width_mode) == (family, feature_type, width_mode)
    features = np.concatenate((episode.X_context, episode.X_query))
    assert np.isnan(features).any()
    assert np.isfinite(episode.y_query).all()
    np.testing.assert_array_equal(np.unique(episode.y_context), np.arange(3))
    if feature_type == "continuous":
        assert not episode.categorical.any()
    elif feature_type == "categorical":
        assert episode.categorical.all()
    else:
        assert 0 < episode.categorical.sum() < 30


def test_ranges_replay_and_cap_classes_to_context():
    config = TrainingConfig(
        n_rows=(10, 35), n_features=(10, 100), n_classes=(2, 10), context_fraction=0.5,
        missing_rate=(0.1, 0.3), missing_pattern="row", size_sampling="uniform",
    )
    episodes = [sample_episode(config, seed=50, index=i) for i in range(12)]
    shape = {(episode.task.n_rows, episode.task.n_features) for episode in episodes}
    assert len(shape) > 1
    assert all(10 <= rows <= 35 and 10 <= features <= 100 for rows, features in shape)
    for episode in episodes:
        assert 0.1 <= episode.missing_rate <= 0.3
        assert 2 <= episode.task.n_classes <= min(len(episode.y_context), 10)
        np.testing.assert_array_equal(np.unique(episode.y_context), np.arange(episode.task.n_classes))
    replay = sample_episode(config, seed=50, index=3)
    np.testing.assert_array_equal(replay.X_context, episodes[3].X_context)
    np.testing.assert_array_equal(replay.y_query, episodes[3].y_query)


def test_regression_and_fully_missing_features():
    config = TrainingConfig(n_rows=10, n_features=10, n_classes=0, missing_rate=1.0)
    episode = sample_episode(config, seed=21)
    assert np.isnan(episode.X_context).all()
    assert np.isnan(episode.X_query).all()
    assert np.issubdtype(episode.y_context.dtype, np.floating)
    assert np.isfinite(episode.y_query).all()


def test_missingness_preserves_episode_targets_and_observed_values():
    config = TrainingConfig(
        n_rows=40, n_features=24, width_modes=("wide",), feature_types=("mixed",),
    )
    clean = sample_episode(config, seed=3)
    masked = sample_episode(replace(config, missing_rate=0.3), seed=3)
    observed = ~np.isnan(masked.X_context)
    np.testing.assert_array_equal(masked.X_context[observed], clean.X_context[observed])
    np.testing.assert_array_equal(masked.y_context, clean.y_context)
    np.testing.assert_array_equal(masked.y_query, clean.y_query)
    np.testing.assert_array_equal(masked.categorical, clean.categorical)


def test_invalid_ranges_and_context_fail_early():
    with pytest.raises(AssertionError, match="Invalid n_features bounds"):
        TrainingConfig(n_features=(20, 10))
    with pytest.raises(AssertionError, match="smallest context"):
        TrainingConfig(n_rows=(10, 100), n_classes=(6, 9), context_fraction=0.5)


def test_class_prevalence_varies_and_context_retains_all_classes():
    config = TrainingConfig(
        n_rows=150, n_features=5, n_classes=2, families=("neural",),
        feature_types=("continuous",), width_modes=("direct",), class_logit_scale=1.2,
    )
    episodes = [sample_episode(config, seed=23, index=i) for i in range(24)]
    prevalences = []
    for episode in episodes:
        np.testing.assert_array_equal(np.unique(episode.y_context), [0, 1])
        y = np.concatenate((episode.y_context, episode.y_query))
        assert abs(y.mean() - episode.task.class_weights[1] / sum(episode.task.class_weights)) <= 1 / len(y)
        prevalences.append(min(y.mean(), 1 - y.mean()))
    assert min(prevalences) < 0.10 and max(prevalences) > 0.40
    replay = sample_episode(config, seed=23, index=4)
    np.testing.assert_array_equal(replay.y_query, episodes[4].y_query)


def test_graph_features_can_share_a_target_node():
    prior = GraphSCMPrior(
        n_nodes=4, channels=1, categorical_fraction=(0, 0), measurement_probability=0,
        target_node_features=(4, 5),
    )
    data = sample(prior, Task(100, 3, 2), seed=18)
    assert np.linalg.matrix_rank(data.X) == 1
    assert abs(np.corrcoef(data.X[:, 0], data.y)[0, 1]) > 0.5


def test_stochastic_binary_labels_vary_counts_at_fixed_prevalence():
    config = TrainingConfig(
        n_rows=150, n_features=5, n_classes=2, families=("neural",),
        feature_types=("continuous",), width_modes=("direct",), binary_label_strength=0,
    )
    counts = []
    for index in range(32):
        episode = sample_episode(config, seed=10, index=index)
        np.testing.assert_array_equal(np.unique(episode.y_context), [0, 1])
        counts.append(episode.y_context.sum() + episode.y_query.sum())
    assert np.std(counts) > 3
    with pytest.raises(AssertionError, match="n_classes=2"):
        replace(config, n_classes=(2, 3))


def test_stochastic_binary_labels_survive_tiny_imbalanced_tasks():
    config = TrainingConfig(
        n_rows=6, n_features=5, n_classes=2, context_fraction=0.5,
        feature_types=("continuous",), width_modes=("direct",),
        class_logit_scale=20, binary_label_strength=(0, 20),
    )
    for index in range(24):
        left = sample_episode(config, seed=11, index=index)
        right = sample_episode(config, seed=11, index=index)
        np.testing.assert_array_equal(np.unique(left.y_context), [0, 1])
        np.testing.assert_array_equal(left.y_query, right.y_query)


def test_stochastic_multiclass_temporal_episodes_keep_query_oracle_and_context_coverage():
    config = TrainingConfig(
        n_rows=12, n_features=40, n_classes=4, families=("scm",),
        feature_types=("mixed",), width_modes=("wide",), missing_rate=0.2,
        split_mode="temporal", context_fraction=0.34, class_logit_scale=5, multiclass_label_strength=0.97,
    )
    for index in range(5):
        episode = sample_episode(config, seed=18, index=index)
        replay = sample_episode(config, seed=18, index=index)
        assert episode.task.context_rows == 4
        np.testing.assert_array_equal(np.sort(episode.y_context), np.arange(4))
        assert episode.query_probabilities.shape == (8, 4)
        np.testing.assert_allclose(episode.query_probabilities.sum(axis=1), 1)
        np.testing.assert_array_equal(episode.query_probabilities, replay.query_probabilities)


def test_ordered_split_does_not_borrow_classes_from_the_query():
    def separated_classes(task, rng):
        labels = (np.arange(task.n_rows) >= task.context_rows).astype(int)
        return Dataset(np.zeros((task.n_rows, task.n_features)), labels, np.zeros(task.n_features, dtype=bool))

    config = TrainingConfig(
        n_rows=12, n_features=6, n_classes=2, families=("scm",), scm=separated_classes,
        feature_types=("continuous",), width_modes=("direct",), split_mode="grouped",
    )
    with pytest.raises(AssertionError, match="context is missing classes"):
        sample_episode(config, seed=11)


@pytest.mark.parametrize("seed", [None, True, -1])
def test_episode_seed_is_required_and_nonnegative(seed):
    with pytest.raises(AssertionError, match="seed must be a nonnegative integer"):
        sample_episode(TrainingConfig(), seed=seed)
    with pytest.raises(AssertionError, match="seed must be a nonnegative integer"):
        sample(GraphSCMPrior(), Task(10, 4, 2), seed=seed)


def test_log_size_respects_lower_rounding_boundary():
    class MinimumUniform:
        def uniform(self, low, high):
            return low

    assert _size((5, 20), MinimumUniform(), log=True) == 5


def test_graph_rounded_measurements_keep_categorical_donors():
    config = TrainingConfig(
        n_rows=10, n_features=20, n_classes=2,
        families=("scm",), feature_types=("categorical",), width_modes=("direct",),
        scm=GraphSCMPrior(categorical_fraction=(0, 0), target_node_features=(0, 3)),
    )
    assert sample_episode(config, seed=1729, index=29).categorical.all()


@pytest.mark.parametrize("width_mode", ("direct", "wide"))
def test_continuous_episodes_disable_native_category_readouts_and_recoding(width_mode):
    graph = GraphSCMPrior(categorical_fraction=(0.5, 0.5))
    prior = AugmentedPrior(graph, recode_probability=1.0)
    config = TrainingConfig(
        n_rows=256, n_features=12, n_classes=3, families=("scm",), scm=prior,
        feature_types=("continuous",), width_modes=(width_mode,),
    )
    direct = sample(prior, Task(256, 12, 3), seed=1729)
    assert direct.categorical.sum() > 6
    episode = sample_episode(config, seed=1729)
    assert not episode.categorical.any()
    assert np.isfinite(episode.X_context).all()
