from dataclasses import replace

import numpy as np
import pytest

from open_tabular_priors import FunctionalSCMPrior, Task, TrainingConfig, sample, sample_episode


@pytest.mark.parametrize("mechanism", ["linear", "additive", "tree", "interaction", "local", "gated"])
@pytest.mark.parametrize("geometry", ["independent", "blocks", "heavy_tail", "subgroups"])
def test_functional_mechanism_geometry_pairs_replay(mechanism, geometry):
    prior = FunctionalSCMPrior(mechanisms=(mechanism,), geometries=(geometry,))
    task = Task(40, 5, 3, (3, 2, 1))
    first = sample(prior, task, seed=17, index=5)
    second = sample(prior, task, seed=17, index=5)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.y, second.y)
    np.testing.assert_array_equal(np.unique(first.y), [0, 1, 2])


def test_functional_prior_composes_with_wide_mixed_and_missingness():
    config = TrainingConfig(
        n_rows=30, n_features=1000, n_classes=2, families=("scm",),
        feature_types=("mixed",), width_modes=("wide",), missing_rate=0.15,
        class_logit_scale=1.2, scm=FunctionalSCMPrior(),
    )
    episode = sample_episode(config, seed=28)
    assert episode.X_context.shape[1] == 1000
    assert 0 < episode.categorical.sum() < 1000
    assert np.isnan(episode.X_context).any()
    np.testing.assert_array_equal(np.unique(episode.y_context), [0, 1])


def test_linear_and_interaction_generate_distinct_regression_surfaces():
    task = Task(3000, 2)
    linear = sample(FunctionalSCMPrior(mechanisms=("linear",), noise=(0.03, 0.03)), task, seed=8)
    interaction = sample(FunctionalSCMPrior(mechanisms=("interaction",), noise=(0.03, 0.03)), task, seed=8)
    for table, expected_error in ((linear, 0.01), (interaction, 0.1)):
        design = np.column_stack((np.ones(task.n_rows), table.X))
        residual = table.y - design @ np.linalg.lstsq(design, table.y, rcond=None)[0]
        error = np.square(residual).mean()
        if table is linear:
            assert error < expected_error
        else:
            assert error > expected_error


def test_block_geometry_generates_correlated_features():
    task = Task(1200, 24)
    independent = sample(FunctionalSCMPrior(geometries=("independent",)), task, seed=32)
    blocks = sample(FunctionalSCMPrior(geometries=("blocks",)), task, seed=32)
    off_diagonal = np.triu_indices(task.n_features, k=1)
    independent_corr = np.abs(np.corrcoef(independent.X.T)[off_diagonal]).mean()
    block_corr = np.abs(np.corrcoef(blocks.X.T)[off_diagonal]).mean()
    assert block_corr > 3 * independent_corr


def test_tree_ensemble_changes_targets_and_replays():
    prior = FunctionalSCMPrior(mechanisms=("tree",), active_features=(3, 3), distractor_weights=(1, 1, 1))
    task = Task(80, 12, 4, multiclass_label_strength=0.8)
    single = sample(prior, task, seed=42)
    fixed_single = sample(replace(prior, tree_count=(1, 1)), task, seed=42)
    ensemble = replace(prior, tree_count=(5, 9))
    first = sample(ensemble, task, seed=42)
    np.testing.assert_array_equal(single.y, fixed_single.y)
    np.testing.assert_array_equal(single.X, first.X)
    assert not np.array_equal(single.class_probabilities, first.class_probabilities)
    np.testing.assert_array_equal(first.y, sample(ensemble, task, seed=42).y)


def test_distractor_width_keeps_target_and_controls_dependence():
    prior = FunctionalSCMPrior(
        mechanisms=("linear",), active_features=(4, 4), distractor_weights=(1, 1, 1),
        shuffle_columns=False, null_block_size=4,
    )
    narrow = sample(prior, Task(2000, 4), seed=109)
    wide = sample(prior, Task(2000, 28), seed=109)
    np.testing.assert_array_equal(wide.y, narrow.y)
    np.testing.assert_array_equal(wide.X[:, :4], narrow.X)
    corr = np.corrcoef(wide.X.T)
    assert corr[12, 13] > 0.75
    assert abs(corr[12, 16]) < 0.07
    assert np.abs(corr[:4, 4:20]).max() < 0.08
    assert np.max(corr[:4, 20:], axis=0).min() > 0.9


@pytest.mark.parametrize("split_mode", ("grouped", "temporal"))
def test_ordered_episodes_apply_covariate_shift_and_target_drift_separately(split_mode):
    prior = FunctionalSCMPrior(mechanisms=("linear",), active_features=(3, 3), distractor_weights=(1, 1, 1),
                               shuffle_columns=False)
    config = TrainingConfig(
        n_rows=40, n_features=12, n_classes=0, context_fraction=0.5, feature_types=("continuous",),
        families=("scm",), width_modes=("direct",), split_mode=split_mode, scm=prior,
    )
    control = sample_episode(config, seed=89)
    shifted_prior = replace(prior, covariate_shift=1.5)
    shifted = sample_episode(replace(config, scm=shifted_prior), seed=89)
    drift = sample_episode(replace(config, scm=replace(shifted_prior, target_drift=2)), seed=89)
    X, X_shifted, X_drift = [np.concatenate((ep.X_context, ep.X_query)) for ep in (control, shifted, drift)]
    np.testing.assert_array_equal(X_shifted, X_drift)
    np.testing.assert_array_equal(X[:, 3:9], X_shifted[:, 3:9])
    assert not np.array_equal(X[:, 9:], X_shifted[:, 9:])
    assert not np.array_equal(shifted.y_query, drift.y_query)
    position = (np.arange(40) >= 20) - 0.5 if split_mode == "grouped" else np.linspace(-0.5, 0.5, 40)
    delta = X_shifted[:, :3] - X[:, :3]
    np.testing.assert_allclose(delta, position[:, None] * (2 * delta[-1]), atol=1e-12)
