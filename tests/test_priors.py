import numpy as np
import pytest

from open_tabular_priors import (
    Dataset,
    Mixture,
    MLPPrior,
    Observation,
    SCMPrior,
    Task,
    WidePrior,
    sample,
)


@pytest.mark.parametrize("prior", [
    Mixture((MLPPrior(), SCMPrior("neural")), (0.5, 0.5)),
    Mixture((SCMPrior("neural"), SCMPrior("tree")), (0.7, 0.3)),
    Observation(SCMPrior(), categorical_fraction=0.25, missing_rate=0.05,
                missing_pattern="row", proxy_fraction=0.25),
    SCMPrior("linear"),
])
@pytest.mark.parametrize("classes", [0, 3])
def test_indexed_sampling(prior, classes):
    task = Task(128, 12, classes)
    first = sample(prior, task, seed=81, index=4)
    other = sample(prior, task, seed=81, index=6)
    replay = sample(prior, task, seed=81, index=4)
    np.testing.assert_array_equal(first.X, replay.X)
    np.testing.assert_array_equal(first.y, replay.y)
    np.testing.assert_array_equal(first.categorical, replay.categorical)
    assert not np.array_equal(first.X, other.X, equal_nan=True)


def test_weighted_classes():
    data = sample(MLPPrior(), Task(1000, 3, 3, (0.1, 0.2, 0.7)), seed=11)
    np.testing.assert_array_equal(np.bincount(data.y), [100, 200, 700])


def test_tied_scores_can_fill_a_multiclass_task():
    task = Task(10, 12, 5)
    prior = SCMPrior("tree", noise=0)
    first = sample(prior, task, seed=1729)
    replay = sample(prior, task, seed=1729)
    np.testing.assert_array_equal(np.unique(first.y), np.arange(5))
    np.testing.assert_array_equal(first.y, replay.y)


def test_multiclass_probabilities_preserve_prevalence_and_query_sampling():
    prevalence = (0.6, 0.29, 0.1, 0.01)
    task = Task(4000, 5, 4, prevalence, multiclass_label_strength=0.85, context_rows=4)
    data = sample(MLPPrior(), task, seed=93)
    p = data.class_probabilities
    np.testing.assert_allclose(p.mean(axis=0), prevalence, atol=1e-12)
    np.testing.assert_array_equal(np.sort(data.y[:4]), np.arange(4))
    np.testing.assert_allclose(np.bincount(data.y[4:], minlength=4) / 3996, p[4:].mean(axis=0), atol=0.025)
    assert p[:, 0].std() > 0.2

    wrapped = Observation(WidePrior(MLPPrior(), base_features=5), missing_rate=0.2)
    wide = sample(wrapped, Task(4000, 8, 4, prevalence, multiclass_label_strength=0.85, context_rows=4), seed=93)
    np.testing.assert_array_equal(wide.y, data.y)
    np.testing.assert_array_equal(wide.class_probabilities, p)


def test_observation_does_not_mutate_input():
    base_data = sample(MLPPrior(), Task(64, 8), seed=40)

    def base(task, rng):
        return base_data

    prior = Observation(base, categorical_fraction=0.5, missing_rate=1.0, proxy_fraction=0.25)
    result = sample(prior, Task(64, 8), seed=41)
    assert result.categorical.sum() == 4
    assert np.isnan(result.X).all()
    assert np.isfinite(base_data.X).all()
    assert not base_data.categorical.any()
    np.testing.assert_array_equal(result.y, base_data.y)


def test_measurement_error_preserves_labels_and_category_codes():
    rng = np.random.default_rng(82)
    continuous = rng.standard_normal(12_000)
    y = (continuous > 0).astype(np.int64)
    X = np.column_stack((continuous, (continuous > 1).astype(float)))
    table = Dataset(X, y, np.array([False, True]))

    def base(task, rng):
        return table

    task = Task(len(X), 2, 2)
    noisy = sample(Observation(base, measurement_retention=(0.6, 0.6)), task, seed=44)
    clean = sample(Observation(base, measurement_retention=(1, 1)), task, seed=44)
    np.testing.assert_array_equal(clean.X, X)
    np.testing.assert_array_equal(noisy.y, y)
    np.testing.assert_array_equal(noisy.X[:, 1], X[:, 1])
    assert np.corrcoef(noisy.X[:, 0], continuous)[0, 1] == pytest.approx(0.6, abs=0.025)


@pytest.mark.parametrize("include_original", [False, True])
def test_wide_mixed_labels_and_metadata(include_original):
    base = Observation(MLPPrior(), categorical_fraction=0.5)
    narrow = sample(base, Task(96, 6, 4), seed=13)
    wide = WidePrior(base, base_features=6, include_original=include_original, max_categories=2)
    expanded = sample(wide, Task(96, 30, 4), seed=13)
    np.testing.assert_array_equal(narrow.y, expanded.y)
    assert expanded.categorical.sum() == 15
    cardinalities = [np.unique(column).size for column in expanded.X[:, expanded.categorical].T]
    assert sum(count > 2 for count in cardinalities) <= (3 if include_original else 0)


def test_noiseless_wide_projection_stays_in_donor_span():
    base = MLPPrior()
    narrow = sample(base, Task(40, 4), seed=8)
    wide = WidePrior(base, base_features=4, density=0.6, noise=0, include_original=False)
    expanded = sample(wide, Task(40, 270), seed=8)
    weights = np.linalg.lstsq(narrow.X, expanded.X, rcond=None)[0]
    np.testing.assert_allclose(narrow.X @ weights, expanded.X, atol=1e-12)


def test_all_categorical_widening():
    base = Observation(MLPPrior(), categorical_fraction=1.0)
    wide = WidePrior(base, base_features=1, include_original=False)
    data = sample(wide, Task(64, 20), seed=41)
    assert data.categorical.all()


def test_zero_density_wide_produces_noise_and_preserves_labels():
    base = MLPPrior()
    narrow = sample(base, Task(200, 4), seed=8)
    wide = WidePrior(base, base_features=4, density=0, noise=1, include_original=False)
    expanded = sample(wide, Task(200, 12), seed=8)
    assert (expanded.X.std(axis=0) > 0.7).all()
    np.testing.assert_array_equal(narrow.y, expanded.y)


def test_invalid_width_and_weights():
    prior = MLPPrior()
    with pytest.raises(AssertionError, match="smaller than the donor width"):
        sample(WidePrior(prior, base_features=4), Task(20, 3), seed=1)
    with pytest.raises(AssertionError, match="Invalid weight sum"):
        Mixture((prior,), (0.0,))


def test_invalid_custom_categorical_output():
    def prior(task, rng):
        return Dataset(np.full((2, 1), 0.5), np.zeros(2), np.ones(1, dtype=bool))

    with pytest.raises(AssertionError, match="integer codes"):
        sample(prior, Task(2, 1), seed=42)
