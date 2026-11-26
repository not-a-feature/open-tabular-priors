# Prior recipes and training

## Episodes and batching

`TrainingConfig` controls table size, feature types, class count, and the
generator. Each episode contains context and query inputs and labels, plus a
categorical column mask. Classification context contains every class in the
episode. A small or imbalanced query can contain just one class. Missing values
are represented by NaN. Use `n_classes=0` for regression.

`sample_episode(config, seed=1729, index=0)` replays a draw for the same config,
seed and index. The PyTorch loader returns lists of episodes by default. Set
`padded=True` to receive tensors with context, query and feature masks. Use
`start_index` to give each stage or epoch a fresh range of episode indices.
The model's initialization and optimizer randomness use a separate seed.

## Recipes and schedules

A `TrainingMixture` selects a complete recipe for each episode using relative
weights. Use it to tie size, labels and feature types together within a recipe.
For example, the second recipe below adds sparse binary columns around an
eight-feature donor table.

```python
from dataclasses import replace

from open_tabular_priors import TrainingConfig, TrainingMixture, sample_episode

narrow = TrainingConfig(
    n_rows=(96, 128), n_features=8, n_classes=2,
    feature_types=("continuous",), width_modes=("direct",),
    class_logit_scale=1.2, binary_label_strength=(2.0, 8.0),
)
sparse = replace(
    narrow, n_features=(32, 64), width_modes=("wide",), wide_base_features=8,
    wide_numerical_mode="binary", wide_zero_fraction=0.85,
)
prior = TrainingMixture((narrow, sparse), (3, 1))
episode = sample_episode(prior, seed=1729, index=3)
```

`wide_numerical_mode="zero_inflated"` adds nonnegative, log-transformed
measurements with a configurable fraction of zeros. Dense numerical projections
are the default for wide mode.

For a schedule, change the mixture between stages and keep the optimizer
running. One sequence is narrow-only, then a 3:1 mix of narrow and wider tables,
then a 1:1 mix. To test whether order helps, compare that schedule with a fixed
mixture containing the same overall proportions. Keep the label law, model,
and training budget fixed, and log the realized recipe counts for each run.

## Compose a generator

Pass a callable as `TrainingConfig(scm=...)` or `TrainingConfig(neural=...)`.
For one table, call `sample(prior, Task(n_rows, n_features, n_classes), seed=1729)`.
Custom generators should respect `Task.numeric_features_only`. Built-in
generators use seeded tie-breaking to keep every requested class when a score
has too few distinct values for quantile labels.

```python
from open_tabular_priors import AugmentedPrior, GraphSCMPrior, Mixture, SCMPrior

neural_tree = Mixture((SCMPrior("neural"), SCMPrior("tree")), (0.7, 0.3))

augmented_graph = AugmentedPrior(
    GraphSCMPrior(
        n_nodes=(2, 32),
        channels=(1, 32),
        categorical_fraction=(0, 0),
        measurement_probability=0.10,
        target_node_features=(0, 3),
    ),
    block_probability=0.30,
    relabel_probability=0.10,
    coarsen_probability=0.10,
    shuffle_columns=True,
)
```

The neural/tree mixture draws on TabICL v1. The augmented graph draws on
Kumo Tabular's noisy measurements, correlated feature blocks and post-processing.
With `TrainingConfig`, use its feature-type and missing-rate settings to
control the episode schema and apply missingness after widening.

`Observation` adds measurement noise, proxies or missing values. `WidePrior`
adds projections while preserving the base table's labels.
`FunctionalSCMPrior` defines how observed features affect the target.
Use `tree_count` for a sum of shallow tree scores and `distractor_weights`
for independent noise, correlated null blocks and noisy copies of informative
features. Grouped and temporal splits can introduce covariate shift or target
drift. For stochastic multiclass episodes, `multiclass_label_strength`
controls how strongly probabilities depend on the latent score. NumPy episodes
expose the query probabilities, and the coverage adjustment applies to
context labels only.
