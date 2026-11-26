# Training and evaluating nanoTabPFN and nanoTabICL

Train each model on two synthetic priors and score its final checkpoint on
the same 30 binary TabArena-Lite tasks. Each task has up to 100 labeled
support rows and 128 features.

- `other` generates binary episodes with 150 rows and five numerical features.
- `weighted` draws from standard SCM, dense wide, sparse binary wide and
  categorical tree recipes with weights 50, 18, 12 and 20. These recipes use
  two to five classes.

Both models train for 2,500 updates per run with 32 episodes per update and
seeds 0, 1 and 2. Each prior has one synthetic episode bank shared by both
models and all three seeds.

## Run the benchmark

From the repository root, set the paths below to a frozen fold directory
with `manifest.json` and its task files, a pinned nanoTabICL source directory,
and a source snapshot of Prior-Evaluation at
`57cf4d10e4ecc4fe5338180dd9c6f7a4462b3a7d`. The snapshot supplies the model
adapters, training loop and scoring code.

```bash
export PYTHONPATH=/path/to/prior-evaluation-snapshot
FOLDS=/path/to/tabarena-lite-folds
UPSTREAM=/path/to/pinned-nanotabicl
RUN=/path/to/output

benchmark() {
    uv run --locked --extra torch --with h5py --with openml --with pandas \
        --with scikit-learn --with schedulefree --with filelock \
        python benchmarks/tabarena-lite/weighted_benchmark.py "$@"
}

benchmark prepare --run "$RUN" --arm other
benchmark prepare --run "$RUN" --arm weighted

for cell in {0..11}; do
    benchmark train --run "$RUN" --folds "$FOLDS" --upstream "$UPSTREAM" --cell "$cell"
done

benchmark summarize --run "$RUN"
```

Cells 0–5 train nanoTabPFN and cells 6–11 train nanoTabICL. For each model,
the first three cells use `other` and the next three use `weighted`. Training
saves a checkpoint and scores it on the frozen folds. The summary reports
means and paired 95% bootstrap intervals over tasks and seeds.

## Results

| Model | Prior | NLL ↓ | AUROC ↑ | Macro-F1 ↑ | MCC ↑ |
|---|---|---:|---:|---:|---:|
| nanoTabPFN | other | 0.4288 | 0.7188 | 0.5419 | 0.1586 |
| nanoTabPFN | weighted | 0.4474 | 0.6365 | 0.4727 | 0.0475 |
| nanoTabICL | other | 0.4133 | 0.7384 | 0.5374 | 0.1579 |
| nanoTabICL | weighted | 0.3965 | 0.7446 | 0.5939 | 0.2499 |

The weighted mix raised macro-F1 and MCC on nanoTabICL and lowered AUROC,
macro-F1 and MCC on nanoTabPFN. The paired intervals for those changes
exclude zero. The experiment changes table size, class count and feature
types together.

[Per-task scores](cells.json) · [Means and paired intervals](summary.json)
