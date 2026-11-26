"""Compare a weighted episode recipe with the narrow other prior on TabArena-Lite."""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch
from filelock import FileLock
from open_tabular_priors import FunctionalSCMPrior, Mixture, SCMPrior, TrainingConfig, TrainingMixture, sample_episode

from prior_evaluation.config import TrainConfig
from prior_evaluation.matched_benchmark import MODELS, _ICLAdapter, _PFNAdapter, _load_icl
from prior_evaluation.pretrain import _train_loop
from prior_evaluation.tabarena_lite import METRICS, score_fold
from prior_evaluation.utils import configure_logging, file_sha256, load_json, rng_for, save_json, set_determinism

ARMS = ("other", "weighted")
RECIPES = ("standard", "dense_wide", "sparse_wide", "categorical")
WEIGHTS = (50, 18, 12, 20)
SCORES = (*METRICS, "mcc")
MAX_CLASSES = 5


def prior_configs(weights, wide_cap):
    control = TrainingConfig(
        n_rows=150, n_features=5, n_classes=2, feature_types=("continuous",), width_modes=("direct",),
        class_logit_scale=1.2, binary_label_strength=(2.0, 8.0), missing_rate=(0.0, 0.10),
    )
    scm = Mixture((control.scm, FunctionalSCMPrior(active_features=(2, 4), tree_count=(2, 8),
                                                  distractor_weights=(1, 1, 1))), (1, 2))
    control = replace(control, scm=scm)
    standard = replace(control, n_rows=(128, 224), n_features=(5, 64), n_classes=(2, MAX_CLASSES),
                       families=("scm",), binary_label_strength=None)
    dense = replace(standard, n_rows=(140, 260), n_features=(64, wide_cap), width_modes=("wide",),
                    wide_base_features=16, wide_noise=0.5)
    sparse = replace(dense, wide_numerical_mode="binary", wide_zero_fraction=0.85)
    categorical = replace(standard, n_features=(8, 64), feature_types=("mixed",),
                          categorical_fraction=(0.45, 0.80), scm=SCMPrior("tree"))
    return control, TrainingMixture((standard, dense, sparse, categorical), weights)


def fixed_batch(recipe, seed, step):
    rng = rng_for(seed, step, 0)
    sizes = {}
    for key in ("n_rows", "n_features", "n_classes"):
        value = vars(recipe)[key]
        if isinstance(value, tuple):
            low, high = value
            if key == "n_features":
                value = min(high, int(round(np.exp(rng.uniform(np.log(low), np.log(high))))))
            else:
                value = int(rng.integers(low, high + 1))
        sizes[key] = value
    split = int(sizes["n_rows"] * rng.uniform(*recipe.context_fraction))
    fraction = math.nextafter(split / sizes["n_rows"], 1.0)
    return replace(recipe, **sizes, context_fraction=fraction), split


def save_atomic(value, path):
    temporary = path.with_suffix(".tmp")
    save_json(value, temporary)
    temporary.replace(path)


def prepare(args):
    control, mix = prior_configs(tuple(args.weights), args.wide_cap)
    path = args.run / f"{args.arm}.h5"
    settings = dict(arm=args.arm, steps=args.steps, batch_size=args.batch_size, data_seed=args.data_seed,
                    weights=args.weights, wide_cap=args.wide_cap, config=repr(control if args.arm=="other" else mix))
    if path.is_file():
        with h5py.File(path) as bank:
            assert json.loads(bank.attrs["settings"]) == settings, "Bank settings changed"
        return
    counts = dict.fromkeys(("other",) if args.arm == "other" else RECIPES, 0)
    temporary = path.with_suffix(".tmp")
    with h5py.File(temporary, "w") as bank:
        bank.attrs["settings"] = json.dumps(settings)
        for step in range(args.steps):
            if args.arm == "other":
                recipe, name = control, "other"
            else:
                selection = mix.select_index(seed=args.data_seed, index=step)
                recipe, name = mix.recipes[selection], RECIPES[selection]
            config, split = fixed_batch(recipe, args.data_seed, step)
            # Each batch shares dimensions and split. Its episodes still draw independent mechanisms.
            X = np.empty((args.batch_size, config.n_rows, config.n_features), dtype=np.float32)
            y = np.empty((args.batch_size, config.n_rows), dtype=np.uint8)
            for row in range(args.batch_size):
                episode = sample_episode(config, seed=args.data_seed, index=step*args.batch_size+row)
                assert len(episode.y_context) == split and episode.task.n_classes == config.n_classes
                values = np.concatenate((episode.X_context, episode.X_query))
                observed = np.isfinite(values[:split])
                means = np.where(observed, values[:split], 0).sum(axis=0) / observed.sum(axis=0).clip(min=1)
                X[row] = np.where(np.isfinite(values), values, means)
                y[row] = np.concatenate((episode.y_context, episode.y_query))
            batch = bank.create_group(f"{step:06d}")
            batch.attrs.update(split=split, n_classes=config.n_classes, recipe=name)
            batch.create_dataset("X", data=X, compression="lzf")
            batch["y"] = y
            counts[name] += args.batch_size
            if (step+1) % 100 == 0:
                print(f"{args.arm}: sampled {step+1}/{args.steps} batches", flush=True)
        bank.attrs["recipe_episode_counts"] = json.dumps(counts)
    temporary.replace(path)
    print(f"{args.arm}: accepted episode counts {counts}", flush=True)


def batches(bank):
    for batch in bank.values():
        yield dict(x=torch.from_numpy(batch["X"][:]), y=torch.from_numpy(batch["y"][:]),
                   train_test_split_index=int(batch.attrs["split"]), n_classes=int(batch.attrs["n_classes"]))


def train(args, model_name, arm, seed):
    directory = args.run / model_name / arm / f"seed-{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "metrics.json"
    with h5py.File(args.run / f"{arm}.h5") as bank:
        settings = json.loads(bank.attrs["settings"])
        assert len(bank) == settings["steps"] == args.steps and settings["batch_size"] == args.batch_size
        assert settings["weights"] == args.weights and settings["wide_cap"] == args.wide_cap
        assert settings["data_seed"] == args.data_seed and settings["arm"] == arm
        manifest_path = args.folds / "manifest.json"
        protocol = dict(bank=settings, recipe_episode_counts=json.loads(bank.attrs["recipe_episode_counts"]),
                        manifest_sha256=file_sha256(manifest_path), seed=seed, model=model_name,
                        lr=args.lr, device=args.device, bf16=args.device=="cuda", microbatch_size=args.microbatch_size)
        if destination.is_file():
            assert load_json(destination)["protocol"] == protocol, "Run settings changed"
            return
        set_determinism(seed)
        if model_name == "nanotabicl":
            model = _ICLAdapter(_load_icl(args.upstream, "model").NanoTabICLv2, MAX_CLASSES)
        else:
            model = _PFNAdapter(MAX_CLASSES)
        config = TrainConfig(steps=args.steps, lr=args.lr, batch_size=args.batch_size, seed=seed,
                             steps_per_eval=min(100, args.steps), device=args.device,
                             warmup_steps=min(250,args.steps//10), weight_decay=1e-5)
        state = directory / "checkpoint.pt"
        if state.is_file():
            saved = torch.load(state, map_location=args.device, weights_only=True)
            assert saved["protocol"] == protocol, "Checkpoint settings changed"
            model.to(args.device).load_state_dict(saved["model"])
            history = saved["training_history"]
            model.eval()
        else:
            model, history = _train_loop(model, batches(bank), config, args.device,
                                         microbatch_size=args.microbatch_size, mask_unused_classes=True,
                                         bf16=args.device=="cuda")
            temporary=state.with_suffix(".tmp")
            torch.save(dict(model=model.state_dict(),protocol=protocol,training_history=history),temporary)
            temporary.replace(state)

    def binary_logits(data, train_test_split_index):
        return model(data, train_test_split_index)[..., :2]

    records = []
    manifest = load_json(manifest_path)
    assert len(manifest["folds"]) == 30
    for task in manifest["folds"]:
        with np.load(args.folds / task["file"], allow_pickle=False) as fold:
            arrays = tuple(fold[name] for name in ("x", "y", "query", "truth"))
        metrics = score_fold(binary_logits, arrays, args.device, 32, metrics=SCORES)
        records.append(dict(task_id=task["task_id"], **metrics))
        print(f"{model_name}/{arm}/{seed}: TabArena-Lite {task['task_id']} {len(records)}/30", flush=True)
    save_atomic(dict(protocol=protocol, tasks=records, mean={key: np.mean([r[key] for r in records]) for key in SCORES}),
                destination)


def summarize(args):
    means = []
    comparisons = []
    rng = rng_for(args.data_seed, 19)
    for model in MODELS:
        data = {}
        for arm in ARMS:
            runs = [load_json(args.run/model/arm/f"seed-{seed}"/"metrics.json") for seed in args.seeds]
            data[arm] = [{row["task_id"]: row for row in run["tasks"]} for run in runs]
            for key in SCORES:
                means.append(dict(model=model, arm=arm, metric=key, mean=np.mean([run["mean"][key] for run in runs])))
        ids = sorted(data["other"][0])
        assert len(ids) == 30 and all(sorted(rows) == ids for arm in ARMS for rows in data[arm])
        seed_draws = rng.integers(len(args.seeds), size=(5000,len(args.seeds)))
        task_draws = rng.integers(len(ids), size=(5000,len(ids)))
        for key in SCORES:
            delta = np.array([[right[task][key]-left[task][key] for task in ids]
                              for left,right in zip(data["other"],data["weighted"],strict=True)])
            distribution = delta[seed_draws[:,:,None],task_draws[:,None,:]].mean(axis=(1,2))
            comparisons.append(dict(model=model,metric=key,weighted_minus_other=np.mean(delta),
                                    ci95=np.quantile(distribution,(0.025,0.975))))
    save_atomic(dict(seeds=args.seeds, means=means, paired=comparisons),args.run/"summary.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","train","summarize"))
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--cell", type=int)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--data-seed", type=int, default=1729)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    parser.add_argument("--weights", nargs=4, type=float, default=list(WEIGHTS))
    parser.add_argument("--wide-cap", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cuda","cpu"), default="cuda")
    args = parser.parse_args()
    assert 1 <= args.microbatch_size <= args.batch_size and args.steps > 0
    assert args.wide_cap >= 64 and len(set(args.seeds)) == len(args.seeds)
    args.run.mkdir(parents=True,exist_ok=True)
    configure_logging()
    torch.set_num_threads(1)
    if args.action == "prepare":
        assert args.arm is not None, "--arm is required"
        with FileLock(args.run / f"{args.arm}.lock", timeout=0):
            prepare(args)
    elif args.action == "train":
        assert args.folds is not None and args.upstream is not None
        cells = [(model,arm,seed) for model in MODELS for arm in ARMS for seed in args.seeds]
        assert args.cell is not None and 0 <= args.cell < len(cells)
        with FileLock(args.run / f"cell-{args.cell}.lock", timeout=0):
            train(args,*cells[args.cell])
    else:
        summarize(args)


if __name__ == "__main__":
    main()
