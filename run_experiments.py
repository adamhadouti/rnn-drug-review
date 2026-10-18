"""Experiment plan. Every run shares the same data splits and evaluation protocol (src/data.py).

    python run_experiments.py --list                         # show all runs
    python run_experiments.py --groups A_arch                # run one group
    python run_experiments.py --shard 0 --num_shards 6       # team member 0 of 6 runs their share
    python run_experiments.py --results_dir /content/drive/MyDrive/rnn_results   # survive Colab resets

Finished runs (results/<name>.json present) are skipped, so a crashed session simply resumes.
"""
import argparse
import os
from dataclasses import replace

from src.cells import expected_cell_params
from src.train import Config, default_name, run

BASE = Config(emb_dim=128, hidden=128, proj=64, layers=1, optimizer="adam", lr=1e-3,
              batch_size=64, epochs=10, patience=3, init="default", train_frac=1.0)
MAIN_CELLS = ["vanilla", "wmc", "cifg"]   # baseline + the two required non-vanilla cells
ALL_CELLS = MAIN_CELLS + ["lstmp"]        # LSTMP as a bonus comparison


def matched_H(cell, target, E=128, P=None):
    return min(range(8, 1024), key=lambda H: abs(expected_cell_params(cell, E, H, P) - target))


def plan():
    runs = []

    def add(group, **kw):
        runs.append(replace(BASE, group=group, **kw))

    # A. Architecture comparison: same everything, 3 seeds -> mean +- std
    for cell in ALL_CELLS:
        for seed in (1, 2, 3):
            add("A_arch", cell=cell, seed=seed)
    # B-E. One hyper-parameter changed at a time from BASE (BASE itself = A_arch seed 1)
    for cell in MAIN_CELLS:
        for lr in (3e-4, 3e-3):
            add("B_lr", cell=cell, lr=lr)
        add("C_optim", cell=cell, optimizer="sgd", lr=0.05)      # each optimiser at its usual lr range
        add("C_optim", cell=cell, optimizer="rmsprop", lr=1e-3)
        add("C_optim", cell=cell, optimizer="adamw", lr=1e-3)
        add("D_init", cell=cell, init="xavier")
        add("D_init", cell=cell, init="orthogonal")
        add("D_init", cell=cell, init="orthogonal", forget_bias=1.0)
        for bs in (32, 256):
            add("E_batch", cell=cell, batch_size=bs)
    # F. Capacity: parameter-matched cells and stacking (stacked vanilla is still "vanilla")
    target = expected_cell_params("vanilla", BASE.emb_dim, BASE.hidden)
    add("F_capacity", cell="cifg", hidden=matched_H("cifg", target))
    add("F_capacity", cell="wmc", hidden=matched_H("wmc", target))
    add("F_capacity", cell="lstmp", hidden=256, proj=64)
    for cell in MAIN_CELLS:
        add("F_capacity", cell=cell, layers=2)
    # G. Class imbalance: inverse-frequency weighted cross-entropy (targets the small neutral class)
    for cell in MAIN_CELLS:
        add("G_imbalance", cell=cell, class_weights=True)
    for r in runs:
        r.name = f"{r.group}__{default_name(r)}" + ("_cw" if r.class_weights else "")
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--groups", nargs="*", help="e.g. A_arch B_lr (default: all)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--results_dir", default="results")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--epochs", type=int, help="override for quick tests")
    p.add_argument("--train_frac", type=float, help="override, e.g. 0.5 if GPU time is short")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()

    runs = [r for r in plan() if not a.groups or r.group in a.groups]
    runs = runs[a.shard::a.num_shards]
    for r in runs:
        r.results_dir, r.data_dir = a.results_dir, a.data_dir
        if a.epochs:
            r.epochs = a.epochs
        if a.train_frac:
            r.train_frac = a.train_frac
    todo = [r for r in runs if not os.path.exists(os.path.join(a.results_dir, f"{r.name}.json"))]
    print(f"{len(runs)} runs in this shard, {len(runs) - len(todo)} already done, {len(todo)} to run")
    for r in runs:
        print(("  [done] " if r not in todo else "  [todo] ") + r.name)
    if a.list:
        return
    for k, r in enumerate(todo, 1):
        print(f"\n=== {k}/{len(todo)}: {r.name}")
        run(r)


if __name__ == "__main__":
    main()
