"""Collect results/*.json -> results/summary.csv, results/summary.md (report-ready tables) and figures.

    python -m src.aggregate --results_dir results
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from .plotstyle import CELL_COLORS, CELL_LABELS, setup

CELL_ORDER = ["vanilla", "wmc", "cifg", "lstmp"]
SWEEPS = {"B_lr": ("lr", "Learning rate"), "C_optim": ("optimizer", "Optimizer"),
          "D_init": ("init_label", "Initialisation"), "E_batch": ("batch_size", "Batch size")}


def load(results_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        r = json.load(open(path))
        c = r["config"]
        rows.append({
            "name": r["name"], "group": r["group"], "cell": c["cell"], "hidden": c["hidden"],
            "proj": c["proj"] if c["cell"] == "lstmp" else None, "layers": c["layers"],
            "optimizer": c["optimizer"], "lr": c["lr"], "batch_size": c["batch_size"], "init": c["init"],
            "forget_bias": c["forget_bias"], "seed": c["seed"], "class_weights": c["class_weights"],
            "init_label": c["init"] + (f"+fb{c['forget_bias']:g}" if c["forget_bias"] else ""),
            "params_total": r["params"]["total"], "params_recurrent": r["params"]["recurrent"],
            "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "sec_per_epoch": r["sec_per_epoch"],
            "val_f1": r["best_val_macro_f1"], "test_acc": r["test"]["acc"], "test_f1": r["test"]["macro_f1"],
            **{f"f1_{k}": v for k, v in r["test"]["per_class_f1"].items()},
            "_history": r["history"], "_cm": r["test"]["confusion_matrix"],
        })
    return pd.DataFrame(rows)


def base_rows(df):
    """Group A seed-1 runs double as the BASE point of every one-at-a-time sweep."""
    b = df[(df.group == "A_arch") & (df.seed == 1)].copy()
    return b


def md_table(d: pd.DataFrame) -> str:
    cols = list(d.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results")
    p.add_argument("--out", default="figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt = setup()
    df = load(a.results_dir)
    if df.empty:
        raise SystemExit(f"No results in {a.results_dir}")
    df.drop(columns=["_history", "_cm"]).to_csv(os.path.join(a.results_dir, "summary.csv"), index=False)
    md = ["# Results summary\n"]

    # ---------------- A: architecture comparison (mean +- std over seeds)
    A = df[df.group == "A_arch"]
    if not A.empty:
        g = A.groupby("cell").agg(n=("seed", "count"), params_total=("params_total", "first"),
                                  params_recurrent=("params_recurrent", "first"),
                                  acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
                                  f1_mean=("test_f1", "mean"), f1_std=("test_f1", "std"),
                                  neutral_f1=("f1_neutral", "mean"), sec_epoch=("sec_per_epoch", "mean"))
        g = g.reindex([c for c in CELL_ORDER if c in g.index]).fillna(0)
        tab = pd.DataFrame({
            "Model": [CELL_LABELS[c] for c in g.index],
            "Recurrent params": [f"{v:,}" for v in g.params_recurrent],
            "Total params": [f"{v:,}" for v in g.params_total],
            "Test acc": [f"{m:.4f} ± {s:.4f}" for m, s in zip(g.acc_mean, g.acc_std)],
            "Test macro-F1": [f"{m:.4f} ± {s:.4f}" for m, s in zip(g.f1_mean, g.f1_std)],
            "Neutral F1": [f"{v:.4f}" for v in g.neutral_f1],
            "s/epoch": [f"{v:.0f}" for v in g.sec_epoch],
            "Seeds": g.n.astype(int).tolist(),
        })
        md += ["## A. Architecture comparison (same data, same protocol, mean ± std over seeds)\n", md_table(tab), ""]

        fig, ax = plt.subplots(figsize=(5.6, 3.6))
        for c, r in g.iterrows():
            ax.errorbar(r.params_recurrent / 1e3, r.f1_mean, yerr=r.f1_std, fmt="o", ms=8, capsize=4,
                        color=CELL_COLORS[c], label=CELL_LABELS[c])
            ax.annotate(CELL_LABELS[c], (r.params_recurrent / 1e3, r.f1_mean), xytext=(8, 4),
                        textcoords="offset points", fontsize=8)
        ax.set(xlabel="Recurrent parameters (thousands)", ylabel="Test macro-F1",
               title="Accuracy vs. recurrent complexity")
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
        ax.margins(x=0.15)
        fig.savefig(os.path.join(a.out, "arch_params_vs_f1.png")); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        for c in [c for c in CELL_ORDER if c in A.cell.values]:
            h = pd.DataFrame(A[(A.cell == c)].sort_values("seed").iloc[0]["_history"])
            axes[0].plot(h.epoch, h.train_loss, color=CELL_COLORS[c], label=CELL_LABELS[c])
            axes[0].plot(h.epoch, h.val_loss, color=CELL_COLORS[c], ls="--")
            axes[1].plot(h.epoch, h.val_macro_f1, color=CELL_COLORS[c], marker="o", ms=4, label=CELL_LABELS[c])
        axes[0].set(xlabel="Epoch", ylabel="Loss", title="Train (solid) / val (dashed) loss")
        axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1", title="Validation macro-F1")
        axes[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "arch_curves.png")); plt.close(fig)

    # ---------------- B-E: one-at-a-time hyper-parameter sweeps
    base = base_rows(df)
    for group, (col, title) in SWEEPS.items():
        S = df[df.group == group]
        if S.empty:
            continue
        S = pd.concat([S, base[base.cell.isin(S.cell.unique())]])
        piv = S.pivot_table(index=col, columns="cell", values="test_f1", aggfunc="mean")
        piv = piv[[c for c in CELL_ORDER if c in piv.columns]]
        piv.index = piv.index.map(lambda v: f"{v:g}" if isinstance(v, float) else str(v))
        t = piv.copy().round(4).reset_index().rename(columns={col: title, **CELL_LABELS})
        md += [f"## {group}: {title} (test macro-F1; other settings = base)\n", md_table(t), ""]

        fig, ax = plt.subplots(figsize=(6, 3.4))
        n, w = len(piv.columns), 0.8 / max(len(piv.columns), 1)
        x = np.arange(len(piv.index))
        for k, c in enumerate(piv.columns):
            ax.bar(x + (k - (n - 1) / 2) * w, piv[c].values, w * 0.92, color=CELL_COLORS[c], label=CELL_LABELS[c])
        lo = np.nanmin(piv.values)
        ax.set_ylim(max(0, lo - 0.05), min(1.0, np.nanmax(piv.values)) + 0.01)
        ax.set(xticks=x, xticklabels=piv.index, xlabel=title, ylabel="Test macro-F1", title=f"Effect of {title.lower()}")
        ax.legend(fontsize=8, ncol=n, loc="lower center", bbox_to_anchor=(0.5, 1.08)); ax.grid(axis="x", visible=False)
        fig.savefig(os.path.join(a.out, f"sweep_{group}.png")); plt.close(fig)

    # ---------------- F: capacity
    F = df[df.group.isin(["F_capacity", "G_imbalance"])]
    if not F.empty:
        t = F[["cell", "hidden", "proj", "layers", "class_weights", "params_recurrent", "test_acc", "test_f1"]].copy()
        t["test_acc"], t["test_f1"] = t.test_acc.round(4), t.test_f1.round(4)
        t.insert(0, "group", F.group.values)
        md += ["## F/G. Capacity (parameter-matched, 2-layer) and class-weighted loss: parameter-matched cells and 2-layer stacks\n", md_table(t.fillna("-")), ""]

    # ---------------- best overall run
    best = df.loc[df.val_f1.idxmax()]  # selected on VALIDATION, reported on test
    md += [f"## Best run (selected on validation macro-F1)\n", f"`{best['name']}` - val F1 {best.val_f1:.4f}, "
           f"test acc {best.test_acc:.4f}, test macro-F1 {best.test_f1:.4f}, "
           f"per-class F1 neg/neu/pos = {best.f1_negative:.3f} / {best.f1_neutral:.3f} / {best.f1_positive:.3f}", ""]
    cm = np.array(best["_cm"], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    labs = ["negative", "neutral", "positive"]
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > 0.5 else "#1f1f1e")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=labs, yticklabels=labs, xlabel="Predicted", ylabel="True",
           title=f"Best model: {CELL_LABELS[best.cell]}")
    ax.grid(False)
    fig.savefig(os.path.join(a.out, "best_confusion.png")); plt.close(fig)

    with open(os.path.join(a.results_dir, "summary.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
