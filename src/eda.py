"""Exploratory data analysis -> figures/eda_*.png and figures/eda_summary.json

    python -m src.eda --data_dir data
"""
import argparse
import json
import os

import numpy as np

from .data import LABELS, dedupe, download, load_raw, tokenize
from .plotstyle import CLASS_COLORS, SEQ, setup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out", default="figures")
    p.add_argument("--max_len", type=int, default=200)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    plt = setup()

    download(a.data_dir)
    train_raw, test_raw = load_raw(a.data_dir, "train"), load_raw(a.data_dir, "test")
    train, test, dup = dedupe(train_raw, test_raw)
    for df in (train, test):
        df["n_tokens"] = df["review"].map(lambda t: len(tokenize(t)))
    lens = train["n_tokens"].values

    summary = {
        "dedup": dup,
        "train_size": len(train), "test_size": len(test),
        "n_drugs": int(train["drugName"].nunique()), "n_conditions": int(train["condition"].nunique()),
        "rating_counts_train": train["rating"].value_counts().sort_index().to_dict(),
        "class_share_train": (train["label"].value_counts(normalize=True).sort_index().round(4)).tolist(),
        "class_share_test": (test["label"].value_counts(normalize=True).sort_index().round(4)).tolist(),
        "tokens": {"mean": float(lens.mean()), "median": float(np.median(lens)),
                   "p90": float(np.percentile(lens, 90)), "p95": float(np.percentile(lens, 95)),
                   "max": int(lens.max()), "share_truncated_at_max_len": float((lens > a.max_len).mean())},
        "mean_tokens_by_class": train.groupby("label")["n_tokens"].mean().round(1).tolist(),
        "top_conditions": train["condition"].value_counts().head(10).to_dict(),
    }

    # 1. rating distribution (1-10), bars coloured by the 3-class mapping
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    rc = train["rating"].value_counts().sort_index()
    colors = [CLASS_COLORS[0 if r <= 4 else 1 if r <= 6 else 2] for r in rc.index]
    ax.bar(rc.index, rc.values, color=colors, width=0.8)
    from matplotlib.patches import Patch
    ax.set(xticks=range(1, 11), xlabel="Rating", ylabel="Reviews (train)", title="Rating distribution and 3-class mapping")
    ax.legend(handles=[Patch(color=CLASS_COLORS[k], label=lab) for k, lab in enumerate(LABELS)], title="Class", loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(a.out, "eda_ratings.png")); plt.close(fig)

    # 2. class share train vs test
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(3)
    tr, te = summary["class_share_train"], summary["class_share_test"]
    ax.bar(x - 0.2, tr, 0.38, color=SEQ, label="train")
    ax.bar(x + 0.2, te, 0.38, color="#86b6ef", label="test")
    for xi, v in zip(x - 0.2, tr):
        ax.text(xi, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    ax.set(xticks=x, xticklabels=LABELS, ylabel="Share of reviews", title="Class imbalance (same in both splits)")
    ax.legend(); ax.grid(axis="x", visible=False)
    fig.savefig(os.path.join(a.out, "eda_classes.png")); plt.close(fig)

    # 3. review length
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.hist(np.clip(lens, 0, 400), bins=80, color=SEQ)
    ax.axvline(a.max_len, color="#1f1f1e", lw=1.2, ls="--")
    ax.text(a.max_len + 5, ax.get_ylim()[1] * 0.85,
            f"max_len={a.max_len}\n{summary['tokens']['share_truncated_at_max_len']:.1%} truncated", fontsize=8)
    ax.set(xlabel="Tokens per review (clipped at 400)", ylabel="Reviews", title="Review length")
    fig.savefig(os.path.join(a.out, "eda_lengths.png")); plt.close(fig)

    # 4. top conditions
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    top = train["condition"].value_counts().head(12)[::-1]
    ax.barh(top.index, top.values, color=SEQ)
    ax.set(xlabel="Reviews (train)", title="Most reviewed conditions"); ax.grid(axis="y", visible=False)
    fig.savefig(os.path.join(a.out, "eda_conditions.png")); plt.close(fig)

    with open(os.path.join(a.out, "eda_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
