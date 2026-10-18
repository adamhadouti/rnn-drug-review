"""Drug Review (Drugs.com, UCI #462) data pipeline.

Protocol (identical for every model and every run):
  * label: rating 1-4 -> negative (0), 5-6 -> neutral (1), 7-10 -> positive (2)
  * duplicates: exact duplicate reviews removed inside each official split, and training reviews
    that also appear in the official test split are removed (prevents train/test leakage)
  * test set  = official drugsComTest_raw (deduplicated), never subsampled
  * val set   = stratified 10% of the official train split, fixed seed (data_seed)
  * train     = the remaining 90%, optionally stratified-subsampled with `train_frac`
  * vocabulary built on the training split only
The processed splits are cached; the cache key contains every data parameter, so all runs that share
data parameters read exactly the same tensors.
"""
import glob
import hashlib
import html
import json
import os
import random
import re
import urllib.request
import zipfile
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Sampler

UCI_URLS = [  # UCI has moved this file before; try current and legacy locations
    "https://archive.ics.uci.edu/static/public/462/drug+review+dataset+drugs+com.zip",
    "https://archive.ics.uci.edu/static/public/462/drug+reviews+drugs+com.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip",
]
# Exact mirror of the two raw UCI files (same rows, columns and HTML-escaped text)
HF_MIRROR = "https://huggingface.co/datasets/lewtun/drug-reviews/resolve/main/{split}.jsonl"
LABELS = ["negative", "neutral", "positive"]
PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?.,;:()]")


# ----------------------------------------------------------------------------- raw data
def download(data_dir: str) -> None:
    if _find(data_dir, "train") and _find(data_dir, "test"):
        return
    os.makedirs(data_dir, exist_ok=True)
    zpath = os.path.join(data_dir, "drug_review.zip")
    for url in UCI_URLS:
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(data_dir)
            for inner in glob.glob(os.path.join(data_dir, "**", "*.zip"), recursive=True):
                if inner != zpath:
                    with zipfile.ZipFile(inner) as z:
                        z.extractall(os.path.dirname(inner))
            if _find(data_dir, "train") and _find(data_dir, "test"):
                return
        except Exception as e:  # 404, bad zip, network...
            print(f"  failed: {e}")
    print("UCI unavailable, using the Hugging Face mirror of the same raw files ...")
    for split, name in (("train", "drugsComTrain_raw.tsv"), ("test", "drugsComTest_raw.tsv")):
        df = pd.read_json(HF_MIRROR.format(split=split), lines=True)
        df.to_csv(os.path.join(data_dir, name), sep="\t", index=False)
        print(f"  {name}: {len(df)} rows")


def _find(data_dir: str, split: str):
    hits = [p for ext in ("tsv", "csv") for p in glob.glob(os.path.join(data_dir, "**", f"*{split.capitalize()}*.{ext}"), recursive=True)]
    hits += [p for ext in ("tsv", "csv") for p in glob.glob(os.path.join(data_dir, "**", f"*{split}*.{ext}"), recursive=True)]
    hits = [h for h in dict.fromkeys(hits) if "processed" not in h]
    return hits[0] if hits else None


def load_raw(data_dir: str, split: str) -> pd.DataFrame:
    path = _find(data_dir, split)
    if path is None:
        raise FileNotFoundError(f"No {split} file found under {data_dir}. Run with --download or place the UCI files there.")
    df = pd.read_csv(path, sep="\t" if path.endswith(".tsv") else ",")
    df = df.rename(columns={df.columns[0]: "uniqueID"}) if df.columns[0].startswith("Unnamed") else df
    df = df.dropna(subset=["review", "rating"]).copy()
    df["review"] = df["review"].astype(str).map(clean_text)
    df["rating"] = df["rating"].astype(int)
    df["label"] = df["rating"].map(rating_to_label)
    return df.reset_index(drop=True)


def clean_text(text: str) -> str:
    text = html.unescape(text).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return re.sub(r"\s+", " ", text).strip()


def rating_to_label(r: int) -> int:
    return 0 if r <= 4 else (1 if r <= 6 else 2)


def tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


# ----------------------------------------------------------------------------- processing
def dedupe(train: pd.DataFrame, test: pd.DataFrame):
    stats = {"train_raw": len(train), "test_raw": len(test)}
    train = train.drop_duplicates(subset="review")
    test = test.drop_duplicates(subset="review")
    stats["train_dup_removed"] = stats["train_raw"] - len(train)
    stats["test_dup_removed"] = stats["test_raw"] - len(test)
    overlap = train["review"].isin(set(test["review"]))
    stats["train_test_overlap_removed"] = int(overlap.sum())
    train = train[~overlap]
    return train.reset_index(drop=True), test.reset_index(drop=True), stats


def build_vocab(token_lists, max_size: int, min_freq: int):
    counts = Counter(tok for toks in token_lists for tok in toks)
    itos = [PAD, UNK] + [w for w, c in counts.most_common(max_size - 2) if c >= min_freq]
    return {w: i for i, w in enumerate(itos)}, itos, counts


def encode(token_lists, stoi, max_len: int, truncation: str = "head"):
    unk = stoi[UNK]
    out = []
    for toks in token_lists:
        if len(toks) > max_len:
            toks = toks[:max_len] if truncation == "head" else toks[-max_len:]
        ids = [stoi.get(t, unk) for t in toks] or [unk]
        out.append(torch.tensor(ids, dtype=torch.long))
    return out


def prepare_data(data_dir: str, max_len: int = 200, max_vocab: int = 30000, min_freq: int = 2,
                 val_frac: float = 0.1, train_frac: float = 1.0, data_seed: int = 42,
                 truncation: str = "head", download_if_missing: bool = True) -> dict:
    params = dict(max_len=max_len, max_vocab=max_vocab, min_freq=min_freq, val_frac=val_frac,
                  train_frac=train_frac, data_seed=data_seed, truncation=truncation)
    key = hashlib.md5(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    cache = os.path.join(data_dir, "processed", f"data_{key}.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)

    if download_if_missing:
        download(data_dir)
    train_df, test_df, stats = dedupe(load_raw(data_dir, "train"), load_raw(data_dir, "test"))
    tr_df, va_df = train_test_split(train_df, test_size=val_frac, stratify=train_df["label"], random_state=data_seed)
    if train_frac < 1.0:
        tr_df, _ = train_test_split(tr_df, train_size=train_frac, stratify=tr_df["label"], random_state=data_seed)

    tok = {name: [tokenize(t) for t in df["review"]] for name, df in (("train", tr_df), ("val", va_df), ("test", test_df))}
    stoi, itos, counts = build_vocab(tok["train"], max_vocab, min_freq)
    total = sum(counts.values())
    covered = sum(counts[w] for w in itos[2:])

    data = {"params": params, "itos": itos, "pad_idx": 0, "labels": LABELS, "stats": stats}
    for name, df in (("train", tr_df), ("val", va_df), ("test", test_df)):
        data[name] = {"x": encode(tok[name], stoi, max_len, truncation),
                      "y": torch.tensor(df["label"].values, dtype=torch.long)}
        stats[f"{name}_size"] = len(df)
        stats[f"{name}_class_counts"] = np.bincount(df["label"].values, minlength=3).tolist()
    stats["vocab_size"] = len(itos)
    stats["train_token_coverage"] = round(covered / max(total, 1), 4)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    torch.save(data, cache)
    print(f"Cached processed data -> {cache}\n{json.dumps(stats, indent=1)}")
    return data


# ----------------------------------------------------------------------------- torch utils
class ReviewDataset(Dataset):
    def __init__(self, split: dict):
        self.x, self.y = split["x"], split["y"]
        self.lengths = [len(s) for s in self.x]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def collate(batch, pad_idx: int = 0):
    seqs, ys = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    x = torch.full((len(seqs), int(lengths.max())), pad_idx, dtype=torch.long)
    for k, s in enumerate(seqs):
        x[k, :len(s)] = s
    return x, lengths, torch.stack(ys)


class BucketBatchSampler(Sampler):
    """Groups reviews of similar length into a batch so padding (and the Python time loop) is short.
    Training: shuffled pools of `pool` batches, sorted by length inside the pool, batches shuffled.
    Evaluation: fully sorted, deterministic."""

    def __init__(self, lengths, batch_size: int, shuffle: bool, seed: int = 0, pool: int = 50):
        self.lengths, self.bs, self.shuffle, self.pool = lengths, batch_size, shuffle, pool
        self.rng = random.Random(seed)

    def __iter__(self):
        idx = list(range(len(self.lengths)))
        if not self.shuffle:
            idx.sort(key=lambda i: self.lengths[i])
            batches = [idx[k:k + self.bs] for k in range(0, len(idx), self.bs)]
        else:
            self.rng.shuffle(idx)
            batches, chunk = [], self.bs * self.pool
            for k in range(0, len(idx), chunk):
                pool = sorted(idx[k:k + chunk], key=lambda i: self.lengths[i])
                batches += [pool[j:j + self.bs] for j in range(0, len(pool), self.bs)]
            self.rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return (len(self.lengths) + self.bs - 1) // self.bs
