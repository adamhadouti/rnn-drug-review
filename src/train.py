"""Train / evaluate one model configuration.

Example:
    python -m src.train --cell wmc --hidden 128 --lr 1e-3 --optimizer adam --seed 1
Every run writes results/<name>.json (config, parameter counts, per-epoch history, test metrics)
and, if tensorboard is installed, logs to runs/<name>.
"""
import argparse
import copy
import json
import os
import random
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from .data import BucketBatchSampler, ReviewDataset, collate, prepare_data
from .model import LSTMClassifier, init_weights, param_breakdown


@dataclass
class Config:
    # model
    cell: str = "vanilla"            # vanilla | wmc | cifg | lstmp
    emb_dim: int = 128
    hidden: int = 128
    proj: int = 64                   # LSTMP projection size P (ignored by other cells)
    layers: int = 1
    dropout: float = 0.3
    emb_dropout: float = 0.2
    pooling: str = "last"            # last | mean | max
    # optimisation
    optimizer: str = "adam"          # adam | adamw | sgd | rmsprop
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9            # sgd only
    batch_size: int = 64
    epochs: int = 10
    patience: int = 3                # early stopping on validation macro-F1
    clip: float = 1.0                # gradient-norm clipping (0 = off)
    init: str = "default"            # default | xavier | orthogonal
    forget_bias: float = 0.0
    class_weights: bool = False      # weight the loss by inverse class frequency
    # data (identical across runs unless changed on purpose)
    data_dir: str = "data"
    max_len: int = 200
    max_vocab: int = 30000
    min_freq: int = 2
    train_frac: float = 1.0
    data_seed: int = 42
    # bookkeeping
    seed: int = 1
    name: str = ""
    group: str = "manual"
    results_dir: str = "results"
    log_dir: str = "runs"
    save_model: bool = False
    num_workers: int = 2


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_name(cfg: Config) -> str:
    parts = [cfg.cell, f"h{cfg.hidden}"]
    if cfg.cell == "lstmp":
        parts.append(f"p{cfg.proj}")
    parts += [f"L{cfg.layers}", cfg.optimizer, f"lr{cfg.lr:g}", f"bs{cfg.batch_size}", cfg.init]
    if cfg.forget_bias:
        parts.append(f"fb{cfg.forget_bias:g}")
    parts.append(f"s{cfg.seed}")
    return "_".join(parts)


def make_optimizer(cfg: Config, params):
    if cfg.optimizer == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay or 0.01)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, nesterov=True, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError(cfg.optimizer)


def make_loader(split, batch_size, shuffle, seed, num_workers):
    ds = ReviewDataset(split)
    sampler = BucketBatchSampler(ds.lengths, batch_size, shuffle=shuffle, seed=seed)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def run_epoch(model, loader, criterion, device, optimizer=None, clip=0.0):
    train = optimizer is not None
    model.train(train)
    tot_loss, n, preds, golds, gnorms = 0.0, 0, [], [], []
    with torch.set_grad_enabled(train):
        for x, lengths, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x, lengths)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gn = nn.utils.clip_grad_norm_(model.parameters(), clip if clip > 0 else float("inf"))
                gnorms.append(float(gn))
                optimizer.step()
            tot_loss += loss.item() * y.size(0)
            n += y.size(0)
            preds.append(logits.argmax(-1).cpu())
            golds.append(y.cpu())
    preds, golds = torch.cat(preds).numpy(), torch.cat(golds).numpy()
    out = {"loss": tot_loss / n, "acc": accuracy_score(golds, preds), "macro_f1": f1_score(golds, preds, average="macro")}
    if gnorms:
        out["grad_norm"] = float(np.mean(gnorms))
    return out, preds, golds


def run(cfg: Config) -> dict:
    cfg.name = cfg.name or default_name(cfg)
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.results_dir, f"{cfg.name}.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = prepare_data(cfg.data_dir, max_len=cfg.max_len, max_vocab=cfg.max_vocab, min_freq=cfg.min_freq,
                        train_frac=cfg.train_frac, data_seed=cfg.data_seed)
    set_seed(cfg.seed)
    train_loader = make_loader(data["train"], cfg.batch_size, True, cfg.seed, cfg.num_workers)
    val_loader = make_loader(data["val"], 256, False, 0, cfg.num_workers)
    test_loader = make_loader(data["test"], 256, False, 0, cfg.num_workers)

    model = LSTMClassifier(len(data["itos"]), len(data["labels"]), cfg.cell, cfg.emb_dim, cfg.hidden,
                           cfg.layers, cfg.proj if cfg.cell == "lstmp" else None, cfg.dropout,
                           cfg.emb_dropout, cfg.pooling, data["pad_idx"])
    init_weights(model, cfg.init, cfg.forget_bias)
    model.to(device)
    params = param_breakdown(model)

    weight = None
    if cfg.class_weights:
        counts = torch.bincount(data["train"]["y"], minlength=len(data["labels"])).float()
        weight = (counts.sum() / (len(counts) * counts)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = make_optimizer(cfg, model.parameters())

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(cfg.log_dir, cfg.name))
    except Exception:
        pass

    print(f"[{cfg.name}] device={device} params={params}")
    history, best_f1, best_state, best_epoch, bad = [], -1.0, None, 0, 0
    t_start = time.time()
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, cfg.clip)
        va, _, _ = run_epoch(model, val_loader, criterion, device)
        rec = {"epoch": epoch, "time": time.time() - t0,
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(rec)
        print(f"  ep{epoch:02d} train loss {tr['loss']:.4f} acc {tr['acc']:.4f} | val loss {va['loss']:.4f} "
              f"acc {va['acc']:.4f} F1 {va['macro_f1']:.4f} | {rec['time']:.0f}s", flush=True)
        if writer:
            for k, v in rec.items():
                if k != "epoch":
                    writer.add_scalar(k.replace("_", "/", 1), v, epoch)
        if not np.isfinite(tr["loss"]):
            print("  loss diverged, stopping")
            break
        if va["macro_f1"] > best_f1:
            best_f1, best_epoch, bad = va["macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"  early stop (best epoch {best_epoch})")
                break
    train_time = time.time() - t_start

    if best_state is not None:
        model.load_state_dict(best_state)
    te, preds, golds = run_epoch(model, test_loader, criterion, device)
    report = classification_report(golds, preds, target_names=data["labels"], output_dict=True, zero_division=0)
    result = {
        "name": cfg.name, "group": cfg.group, "config": asdict(cfg), "params": params, "device": str(device),
        "best_epoch": best_epoch, "epochs_run": len(history), "train_time_s": train_time,
        "sec_per_epoch": train_time / max(len(history), 1), "best_val_macro_f1": best_f1,
        "test": {"loss": te["loss"], "acc": te["acc"], "macro_f1": te["macro_f1"],
                 "weighted_f1": f1_score(golds, preds, average="weighted"),
                 "per_class_f1": {c: report[c]["f1-score"] for c in data["labels"]},
                 "confusion_matrix": confusion_matrix(golds, preds).tolist()},
        "history": history, "data_stats": data["stats"],
    }
    if writer:
        writer.add_hparams({k: v for k, v in asdict(cfg).items() if isinstance(v, (int, float, str, bool))},
                           {"hparam/test_macro_f1": te["macro_f1"], "hparam/test_acc": te["acc"]})
        writer.close()
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    if cfg.save_model:
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), os.path.join("checkpoints", f"{cfg.name}.pt"))
    print(f"[{cfg.name}] TEST acc {te['acc']:.4f} macro-F1 {te['macro_f1']:.4f} -> {out_path}")
    return result


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for f in fields(Config):
        if f.type is bool or f.type == "bool":
            p.add_argument(f"--{f.name}", action="store_true", default=f.default)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type, f.type) if isinstance(f.type, str) else f.type
            p.add_argument(f"--{f.name}", type=typ, default=f.default)
    return Config(**vars(p.parse_args(argv)))


if __name__ == "__main__":
    run(parse_args())
