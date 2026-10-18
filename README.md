# LSTM Variants for Drug Review Sentiment Classification

Advanced NLP lab project (UIR, 2026/2027). We compare a **vanilla LSTM** baseline with two non-vanilla cells, **Working Memory Connections (WMC)** and the **Coupled Input-Forget Gate (CIFG)** LSTM, plus **LSTMP** (projection) as a bonus. The task is 3-class sentiment classification on the UCI Drugs.com review dataset. Every cell is written from scratch with `nn.Linear`. No `nn.LSTM` or `nn.LSTMCell` is used anywhere in the models.

## Task and protocol

| | |
|---|---|
| Data | [UCI Drug Review #462](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com), downloaded automatically |
| Labels | rating 1–4 → negative, 5–6 → neutral, 7–10 → positive |
| Cleaning | HTML entities unescaped; exact duplicate reviews removed; train reviews that also appear in the test split removed (no leakage) |
| Splits | official test split = **test** (never subsampled); official train split → stratified 90 % **train** / 10 % **val**, fixed `data_seed=42` |
| Vocabulary | built on train only, top 30k tokens, min freq 2; reviews truncated to 200 tokens |
| Selection | early stopping and model selection on **validation macro-F1**; test is touched once per run |
| Metrics | accuracy, **macro-F1** (main metric, because of class imbalance), per-class F1, confusion matrix |
| Complexity | total / embedding / recurrent / classifier trainable parameters reported for every run |

All runs read the same cached tensors, so every model sees identical splits.

## Repository layout

```
src/cells.py        the four cells (eqs 1–25) + time unroller
src/model.py        Embedding → stacked cells → pooling → Linear; weight-init schemes; param counts
src/data.py         download, cleaning, dedup, fixed splits, vocab, length-bucketed batching
src/train.py        train/evaluate one config → results/<name>.json + TensorBoard logs
run_experiments.py  the full experiment plan (51 runs), shardable across team members, resumable
src/eda.py          EDA figures + statistics for the report
src/aggregate.py    results → summary.csv, summary.md (report tables) and figures
tests/              correctness tests (cells vs. equations), synthetic-data generator
colab_runner.ipynb  one-click Colab workflow
```

## Mapping from equations to code

| Cell | Gates / state | Recurrent parameters (one layer, input E, hidden H) |
|---|---|---|
| Vanilla (1–6) | i, f, o, g | 4H(E+H+1) |
| WMC (7–12) | i, f use tanh(W_c c_{t-1}); **o uses tanh(W_co c_t)**, so c_t is computed before o_t | 4H(E+H+1) + 3H² |
| CIFG (13–18) | i_t = 1 − f_t (no input-gate weights) | 3H(E+H+1) |
| LSTMP (19–25) | h_t = W_p m_t + b_p, recurrence on the P-dim h | 4H(E+P+1) + PH + P |

Two implementation choices are mathematically equivalent to the equations, just faster:

- The gate matrices are block-stacked into one `nn.Linear` per input (x → gates, h → gates).
- W_x x_t is computed for all time steps at once, before the time loop.

Only one bias per gate is used, as in the equations. `tests/test_cells.py` recomputes every cell directly from its equations and checks:

- the vanilla cell matches `torch.nn.LSTMCell` (used only as a test reference);
- the parameter counts above;
- padding invariance.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                      # cell correctness
python -m src.eda                              # downloads data, writes figures/eda_*.png
python -m src.train --cell wmc --epochs 10     # one run
python run_experiments.py --list               # the full plan
python run_experiments.py --shard 0 --num_shards 6 --results_dir <shared folder>
python -m src.aggregate --results_dir <shared folder>
```

On Colab, open `colab_runner.ipynb`, set your shard index, and run all cells. Results are written to Google Drive, and finished runs are skipped, so a disconnect just resumes. Offline smoke test: `python tests/make_fake_data.py data_fake && python run_experiments.py --data_dir data_fake --epochs 1`.

## Experiment plan

Base config: emb 128, H 128 (LSTMP P 64), 1 layer, Adam lr 1e-3, batch 64, dropout 0.3, PyTorch default init, gradient clipping 1.0, ≤10 epochs with patience 3. Each sweep changes **one** setting from the base. The base point of every sweep is the seed-1 run of group A.

| Group | What varies | Cells | Runs |
|---|---|---|---|
| A_arch | architecture, seeds 1/2/3 (mean ± std) | vanilla, WMC, CIFG, LSTMP | 12 |
| B_lr | learning rate 3e-4 / 1e-3 / 3e-3 | vanilla, WMC, CIFG | 6 |
| C_optim | Adam / AdamW / RMSprop / SGD-Nesterov (lr 0.05) | " | 9 |
| D_init | default / Xavier / orthogonal / orthogonal + forget-bias 1 | " | 9 |
| E_batch | batch size 32 / 64 / 256 | " | 6 |
| F_capacity | parameter-matched hidden sizes; LSTMP H256-P64; 2-layer stacks | all | 6 |
| G_imbalance | class-weighted cross-entropy | vanilla, WMC, CIFG | 3 |

If GPU time is short, add `--train_frac 0.5`. This subsamples the training split stratified by class, so the class balance is preserved, and validation and test stay complete. Everyone must use the same value.

## Outputs for the report

- `figures/eda_*.png`: rating and class distribution, review lengths, top conditions.
- `figures/arch_*.png`: accuracy vs. recurrent parameters, and learning curves.
- `figures/sweep_*.png`: one figure per hyperparameter sweep.
- `figures/best_confusion.png`: confusion matrix of the best model.
- `results/summary.md`: report-ready tables. `results/summary.csv` holds every run.

## References

- Landi et al., *Working Memory Connections for LSTM*, Neural Networks 2021.
- Greff et al., *LSTM: A Search Space Odyssey*, 2015 (CIFG).
- Sak et al., *LSTM-based RNN architectures for large vocabulary speech recognition*, Interspeech 2014 (LSTMP).
- Gräßer et al., *Aspect-Based Sentiment Analysis of Drug Reviews*, 2018 (dataset).
