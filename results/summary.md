# Results summary

## A. Architecture comparison (same data, same protocol, mean ± std over seeds)

| Model | Recurrent params | Total params | Test acc | Test macro-F1 | Neutral F1 | s/epoch | Seeds |
|---|---|---|---|---|---|---|---|
| Vanilla LSTM | 131,584 | 2,398,339 | 0.7917 ± 0.0046 | 0.5298 ± 0.0033 | 0.0092 | 39 | 3 |
| WMC-LSTM | 180,736 | 2,447,491 | 0.7958 ± 0.0017 | 0.5350 ± 0.0037 | 0.0093 | 61 | 3 |
| CIFG-LSTM | 98,688 | 2,365,443 | 0.7950 ± 0.0026 | 0.5391 ± 0.0081 | 0.0264 | 39 | 3 |
| LSTMP | 107,072 | 2,373,635 | 0.7936 ± 0.0006 | 0.5307 ± 0.0032 | 0.0129 | 47 | 3 |

## B_lr: Learning rate (test macro-F1; other settings = base)

| Learning rate | Vanilla LSTM | WMC-LSTM | CIFG-LSTM |
|---|---|---|---|
| 0.0003 | 0.4986 | 0.5086 | 0.5048 |
| 0.001 | 0.5319 | 0.537 | 0.544 |
| 0.003 | 0.5518 | 0.5627 | 0.5694 |

## C_optim: Optimizer (test macro-F1; other settings = base)

| Optimizer | Vanilla LSTM | WMC-LSTM | CIFG-LSTM |
|---|---|---|---|
| adam | 0.5319 | 0.537 | 0.544 |
| adamw | 0.5317 | 0.5374 | 0.5453 |
| rmsprop | 0.5294 | 0.5371 | 0.5406 |
| sgd | 0.4959 | 0.5071 | 0.5046 |

## D_init: Initialisation (test macro-F1; other settings = base)

| Initialisation | Vanilla LSTM | WMC-LSTM | CIFG-LSTM |
|---|---|---|---|
| default | 0.5319 | 0.537 | 0.544 |
| orthogonal | 0.5311 | 0.5362 | 0.5486 |
| orthogonal+fb1 | 0.541 | 0.5552 | 0.5609 |
| xavier | 0.5301 | 0.5376 | 0.5374 |

## F/G. Capacity (parameter-matched, 2-layer) and class-weighted loss: parameter-matched cells and 2-layer stacks

| group | cell | hidden | proj | layers | class_weights | params_recurrent | test_acc | test_f1 |
|---|---|---|---|---|---|---|---|---|
| G_imbalance | cifg | 128 | - | 1 | True | 98688 | 0.7082 | 0.5828 |
| G_imbalance | vanilla | 128 | - | 1 | True | 131584 | 0.7362 | 0.5666 |
| G_imbalance | wmc | 128 | - | 1 | True | 180736 | 0.7151 | 0.5854 |

## Best run (selected on validation macro-F1)

`G_imbalance__cifg_h128_L1_adam_lr0.001_bs64_default_s1_cw` - val F1 0.5939, test acc 0.7082, test macro-F1 0.5828, per-class F1 neg/neu/pos = 0.666 / 0.252 / 0.831
