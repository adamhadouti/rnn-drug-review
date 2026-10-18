"""Sentiment classifier: Embedding -> stacked custom LSTM layers -> pooling -> Linear."""
import math

import torch
import torch.nn as nn

from .cells import RNNLayer, make_cell


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        cell_type: str = "vanilla",
        emb_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        proj_size: int = None,
        dropout: float = 0.3,
        emb_dropout: float = 0.2,
        pooling: str = "last",
        pad_idx: int = 0,
    ):
        super().__init__()
        self.pooling = pooling
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_dropout = nn.Dropout(emb_dropout)
        layers, in_dim = [], emb_dim
        for _ in range(num_layers):
            cell = make_cell(cell_type, in_dim, hidden_size, proj_size)
            layers.append(RNNLayer(cell))
            in_dim = cell.output_size
        self.rnn = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.emb_dropout(self.embedding(tokens))
        for k, layer in enumerate(self.rnn):
            x, _ = layer(x)
            if k < len(self.rnn) - 1:
                x = self.dropout(x)
        return self.fc(self.dropout(self.pool(x, lengths)))

    def pool(self, out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = out.shape
        lengths = lengths.to(out.device)
        if self.pooling == "last":  # hidden state at the last *valid* time step
            return out[torch.arange(B, device=out.device), lengths - 1]
        mask = (torch.arange(T, device=out.device)[None, :] < lengths[:, None]).unsqueeze(-1)
        if self.pooling == "mean":
            return (out * mask).sum(1) / lengths[:, None].clamp(min=1)
        if self.pooling == "max":
            return out.masked_fill(~mask, float("-inf")).max(1).values
        raise ValueError(self.pooling)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def param_breakdown(model: LSTMClassifier) -> dict:
    return {
        "total": count_params(model),
        "embedding": count_params(model.embedding),
        "recurrent": count_params(model.rnn),
        "classifier": count_params(model.fc),
    }


@torch.no_grad()
def init_weights(model: LSTMClassifier, scheme: str = "default", forget_bias: float = 0.0):
    """Weight initialisation schemes compared in the hyper-parameter study.

    default    : PyTorch nn.Linear default (Kaiming-uniform weights, uniform bias)
    xavier     : Xavier-uniform on every recurrent-cell weight, zero biases
    orthogonal : orthogonal init per gate block on recurrent (h->gates, c->gates) weights,
                 Xavier-uniform on input weights, zero biases
    forget_bias: value added to the forget-gate bias (1.0 is the classic Jozefowicz et al. trick)
    """
    for layer in model.rnn:
        cell = layer.cell
        H = cell.hidden_size
        if scheme in ("xavier", "orthogonal"):
            for name, lin in cell.named_children():
                if not isinstance(lin, nn.Linear):
                    continue
                recurrent = name != "x2h" and name != "proj"
                if scheme == "orthogonal" and recurrent:
                    n_blocks = lin.weight.shape[0] // H
                    for b in range(n_blocks):  # each gate block gets its own orthogonal matrix
                        nn.init.orthogonal_(lin.weight[b * H:(b + 1) * H])
                else:
                    nn.init.xavier_uniform_(lin.weight)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
        elif scheme != "default":
            raise ValueError(f"Unknown init scheme '{scheme}'")
        if forget_bias:
            k = cell.forget_gate_index
            cell.x2h.bias[k * H:(k + 1) * H] += forget_bias


def matched_hidden_size(cell_type: str, target_params: int, E: int, P: int = None) -> int:
    """Hidden size H giving a 1-layer cell with roughly `target_params` parameters
    (used for the parameter-matched comparison)."""
    from .cells import expected_cell_params

    best, best_gap = 1, math.inf
    for H in range(1, 2048):
        gap = abs(expected_cell_params(cell_type, E, H, P) - target_params)
        if gap < best_gap:
            best, best_gap = H, gap
    return best
