"""Correctness tests for the custom cells. Run: python -m pytest tests -q   (or python tests/test_cells.py)

nn.LSTMCell is used ONLY here, as an independent reference to check the vanilla cell."""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cells import CIFGLSTMCell, LSTMPCell, RNNLayer, VanillaLSTMCell, WMCLSTMCell, expected_cell_params, make_cell  # noqa: E402
from src.model import LSTMClassifier, count_params, init_weights  # noqa: E402

torch.manual_seed(0)
B, E, H, P = 5, 7, 6, 3
sig, tanh = torch.sigmoid, torch.tanh


def blocks(lin, n):
    return lin.weight.chunk(n, 0)


def test_vanilla_matches_torch_lstmcell():
    cell = VanillaLSTMCell(E, H)
    ref = nn.LSTMCell(E, H)  # PyTorch gate order is also [i, f, g, o]
    with torch.no_grad():
        ref.weight_ih.copy_(cell.x2h.weight); ref.bias_ih.copy_(cell.x2h.bias)
        ref.weight_hh.copy_(cell.h2h.weight); ref.bias_hh.zero_()
    x, h, c = torch.randn(B, E), torch.randn(B, H), torch.randn(B, H)
    h1, (_, c1) = cell(x, (h, c))
    h2, c2 = ref(x, (h, c))
    assert torch.allclose(h1, h2, atol=1e-6) and torch.allclose(c1, c2, atol=1e-6)


def test_wmc_matches_equations():
    cell = WMCLSTMCell(E, H)
    x, h, c = torch.randn(B, E), torch.randn(B, H), torch.randn(B, H)
    Wxi, Wxf, Wxg, Wxo = blocks(cell.x2h, 4); bi, bf, bg, bo = cell.x2h.bias.chunk(4)
    Whi, Whf, Whg, Who = blocks(cell.h2h, 4)
    Wci, Wcf = blocks(cell.c2if, 2); Wco = cell.c2o.weight
    i = sig(x @ Wxi.T + h @ Whi.T + tanh(c @ Wci.T) + bi)
    f = sig(x @ Wxf.T + h @ Whf.T + tanh(c @ Wcf.T) + bf)
    g = tanh(h @ Whg.T + x @ Wxg.T + bg)
    c_new = f * c + i * g
    o = sig(x @ Wxo.T + h @ Who.T + tanh(c_new @ Wco.T) + bo)  # eq (9): current c_t
    h_new = o * tanh(c_new)
    h1, (_, c1) = cell(x, (h, c))
    assert torch.allclose(h1, h_new, atol=1e-6) and torch.allclose(c1, c_new, atol=1e-6)


def test_cifg_matches_equations():
    cell = CIFGLSTMCell(E, H)
    x, h, c = torch.randn(B, E), torch.randn(B, H), torch.randn(B, H)
    Wxf, Wxg, Wxo = blocks(cell.x2h, 3); bf, bg, bo = cell.x2h.bias.chunk(3)
    Whf, Whg, Who = blocks(cell.h2h, 3)
    f = sig(h @ Whf.T + x @ Wxf.T + bf)
    i = 1 - f
    o = sig(h @ Who.T + x @ Wxo.T + bo)
    g = tanh(h @ Whg.T + x @ Wxg.T + bg)
    c_new = f * c + i * g
    h1, (_, c1) = cell(x, (h, c))
    assert torch.allclose(h1, o * tanh(c_new), atol=1e-6) and torch.allclose(c1, c_new, atol=1e-6)


def test_lstmp_matches_equations():
    cell = LSTMPCell(E, H, P)
    x, h, c = torch.randn(B, E), torch.randn(B, P), torch.randn(B, H)
    Wxi, Wxf, Wxg, Wxo = blocks(cell.x2h, 4); bi, bf, bg, bo = cell.x2h.bias.chunk(4)
    Whi, Whf, Whg, Who = blocks(cell.h2h, 4)
    i, f, o = sig(x @ Wxi.T + h @ Whi.T + bi), sig(x @ Wxf.T + h @ Whf.T + bf), sig(x @ Wxo.T + h @ Who.T + bo)
    g = tanh(x @ Wxg.T + h @ Whg.T + bg)
    c_new = f * c + i * g
    m = o * tanh(c_new)
    h_new = m @ cell.proj.weight.T + cell.proj.bias
    h1, (_, c1) = cell(x, (h, c))
    assert h1.shape == (B, P) and torch.allclose(h1, h_new, atol=1e-6) and torch.allclose(c1, c_new, atol=1e-6)


def test_param_counts():
    for name in ("vanilla", "wmc", "cifg", "lstmp"):
        cell = make_cell(name, E, H, P)
        assert count_params(cell) == expected_cell_params(name, E, H, P), name


def test_no_builtin_lstm_used():
    for name in ("vanilla", "wmc", "cifg", "lstmp"):
        m = LSTMClassifier(50, 3, name, E, H, 2, P if name == "lstmp" else None)
        assert not any(isinstance(mod, (nn.LSTM, nn.LSTMCell, nn.RNNBase)) for mod in m.modules())


def test_layer_unroll_equals_steps():
    cell = WMCLSTMCell(E, H)
    x = torch.randn(B, 4, E)
    out, _ = RNNLayer(cell)(x)
    state = cell.init_state(B, x.device, x.dtype)
    for t in range(4):
        h, state = cell(x[:, t], state)
        assert torch.allclose(out[:, t], h, atol=1e-6)


def test_padding_invariance():
    """A review must get the same logits alone or padded inside a batch with a longer review."""
    for name in ("vanilla", "wmc", "cifg", "lstmp"):
        for pooling in ("last", "mean", "max"):
            m = LSTMClassifier(30, 3, name, E, H, 2, P if name == "lstmp" else None, pooling=pooling).eval()
            short, long_ = torch.randint(2, 30, (1, 4)), torch.randint(2, 30, (1, 9))
            alone = m(short, torch.tensor([4]))
            batch = torch.zeros(2, 9, dtype=torch.long)
            batch[0, :4], batch[1] = short[0], long_[0]
            padded = m(batch, torch.tensor([4, 9]))[0:1]
            assert torch.allclose(alone, padded, atol=1e-5), (name, pooling)


def test_inits_and_backward():
    for name in ("vanilla", "wmc", "cifg", "lstmp"):
        for scheme in ("default", "xavier", "orthogonal"):
            m = LSTMClassifier(30, 3, name, E, H, 2, P if name == "lstmp" else None)
            init_weights(m, scheme, forget_bias=1.0)
            k = m.rnn[0].cell.forget_gate_index
            if scheme != "default":
                assert torch.allclose(m.rnn[0].cell.x2h.bias[k * H:(k + 1) * H], torch.ones(H))
            loss = m(torch.randint(2, 30, (3, 6)), torch.tensor([6, 3, 1])).sum()
            loss.backward()
            assert all(p.grad is not None for p in m.rnn.parameters())


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
