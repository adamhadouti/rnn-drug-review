"""
LSTM cell variants implemented from scratch with nn.Linear only (no nn.LSTM / nn.LSTMCell).

Implementation notes
--------------------
* Gate matrices are *fused*: one nn.Linear maps x_t -> [all gates] and one maps h_{t-1} -> [all gates].
  This is mathematically identical to separate W_xi, W_xf, ... matrices (it is a block-stacked matrix),
  just faster. Gate order inside the fused output is documented per cell.
* The input projection W_x x_t (+ bias) does not depend on the recurrence, so it is computed for ALL
  time steps in a single nn.Linear call before the time loop (`input_projection`). Only W_h h_{t-1}
  (and the WMC cell connections) are computed inside the loop.
* Only one bias per gate (on the x-projection), exactly as in the equations. The recurrent Linear has
  bias=False, so parameter counts match the equations.
"""
import torch
import torch.nn as nn


class BaseCell(nn.Module):
    """Common interface: input_projection (hoisted), step (one time step), init_state."""

    n_gates: int = 4
    forget_gate_index: int = 1  # position of the forget gate inside the fused gate vector

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

    @property
    def output_size(self) -> int:  # dimension of h_t (what the next layer / classifier sees)
        return self.hidden_size

    @property
    def recurrent_size(self) -> int:  # dimension of h_{t-1} fed back into the gates
        return self.hidden_size

    def input_projection(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, E) -> (B, T, n_gates*H). Equals W_x* x_t + b_* for every t."""
        return self.x2h(x)

    def init_state(self, batch: int, device, dtype):
        h = torch.zeros(batch, self.recurrent_size, device=device, dtype=dtype)
        c = torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
        return h, c

    def step(self, xw: torch.Tensor, state):
        raise NotImplementedError

    def forward(self, x_t: torch.Tensor, state=None):
        """Single-step convenience API: x_t (B, E) -> h_t, (h_t, c_t)."""
        if state is None:
            state = self.init_state(x_t.size(0), x_t.device, x_t.dtype)
        return self.step(self.x2h(x_t), state)


class VanillaLSTMCell(BaseCell):
    """Eqs (1)-(6). Fused gate order: [i, f, g, o]."""

    n_gates = 4
    forget_gate_index = 1

    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        self.x2h = nn.Linear(input_size, 4 * hidden_size)               # W_xi, W_xf, W_xg, W_xo and b_*
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)  # W_hi, W_hf, W_hg, W_ho

    def step(self, xw, state):
        h_prev, c_prev = state
        z = xw + self.h2h(h_prev)
        zi, zf, zg, zo = z.chunk(4, dim=-1)
        i = torch.sigmoid(zi)                       # (1)
        f = torch.sigmoid(zf)                       # (2)
        o = torch.sigmoid(zo)                       # (3)
        g = torch.tanh(zg)                          # (4)
        c = f * c_prev + i * g                      # (5)
        h = o * torch.tanh(c)                       # (6)
        return h, (h, c)


class WMCLSTMCell(BaseCell):
    """Working Memory Connections, eqs (7)-(12). Fused gate order: [i, f, g, o].

    The cell state is fed to the gates through a learnable matrix squashed by tanh:
      i_t, f_t use c_{t-1};  o_t uses the *current* c_t (so c_t is computed before o_t).
    """

    n_gates = 4
    forget_gate_index = 1

    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        H = hidden_size
        self.x2h = nn.Linear(input_size, 4 * H)               # W_x* and b_*
        self.h2h = nn.Linear(H, 4 * H, bias=False)            # W_h*
        self.c2if = nn.Linear(H, 2 * H, bias=False)           # [W_ci; W_cf]  (applied to c_{t-1})
        self.c2o = nn.Linear(H, H, bias=False)                # W_co          (applied to c_t)

    def step(self, xw, state):
        h_prev, c_prev = state
        z = xw + self.h2h(h_prev)
        zi, zf, zg, zo = z.chunk(4, dim=-1)
        ci, cf = torch.tanh(self.c2if(c_prev)).chunk(2, dim=-1)
        i = torch.sigmoid(zi + ci)                  # (7)
        f = torch.sigmoid(zf + cf)                  # (8)
        g = torch.tanh(zg)                          # (10)
        c = f * c_prev + i * g                      # (11)
        o = torch.sigmoid(zo + torch.tanh(self.c2o(c)))  # (9)  uses current c_t
        h = o * torch.tanh(c)                       # (12)
        return h, (h, c)


class CIFGLSTMCell(BaseCell):
    """Coupled Input-Forget Gate, eqs (13)-(18). Fused gate order: [f, g, o] (no input gate params)."""

    n_gates = 3
    forget_gate_index = 0

    def __init__(self, input_size, hidden_size):
        super().__init__(input_size, hidden_size)
        self.x2h = nn.Linear(input_size, 3 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=False)

    def step(self, xw, state):
        h_prev, c_prev = state
        z = xw + self.h2h(h_prev)
        zf, zg, zo = z.chunk(3, dim=-1)
        f = torch.sigmoid(zf)                       # (13)
        i = 1.0 - f                                 # (14)
        o = torch.sigmoid(zo)                       # (15)
        g = torch.tanh(zg)                          # (16)
        c = f * c_prev + i * g                      # (17)
        h = o * torch.tanh(c)                       # (18)
        return h, (h, c)


class LSTMPCell(BaseCell):
    """LSTM with recurrent projection, eqs (19)-(25). Fused gate order: [i, f, g, o].

    c_t, g_t, m_t in R^H ; h_t in R^P with h_t = W_p m_t + b_p  (P < H typically).
    The recurrence uses the projected h_{t-1} (size P), which is where the parameter savings come from.
    """

    n_gates = 4
    forget_gate_index = 1

    def __init__(self, input_size, hidden_size, proj_size):
        super().__init__(input_size, hidden_size)
        self.proj_size = proj_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(proj_size, 4 * hidden_size, bias=False)
        self.proj = nn.Linear(hidden_size, proj_size)                   # W_p, b_p

    @property
    def output_size(self):
        return self.proj_size

    @property
    def recurrent_size(self):
        return self.proj_size

    def step(self, xw, state):
        h_prev, c_prev = state
        z = xw + self.h2h(h_prev)
        zi, zf, zg, zo = z.chunk(4, dim=-1)
        i = torch.sigmoid(zi)                       # (19)
        f = torch.sigmoid(zf)                       # (20)
        o = torch.sigmoid(zo)                       # (21)
        g = torch.tanh(zg)                          # (22)
        c = f * c_prev + i * g                      # (23)
        m = o * torch.tanh(c)                       # (24)
        h = self.proj(m)                            # (25)
        return h, (h, c)


CELLS = {
    "vanilla": VanillaLSTMCell,
    "wmc": WMCLSTMCell,
    "cifg": CIFGLSTMCell,
    "lstmp": LSTMPCell,
}


def make_cell(cell_type: str, input_size: int, hidden_size: int, proj_size: int = None) -> BaseCell:
    if cell_type not in CELLS:
        raise ValueError(f"Unknown cell '{cell_type}'. Choose from {list(CELLS)}")
    if cell_type == "lstmp":
        if not proj_size:
            raise ValueError("lstmp needs proj_size")
        return LSTMPCell(input_size, hidden_size, proj_size)
    return CELLS[cell_type](input_size, hidden_size)


def expected_cell_params(cell_type: str, E: int, H: int, P: int = None) -> int:
    """Closed-form parameter counts (one layer), used in tests and in the report."""
    if cell_type == "vanilla":
        return 4 * H * (E + H + 1)
    if cell_type == "cifg":
        return 3 * H * (E + H + 1)
    if cell_type == "wmc":
        return 4 * H * (E + H + 1) + 3 * H * H
    if cell_type == "lstmp":
        return 4 * H * (E + P + 1) + P * H + P
    raise ValueError(cell_type)


class RNNLayer(nn.Module):
    """Unrolls a cell over time. Sequences are right-padded; outputs after a sequence's end are
    ignored downstream (pooling uses lengths), so padding never leaks into valid positions."""

    def __init__(self, cell: BaseCell):
        super().__init__()
        self.cell = cell

    def forward(self, x: torch.Tensor):
        B, T, _ = x.shape
        xw = self.cell.input_projection(x)          # (B, T, G*H), all steps at once
        state = self.cell.init_state(B, x.device, x.dtype)
        outputs = []
        for t in range(T):
            h, state = self.cell.step(xw[:, t], state)
            outputs.append(h)
        return torch.stack(outputs, dim=1), state
