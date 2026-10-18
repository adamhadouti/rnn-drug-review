"""Shared figure style for the report: fixed colour per cell (colour follows the entity, never its rank)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

CELL_COLORS = {"vanilla": "#2a78d6", "wmc": "#eb6834", "cifg": "#1baf7a", "lstmp": "#eda100"}
CELL_LABELS = {"vanilla": "Vanilla LSTM", "wmc": "WMC-LSTM", "cifg": "CIFG-LSTM", "lstmp": "LSTMP"}
CLASS_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # negative, neutral, positive
SEQ = "#2a78d6"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "lines.linewidth": 2, "legend.frameon": False,
    })
    return plt
