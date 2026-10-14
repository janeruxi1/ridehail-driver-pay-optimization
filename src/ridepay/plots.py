"""One plotting style for every notebook: recessive axes, thin marks, a fixed categorical order."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical hues, always assigned in this order (validated for colour-vision deficiency).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#8a8985"
INK = "#0b0b0b"
INK_2 = "#52514e"
SEQUENTIAL = "Blues"
DIVERGING = "RdBu_r"


def use_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "axes.edgecolor": "#c9c8c3", "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 8,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def subplots(*args, **kwargs):
    use_style()
    return plt.subplots(*args, **kwargs)
