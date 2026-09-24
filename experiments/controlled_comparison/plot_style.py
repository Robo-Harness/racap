"""Shared, publication-oriented Matplotlib styling for experiment figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure


METHOD_COLORS = {
    "capx": "#707070",
    "rats_base": "#56B4E9",
    "rats_90": "#0072B2",
    "racap_phase1": "#E69F00",
    "racap_phase2": "#D55E00",
}


def apply_paper_style() -> None:
    """Install a compact style that remains legible in a two-column paper."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "lines.linewidth": 1.6,
            "lines.markersize": 5.0,
            "errorbar.capsize": 2.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(axis: Axes, *, grid_axis: str | None = "y") -> None:
    """Remove chart junk while retaining a light quantitative guide."""

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(direction="out", length=3, width=0.7)
    if grid_axis:
        axis.grid(axis=grid_axis, color="#D9D9D9", linewidth=0.6, alpha=0.7)
        axis.set_axisbelow(True)


def save_figure(fig: Figure, output: Path) -> None:
    """Write editable vector PDF plus a high-resolution inspection PNG."""

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(
        output.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )


apply_paper_style()
