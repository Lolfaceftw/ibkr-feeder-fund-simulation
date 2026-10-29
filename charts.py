"""Render terminal wealth distributions and annual portfolio paths."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


def render_charts(
    vwra: np.ndarray, sb: np.ndarray, output_dir: Path, df: float
) -> None:
    """Write histogram and wealth-path charts with fund and percentile styling.

    Both arrays contain complete, paired simulation results. Percentiles are
    calculated from every trial before the chart viewing limits are
    applied. The histogram intentionally shows only USD 0–410,000, and the path
    chart shows only USD 0–600,000; observations outside those windows are not
    drawn. The caller remains responsible for retaining the complete data and
    reporting any observations outside the display windows.

    Blue/red translucent distributions and paths identify VWRA/SB respectively.
    The terminal histogram uses cyan, black, and orange percentile markers.
    The evolution chart uses green, black, and orange percentile curves, with
    solid VWRA curves and dashed SB curves. Labels report simulated results.

    Args:
        vwra: Finite, nonnegative portfolio values in USD, shaped
            ``(trials, years + 1)``. Column zero must contain the initial zeros.
        sb: Portfolio values with the same shape and initial condition as
            ``vwra``. Rows must correspond to the same simulated market paths.
        output_dir: Directory receiving ``terminal_wealth.png``,
            ``terminal_wealth.svg``, ``wealth_evolution.png``, and
            ``wealth_evolution.svg``. It is created if needed.
        df: Student-t degrees of freedom displayed in the histogram title.

    Returns:
        None. Figures are saved at 100 DPI and closed after export.

    Raises:
        ValueError: Inputs are empty, have incompatible shapes, contain invalid
            wealth values, or do not start at zero; or ``df`` is not above two.
        OSError: The output directory or chart files cannot be written.

    Example:
        After running the simulator, pass its full arrays directly::

            render_charts(vwra_paths, sb_paths, Path("outputs"), df=3)
    """
    if vwra.ndim != 2 or sb.shape != vwra.shape:
        raise ValueError("Wealth arrays must have the same two-dimensional shape.")
    if vwra.shape[0] < 1 or vwra.shape[1] < 2:
        raise ValueError("At least one trial and one simulated year are required.")
    if not np.isfinite(df) or df <= 2:
        raise ValueError("Student-t degrees of freedom must be finite and above two.")
    for values in (vwra, sb):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Wealth values must be finite and nonnegative.")
        if np.any(values[:, 0] != 0):
            raise ValueError("Column zero must contain zero initial wealth.")

    output_dir.mkdir(parents=True, exist_ok=True)
    percentiles = (5, 50, 95)
    vwra_percentiles = np.percentile(vwra, percentiles, axis=0)
    sb_percentiles = np.percentile(sb, percentiles, axis=0)
    trial_count, year_columns = vwra.shape
    horizon = year_columns - 1

    with plt.rc_context({"font.size": 10, "axes.titlesize": 15,
                         "axes.labelsize": 12, "svg.hashsalt": "wealth-charts"}):
        fig, ax = plt.subplots(figsize=(16.02, 8.67), dpi=100)
        bins = np.linspace(0, 410_000, 101)
        ax.hist(vwra[:, -1], bins=bins, color="blue", alpha=0.5,
                label="VWRA (Ireland)")
        ax.hist(sb[:, -1], bins=bins, color="red", alpha=0.5,
                label="SB Global (PH)")
        for index, (percentile, color) in enumerate(
            zip(percentiles, ("cyan", "black", "orange"), strict=True)
        ):
            vwra_value = vwra_percentiles[index, -1]
            sb_value = sb_percentiles[index, -1]
            vwra_style = "-" if percentile == 50 else "--"
            ax.axvline(vwra_value, color=color, linestyle=vwra_style, linewidth=2,
                       label=f"VWRA {percentile}th: ${vwra_value / 1000:.1f}k")
            ax.axvline(sb_value, color=color, linestyle=":", linewidth=2,
                       label=f"SB {percentile}th: ${sb_value / 1000:.1f}k")
        ax.set_title(f"Terminal Wealth Distribution (N={trial_count}, df={df:g})")
        ax.set_xlabel(f"Wealth at Year {horizon} [USD]")
        ax.set_ylabel("Frequency")
        ax.set_xlim(-20_000, 430_000)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(loc="upper right", fontsize=9)
        fig.text(0.5, 0.012,
                 "Display: $0–410,000; percentiles use all trials",
                 ha="center", fontsize=8, color="#555555")
        fig.subplots_adjust(left=0.058, right=0.987, top=0.93, bottom=0.09)
        for suffix in ("png", "svg"):
            fig.savefig(output_dir / f"terminal_wealth.{suffix}",
                        metadata={"Creator": "ibkr-feeder-fund-simulation"})
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(16.58, 8.68), dpi=100)
        years = np.arange(year_columns)
        for values, color in ((vwra, "blue"), (sb, "red")):
            points = np.empty((trial_count, year_columns, 2))
            points[:, :, 0] = years
            points[:, :, 1] = values
            ax.add_collection(LineCollection(points, colors=color, alpha=0.02,
                                             linewidths=0.6, rasterized=True))
        for index, (percentile, color) in enumerate(
            zip(percentiles, ("green", "black", "darkorange"), strict=True)
        ):
            ax.plot(years, vwra_percentiles[index], color=color, linewidth=2.5,
                    label=f"VWRA {percentile}th Percentile")
            ax.plot(years, sb_percentiles[index], color=color, linestyle="--",
                    linewidth=2.5, label=f"SB {percentile}th Percentile")
            for values, label_color, offset in (
                (vwra_percentiles, "blue", 5), (sb_percentiles, "red", -5)
            ):
                terminal = values[index, -1]
                ax.annotate(f"${terminal / 1000:.1f}k", (horizon, terminal),
                            xytext=(6, offset), textcoords="offset points",
                            color=label_color, fontsize=10, fontweight="bold",
                            va="center")
        ax.set_title(f"{horizon}-Year Wealth Evolution: All Stochastic Paths")
        ax.set_xlabel("Years")
        ax.set_ylabel("Portfolio Value [USD]")
        ax.set_xlim(0, horizon + 3)
        ax.set_ylim(0, 600_000)
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", ncol=2, fontsize=9)
        fig.text(0.5, 0.012,
                 "All paths plotted; vertical display limit: $600,000",
                 ha="center", fontsize=8, color="#555555")
        fig.subplots_adjust(left=0.094, right=0.991, top=0.945, bottom=0.09)
        for suffix in ("png", "svg"):
            fig.savefig(output_dir / f"wealth_evolution.{suffix}",
                        metadata={"Creator": "ibkr-feeder-fund-simulation"})
        plt.close(fig)
