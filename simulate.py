"""Simulate long-term VWRA and SB Global wealth using paired GARCH-t returns."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from charts import render_charts


@dataclass(frozen=True)
class Config:
    """Declare all numerical inputs; rates are fractions and amounts are USD.

    The JSON configuration must provide every field. Annual volatility, GARCH
    coefficients, dividend yield, seed, and contribution timing are explicit
    modeling assumptions recorded alongside every run.
    """

    trials: int
    years: int
    seed: int
    annual_contribution: float
    annual_mean_return: float
    annual_volatility: float
    garch_alpha: float
    garch_beta: float
    student_df: float
    dividend_yield: float
    vwra_expense_ratio: float
    sb_expense_ratio: float
    vwra_dividend_withholding: float
    sb_dividend_withholding: float
    wise_fee_min: float
    wise_fee_max: float

    def validate(self) -> None:
        """Reject invalid inputs before allocating arrays or drawing randomness.

        Raises:
            ValueError: A count, rate, variance, or stationarity condition fails.
        """
        for name in ("trials", "years", "seed"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be an integer in its valid range")
        for name, value in asdict(self).items():
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.annual_contribution <= 0 or self.annual_volatility < 0:
            raise ValueError("Contribution must be positive; volatility nonnegative")
        if self.student_df <= 2:
            raise ValueError("Student degrees of freedom must exceed two")
        if min(self.garch_alpha, self.garch_beta) < 0:
            raise ValueError("GARCH coefficients must be nonnegative")
        if self.garch_alpha + self.garch_beta >= 1:
            raise ValueError("GARCH alpha + beta must be less than one")
        for name in ("dividend_yield", "vwra_expense_ratio", "sb_expense_ratio",
                     "vwra_dividend_withholding", "sb_dividend_withholding",
                     "wise_fee_min", "wise_fee_max"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.wise_fee_min > self.wise_fee_max:
            raise ValueError("Wise fee minimum must not exceed maximum")


def simulate(config: Config) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate paired annual GARCH-t market paths and invest annual contributions.

    Args:
        config: Validated or unvalidated complete numerical configuration.

    Returns:
        VWRA and SB wealth arrays of shape (trials, years + 1), including zero
        initial wealth, and diagnostics counting nonpositive annual gross factors.
        Both strategies experience exactly the same market innovations. Each
        year's contribution arrives before that year's return and costs.

    Raises:
        ValueError: Configuration is invalid.
        FloatingPointError: The requested simulation produces nonfinite values.
    """
    config.validate()
    market_seed, conversion_seed = np.random.SeedSequence(config.seed).spawn(2)
    market_rng = np.random.default_rng(market_seed)
    conversion_rng = np.random.default_rng(conversion_seed)
    vwra = np.zeros((config.trials, config.years + 1))
    sb = np.zeros_like(vwra)
    variance = np.full(config.trials, config.annual_volatility ** 2)
    omega = config.annual_volatility ** 2 * (
        1 - config.garch_alpha - config.garch_beta
    )
    scale = np.sqrt((config.student_df - 2) / config.student_df)
    wipeouts = {"vwra_nonpositive_gross_factors": 0,
                "sb_nonpositive_gross_factors": 0}
    with np.errstate(over="raise", invalid="raise"):
        for year in range(1, config.years + 1):
            shock = np.sqrt(variance) * market_rng.standard_t(
                config.student_df, config.trials
            ) * scale
            market_return = config.annual_mean_return + shock
            conversion_fee = conversion_rng.uniform(
                config.wise_fee_min, config.wise_fee_max, config.trials
            )
            for name, wealth, expense, withholding, deposit in (
                ("vwra", vwra, config.vwra_expense_ratio,
                 config.vwra_dividend_withholding,
                 config.annual_contribution * (1 - conversion_fee)),
                ("sb", sb, config.sb_expense_ratio,
                 config.sb_dividend_withholding, config.annual_contribution),
            ):
                gross = 1 + market_return - config.dividend_yield * withholding
                wipeouts[f"{name}_nonpositive_gross_factors"] += int(
                    np.count_nonzero(gross <= 0)
                )
                # Explicit limited liability: a <= -100% annual net return wipes
                # out that year's invested wealth. Later deposits may invest again.
                wealth[:, year] = (wealth[:, year - 1] + deposit) * (
                    np.maximum(gross, 0) * (1 - expense)
                )
            variance = omega + config.garch_alpha * shock ** 2 + (
                config.garch_beta * variance
            )
    if not np.isfinite(vwra).all() or not np.isfinite(sb).all():
        raise FloatingPointError("Simulation generated nonfinite wealth")
    return vwra, sb, wipeouts


def export_results(config: Config, output_dir: Path) -> dict:
    """Run the simulation and export complete data, comparison tables, and charts.

    Args:
        config: Full simulation inputs.
        output_dir: Destination directory, created if absent.

    Returns:
        JSON-compatible summary containing terminal percentiles, median gap,
        display-range counts and model diagnostics.
    """
    vwra, sb, diagnostics = simulate(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "paths.npz", vwra=vwra, sb=sb)
    quantiles = np.array([5, 50, 95])
    values = {"vwra": np.percentile(vwra, quantiles, axis=0),
              "sb": np.percentile(sb, quantiles, axis=0)}
    summary = {"config": asdict(config), "diagnostics": diagnostics,
               "terminal_percentiles": []}
    for name in ("vwra", "sb"):
        for index, percentile in enumerate(quantiles):
            actual = float(values[name][index, -1])
            summary["terminal_percentiles"].append({
                "fund": name, "percentile": int(percentile),
                "wealth_usd": actual,
            })
    median_vwra = float(values["vwra"][1, -1])
    median_sb = float(values["sb"][1, -1])
    summary["median_gap_usd"] = median_vwra - median_sb
    summary["median_gap_percent"] = 100 * (median_vwra - median_sb) / median_vwra
    summary["display_range_counts"] = {
        name: {"terminal_above_410000": int(np.count_nonzero(w[:, -1] > 410000)),
               "path_points_above_600000": int(np.count_nonzero(w > 600000))}
        for name, w in (("vwra", vwra), ("sb", sb))
    }
    with (output_dir / "percentiles.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "vwra_p5", "vwra_p50", "vwra_p95",
                         "sb_p5", "sb_p50", "sb_p95"])
        for year in range(config.years + 1):
            writer.writerow([year, *values["vwra"][:, year], *values["sb"][:, year]])
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    render_charts(vwra, sb, output_dir, config.student_df)
    return summary


def main() -> None:
    """Parse CLI paths, require a complete JSON configuration, and generate outputs.

    Examples:
        Run ``python simulate.py --config config.json --output results`` from
        the checkout. Missing files and unknown configuration keys raise errors.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    config = Config(**json.loads(args.config.read_text(encoding="utf-8")))
    summary = export_results(config, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
