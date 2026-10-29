"""Check saved simulation data with a fresh execution of the configured model."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from simulate import Config, simulate


def main() -> None:
    """Verify deterministic outputs and a zero-volatility closed-form case.

    Reads the real project configuration and saved outputs, reruns its 10,000
    paths, and compares all saved wealth values and reported percentiles.
    Also exercises the model at zero volatility against an independently
    calculated annuity with beginning-of-year contributions and fixed fees.

    Raises:
        AssertionError: Saved data, reproducibility, or accounting differs.
    """
    config = Config(**json.loads(Path("config.json").read_text(encoding="utf-8")))
    vwra, sb, diagnostics = simulate(config)
    with np.load("results/paths.npz") as stored:
        np.testing.assert_array_equal(vwra, stored["vwra"])
        np.testing.assert_array_equal(sb, stored["sb"])
    summary = json.loads(Path("results/summary.json").read_text(encoding="utf-8"))
    assert diagnostics == summary["diagnostics"]
    for row in summary["terminal_percentiles"]:
        data = vwra if row["fund"] == "vwra" else sb
        np.testing.assert_allclose(
            np.percentile(data[:, -1], row["percentile"]), row["wealth_usd"]
        )
    deterministic = replace(config, annual_volatility=0.0,
                            wise_fee_max=config.wise_fee_min)
    dv, ds, _ = simulate(deterministic)
    for actual, expense, withholding, deposit in (
        (dv, config.vwra_expense_ratio, config.vwra_dividend_withholding,
         config.annual_contribution * (1 - config.wise_fee_min)),
        (ds, config.sb_expense_ratio, config.sb_dividend_withholding,
         config.annual_contribution),
    ):
        growth = (1 + config.annual_mean_return - config.dividend_yield * withholding)
        growth *= 1 - expense
        years = np.arange(config.years + 1)
        expected = deposit * growth * (growth ** years - 1) / (growth - 1)
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape))
    print("PASS: full-path repeatability, saved percentiles, and annuity accounting")


if __name__ == "__main__":
    main()
