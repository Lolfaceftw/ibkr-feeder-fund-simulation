# Simulation methods

## Annual market returns

For independent trials, each annual innovation is drawn from Student-t with
`nu = 3`, then scaled to variance one:

```text
z[t] = t_nu * sqrt((nu - 2) / nu)
epsilon[t] = sqrt(h[t]) * z[t]
R[t] = mu + epsilon[t]
h[t+1] = omega + alpha * epsilon[t]^2 + beta * h[t]
omega = (1 - alpha - beta) * sigma^2
h[1] = sigma^2
```

The annual time step, volatility, GARCH coefficients, dividend yield,
contribution timing, fee application order, and seed are modeling choices.
They are all explicit in the configuration or equations. No parameters are
estimated from historical prices. A single seed creates separate random
streams for market returns and conversion charges.

This is a constant-mean GARCH(1,1) model with standardized Student-t errors;
see the [ARCH documentation](https://bashtage.github.io/arch/doc/univariate/univariate_volatility_modeling.html).
Both strategies use identical market draws, isolating the modeled cost differences.

## Contributions and costs

With annual contribution `C`, conversion fee `u`, dividend yield `d`, dividend
withholding rate `w`, and annual expense ratio `e`:

```text
VWRA deposit[t] = C * (1 - u[t])
SB deposit[t] = C
gross[t] = 1 + R[t] - d * w
wealth[t] = (wealth[t-1] + deposit[t]) * max(gross[t], 0) * (1 - e)
```

`R[t]` includes gross dividends. Withholding reduces the dividend component
once; dividends are not added a second time. The return location is 10%
before costs and the limited-liability rule.

Student-t simple returns are unbounded. A nonpositive gross factor wipes out
the invested balance for that year. Subsequent contributions can invest again.
These events are counted in `summary.json`; they are not redrawn. Limited
liability changes the realized expected return from the underlying return
location. No other truncation is applied to market returns or wealth data.

No inflation adjustment, FX movement, broker commissions, transfer minimums,
redemption taxes, or changing fund costs are modeled. The 0.755% SB input uses
the rounded total in the linked analysis; its listed components sum to 0.7555%.
Dividend withholding is a simplified portfolio-wide drag. Actual treatment
depends on holdings, domicile, and applicable rules.

## Statistics and plotting

Percentiles are computed across every trial independently at each year. A
percentile curve does not describe one investor's continuous path. The 5th
and 95th percentiles leave 5% of outcomes below and above them, respectively.

The histogram uses 100 shared bins between $0 and $410,000. The path chart
shows all trials with a $600,000 vertical display limit. Observations outside
these display ranges remain in `paths.npz` and in percentile calculations;
their counts appear in `summary.json`.

The median gap compares the two marginal medians. Expressing that gap as a
multiple of annual contributions is an amount comparison, not extra working
years or a time-to-goal estimate.
