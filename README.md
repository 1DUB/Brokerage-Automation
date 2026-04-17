# TAA Monthly Signal Generator

Automated monthly signal computation for a 3-strategy Tactical Asset Allocation portfolio, as specified in the Investment Policy Statement (IPS v1.2).

## Portfolio

| Strategy | Weight | Status |
|---|---|---|
| HAA-Balanced (Keller & Keuning 2023) | 40% | ✅ Implemented |
| KDA (Kipnis 2019) | 40% | ✅ Implemented |
| Vitral MAM (Zambrano & Rizzolo 2022) | 20% | ✅ Implemented |

## How it works

1. GitHub Actions runs on the last trading day of each month and the first trading day of the next month.
2. The script fetches price data from Yahoo Finance, computes momentum signals per each strategy's published rules, and produces a target allocation.
3. An email report is sent via Resend with the target allocation, sanity checks, and strategy breakdown.
4. You read the email and execute trades manually through your broker.

## Setup

### 1. Fork/clone this repository (private)

### 2. Set GitHub Secrets

Go to Settings → Secrets and variables → Actions, and add:

| Secret | Description |
|---|---|
| `RESEND_API_KEY` | Your Resend API key (get one at https://resend.com) |
| `TAA_EMAIL_TO` | Your email address |
| `TAA_EMAIL_FROM` | Sender address (must be verified in Resend, or use `signals@resend.dev` for testing) |

### 3. Test locally

```bash
pip install -r requirements.txt
python run_monthly.py --force
```

This will compute signals and print the report to console (without sending email, since RESEND_API_KEY won't be set locally unless you export it).

### 4. Run tests

```bash
python tests/test_haa.py
python tests/test_kda.py
python tests/test_vitral.py
```

### 5. Calibrate Vitral against the paper (recommended, one-time)

Before fully trusting the Vitral implementation, validate it against the paper's reported backtest metrics:

```bash
python calibrate_vitral.py
```

This fetches ~20 years of daily data and runs a backtest from 12/31/2003 to 5/31/2022, comparing to the paper's reported 590.66% total return. Passes if within ±10%.

## Project Structure

```
taa-signals/
├── run_monthly.py          # Main orchestrator
├── calibrate_vitral.py     # Vitral paper calibration (run locally)
├── requirements.txt        # Python dependencies
├── strategies/
│   ├── haa.py              # HAA-Balanced
│   ├── kda.py              # KDA
│   └── vitral.py           # Vitral MAM
├── lib/
│   ├── data.py             # Yahoo Finance data fetching (monthly + daily)
│   ├── momentum.py         # Shared momentum formulas (13612U, 13612W)
│   ├── optimization.py     # Min-variance optimizer for KDA
│   ├── report.py           # Email report formatting (plain text + HTML)
│   └── notify.py           # Resend email delivery
├── tests/
│   ├── test_haa.py
│   ├── test_kda.py
│   └── test_vitral.py
└── .github/
    └── workflows/
        └── monthly.yml     # GitHub Actions schedule
```

## Ticker Substitutions

If an ETF is delisted or renamed, update the `TICKER_ALIASES` dict in `lib/data.py`. No other code changes needed.

## References

- Keller & Keuning (2023), *Hybrid Asset Allocation (HAA)*, SSRN 4346906
- Kipnis (2019), *KDA Asset Allocation*, quantstrattrader.wordpress.com
- Zambrano & Rizzolo (2022), *Multi-Asset Momentum*, SSRN 4199648
- Investment Policy Statement v1.2 (maintained separately)
