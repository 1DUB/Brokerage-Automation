# Brokerage Model

**Tax-efficient monthly Tactical Asset Allocation (TAA) for taxable brokerage accounts.**

Automated monthly signal computation for a 3-strategy portfolio optimized for **low turnover and tax efficiency** while maintaining strong risk-adjusted returns.

## Portfolio

| Strategy | Weight | Status |
|---|---|---|
| NLX Hybrid AA 60/40 | 40% | ✅ Implemented |
| Stoken’s ACA [Dynamic Bond] | 40% | ✅ Implemented |
| Lethargic Asset Allocation | 20% | ✅ Implemented |

## How it works

1. GitHub Actions runs on the last trading day of each month and the first trading day of the next month.
2. The script fetches price data from Yahoo Finance + FRED (unemployment rate for Lethargic), computes signals per each strategy's published rules, and produces a target allocation.
3. An email report is sent via Resend with the target allocation, sanity checks, and strategy breakdown.
4. You read the email and execute trades manually through your broker.

**Designed specifically for taxable brokerage accounts** — moderate turnover, preference for long-term gains, and liquid ETFs only.

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

### 4. Run tests

```bash
python tests/test_nlx.py
python tests/test_stoken.py
python tests/test_lethargic.py
```

### 5. Calibrate Lethargic against the paper (recommended, one-time)

The Lethargic implementation includes the exact Growth-Trend Timing rule using official FRED unemployment data.

## Project Structure

```
brokerage-model/
├── run_monthly.py          # Main orchestrator (40/40/20 weights)
├── requirements.txt        # Python dependencies
├── strategies/
│   ├── nlx.py              # NLX Hybrid AA 60/40
│   ├── stoken.py           # Stoken’s ACA [Dynamic Bond]
│   └── lethargic.py        # Lethargic Asset Allocation
├── lib/
│   ├── data.py             # Yahoo Finance + FRED unemployment data
│   ├── momentum.py         # Shared momentum formulas
│   ├── report.py           # Email report formatting
│   └── notify.py           # Resend email delivery
├── tests/
│   ├── test_nlx.py
│   ├── test_stoken.py
│   └── test_lethargic.py
└── .github/
    └── workflows/
        └── monthly.yml     # GitHub Actions schedule
```

## Ticker Substitutions

If an ETF is delisted or renamed, update the TICKER_ALIASES dict in lib/data.py.

## References

- NLX Finance — “The HAA Strategy Revisited” (Hybrid AA 60/40)
- Wouter Keller — “Growth-Trend Timing and 60-40 Variations: Lethargic Asset Allocation”
- Dick Stoken — “Survival of the Fittest for Investors” (Active Combined Asset with Dynamic Bond variation)
- Investment Policy Statement v1.2 (maintained separately)
