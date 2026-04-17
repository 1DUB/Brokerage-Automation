#!/usr/bin/env python3
"""
Brokerage Model Monthly Signal Generator

40% NLX + 40% Stoken ACA [Dynamic Bond] + 20% Lethargic AA

Run manually:   python run_monthly.py
Run via GitHub Actions: see .github/workflows/monthly.yml

Environment variables required:
  RESEND_API_KEY   - Resend API key for email delivery
  TAA_EMAIL_TO     - Your email address
  TAA_EMAIL_FROM   - Sender address (must be verified in Resend)
"""

import pandas as pd
from lib.data import fetch_monthly_prices, fetch_daily_prices
from strategies.nlx import compute_nlx_signals
from strategies.stoken import compute_stoken_signals
from strategies.lethargic import compute_lethargic_signals
from lib.report import generate_report
from lib.notify import send_email

# Weights
WEIGHTS = {"NLX": 0.40, "Stoken": 0.40, "Lethargic": 0.20}

def main(force: bool = False):
    monthly = fetch_monthly_prices([
        "SPY", "IWM", "IEFA", "IEMG", "VNQ", "PDBC", "IEF", "TLT",
        "BIL", "TIP", "GLD", "VTV", "QQQ"
    ])
    daily = fetch_daily_prices(["SPY", "IEF", "GLD", "TLT", "VNQ"])  # used by Stoken

    nlx_sig = compute_nlx_signals(monthly)
    stoken_sig = compute_stoken_signals(monthly)
    lethargic_sig = compute_lethargic_signals(monthly)

    # Blend
    target = {}
    for ticker, w in nlx_sig.allocation.items():
        target[ticker] = target.get(ticker, 0) + w * WEIGHTS["NLX"]
    for ticker, w in stoken_sig.allocation.items():
        target[ticker] = target.get(ticker, 0) + w * WEIGHTS["Stoken"]
    for ticker, w in lethargic_sig.allocation.items():
        target[ticker] = target.get(ticker, 0) + w * WEIGHTS["Lethargic"]

    # Normalize
    total = sum(target.values())
    target = {k: v / total for k, v in target.items() if v > 0.0001}

    report = generate_report(target, [nlx_sig, stoken_sig, lethargic_sig], WEIGHTS)
    send_email(report)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    main(force=args.force)
