#!/usr/bin/env python3
"""
Brokerage Model Monthly Signal Generator

Stoken 40% + Composite Dual Momentum 25% + Lethargic 20%  + NLX 15%

Single-day execution per IPS v2.1 Section 4.2 — no tranching.
Runs on the last trading day of each month only.
"""

import sys
import os
import logging
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.data import fetch_monthly_prices, fetch_daily_prices, is_last_trading_day
from lib.report import format_report, run_sanity_checks
from lib.notify import send_email, send_failure_alert

from strategies.stoken import compute_stoken_signals
from strategies.cdm import compute_cdm_signals
from strategies.nlx import compute_nlx_signals
from strategies.lethargic import compute_lethargic_signals

logger = logging.getLogger("brokerage-model")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Strategy weights (per IPS v2.1)
STRATEGY_WEIGHTS = {
    "Stoken's ACA [Dynamic Bond]": 0.40,
    "Composite Dual Momentum": 0.25,
    "NLX Hybrid AA 60/40": 0.15,
    "Lethargic Asset Allocation": 0.20,
}


def main(force: bool = False):
    today = datetime.now()

    if is_last_trading_day(today):
        pass  # proceed
    elif force:
        logger.info("FORCE mode: running despite not being the last trading day.")
    else:
        logger.info(
            f"Today ({today.strftime('%Y-%m-%d')}) is not the last trading day "
            f"of the month. Use --force to run anyway."
        )
        return

    logger.info("Running Brokerage Model — single-day execution")

    # Fetch data (all tickers from all strategies)
    all_tickers = ["SPY", "IEF", "GLD", "TLT", "VNQ", "BIL", "TIP", "IEFA", "LQD", "HYG", "REM", "VTV", "QQQ"]
    monthly_prices = fetch_monthly_prices(all_tickers, months_history=15)
    daily_prices = fetch_daily_prices(["SPY", "IEF", "GLD", "TLT", "VNQ"], months_history=14)

    # Compute signals
    strategy_allocations = {}
    strategy_summaries = {}

    stoken_sig = compute_stoken_signals(monthly_prices)
    strategy_allocations["Stoken's ACA [Dynamic Bond]"] = stoken_sig.allocation
    strategy_summaries["Stoken's ACA [Dynamic Bond] (40% sleeve)"] = stoken_sig.summary()

    cdm_sig = compute_cdm_signals(monthly_prices)
    strategy_allocations["Composite Dual Momentum"] = cdm_sig.allocation
    strategy_summaries["Composite Dual Momentum (25% sleeve)"] = cdm_sig.summary()

    nlx_sig = compute_nlx_signals(monthly_prices)
    strategy_allocations["NLX Hybrid AA 60/40"] = nlx_sig.allocation
    strategy_summaries["NLX Hybrid AA 60/40 (15% sleeve)"] = nlx_sig.summary()

    lethargic_sig = compute_lethargic_signals(monthly_prices)
    strategy_allocations["Lethargic Asset Allocation"] = lethargic_sig.allocation
    strategy_summaries["Lethargic Asset Allocation (20% sleeve)"] = lethargic_sig.summary()

    # Combine
    combined = {}
    for name, alloc in strategy_allocations.items():
        weight = STRATEGY_WEIGHTS[name]
        for ticker, w in alloc.items():
            combined[ticker] = combined.get(ticker, 0) + w * weight

    # Sanity checks & report
    sanity_errors = run_sanity_checks(combined)
    signal_date = monthly_prices.index[-1].strftime("%Y-%m-%d")
    subject, body = format_report(
        signal_date=signal_date,
        combined_allocation=combined,
        strategy_summaries=strategy_summaries,
        previous_allocation=None,
        sanity_errors=sanity_errors if sanity_errors else None,
    )

    print("\n" + "=" * 60)
    print(body)
    print("=" * 60 + "\n")

    if os.environ.get("RESEND_API_KEY"):
        success = send_email(subject, body)
        logger.info("Email sent" if success else "Failed to send email")
    else:
        logger.info("No RESEND_API_KEY — report printed to console (local test).")

    logger.info("Done.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    try:
        main(force=force)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        tb = traceback.format_exc()
        logger.error(tb)
        try:
            send_failure_alert(str(e), tb)
        except:
            pass
        sys.exit(1)
