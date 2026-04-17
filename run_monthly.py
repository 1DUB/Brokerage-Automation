#!/usr/bin/env python3
"""
Brokerage Model Monthly Signal Generator

40% NLX Hybrid AA 60/40
40% Stoken’s ACA [Dynamic Bond]
20% Lethargic Asset Allocation

Optimized for taxable brokerage accounts (lower turnover).

Run manually:   python run_monthly.py
Run via GitHub Actions: see .github/workflows/monthly.yml

Environment variables required:
  RESEND_API_KEY   - Resend API key for email delivery
  TAA_EMAIL_TO     - Your email address
  TAA_EMAIL_FROM   - Sender address (must be verified in Resend)
"""

import sys
import os
import logging
import traceback
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.data import (
    fetch_monthly_prices,
    fetch_daily_prices,
    is_last_trading_day,
    is_first_trading_day,
)
from lib.report import format_report, run_sanity_checks
from lib.notify import send_email, send_failure_alert

# New strategies
from strategies.nlx import compute_nlx_signals
from strategies.stoken import compute_stoken_signals
from strategies.lethargic import compute_lethargic_signals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("brokerage-model")

# ── Strategy weights (per your request) ─────────────────────────────
STRATEGY_WEIGHTS = {
    "NLX Hybrid AA 60/40": 0.40,
    "Stoken’s ACA [Dynamic Bond]": 0.40,
    "Lethargic Asset Allocation": 0.20,
}


def main(force: bool = False):
    """
    Main entry point for the Brokerage Model.
    """
    today = datetime.now()

    # Determine tranche
    if is_last_trading_day(today):
        tranche = 1
    elif is_first_trading_day(today):
        tranche = 2
    elif force:
        tranche = 1
        logger.info("FORCE mode: running despite not being a trading day.")
    else:
        logger.info(
            f"Today ({today.strftime('%Y-%m-%d')}) is not a scheduled trading day. "
            "Use --force to run anyway."
        )
        return

    logger.info(f"Running Brokerage Model signal computation — Tranche {tranche}")

    # ── Fetch data ───────────────────────────────────────────────────
    all_tickers = [
        "SPY", "IWM", "IEFA", "IEMG", "VNQ", "PDBC", "IEF", "TLT",
        "BIL", "TIP", "GLD", "VTV", "QQQ"
    ]

    logger.info(f"Fetching monthly prices for {len(all_tickers)} tickers...")
    monthly_prices = fetch_monthly_prices(all_tickers, months_history=15)

    # Only Stoken needs daily prices
    daily_tickers = ["SPY", "IEF", "GLD", "TLT", "VNQ"]
    logger.info(f"Fetching daily prices for {len(daily_tickers)} tickers...")
    daily_prices = fetch_daily_prices(daily_tickers, months_history=14)

    # ── Compute strategy signals ─────────────────────────────────────
    strategy_allocations = {}
    strategy_summaries = {}

    # NLX (40%)
    logger.info("Computing NLX Hybrid AA 60/40 signals...")
    nlx_sig = compute_nlx_signals(monthly_prices)
    strategy_allocations["NLX Hybrid AA 60/40"] = nlx_sig.allocation
    strategy_summaries["NLX Hybrid AA 60/40 (40% sleeve)"] = nlx_sig.summary()

    # Stoken (40%)
    logger.info("Computing Stoken’s ACA [Dynamic Bond] signals...")
    stoken_sig = compute_stoken_signals(monthly_prices)
    strategy_allocations["Stoken’s ACA [Dynamic Bond]"] = stoken_sig.allocation
    strategy_summaries["Stoken’s ACA [Dynamic Bond] (40% sleeve)"] = stoken_sig.summary()

    # Lethargic (20%)
    logger.info("Computing Lethargic Asset Allocation signals...")
    lethargic_sig = compute_lethargic_signals(monthly_prices)
    strategy_allocations["Lethargic Asset Allocation"] = lethargic_sig.allocation
    strategy_summaries["Lethargic Asset Allocation (20% sleeve)"] = lethargic_sig.summary()

    # ── Combine allocations ──────────────────────────────────────────
    combined = {}
    for name, alloc in strategy_allocations.items():
        weight = STRATEGY_WEIGHTS[name]
        for ticker, w in alloc.items():
            combined[ticker] = combined.get(ticker, 0) + w * weight

    # ── Sanity checks ────────────────────────────────────────────────
    sanity_errors = run_sanity_checks(combined)
    if sanity_errors:
        logger.warning(f"Sanity check failures: {sanity_errors}")
    else:
        logger.info("All sanity checks passed.")

    # ── Format report ────────────────────────────────────────────────
    signal_date = monthly_prices.index[-1].strftime("%Y-%m-%d")
    subject, body = format_report(
        signal_date=signal_date,
        tranche=tranche,
        combined_allocation=combined,
        strategy_summaries=strategy_summaries,
        previous_allocation=None,          # TODO: add persistence later if desired
        sanity_errors=sanity_errors if sanity_errors else None,
    )

    # ── Send email ───────────────────────────────────────────────────
    logger.info("Sending signal report email...")
    print("\n" + "=" * 60)
    print(body)
    print("=" * 60 + "\n")

    if os.environ.get("RESEND_API_KEY"):
        success = send_email(subject, body)
        if success:
            logger.info("Email sent successfully.")
        else:
            logger.error("Failed to send email.")
    else:
        logger.info("No RESEND_API_KEY set — email not sent (local test mode).")

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
        except Exception:
            pass

        sys.exit(1)
