#!/usr/bin/env python3
"""
TAA Monthly Signal Generator

Phase 1: HAA-Balanced (40% of portfolio) ✅
Phase 2: KDA (40% of portfolio)          ✅
Phase 3: Vitral MAM (20% of portfolio)   ✅

Full 40/40/20 portfolio now active per IPS v1.2.

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
from strategies.haa import compute_haa_signals, ALL_TICKERS as HAA_TICKERS
from strategies.kda import compute_kda_signals, ALL_TICKERS as KDA_TICKERS
from strategies.vitral import compute_vitral_signals, ALL_TICKERS as VITRAL_TICKERS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("taa-signals")

# ── Strategy weights (per IPS v1.2) ──────────────────────────────────
# Final target weights per Investment Policy Statement:
#   HAA: 40%, KDA: 40%, Vitral: 20%
TARGET_WEIGHTS = {
    "HAA-Balanced": 0.40,
    "KDA":          0.40,
    "Vitral-MAM":   0.20,
}

# All three strategies are now active (Phase 3 complete).
ACTIVE_STRATEGIES = ["HAA-Balanced", "KDA", "Vitral-MAM"]

# Renormalize active weights so they sum to 1.0
# (no-op here since all three are active and already sum to 1.0)
_active_total = sum(TARGET_WEIGHTS[s] for s in ACTIVE_STRATEGIES)
STRATEGY_WEIGHTS = {
    s: TARGET_WEIGHTS[s] / _active_total for s in ACTIVE_STRATEGIES
}


def combine_allocations(
    strategy_allocations: dict,  # {strategy_name: {ticker: weight}}
    strategy_weights: dict,  # {strategy_name: portfolio_weight}
) -> dict:
    """
    Combine multiple strategy allocations into a single portfolio allocation.
    
    Each strategy's allocation (summing to 1.0 within the strategy) is scaled
    by the strategy's portfolio weight, then summed across strategies.
    """
    combined = {}
    
    for strategy_name, alloc in strategy_allocations.items():
        sleeve_weight = strategy_weights.get(strategy_name, 0)
        for ticker, weight in alloc.items():
            combined[ticker] = combined.get(ticker, 0) + weight * sleeve_weight
    
    return combined


def main(force: bool = False):
    """
    Main entry point. Computes signals and sends email.
    
    Args:
        force: If True, skip the trading-day check (useful for testing).
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
    
    logger.info(f"Running TAA signal computation — Tranche {tranche}")
    
    # ── Fetch data ───────────────────────────────────────────────────
    # Union of all tickers needed across strategies
    all_tickers = list(set(HAA_TICKERS + KDA_TICKERS + VITRAL_TICKERS))
    
    logger.info(f"Fetching monthly prices for {len(all_tickers)} tickers...")
    monthly_prices = fetch_monthly_prices(all_tickers, months_history=15)
    logger.info(
        f"Got {len(monthly_prices)} months of data, "
        f"latest: {monthly_prices.index[-1].strftime('%Y-%m-%d')}"
    )
    
    # KDA and Vitral need daily prices
    # Vitral's 200-day SMA needs ~12 months of daily history
    daily_tickers = list(set(KDA_TICKERS + VITRAL_TICKERS))
    logger.info(f"Fetching daily prices for {len(daily_tickers)} tickers...")
    daily_prices = fetch_daily_prices(daily_tickers, months_history=14)
    logger.info(
        f"Got {len(daily_prices)} days of data, "
        f"latest: {daily_prices.index[-1].strftime('%Y-%m-%d')}"
    )
    
    # ── Compute strategy signals ─────────────────────────────────────
    strategy_allocations = {}
    strategy_summaries = {}
    
    # HAA
    logger.info("Computing HAA-Balanced signals...")
    haa_signal = compute_haa_signals(monthly_prices)
    strategy_allocations["HAA-Balanced"] = haa_signal.allocation
    haa_pct = STRATEGY_WEIGHTS["HAA-Balanced"] * 100
    strategy_summaries[f"HAA-Balanced ({haa_pct:.0f}% sleeve)"] = haa_signal.summary()
    
    # KDA
    logger.info("Computing KDA signals...")
    kda_signal = compute_kda_signals(monthly_prices, daily_prices)
    strategy_allocations["KDA"] = kda_signal.allocation
    kda_pct = STRATEGY_WEIGHTS["KDA"] * 100
    strategy_summaries[f"KDA ({kda_pct:.0f}% sleeve)"] = kda_signal.summary()
    
    # Vitral
    logger.info("Computing Vitral MAM signals...")
    vitral_signal = compute_vitral_signals(monthly_prices, daily_prices)
    strategy_allocations["Vitral-MAM"] = vitral_signal.allocation
    vitral_pct = STRATEGY_WEIGHTS["Vitral-MAM"] * 100
    strategy_summaries[f"Vitral MAM ({vitral_pct:.0f}% sleeve)"] = vitral_signal.summary()
    
    # ── Combine allocations ──────────────────────────────────────────
    combined = combine_allocations(strategy_allocations, STRATEGY_WEIGHTS)
    
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
        previous_allocation=None,  # TODO: load from saved state
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
        logger.info(
            "No RESEND_API_KEY set — email not sent. "
            "Report printed to console above."
        )
    
    logger.info("Done.")


if __name__ == "__main__":
    force = "--force" in sys.argv
    
    try:
        main(force=force)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        tb = traceback.format_exc()
        logger.error(tb)
        
        # Try to send failure alert
        try:
            send_failure_alert(str(e), tb)
        except Exception:
            pass  # Don't crash on notification failure
        
        sys.exit(1)
