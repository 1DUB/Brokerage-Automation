"""
Format the monthly signal report for email delivery — Brokerage Model
"""

from datetime import datetime
from typing import Dict, List, Optional, Any


def format_report(
    signal_date: str,
    tranche: int,  # 1 or 2
    combined_allocation: Dict[str, float],
    strategy_summaries: Dict[str, str],  # strategy_name -> summary text
    previous_allocation: Optional[Dict[str, float]] = None,
    sanity_errors: Optional[list] = None,
    phase_notice: Optional[str] = None,
) -> tuple:
    """
    Format the complete email report for the Brokerage Model.
    
    Returns:
        (subject, body) tuple of strings.
    """
    subject = f"[Brokerage Model] Tranche {tranche} Signals — {signal_date}"
    
    lines = []
    lines.append("=" * 60)
    lines.append(f"  Brokerage Model Monthly Signal Report — Tranche {tranche} of 2")
    lines.append(f"  Signal date: {signal_date}")
    
    if phase_notice:
        lines.append("")
        lines.append(f"  ⚠ {phase_notice}")
    
    if tranche == 1:
        lines.append("  Action: Execute 50% of rebalancing trades at today's close.")
        lines.append("  Tranche 2 (same allocations) fires next trading day.")
    else:
        lines.append("  Action: Execute remaining 50% of rebalancing trades.")
        lines.append("  Same allocations as Tranche 1 — no new signals computed.")
    
    lines.append("=" * 60)
    lines.append("")

    # ── Sanity checks ────────────────────────────────────────────────
    if sanity_errors:
        lines.append("SANITY CHECKS:  ✗ FAILED — DO NOT TRADE")
        for err in sanity_errors:
            lines.append(f"  ✗ {err}")
        lines.append("")
        lines.append("  Review the errors above before executing any trades.")
        lines.append("  Per IPS Section 5.4, do not trade until resolved.")
        lines.append("")
    else:
        lines.append("SANITY CHECKS:  ✓ All passed")
        total = sum(combined_allocation.values())
        lines.append(f"  ✓ Weights sum to {total:.1%}")
        lines.append("  ✓ No negative weights")
        max_pos = max(combined_allocation.values()) if combined_allocation else 0
        lines.append(f"  ✓ Largest position: {max_pos:.1%} (< 40% limit)")
        lines.append("")

    # ── Combined target allocation ───────────────────────────────────
    lines.append("TARGET ALLOCATION (% of total portfolio):")
    lines.append("")

    # Sort: largest positions first, cash last
    sorted_alloc = sorted(
        combined_allocation.items(),
        key=lambda x: (x[0] in ("CASH", "BIL"), -x[1]),
    )
    
    for ticker, weight in sorted_alloc:
        if weight > 0.001:
            bar = "█" * int(weight * 40)
            lines.append(f"  {ticker:6s} {weight:6.1%}  {bar}")
    
    lines.append(f"  {'------':6s} ------")
    lines.append(f"  {'TOTAL':6s} {sum(combined_allocation.values()):6.1%}")
    lines.append("")

    # ── Changes from previous month ──────────────────────────────────
    if previous_allocation:
        lines.append("CHANGES FROM PREVIOUS MONTH:")
        lines.append("")
        
        all_tickers = sorted(
            set(list(combined_allocation.keys()) + list(previous_allocation.keys()))
        )
        
        changes = []
        for ticker in all_tickers:
            new_wt = combined_allocation.get(ticker, 0)
            old_wt = previous_allocation.get(ticker, 0)
            diff = new_wt - old_wt
            if abs(diff) > 0.005:  # only show meaningful changes
                direction = "BUY" if diff > 0 else "SELL"
                changes.append(
                    f"  {ticker:6s} {old_wt:5.1%} → {new_wt:5.1%}  "
                    f"({direction} {abs(diff):.1%})"
                )
        
        if changes:
            for line in changes:
                lines.append(line)
        else:
            lines.append("  No changes from last month.")
        
        lines.append("")

    # ── Strategy breakdown ───────────────────────────────────────────
    lines.append("SIGNAL BREAKDOWN BY STRATEGY:")
    lines.append("")
    
    for strategy_name, summary in strategy_summaries.items():
        lines.append(f"[{strategy_name}]")
        lines.append(summary)
        lines.append("")

    # ── Footer ───────────────────────────────────────────────────────
    lines.append("-" * 60)
    lines.append("EXECUTION REMINDER:")
    lines.append("  Per IPS v1.2 Section 4.4, execute 50% of rebalancing")
    lines.append("  trades per tranche. Minimum trade threshold: for")
    lines.append("  positions being adjusted, skip the trade if drift is")
    lines.append("  less than 20% of the target allocation. Always execute")
    lines.append("  additions (from 0%) and removals (to 0%).")
    lines.append("")
    lines.append(f"  Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("=" * 60)
    
    body = "\n".join(lines)
    return subject, body


def run_sanity_checks(allocation: Dict[str, float]) -> list:
    """
    Run sanity checks on the combined allocation.
    Returns list of error strings. Empty list = all checks passed.
    """
    errors = []
    
    total = sum(allocation.values())
    if abs(total - 1.0) > 0.02:
        errors.append(f"Weights sum to {total:.3f}, expected 1.0 (±0.02)")
    
    for ticker, weight in allocation.items():
        if weight < -0.001:
            errors.append(f"Negative weight: {ticker} = {weight:.3f}")
    
    for ticker, weight in allocation.items():
        if weight > 0.40:
            errors.append(
                f"Position too large: {ticker} = {weight:.1%} (> 40% limit)"
            )
    
    if not allocation or all(v < 0.001 for v in allocation.values()):
        errors.append("Allocation is empty or all-zero")
    
    return errors
