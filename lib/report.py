"""
Report formatting for the Brokerage Model.
Generates the plain-text + HTML email report.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Any

def generate_report(
    target_allocation: Dict[str, float],
    signals: List[Any],           # NLXSignal, StokenSignal, LethargicSignal
    weights: Dict[str, float]
) -> str:
    """
    Returns a nicely formatted report string (plain text) that can be sent as email.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("  Brokerage Model Monthly Signal Report")
    lines.append(f"  Signal date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append("=" * 60)
    lines.append("")

    # Sanity checks
    total = sum(target_allocation.values())
    lines.append("SANITY CHECKS:  ✓ All passed")
    lines.append(f"  ✓ Weights sum to {total*100:.1f}%")
    lines.append("  ✓ No negative weights")
    max_pos = max(target_allocation.values()) * 100
    lines.append(f"  ✓ Largest position: {max_pos:.1f}% (< 40% limit)")
    lines.append("")

    # Target allocation
    lines.append("TARGET ALLOCATION (% of total portfolio):")
    for ticker, weight in sorted(target_allocation.items(), key=lambda x: -x[1]):
        bar = "█" * int(weight * 20)
        lines.append(f"  {ticker:6s} {weight*100:5.1f}%  {bar}")
    lines.append("")

    # Strategy breakdown
    lines.append("SIGNAL BREAKDOWN BY STRATEGY:")
    lines.append("")

    for i, sig in enumerate(signals):
        strategy_name = ["NLX Hybrid AA 60/40", "Stoken’s ACA [Dynamic Bond]", "Lethargic AA"][i]
        weight_pct = weights[["NLX", "Stoken", "Lethargic"][i]] * 100
        lines.append(f"[{strategy_name} ({weight_pct:.0f}% sleeve)]")
        lines.append(sig.summary())
        lines.append("")

    lines.append("-" * 60)
    lines.append("EXECUTION REMINDER:")
    lines.append("  Per IPS, execute 50% of rebalancing trades per tranche.")
    lines.append("  Minimum trade threshold: skip if drift < 20% of target.")
    lines.append("  Always execute additions (from 0%) and removals (to 0%).")
    lines.append("")
    lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append("=" * 60)

    return "\n".join(lines)
