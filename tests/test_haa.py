"""
Smoke tests for HAA strategy implementation.

These tests verify the mechanics are correct, not the historical performance.
For historical performance validation, compare against Allocate Smartly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from lib.momentum import momentum_13612u, momentum_13612w, total_return
from strategies.haa import compute_haa_signals


def make_prices(monthly_returns: list, start_price: float = 100.0) -> pd.Series:
    """Helper: create a price series from a list of monthly returns."""
    prices = [start_price]
    for r in monthly_returns:
        prices.append(prices[-1] * (1 + r))
    dates = pd.date_range("2020-01-31", periods=len(prices), freq="ME")
    return pd.Series(prices, index=dates)


def test_total_return():
    """Test basic total return calculation."""
    prices = make_prices([0.05, 0.03, -0.02, 0.04])
    # 1-month return should be the last return
    r1 = total_return(prices, 1)
    assert abs(r1 - 0.04) < 0.0001, f"Expected ~0.04, got {r1}"
    print("  ✓ total_return works correctly")


def test_momentum_13612u():
    """Test unweighted momentum formula."""
    # 13 months of prices needed for 12-month lookback
    returns = [0.02] * 12  # steady 2% monthly gains
    prices = make_prices(returns)
    
    mom = momentum_13612u(prices)
    
    # r1 ≈ 0.02, r3 ≈ 0.0612, r6 ≈ 0.1262, r12 ≈ 0.2682
    # Average ≈ (0.02 + 0.0612 + 0.1262 + 0.2682) / 4 ≈ 0.1189
    assert mom is not None and not np.isnan(mom)
    assert mom > 0, f"Expected positive momentum, got {mom}"
    print(f"  ✓ momentum_13612u = {mom:.4f} (expected ~0.12)")


def test_momentum_13612w():
    """Test weighted momentum formula."""
    returns = [0.02] * 12
    prices = make_prices(returns)
    
    mom = momentum_13612w(prices)
    
    # 12*r1 + 4*r3 + 2*r6 + 1*r12
    # ≈ 12*0.02 + 4*0.0612 + 2*0.1262 + 1*0.2682 ≈ 0.9058
    assert mom is not None and not np.isnan(mom)
    assert mom > 0
    print(f"  ✓ momentum_13612w = {mom:.4f} (expected ~0.91)")


def test_haa_risk_on():
    """Test HAA produces a risk-on allocation when TIP momentum is positive."""
    # Build a prices DataFrame with all HAA tickers
    np.random.seed(42)
    n_months = 14
    dates = pd.date_range("2023-01-31", periods=n_months, freq="ME")
    
    tickers = ["SPY", "IWM", "IEFA", "IEMG", "VNQ", "PDBC", "IEF", "TLT", "TIP", "BIL"]
    
    # All assets trending up → TIP positive → risk-on
    data = {}
    for t in tickers:
        start = 100
        prices = [start]
        for _ in range(n_months - 1):
            prices.append(prices[-1] * (1 + np.random.uniform(0.005, 0.03)))
        data[t] = prices
    
    prices_df = pd.DataFrame(data, index=dates)
    
    signal = compute_haa_signals(prices_df)
    
    assert signal.is_risk_on, "Expected risk-on with all assets trending up"
    assert len(signal.allocation) > 0, "Expected non-empty allocation"
    total_weight = sum(signal.allocation.values())
    assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, expected 1.0"
    print(f"  ✓ HAA risk-on: {len(signal.allocation)} assets, total weight {total_weight:.2f}")


def test_haa_risk_off():
    """Test HAA goes to defensive when TIP momentum is negative."""
    n_months = 14
    dates = pd.date_range("2023-01-31", periods=n_months, freq="ME")
    
    tickers = ["SPY", "IWM", "IEFA", "IEMG", "VNQ", "PDBC", "IEF", "TLT", "TIP", "BIL"]
    
    data = {}
    for t in tickers:
        start = 100
        prices = [start]
        for _ in range(n_months - 1):
            if t == "TIP":
                # TIP declining → negative momentum → risk-off
                prices.append(prices[-1] * (1 - np.random.uniform(0.01, 0.03)))
            else:
                prices.append(prices[-1] * (1 + np.random.uniform(0.005, 0.03)))
        data[t] = prices
    
    prices_df = pd.DataFrame(data, index=dates)
    
    signal = compute_haa_signals(prices_df)
    
    assert not signal.is_risk_on, "Expected risk-off with TIP declining"
    # Should be 100% in either IEF or BIL
    assert len(signal.allocation) == 1, f"Expected 1 position, got {len(signal.allocation)}"
    defensive_ticker = list(signal.allocation.keys())[0]
    assert defensive_ticker in ("IEF", "BIL"), f"Expected IEF or BIL, got {defensive_ticker}"
    print(f"  ✓ HAA risk-off: 100% {defensive_ticker}")


def test_haa_weights_sum_to_one():
    """Test that HAA allocations always sum to 1.0."""
    np.random.seed(123)
    n_months = 14
    dates = pd.date_range("2023-01-31", periods=n_months, freq="ME")
    
    tickers = ["SPY", "IWM", "IEFA", "IEMG", "VNQ", "PDBC", "IEF", "TLT", "TIP", "BIL"]
    
    # Mix of positive and negative trends
    data = {}
    for i, t in enumerate(tickers):
        start = 100
        prices = [start]
        for _ in range(n_months - 1):
            drift = 0.02 if i % 3 != 0 else -0.01
            prices.append(prices[-1] * (1 + drift + np.random.normal(0, 0.01)))
        data[t] = prices
    
    prices_df = pd.DataFrame(data, index=dates)
    signal = compute_haa_signals(prices_df)
    
    total = sum(signal.allocation.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"
    print(f"  ✓ Weights sum check: {total:.4f}")


def test_summary_format():
    """Test that the summary string is non-empty and readable."""
    np.random.seed(42)
    n_months = 14
    dates = pd.date_range("2023-01-31", periods=n_months, freq="ME")
    tickers = ["SPY", "IWM", "IEFA", "IEMG", "VNQ", "PDBC", "IEF", "TLT", "TIP", "BIL"]
    
    data = {}
    for t in tickers:
        prices = [100]
        for _ in range(n_months - 1):
            prices.append(prices[-1] * 1.015)
        data[t] = prices
    
    prices_df = pd.DataFrame(data, index=dates)
    signal = compute_haa_signals(prices_df)
    
    summary = signal.summary()
    assert len(summary) > 50, "Summary too short"
    assert "TIP" in summary or "momentum" in summary.lower()
    print(f"  ✓ Summary format OK ({len(summary)} chars)")


if __name__ == "__main__":
    print("Running HAA tests...\n")
    
    test_total_return()
    test_momentum_13612u()
    test_momentum_13612w()
    test_haa_risk_on()
    test_haa_risk_off()
    test_haa_weights_sum_to_one()
    test_summary_format()
    
    print("\n✓ All tests passed.")
