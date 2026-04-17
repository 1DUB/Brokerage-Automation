"""
Smoke tests for Vitral Multi-Asset Momentum implementation.

These tests verify the mechanics are correct, not historical performance.
The calibration against the paper's 590.66% target is a separate validation
done via a backtest script.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

from strategies.vitral import (
    compute_vitral_signals,
    compute_raw_signals,
    compute_correlations_with_portfolio,
    apply_correlation_adjustment,
    rank_assets,
    count_negative_signals,
    _total_return_days,
    _price_minus_sma,
    _risk_adjusted_momentum,
    RISK_ON_UNIVERSE,
    RISK_OFF_UNIVERSE,
    ALL_TICKERS as VITRAL_TICKERS,
    SIGNAL_SPECS,
    TOP_N,
)


def make_synthetic_vitral_data(
    seed=42,
    trend_overrides=None,
    all_trending_up=True,
    all_trending_down=False,
):
    """
    Generate synthetic daily price data for all Vitral tickers.
    
    Returns:
        (monthly_prices, daily_prices) tuple of DataFrames.
    """
    np.random.seed(seed)
    
    n_days = 400  # > 252 needed for 12-month lookback + buffer
    daily_dates = pd.bdate_range("2024-01-01", periods=n_days)
    
    daily_data = {}
    
    for i, ticker in enumerate(VITRAL_TICKERS):
        if trend_overrides is not None and ticker in trend_overrides:
            drift = trend_overrides[ticker]
        elif all_trending_down:
            drift = -0.02  # Monthly drift
        elif all_trending_up:
            drift = 0.02
        else:
            drift = 0.0
        
        daily_drift = drift / 21  # 21 trading days per month
        
        # Add small noise but keep trend signs dominant
        noise_scale = 0.003
        asset_offset = i * 0.00003  # tiny per-asset differentiation
        daily_returns = np.random.normal(
            daily_drift + asset_offset, noise_scale, n_days
        )
        
        prices = 100 * np.cumprod(1 + daily_returns)
        daily_data[ticker] = prices
    
    daily_df = pd.DataFrame(daily_data, index=daily_dates)
    monthly_df = daily_df.resample("ME").last()
    
    return monthly_df, daily_df


# ── Individual signal function tests ─────────────────────────────────

def test_total_return_days():
    """Basic total return calculation."""
    # Series where price doubled over 21 days
    prices = pd.Series([100, 100, 100] + [200] * 19, 
                        index=pd.bdate_range("2024-01-01", periods=22))
    r = _total_return_days(prices, 21)
    # Price now is 200, 21 days ago was 100
    assert abs(r - 1.0) < 0.001, f"Expected ~1.0, got {r}"
    print(f"  ✓ _total_return_days works: {r:.4f}")


def test_price_minus_sma():
    """Price minus SMA calculation."""
    # Flat prices: PMA should be 0
    prices = pd.Series([100] * 100, index=pd.bdate_range("2024-01-01", periods=100))
    r = _price_minus_sma(prices, 50)
    assert abs(r) < 0.001, f"Expected ~0 for flat prices, got {r}"
    
    # Rising prices: PMA should be positive
    prices = pd.Series(np.linspace(100, 200, 100),
                        index=pd.bdate_range("2024-01-01", periods=100))
    r = _price_minus_sma(prices, 50)
    assert r > 0, f"Expected positive PMA for rising prices, got {r}"
    print(f"  ✓ _price_minus_sma works: flat=0, rising={r:.4f}")


def test_risk_adjusted_momentum():
    """Risk-adjusted momentum penalizes choppy paths."""
    # Smooth upward path: should give high RA score (near 1.0 max)
    smooth = pd.Series(np.linspace(100, 120, 100),
                      index=pd.bdate_range("2024-01-01", periods=100))
    ra_smooth = _risk_adjusted_momentum(smooth, 63)
    
    # Choppy but ending at same place: should give lower RA score
    np.random.seed(42)
    choppy_values = 100 + np.cumsum(np.random.normal(0.2, 2.0, 100))
    # Force endpoint to match smooth
    choppy_values = choppy_values + (120 - choppy_values[-1])
    choppy = pd.Series(choppy_values, index=pd.bdate_range("2024-01-01", periods=100))
    ra_choppy = _risk_adjusted_momentum(choppy, 63)
    
    assert ra_smooth > ra_choppy, (
        f"Smooth RA ({ra_smooth:.3f}) should be > choppy RA ({ra_choppy:.3f})"
    )
    print(f"  ✓ _risk_adjusted_momentum: smooth={ra_smooth:.4f}, choppy={ra_choppy:.4f}")


# ── Signal aggregation tests ─────────────────────────────────────────

def test_compute_raw_signals():
    """Raw signals computation produces 9 scores per asset."""
    monthly, daily = make_synthetic_vitral_data()
    signals = compute_raw_signals(daily, RISK_ON_UNIVERSE)
    
    assert signals.shape == (13, 9), f"Expected (13, 9), got {signals.shape}"
    assert not signals.isna().all().all(), "All signals are NaN — something broke"
    
    # In an uptrend, most signals should be positive
    positive_count = (signals > 0).sum().sum()
    total_count = signals.size
    assert positive_count > total_count * 0.7, (
        f"Expected most signals positive in uptrend, got {positive_count}/{total_count}"
    )
    print(f"  ✓ compute_raw_signals: {signals.shape}, {positive_count}/{total_count} positive")


def test_correlation_adjustment():
    """Correlation adjustment divides by (1 + rho)."""
    # Manually construct a toy example
    raw = pd.DataFrame({
        "sig1": [1.0, 1.0, 1.0],
        "sig2": [0.5, 0.5, 0.5],
    }, index=["A", "B", "C"])
    
    # Perfect positive correlation (rho=1) → divide by 2 → halve
    # Zero correlation (rho=0) → divide by 1 → unchanged
    # Negative correlation (rho=-0.5) → divide by 0.5 → double
    corrs = {"A": 1.0, "B": 0.0, "C": -0.5}
    
    adjusted = apply_correlation_adjustment(raw, corrs)
    
    assert abs(adjusted.loc["A", "sig1"] - 0.5) < 0.001
    assert abs(adjusted.loc["B", "sig1"] - 1.0) < 0.001
    assert abs(adjusted.loc["C", "sig1"] - 2.0) < 0.001
    print("  ✓ Correlation adjustment works correctly")


def test_rank_assets():
    """Rank scoring assigns 13 to best, 1 to worst per signal."""
    # 3 assets, 2 signals — use 3 to get 3-2-1 scoring
    signals = pd.DataFrame({
        "s1": [10.0, 5.0, 1.0],  # A best, B mid, C worst
        "s2": [1.0, 5.0, 10.0],  # C best, B mid, A worst
    }, index=["A", "B", "C"])
    
    scores = rank_assets(signals)
    # A: 3 + 1 = 4
    # B: 2 + 2 = 4
    # C: 1 + 3 = 4
    # All equal! Check actual values:
    assert scores["A"] == 4.0, f"A: {scores['A']}"
    assert scores["B"] == 4.0, f"B: {scores['B']}"
    assert scores["C"] == 4.0, f"C: {scores['C']}"
    
    # Now a clearer case: A dominates both signals
    signals2 = pd.DataFrame({
        "s1": [10.0, 5.0, 1.0],
        "s2": [10.0, 5.0, 1.0],
    }, index=["A", "B", "C"])
    scores2 = rank_assets(signals2)
    assert scores2["A"] == 6.0  # 3+3
    assert scores2["B"] == 4.0  # 2+2
    assert scores2["C"] == 2.0  # 1+1
    print(f"  ✓ rank_assets produces correct scores")


def test_count_negative_signals():
    """Negative signal counting."""
    signals = pd.DataFrame({
        "s1": [1.0, -1.0, -1.0],
        "s2": [-0.5, -0.5, 0.5],
        "s3": [0.0, -0.2, 0.3],  # Note: 0 counts as negative per ≤0
    }, index=["A", "B", "C"])
    
    counts = count_negative_signals(signals)
    assert counts["A"] == 2, f"A: {counts['A']}"  # s2 and s3 (zero)
    assert counts["B"] == 3, f"B: {counts['B']}"  # all three
    assert counts["C"] == 1, f"C: {counts['C']}"  # only s1
    print(f"  ✓ count_negative_signals works")


# ── Full Vitral strategy tests ───────────────────────────────────────

def test_vitral_risk_on_uptrend():
    """When all assets trend up, Vitral should be fully risk-on."""
    monthly, daily = make_synthetic_vitral_data(all_trending_up=True)
    signal = compute_vitral_signals(monthly, daily)
    
    assert signal.n_negative_assets == 0, (
        f"Expected 0 negative assets in uptrend, got {signal.n_negative_assets}"
    )
    assert signal.pct_risk_on == 1.0, (
        f"Expected 100% risk-on, got {signal.pct_risk_on*100:.0f}%"
    )
    assert len(signal.selected_assets) == TOP_N
    
    total = sum(signal.allocation.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"
    
    # Each selected asset should get 1/5 = 20%
    for ticker in signal.selected_assets:
        w = signal.allocation[ticker]
        assert abs(w - 0.2) < 0.005, f"{ticker}: {w}"
    print(f"  ✓ Vitral risk-on: {len(signal.selected_assets)} assets @ 20% each")


def test_vitral_risk_off_downtrend():
    """When all assets trend down, Vitral should be fully risk-off."""
    monthly, daily = make_synthetic_vitral_data(
        all_trending_up=False, all_trending_down=True
    )
    signal = compute_vitral_signals(monthly, daily)
    
    # With everything trending down hard, most assets should be classified negative
    assert signal.n_negative_assets >= 10, (
        f"Expected most assets negative, got {signal.n_negative_assets}/13"
    )
    # With n ≥ 10, 1 - n/9.75 should be ≤ 0, so 0% risk-on
    assert signal.pct_risk_on == 0.0, (
        f"Expected 0% risk-on, got {signal.pct_risk_on*100:.0f}%"
    )
    
    total = sum(signal.allocation.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"
    
    # Allocation should be 100% in risk-off
    assert signal.risk_off_choice in signal.allocation
    assert signal.allocation[signal.risk_off_choice] >= 0.99
    print(f"  ✓ Vitral risk-off: 100% {signal.risk_off_choice}")


def test_vitral_partial_risk_on():
    """Mixed regime: some assets trending up, some down."""
    # 5 assets trending down strongly, 8 trending up
    overrides = {}
    down_tickers = ["VGK", "EWJ", "VNQ", "HYG", "TLT"]
    for t in down_tickers:
        overrides[t] = -0.025
    for t in RISK_ON_UNIVERSE:
        if t not in overrides:
            overrides[t] = 0.02
    # Risk-off assets flat
    for t in RISK_OFF_UNIVERSE:
        overrides[t] = 0.001
    
    monthly, daily = make_synthetic_vitral_data(trend_overrides=overrides)
    signal = compute_vitral_signals(monthly, daily)
    
    # Some but not all assets should be negative
    assert 3 <= signal.n_negative_assets <= 7, (
        f"Expected mixed regime, got {signal.n_negative_assets}/13 negative"
    )
    # Partial risk-on
    assert 0.0 < signal.pct_risk_on < 1.0, (
        f"Expected partial risk-on, got {signal.pct_risk_on*100:.0f}%"
    )
    
    total = sum(signal.allocation.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"
    print(
        f"  ✓ Vitral partial: {signal.n_negative_assets}/13 negative, "
        f"risk-on {signal.pct_risk_on*100:.0f}%"
    )


def test_vitral_weights_sum_to_one():
    """Weights always sum to 1 across various seeds."""
    for seed in range(5):
        monthly, daily = make_synthetic_vitral_data(seed=seed)
        signal = compute_vitral_signals(monthly, daily)
        total = sum(signal.allocation.values())
        assert abs(total - 1.0) < 0.01, (
            f"Seed {seed}: weights sum to {total}"
        )
    print(f"  ✓ Weights consistently sum to 1.0 across 5 random scenarios")


def test_vitral_summary():
    """Summary string is non-empty and mentions key fields."""
    monthly, daily = make_synthetic_vitral_data()
    signal = compute_vitral_signals(monthly, daily)
    summary = signal.summary()
    
    assert len(summary) > 100
    assert "Risk-on" in summary or "risk-on" in summary.lower()
    assert "allocation" in summary.lower()
    print(f"  ✓ Vitral summary OK ({len(summary)} chars)")


def test_vitral_signal_count():
    """Check the signal specification is exactly 9."""
    assert len(SIGNAL_SPECS) == 9, f"Expected 9 signal specs, got {len(SIGNAL_SPECS)}"
    
    measures = set(spec[0] for spec in SIGNAL_SPECS)
    assert measures == {"TR", "PMA", "RA"}, f"Measures: {measures}"
    
    # 3 of each measure
    for measure in ["TR", "PMA", "RA"]:
        count = sum(1 for spec in SIGNAL_SPECS if spec[0] == measure)
        assert count == 3, f"{measure}: {count} signals (expected 3)"
    print(f"  ✓ Signal spec: 9 total, 3 each of TR/PMA/RA")


if __name__ == "__main__":
    print("Running Vitral tests...\n")
    
    print("Signal function tests:")
    test_total_return_days()
    test_price_minus_sma()
    test_risk_adjusted_momentum()
    
    print("\nSignal aggregation tests:")
    test_compute_raw_signals()
    test_correlation_adjustment()
    test_rank_assets()
    test_count_negative_signals()
    
    print("\nFull strategy tests:")
    test_vitral_signal_count()
    test_vitral_risk_on_uptrend()
    test_vitral_risk_off_downtrend()
    test_vitral_partial_risk_on()
    test_vitral_weights_sum_to_one()
    test_vitral_summary()
    
    print("\n✓ All tests passed.")
