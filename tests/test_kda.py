"""
Smoke tests for KDA strategy implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from lib.optimization import min_variance_weights, build_covariance
from strategies.kda import (
    compute_kda_signals,
    compute_blended_correlation,
    compute_volatilities,
    RISKY_UNIVERSE,
    CANARY_UNIVERSE,
    ALL_TICKERS as KDA_TICKERS,
)


def make_synthetic_data(seed=42, monthly_drift=None, trend_canaries=True, deterministic=True):
    """
    Generate synthetic monthly + daily price data for all KDA tickers.
    
    By default uses near-deterministic price paths (small noise) so test
    outcomes are predictable but covariance matrices are non-singular.
    Set deterministic=False for fully noisy random walks.
    
    Returns:
        (monthly_prices, daily_prices) tuple of DataFrames.
    """
    np.random.seed(seed)
    
    # Generate ~14 months of daily data (about 320 trading days)
    n_days = 320
    daily_dates = pd.bdate_range("2025-01-01", periods=n_days)
    
    daily_data = {}
    
    for i, ticker in enumerate(KDA_TICKERS):
        if monthly_drift is not None:
            drift = monthly_drift.get(ticker, 0.025)
        elif ticker in CANARY_UNIVERSE and not trend_canaries:
            drift = -0.025  # Strong negative trend for canaries (risk-off scenario)
        else:
            drift = 0.025   # Strong positive drift (dominates noise)
        
        daily_drift = drift / 21  # ~21 trading days per month
        
        if deterministic:
            # Strong trend + small per-asset noise.
            # Noise is small enough to not flip momentum signs but large
            # enough to give a non-singular covariance matrix.
            noise_scale = 0.002
            asset_offset = i * 0.00005
            daily_returns = np.random.normal(
                daily_drift + asset_offset, noise_scale, n_days
            )
        else:
            # Noisy random walk
            daily_returns = np.random.normal(daily_drift, 0.008, n_days)
        
        prices = 100 * np.cumprod(1 + daily_returns)
        daily_data[ticker] = prices
    
    daily_df = pd.DataFrame(daily_data, index=daily_dates)
    
    # Resample to month-end
    monthly_df = daily_df.resample("ME").last()
    
    return monthly_df, daily_df


# ── Optimization tests ───────────────────────────────────────────────

def test_min_variance_two_assets():
    """Min-variance with two uncorrelated assets of different vols."""
    # Asset 0 has vol 0.1, Asset 1 has vol 0.2, uncorrelated
    cov = np.array([[0.01, 0.0], [0.0, 0.04]])
    w = min_variance_weights(cov)
    
    # Theoretical answer: w_i ∝ 1/var_i, so w = [4, 1] / 5 = [0.8, 0.2]
    assert abs(w[0] - 0.8) < 0.01, f"Expected 0.8, got {w[0]}"
    assert abs(w[1] - 0.2) < 0.01, f"Expected 0.2, got {w[1]}"
    print("  ✓ Min-var prefers low-volatility asset correctly")


def test_min_variance_long_only():
    """Min-variance produces all non-negative weights."""
    np.random.seed(123)
    A = np.random.randn(7, 7)
    cov = A @ A.T  # PSD
    w = min_variance_weights(cov)
    
    assert all(w >= -1e-9), f"Negative weight found: {w}"
    assert abs(w.sum() - 1.0) < 1e-6, f"Weights sum to {w.sum()}"
    print(f"  ✓ Min-var long-only constraint respected (7 assets)")


def test_build_covariance():
    """Covariance construction from correlation + volatilities."""
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    vols = np.array([0.1, 0.2])
    cov = build_covariance(corr, vols)
    
    # Cov[0,0] = 0.1 * 0.1 * 1.0 = 0.01
    # Cov[1,1] = 0.2 * 0.2 * 1.0 = 0.04
    # Cov[0,1] = 0.1 * 0.2 * 0.5 = 0.01
    assert abs(cov[0,0] - 0.01) < 1e-9
    assert abs(cov[1,1] - 0.04) < 1e-9
    assert abs(cov[0,1] - 0.01) < 1e-9
    print("  ✓ Covariance construction correct")


# ── KDA strategy tests ───────────────────────────────────────────────

def test_kda_risk_on():
    """KDA should be fully risk-on when both canaries have positive momentum."""
    # All tickers trending up -> both canaries positive -> 100% risk-on
    monthly, daily = make_synthetic_data(seed=42)
    
    signal = compute_kda_signals(monthly, daily)
    
    assert signal.n_positive_canaries == 2, (
        f"Expected 2 positive canaries, got {signal.n_positive_canaries}"
    )
    assert signal.pct_aggressive == 1.0, (
        f"Expected 100% risk-on, got {signal.pct_aggressive*100:.0f}%"
    )
    assert len(signal.selected_assets) > 0, "Expected some risky assets selected"
    
    total = sum(signal.allocation.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"
    print(
        f"  ✓ KDA risk-on: {len(signal.selected_assets)} assets selected, "
        f"weights sum to {total:.3f}"
    )


def test_kda_risk_off():
    """KDA should be fully defensive when both canaries are negative."""
    monthly, daily = make_synthetic_data(seed=42, trend_canaries=False)
    
    signal = compute_kda_signals(monthly, daily)
    
    assert signal.n_positive_canaries == 0, (
        f"Expected 0 positive canaries, got {signal.n_positive_canaries}"
    )
    assert signal.pct_aggressive == 0.0, (
        f"Expected 0% risk-on, got {signal.pct_aggressive*100:.0f}%"
    )
    
    # Should be 100% in IEF or CASH
    total = sum(signal.allocation.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"
    
    # The defensive destination should be either IEF or CASH
    defensive_keys = {"IEF", "CASH"}
    has_defensive = any(k in signal.allocation for k in defensive_keys)
    assert has_defensive, f"Expected IEF or CASH in allocation, got {signal.allocation}"
    print(f"  ✓ KDA risk-off: 100% defensive, allocation = {signal.allocation}")


def test_kda_partial_risk_on():
    """KDA should be 50/50 when exactly one canary is positive."""
    # One canary positive, one negative
    monthly_drift = {t: 0.015 for t in KDA_TICKERS}
    monthly_drift["VWO"] = 0.015    # positive
    monthly_drift["BND"] = -0.015   # negative
    
    monthly, daily = make_synthetic_data(seed=42, monthly_drift=monthly_drift)
    
    signal = compute_kda_signals(monthly, daily)
    
    assert signal.n_positive_canaries == 1, (
        f"Expected 1 positive canary, got {signal.n_positive_canaries}"
    )
    assert signal.pct_aggressive == 0.5, (
        f"Expected 50% risk-on, got {signal.pct_aggressive*100:.0f}%"
    )
    
    total = sum(signal.allocation.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}"
    print(f"  ✓ KDA 50/50: weights sum to {total:.3f}, allocation has {len(signal.allocation)} positions")


def test_kda_summary_format():
    """KDA summary string is non-empty and readable."""
    monthly, daily = make_synthetic_data(seed=42)
    signal = compute_kda_signals(monthly, daily)
    
    summary = signal.summary()
    assert len(summary) > 100
    assert "canary" in summary.lower() or "Canaries" in summary
    print(f"  ✓ KDA summary OK ({len(summary)} chars)")


def test_kda_weights_always_sum_to_one():
    """Test KDA weights sum to 1.0 across many random scenarios."""
    for seed in range(10):
        monthly, daily = make_synthetic_data(seed=seed)
        signal = compute_kda_signals(monthly, daily)
        total = sum(signal.allocation.values())
        assert abs(total - 1.0) < 0.01, (
            f"Seed {seed}: weights sum to {total}, expected 1.0"
        )
    print("  ✓ KDA weights consistently sum to 1.0 across 10 random scenarios")


def test_kda_min_variance_prefers_lower_vol():
    """
    Test the min-variance optimizer directly on a covariance matrix
    with clearly different volatilities. This is more reliable than
    testing via the full KDA pipeline where asset selection randomness
    can produce same-volatility selections.
    """
    # Build a 5x5 covariance matrix with 2 low-vol assets and 3 high-vol assets
    # Uncorrelated, so min-var should strongly prefer the low-vol ones
    vols = np.array([0.05, 0.05, 0.30, 0.30, 0.30])  # 5%, 5%, 30%, 30%, 30%
    corr = np.eye(5)  # identity → uncorrelated
    cov = build_covariance(corr, vols)
    
    weights = min_variance_weights(cov)
    
    # The two low-vol assets should together dominate the allocation
    low_vol_weight = weights[0] + weights[1]
    high_vol_weight = weights[2] + weights[3] + weights[4]
    
    assert low_vol_weight > high_vol_weight, (
        f"Low-vol total {low_vol_weight:.3f} should exceed "
        f"high-vol total {high_vol_weight:.3f}"
    )
    assert low_vol_weight > 0.9, (
        f"Low-vol assets should dominate, got {low_vol_weight:.3f}"
    )
    print(
        f"  ✓ Min-var prefers low-vol: "
        f"{low_vol_weight*100:.0f}% low-vol vs {high_vol_weight*100:.0f}% high-vol"
    )


if __name__ == "__main__":
    print("Running KDA tests...\n")
    
    print("Optimizer tests:")
    test_min_variance_two_assets()
    test_min_variance_long_only()
    test_build_covariance()
    
    print("\nKDA strategy tests:")
    test_kda_risk_on()
    test_kda_risk_off()
    test_kda_partial_risk_on()
    test_kda_summary_format()
    test_kda_weights_always_sum_to_one()
    test_kda_min_variance_prefers_lower_vol()
    
    print("\n✓ All tests passed.")
