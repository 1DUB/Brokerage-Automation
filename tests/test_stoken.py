"""
Test for Stoken’s ACA [Dynamic Bond]
"""

import unittest
import pandas as pd
from strategies.stoken import compute_stoken_signals
from lib.data import fetch_monthly_prices


class TestStoken(unittest.TestCase):
    def test_compute_signals(self):
        monthly = fetch_monthly_prices(
            ["SPY", "IEF", "GLD", "TLT", "VNQ"],
            months_history=15
        )
        
        sig = compute_stoken_signals(monthly)
        
        self.assertIsNotNone(sig)
        self.assertIsInstance(sig.signal_date, str)
        
        # Allocation must sum to 100%
        total = sum(sig.allocation.values())
        self.assertAlmostEqual(total, 1.0, places=4)
        
        # Should contain at least one of the assets from the three pairs
        allowed = {"SPY", "IEF", "GLD", "TLT", "VNQ", "CASH"}
        self.assertTrue(any(t in allowed for t in sig.allocation))
        
        # No negative weights
        for weight in sig.allocation.values():
            self.assertGreaterEqual(weight, 0.0)


if __name__ == "__main__":
    unittest.main()
