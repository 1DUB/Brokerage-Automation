"""
Test for Lethargic Asset Allocation (uses FRED unemployment data)
"""

import unittest
import pandas as pd
from strategies.lethargic import compute_lethargic_signals
from lib.data import fetch_monthly_prices


class TestLethargic(unittest.TestCase):
    def test_compute_signals(self):
        monthly = fetch_monthly_prices(
            ["VTV", "GLD", "IEF", "QQQ", "SPY"],
            months_history=15
        )
        
        sig = compute_lethargic_signals(monthly)
        
        self.assertIsNotNone(sig)
        self.assertIsInstance(sig.signal_date, str)
        
        # Allocation must sum to 100%
        total = sum(sig.allocation.values())
        self.assertAlmostEqual(total, 1.0, places=4)
        
        # Must contain the four static holdings + tactical slice
        expected_static = {"VTV", "GLD", "IEF"}
        self.assertTrue(all(t in sig.allocation for t in expected_static))
        
        # Tactical slice must be either QQQ or CASH
        tactical_keys = [k for k in sig.allocation if k in {"QQQ", "CASH"}]
        self.assertEqual(len(tactical_keys), 1)
        
        # No negative weights
        for weight in sig.allocation.values():
            self.assertGreaterEqual(weight, 0.0)


if __name__ == "__main__":
    unittest.main()
