"""
Test for NLX Hybrid AA 60/40
"""

import unittest
import pandas as pd
from strategies.nlx import compute_nlx_signals
from lib.data import fetch_monthly_prices


class TestNLX(unittest.TestCase):
    def test_compute_signals(self):
        # Fetch enough history for 12-month momentum
        monthly = fetch_monthly_prices(
            ["SPY", "IWM", "IEFA", "IEMG", "VNQ", "PDBC", "IEF", "TLT", "BIL", "TIP"],
            months_history=15
        )
        
        sig = compute_nlx_signals(monthly)
        
        self.assertIsNotNone(sig)
        self.assertIsInstance(sig.signal_date, str)
        
        # Allocation must sum to 100%
        total = sum(sig.allocation.values())
        self.assertAlmostEqual(total, 1.0, places=4)
        
        # On risk-on it should be 60/40 SPY/IEF
        # On risk-off it should be 100% defensive (BIL or IEF)
        if sig.is_risk_on:
            self.assertIn("SPY", sig.allocation)
            self.assertIn("IEF", sig.allocation)
            self.assertAlmostEqual(sig.allocation["SPY"], 0.60, places=2)
            self.assertAlmostEqual(sig.allocation["IEF"], 0.40, places=2)
        else:
            self.assertTrue(any(t in ["BIL", "IEF"] for t in sig.allocation))


if __name__ == "__main__":
    unittest.main()
