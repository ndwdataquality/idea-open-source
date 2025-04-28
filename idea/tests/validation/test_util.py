import unittest

import numpy as np
import pandas as pd

from idea.exceptions import IDEAError
from idea.validation.util import (
    calculate_minutes_no_coverage,
    calculate_running_mean,
    determine_coverage_profile_value,
    handle_profile_value,
    match_no_coverage_profile,
    sanitize_cov_values,
    set_segment_closure_status,
    update_counter,
    update_no_coverage_counters,
)


class TestCoverageFunctions(unittest.TestCase):
    def test_fcd_zero(self):
        # fcd = 0: Both counters must be updated with +1
        self.assertEqual(update_no_coverage_counters(0, 0, 0), (1, 1))
        self.assertEqual(update_no_coverage_counters(0, 2, 3), (3, 4))

    def test_fcd_one(self):
        # fcd = 1: FCD==0-counter reset, FCD in (0, 1)-counter is upped.
        self.assertEqual(update_no_coverage_counters(1, 5, 2), (0, 3))

    def test_fcd_not_zero_or_one(self):
        # fcd != 0 en fcd != 1: Both counters will be reset.
        self.assertEqual(update_no_coverage_counters(2, 4, 6), (0, 0))
        self.assertEqual(update_no_coverage_counters(5, 1, 1), (0, 0))

    def test_sequence(self):
        # Test een sequence of values.
        counters = (0, 0)
        for fcd in [0, 0, 1, 2]:
            counters = update_no_coverage_counters(fcd, *counters)
        self.assertEqual(counters, (0, 0))

    def test_fcd_out_of_range(self):
        # Test out of range fcd value
        with self.assertRaises(IDEAError):
            update_no_coverage_counters(11, 0, 0)
        with self.assertRaises(IDEAError):
            update_no_coverage_counters(-1, 0, 0)

    def test_update_counter(self):
        self.assertEqual(update_counter(True, 2), 3)
        self.assertEqual(update_counter(False, 4), 0)

    def test_calculate_minutes_no_coverage(self):
        df = pd.DataFrame(
            {"fcd": [0, 0, 1, 2, np.nan, 0]}, index=pd.date_range("2024-01-01", periods=6, freq="T")
        )
        result = calculate_minutes_no_coverage(df)
        self.assertListEqual(result["consecutive_zeros"].tolist(), [1, 2, 0, 0, 0, 1])
        self.assertListEqual(result["consecutive_low"].tolist(), [1, 2, 3, 0, 0, 1])

    def test_match_no_coverage_profile(self):
        coverage_df = pd.DataFrame(
            {"fcd": [0.2], "consecutive_zeros": [1], "consecutive_low": [2]},
            index=pd.to_datetime(["2024-01-01 10:00"]),
        )
        profile_df = pd.DataFrame(
            {
                "day_of_week": ["Monday"],
                "hour_of_day": [10],
                "max_consecutive_zeros_q95": [2],
                "max_consecutive_zeros_or_ones_q95": [4],
                "fcd_mean_median": [0.5],
            }
        )
        result = match_no_coverage_profile(coverage_df, profile_df)
        self.assertIn("max_consecutive_zeros_q95", result.columns)
        self.assertEqual(result.shape[0], 1)

    def test_determine_coverage_profile_value(self):
        row = pd.Series(
            {
                "fcd_mean_median": 0.3,
                "consecutive_low": 5,
                "consecutive_zeros": 2,
                "max_consecutive_zeros_or_ones_q95": 8,
                "max_consecutive_zeros_q95": 4,
            }
        )
        prev = row.copy()
        val = determine_coverage_profile_value(row, prev, cov_threshold_zeros_or_one_values=0.2)
        self.assertEqual(val, (5, 5, 8))  # Uses "low" branch since 0.3 > 0.2

    def test_calculate_running_mean(self):
        result = calculate_running_mean(5, 0.6, 2, 0.3)
        expected = (5 * 0.6 + 2 * 0.3) / (5 + 2)
        self.assertAlmostEqual(result, expected)

    def test_handle_profile_value(self):
        self.assertEqual(handle_profile_value(np.nan), 60)
        self.assertGreaterEqual(handle_profile_value(1), 1)

    def test_sanitize_cov_values(self):
        self.assertEqual(sanitize_cov_values(np.nan, 2), (0, 2))
        self.assertEqual(sanitize_cov_values(2, np.nan), (2, 0))

    def test_set_segment_closure_status(self):
        df = pd.DataFrame({"running_mean": [0.10, 0.80, 0.4]})
        result = set_segment_closure_status(df)
        expected = ["open", "closed", "undetermined"]
        self.assertListEqual(result["segment_closure_status"].tolist(), expected)
