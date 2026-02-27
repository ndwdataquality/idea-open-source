import math
import unittest

import numpy as np
import pandas as pd

from idea.constants import K_MAX, K_START, MINIMUM_PROFILE_VALUE
from idea.exceptions import IDEAError
from idea.validation.util import (
    apply_momentum,
    calculate_minutes_no_coverage,
    calculate_running_mean,
    determine_coverage_profile_value,
    handle_profile_value,
    match_no_coverage_profile,
    sanitize_cov_values,
    set_segment_closure_status,
    update_counter,
    update_k,
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
            {"fcd": [0, 0, 1, 2, np.nan, 0]},
            index=pd.date_range("2024-01-01", periods=6, freq="min"),
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


class TestUpdateK(unittest.TestCase):
    def test_resets_to_k_start_when_no_vehicles(self):
        # coverage == 0: always resets regardless of other inputs
        self.assertAlmostEqual(update_k(5.0, 0, 3, 2, 5.0), K_START)

    def test_doubles_when_event_within_decay_window(self):
        # current_minutes_no_low == 0 (event now), previous <= decay_window
        self.assertAlmostEqual(update_k(2.0, 3, 0, 4, 5.0), 4.0)

    def test_unchanged_when_event_outside_decay_window(self):
        # current_minutes_no_low == 0 but previous > decay_window
        self.assertAlmostEqual(update_k(2.0, 3, 0, 6, 5.0), 2.0)

    def test_unchanged_when_no_current_event(self):
        # current_minutes_no_low > 0: neither reset nor boost
        self.assertAlmostEqual(update_k(2.0, 3, 2, 1, 5.0), 2.0)

    def test_capped_at_k_max(self):
        # k * 2 would exceed K_MAX
        self.assertAlmostEqual(update_k(K_MAX, 3, 0, 2, 5.0), K_MAX)

    def test_no_vehicles_takes_priority_over_event(self):
        # coverage == 0 always resets even if current_minutes_no_low == 0
        self.assertAlmostEqual(update_k(5.0, 0, 0, 2, 5.0), K_START)


class TestApplyMomentum(unittest.TestCase):
    def test_no_momentum_when_high_coverage(self):
        # coverage >= COV_HIGH (6): momentum conditions not met, values unchanged
        running_mean, k = apply_momentum(
            coverage=7.0,
            coverage_profile_value=8.0,
            current_minutes_no_low=0,
            previous_minutes_no_low=0,
            profile_cov_ones=10.0,
            k=2.0,
            running_mean=0.8,
        )
        self.assertAlmostEqual(running_mean, 0.8)
        self.assertAlmostEqual(k, 2.0)

    def test_no_momentum_when_sudden_drop(self):
        # (coverage_profile_value - coverage) > COV_DROP_LIMIT (8): momentum not applied
        running_mean, k = apply_momentum(
            coverage=1.0,
            coverage_profile_value=10.0,  # 10 - 1 = 9 > 8
            current_minutes_no_low=0,
            previous_minutes_no_low=0,
            profile_cov_ones=10.0,
            k=2.0,
            running_mean=0.8,
        )
        self.assertAlmostEqual(running_mean, 0.8)
        self.assertAlmostEqual(k, 2.0)

    def test_alpha_scales_running_mean(self):
        # Verify the running mean is multiplied by exp(-k * echo)
        # profile_cov_ones=10 -> decay_window=5, current_minutes_no_low=0 -> echo=1.0
        # coverage=2 (no vehicles -> k stays at 2.0), alpha=exp(-2.0)
        running_mean, _ = apply_momentum(
            coverage=2.0,
            coverage_profile_value=5.0,
            current_minutes_no_low=0,
            previous_minutes_no_low=6,  # outside window, k unchanged
            profile_cov_ones=10.0,
            k=2.0,
            running_mean=0.8,
        )
        expected = 0.8 * math.exp(-2.0 * 1.0)
        self.assertAlmostEqual(running_mean, expected)

    def test_echo_zero_when_minutes_exceed_decay_window(self):
        # current_minutes_no_low >= decay_window: echo clamps to 0, alpha = 1, no scaling
        # profile_cov_ones=10 -> decay_window=5; current_minutes_no_low=10 >= 5
        running_mean, _ = apply_momentum(
            coverage=2.0,
            coverage_profile_value=5.0,
            current_minutes_no_low=10,
            previous_minutes_no_low=9,
            profile_cov_ones=10.0,
            k=2.0,
            running_mean=0.8,
        )
        self.assertAlmostEqual(running_mean, 0.8)

    def test_nan_profile_cov_ones_uses_minimum_profile_value(self):
        # NaN profile_cov_ones is clamped to MINIMUM_PROFILE_VALUE (5)
        # decay_window = 0.5 * 5 = 2.5, current_minutes_no_low=0 -> echo=1.0
        running_mean_nan, k_nan = apply_momentum(
            coverage=2.0,
            coverage_profile_value=5.0,
            current_minutes_no_low=0,
            previous_minutes_no_low=6,
            profile_cov_ones=np.nan,
            k=2.0,
            running_mean=0.8,
        )
        running_mean_min, k_min = apply_momentum(
            coverage=2.0,
            coverage_profile_value=5.0,
            current_minutes_no_low=0,
            previous_minutes_no_low=6,
            profile_cov_ones=float(MINIMUM_PROFILE_VALUE),
            k=2.0,
            running_mean=0.8,
        )
        self.assertAlmostEqual(running_mean_nan, running_mean_min)
        self.assertAlmostEqual(k_nan, k_min)
