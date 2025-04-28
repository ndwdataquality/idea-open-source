import os
from pathlib import Path
from unittest import TestCase

import pandas as pd
from parameterized import parameterized

from idea.profile.profile import calculate_profile
from idea.validation.validation import validate_roadwork

RESOURCES_PATH = Path(__file__).parents[2] / "tests" / "resources" / "validation"
TESTS = [x[0] for x in os.walk(RESOURCES_PATH) if x[0]][1::]


class TestIdeaValidation(TestCase):
    @parameterized.expand(TESTS)
    def test_validate_roadwork(self, test_path):
        """Test with parameterized input files and expected output."""
        # Load the input data
        input_fcd = pd.read_csv(f"{test_path}/fcd_profile_input.csv", sep=";", index_col=0)
        input_fcd.index = pd.to_datetime(input_fcd.index, utc=True)

        fcd_during_roadwork = pd.read_csv(
            f"{test_path}/fcd_during_roadwork.csv", sep=";", index_col=0
        )
        fcd_during_roadwork.index = pd.to_datetime(fcd_during_roadwork.index, utc=True)
        start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        end = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

        # Load the expected result
        expected_result = pd.read_csv(f"{test_path}/expected.csv", sep=";", index_col=0)
        expected_result.index = pd.to_datetime(expected_result.index, utc=True)
        expected_result = expected_result.reset_index()

        # Run the profile calculation and the validation algorithm
        profile = calculate_profile(input_fcd, start, end)
        result = validate_roadwork(fcd_during_roadwork, profile)

        # Compare the result with the expected output
        pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)
