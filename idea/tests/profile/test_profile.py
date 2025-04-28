import os
from pathlib import Path
from unittest import TestCase

import pandas as pd
from parameterized import parameterized

from idea.profile import profile

RESOURCES_PATH = Path(__file__).parents[2] / "tests" / "resources" / "profile"
TESTS = [x[0] for x in os.walk(RESOURCES_PATH) if x[0]][1::]


class TestIdeaProfile(TestCase):
    @parameterized.expand(TESTS)
    def test_calculate_profile(self, test_path):
        """Test with parameterized input files and expected output."""

        # Load the input data
        input_fcd = pd.read_csv(f"{test_path}/input_fcd.csv", sep=";", index_col=0)
        input_fcd.index = pd.to_datetime(input_fcd.index, utc=True)

        # Load the expected result
        expected_result = pd.read_csv(f"{test_path}/expected.csv", sep=";")

        # Define the time range for the test
        start = pd.to_datetime("2024-01-01", utc=True)
        end = pd.to_datetime("2025-01-01", utc=True)

        # Run the profile calculation
        result = profile.calculate_profile(input_fcd, start, end)

        # Compare the result with the expected output
        pd.testing.assert_frame_equal(result, expected_result, check_dtype=False)
