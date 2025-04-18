import datetime as dt
from unittest import TestCase

import numpy as np
import pandas as pd

from idea.profile import util


class TestIdeaUtil(TestCase):
    def test_quantile_95_empty(self):
        """Test with an empty series."""
        series = pd.Series([], dtype=float)
        result = util.quantile_95(series)
        self.assertTrue(pd.isna(result))

    def test_series_with_nan(self):
        """Test with a series containing NaN values."""
        series = pd.Series([1, 2, 3, 4, 5, float("nan")])
        result = util.quantile_95(series)
        expected = 4.8
        self.assertAlmostEqual(result, expected)

    def test_negative_values(self):
        """Test with a series containing negative values."""
        series = pd.Series([-10, -5, 0, 5, 10])
        result = util.quantile_95(series)
        expected = 9
        self.assertAlmostEqual(result, expected)

    def test_interpolate_missing_minutes(self):
        """Test interpolate_missing_minutes function with various input cases."""
        # Define test cases
        tests = [
            # Case with missing minutes in the time range
            {
                "input": pd.DataFrame(
                    {"value": [1, 2, 3, 4]},
                    index=pd.to_datetime(
                        [
                            "2024-01-01 00:00",
                            "2024-01-01 00:01",
                            "2024-01-01 00:03",
                            "2024-01-01 00:04",
                        ]
                    ),
                ),
                "expected": pd.DataFrame(
                    {"value": [1, 2, np.nan, 3, 4, np.nan]},
                    index=pd.date_range("2024-01-01 00:00", "2024-01-01 00:05", freq="min"),
                ),
                "description": "Missing minutes in time range",
            },
            # Case with no missing minutes (complete DataFrame)
            {
                "input": pd.DataFrame(
                    {"value": [1, 2, 3, 4, 5, 6]},
                    index=pd.to_datetime(
                        [
                            "2024-01-01 00:00",
                            "2024-01-01 00:01",
                            "2024-01-01 00:02",
                            "2024-01-01 00:03",
                            "2024-01-01 00:04",
                            "2024-01-01 00:05",
                        ]
                    ),
                ),
                "expected": pd.DataFrame(
                    {"value": [1, 2, 3, 4, 5, 6]},
                    index=pd.date_range("2024-01-01 00:00", "2024-01-01 00:05", freq="min"),
                ),
                "description": "No missing minutes",
            },
        ]

        # Test each case
        for i, test_case in enumerate(tests):
            with self.subTest(f"Case {i+1}: {test_case['description']}"):
                result = util.interpolate_missing_minutes(
                    test_case["input"], dt.datetime(2024, 1, 1, 0, 0), dt.datetime(2024, 1, 1, 0, 5)
                )

                # Check if the result has the correct values and indexes
                pd.testing.assert_frame_equal(result, test_case["expected"], check_dtype=False)

    def test_fill_nan_in_subset(self):
        """Test filling NaN values in specified columns."""
        df = pd.DataFrame(
            {
                "col1": [1, np.nan, 3],
                "col2": [np.nan, 5, np.nan],
                "col3": [7, 8, 9],
            }
        )
        column_subset = ["col1", "col2"]
        expected = pd.DataFrame(
            {
                "col1": [1, 0, 3],
                "col2": [0, 5, 0],
                "col3": [7, 8, 9],
            }
        ).astype({"col1": "int16", "col2": "int16"})

        result = util.fill_nan_columns_with_zeros(df, column_subset)
        pd.testing.assert_frame_equal(result, expected)

    def test_no_nan_in_subset(self):
        """Test when there are no NaN values in the specified columns."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
                "col3": [7, 8, 9],
            }
        )
        column_subset = ["col1", "col2"]
        expected = df.astype({"col1": "int16", "col2": "int16"})

        result = util.fill_nan_columns_with_zeros(df, column_subset)
        pd.testing.assert_frame_equal(result, expected)

    def test_empty_dataframe(self):
        """Test behavior with an empty DataFrame."""
        df = pd.DataFrame(columns=["col1", "col2", "col3"])
        column_subset = ["col1", "col2"]
        expected = pd.DataFrame(columns=["col1", "col2", "col3"]).astype(
            {"col1": "int16", "col2": "int16"}
        )

        result = util.fill_nan_columns_with_zeros(df, column_subset)
        pd.testing.assert_frame_equal(result, expected)

    def test_subset_not_in_dataframe(self):
        """Test behavior when specified columns are not in the DataFrame."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3],
                "col2": [4, 5, 6],
            }
        )
        column_subset = ["col3"]
        with self.assertRaises(KeyError):
            util.fill_nan_columns_with_zeros(df, column_subset)

    def test_non_numeric_columns(self):
        """Test behavior when non-numeric columns are included in the subset."""
        df = pd.DataFrame(
            {
                "col1": [1, np.nan, 3],
                "col2": ["a", "b", "c"],
                "col3": [7, 8, 9],
            }
        )
        column_subset = ["col1", "col2"]
        with self.assertRaises(ValueError):
            util.fill_nan_columns_with_zeros(df, column_subset)

    def test_add_periods_valid_index(self):
        """Test adding period columns with a valid DatetimeIndex."""
        df = pd.DataFrame(
            {"value": [10, 20, 30]},
            index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 23:00"]),
        )
        expected = pd.DataFrame(
            {
                "value": [10, 20, 30],
                "day_of_week": [0, 0, 1],
                "hour_of_day": [0, 12, 23],
            },
            index=pd.to_datetime(["2024-01-01 00:00", "2024-01-01 12:00", "2024-01-02 23:00"]),
        )
        expected["day_of_week"] = expected["day_of_week"].astype("int32")
        expected["hour_of_day"] = expected["hour_of_day"].astype("int32")
        result = util.add_periods(df)
        pd.testing.assert_frame_equal(result, expected)

    def test_filter_max_consecutive_60(self):
        # Define the expected values for different scenarios
        tests = [
            # No changes
            {
                "input": {
                    "max_consecutive_zeros_or_ones": [0, 1, 2, 3],
                    "hour_of_date": [
                        dt.datetime(2024, 1, 1, 0, 0),
                        dt.datetime(2024, 1, 1, 1, 0),
                        dt.datetime(2024, 1, 1, 2, 0),
                        dt.datetime(2024, 1, 1, 3, 0),
                    ],
                },
                "n": 2,
                "output": {
                    "max_consecutive_zeros_or_ones": [0, 1, 2, 3],
                },
            },
            # Exact N 60
            {
                "input": {
                    "max_consecutive_zeros_or_ones": [0, 60, 60, 0],
                    "hour_of_date": [
                        dt.datetime(2024, 1, 1, 0, 0),
                        dt.datetime(2024, 1, 1, 1, 0),
                        dt.datetime(2024, 1, 1, 2, 0),
                        dt.datetime(2024, 1, 1, 3, 0),
                    ],
                },
                "n": 2,
                "output": {
                    "max_consecutive_zeros_or_ones": [0, np.nan, np.nan, 0],
                },
            },
            # No filter needed but 60 available
            {
                "input": {
                    "max_consecutive_zeros_or_ones": [0, 60, 1, 60, 1],
                    "hour_of_date": [
                        dt.datetime(2024, 1, 1, 0, 0),
                        dt.datetime(2024, 1, 1, 1, 0),
                        dt.datetime(2024, 1, 1, 2, 0),
                        dt.datetime(2024, 1, 1, 3, 0),
                        dt.datetime(2024, 1, 1, 4, 0),
                    ],
                },
                "n": 2,
                "output": {
                    "max_consecutive_zeros_or_ones": [0, 60, 1, 60, 1],
                },
            },
            # All values need to be filtered.
            {
                "input": {
                    "max_consecutive_zeros_or_ones": [60, 60, 60, 60, 60],
                    "hour_of_date": [
                        dt.datetime(2024, 1, 1, 0, 0),
                        dt.datetime(2024, 1, 1, 1, 0),
                        dt.datetime(2024, 1, 1, 2, 0),
                        dt.datetime(2024, 1, 1, 3, 0),
                        dt.datetime(2024, 1, 1, 4, 0),
                    ],
                },
                "n": 2,
                "output": {
                    "max_consecutive_zeros_or_ones": [
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                        np.nan,
                    ],
                },
            },
            # N of zero, no filter needed.
            {
                "input": {
                    "max_consecutive_zeros_or_ones": [1, 1, 1, 1, 1],
                    "hour_of_date": [
                        dt.datetime(2024, 1, 1, 0, 0),
                        dt.datetime(2024, 1, 1, 1, 0),
                        dt.datetime(2024, 1, 1, 2, 0),
                        dt.datetime(2024, 1, 1, 3, 0),
                        dt.datetime(2024, 1, 1, 4, 0),
                    ],
                },
                "n": 0,
                "output": {
                    "max_consecutive_zeros_or_ones": [1, 1, 1, 1, 1],
                },
            },
            # N of zero, filter needed.
            {
                "input": {
                    "max_consecutive_zeros_or_ones": [60, 1],
                    "hour_of_date": [
                        dt.datetime(2024, 1, 1, 0, 0),
                        dt.datetime(2024, 1, 1, 1, 0),
                    ],
                },
                "n": 0,
                "output": {
                    "max_consecutive_zeros_or_ones": [np.nan, 1],
                },
            },
        ]

        for i, test_values in enumerate(tests):
            with self.subTest(f"Scenario {i + 1}"):
                # Create input DataFrame
                input_df = pd.DataFrame(test_values["input"])

                # Apply the filter function
                filtered_df = util.filter_max_consecutive_60(
                    input_df,
                    n=test_values["n"],
                    column_to_replace_with_nan=["max_consecutive_zeros_or_ones"],
                )

                # Define the expected output DataFrame
                expected_df = pd.DataFrame(test_values["output"])

                # Set the 'hour_of_date' column as index for expected output only
                expected_df = expected_df.set_index(
                    pd.to_datetime(test_values["input"]["hour_of_date"])
                )
                expected_df.index.name = "hour_of_date"

                # Compare the DataFrames
                pd.testing.assert_frame_equal(filtered_df, expected_df, check_dtype=False)

    def test_aggregate_by_hour(self):
        tests = [
            # Test case 1: Simple input with no missing values
            {
                "input": {
                    "datetime": pd.date_range("2023-01-01 00:00:00", periods=6, freq="30min"),
                    "fcd": [0, 0, 1, 0, 0, 1],
                },
                "expected": {
                    "hour_of_date": pd.to_datetime(
                        ["2023-01-01 00:00:00", "2023-01-01 01:00:00", "2023-01-01 02:00:00"]
                    ),
                    "fcd_mean": [0.0, 0.5, 0.5],
                    "max_consecutive_zeros": [2, 1, 1],
                    "max_consecutive_zeros_or_ones": [2, 2, 2],
                },
            },
            # Test case 2: All zeros in 0_5 column
            {
                "input": {
                    "datetime": pd.date_range("2023-01-01 00:00:00", periods=4, freq="30min"),
                    "fcd": [0, 0, 0, 0],
                },
                "expected": {
                    "hour_of_date": pd.to_datetime(["2023-01-01 00:00:00", "2023-01-01 01:00:00"]),
                    "fcd_mean": [0.0, 0.0],
                    "max_consecutive_zeros": [2, 2],
                    "max_consecutive_zeros_or_ones": [2, 2],
                },
            },
            # Test case 3: Multiple target segments
            {
                "input": {
                    "datetime": pd.date_range("2023-01-01 00:00:00", periods=6, freq="10min"),
                    "fcd": [0, 1, 0, 0, 1, 1],
                },
                "expected": {
                    "hour_of_date": pd.to_datetime(["2023-01-01 00:00:00"]),
                    "fcd_mean": [0.5],
                    "max_consecutive_zeros": [2],
                    "max_consecutive_zeros_or_ones": [6],
                },
            },
        ]

        for i, test_values in enumerate(tests):
            with self.subTest(f"Scenario {i + 1}"):
                input_df = pd.DataFrame(test_values["input"])
                input_df["datetime"] = pd.to_datetime(input_df["datetime"])
                input_df.set_index("datetime", inplace=True)
                expected_df = pd.DataFrame(test_values["expected"])
                result_df = util.aggregate_by_hour(input_df)
                pd.testing.assert_frame_equal(result_df, expected_df)

    def test_max_consecutive_true_streak(self):
        tests = [
            # No True values
            {
                "input": [False, False, False],
                "output": 0,
            },
            # All True values
            {
                "input": [True, True, True],
                "output": 3,
            },
            # Alternating True and False
            {
                "input": [False, True, False, True],
                "output": 1,
            },
            # Single streak of True values
            {
                "input": [False, True, True, False],
                "output": 2,
            },
            # Multiple streaks of varying lengths with longer 1 streak then 0.
            {
                "input": [False, False, True, True, True, True, False, False, False, True, False],
                "output": 4,
            },
            # Leading and trailing False values
            {
                "input": [False, False, True, True, False, False],
                "output": 2,
            },
        ]

        for i, test_values in enumerate(tests):
            with self.subTest(f"Scenario {i + 1}"):
                sr = pd.Series(test_values["input"])
                result = util.max_consecutive_true_streak(sr)
                self.assertEqual(result, test_values["output"])

    def test_max_consecutive_zeros(self):
        tests = [
            # No True values
            {
                "input": [1, 1, 1],
                "output": 0,
            },
            # All True values
            {
                "input": [0, 0, 0],
                "output": 3,
            },
            # Alternating True and False
            {
                "input": [0, 1, 0, 1],
                "output": 1,
            },
            # Single streak of True values
            {
                "input": [2, 0, 0, 2],
                "output": 2,
            },
            # Multiple streaks of varying lengths with longer 1 streak then 0.
            {
                "input": [0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0],
                "output": 3,
            },
            # Leading and trailing False values
            {
                "input": [1, 1, 0, 0, 1, 1],
                "output": 2,
            },
        ]

        for i, test_values in enumerate(tests):
            with self.subTest(f"Scenario {i + 1}"):
                sr = pd.Series(test_values["input"])
                result = util.max_consecutive_zeros(sr)
                self.assertEqual(result, test_values["output"])

    def test_max_consecutive_zeros_or_ones(self):
        tests = [
            # Single zero value
            {
                "input": [2, 0, 2],
                "output": 1,
            },
            # Mixed zeros and ones values
            {
                "input": [1, 0, 1],
                "output": 3,
            },
            #  Mixed zeros and ones values and non zeros or ones values
            {
                "input": [2, 1, 0, 1, 2],
                "output": 3,
            },
            # Single streak of True values
            {
                "input": [2, 0, 0, 2],
                "output": 2,
            },
            # Multiple streaks of varying lengths with longer 1 streak then 0.
            {
                "input": [2, 0, 1, 2, 1, 1, 0, 0, 2, 1, 0],
                "output": 4,
            },
            # Leading and trailing False values
            {
                "input": [2, 1, 0, 0, 1, 2],
                "output": 4,
            },
        ]

        for i, test_values in enumerate(tests):
            with self.subTest(f"Scenario {i + 1}"):
                sr = pd.Series(test_values["input"])
                result = util.max_consecutive_zeros_or_ones(sr)
                self.assertEqual(result, test_values["output"])

    def test_fill_missing_values_with_values(self):
        """Test fill_missing_values_with_values function with various input cases."""
        # Define test cases
        tests = [
            # Case with missing values in all columns
            {
                "input": pd.DataFrame(
                    {
                        "max_consecutive_zeros_or_ones_q95": [10.0, np.nan, 20.0],
                        "max_consecutive_zeros_q95": [5.0, 15.0, np.nan],
                        "fcd_mean_median": [np.nan, 0.5, 0.8],
                        "other_column": [1, 2, 3],
                    }
                ),
                "expected": pd.DataFrame(
                    {
                        "max_consecutive_zeros_or_ones_q95": [10.0, 60.0, 20.0],
                        "max_consecutive_zeros_q95": [5.0, 15.0, 60.0],
                        "fcd_mean_median": [0.0, 0.5, 0.8],
                        "other_column": [1, 2, 3],
                    }
                ),
                "description": "Missing values in all fillable columns",
            },
            # Case with no missing values
            {
                "input": pd.DataFrame(
                    {
                        "max_consecutive_zeros_or_ones_q95": [10.0, 20.0, 30.0],
                        "max_consecutive_zeros_q95": [5.0, 15.0, 25.0],
                        "fcd_mean_median": [0.2, 0.5, 0.8],
                        "other_column": [1, 2, 3],
                    }
                ),
                "expected": pd.DataFrame(
                    {
                        "max_consecutive_zeros_or_ones_q95": [10.0, 20.0, 30.0],
                        "max_consecutive_zeros_q95": [5.0, 15.0, 25.0],
                        "fcd_mean_median": [0.2, 0.5, 0.8],
                        "other_column": [1, 2, 3],
                    }
                ),
                "description": "No missing values",
            },
            # Empty DataFrame case
            {
                "input": pd.DataFrame(
                    columns=[
                        "max_consecutive_zeros_or_ones_q95",
                        "max_consecutive_zeros_q95",
                        "fcd_mean_median",
                    ]
                ),
                "expected": pd.DataFrame(
                    columns=[
                        "max_consecutive_zeros_or_ones_q95",
                        "max_consecutive_zeros_q95",
                        "fcd_mean_median",
                    ]
                ),
                "description": "Empty DataFrame",
            },
            # Case with columns containing only NaN values
            {
                "input": pd.DataFrame(
                    {
                        "max_consecutive_zeros_or_ones_q95": [np.nan, np.nan],
                        "max_consecutive_zeros_q95": [np.nan, np.nan],
                        "fcd_mean_median": [np.nan, np.nan],
                    }
                ),
                "expected": pd.DataFrame(
                    {
                        "max_consecutive_zeros_or_ones_q95": [60.0, 60.0],
                        "max_consecutive_zeros_q95": [60.0, 60.0],
                        "fcd_mean_median": [0.0, 0.0],
                    }
                ),
                "description": "All NaN values",
            },
        ]

        for i, test_case in enumerate(tests):
            with self.subTest(f"Case {i+1}: {test_case['description']}"):
                result = util.fill_missing_values_with_values(test_case["input"].copy())
                pd.testing.assert_frame_equal(result, test_case["expected"], check_dtype=False)
