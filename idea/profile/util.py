import datetime as dt
import itertools
import logging

import numpy as np
import pandas as pd

from idea.constants import (
    COLUMNS_TO_REPLACE_VALUES_WITH_NAN,
    CONSECUTIVE_60_MINUTES,
    DAYS_OF_WEEK,
    FCD_MEAN_MEDIAN_MISSING_REPLACEMENT_VALUE,
    MAX_ACCEPTABLE_CONSECUTIVE_ZEROS_Q95,
    MAX_CONSECUTIVE_ZEROS_OR_ONES_Q95_REPLACEMENT_VALUE,
    MINIMUM_HOURS_NO_TRAFFIC_FOR_PROFILE,
    THRESHOLD_OF_USEFUL_DATA_PROFILE,
)
from idea.exceptions import IDEAError

logger = logging.getLogger(__name__)


def quantile_95(x: pd.Series) -> float:
    """Get the 95th percentile of a pandas series."""
    return x.quantile(0.95)


def interpolate_missing_minutes(
    df: pd.DataFrame,
    start: dt.datetime,
    end: dt.datetime,
) -> pd.DataFrame:
    """Interpolate missing minutes, so we have a record for each segment for each minute."""
    complete_index = pd.date_range(start=start, end=end, freq="min")
    return df.reindex(complete_index)


def fill_nan_columns_with_zeros(df: pd.DataFrame, column_subset: list):
    """Fills NaN values in specified columns of the DataFrame with 0."""
    df[column_subset] = df[column_subset].fillna(0).astype("int16")
    return df


def add_periods(df: pd.DataFrame):
    """Adds period-related columns to the DataFrame based on its DatetimeIndex."""
    df["day_of_week"] = df.index.day_of_week
    df["hour_of_day"] = df.index.hour
    return df


def filter_max_consecutive_60(
    df: pd.DataFrame,
    n: int = MINIMUM_HOURS_NO_TRAFFIC_FOR_PROFILE,
    column_to_replace_with_nan: list = COLUMNS_TO_REPLACE_VALUES_WITH_NAN,
):
    """
    Sets data values to NaN where the maximum number of consecutive 60s
    in 'max_consecutive_zeros_or_ones' exceeds the given threshold 'n'.
    In fill_missing_values_with_values, the values are set with a predefined constant.

    Parameters
    ----------
    df (pd.DataFrame): The input DataFrame.
    n (int): The threshold for the maximum consecutive 60s.
    Default is 5 (MINIMUM_HOURS_NO_TRAFFIC_FOR_PROFILE, in filter_missing_periods).

    Returns
    -------
    pd.DataFrame:
    DataFrame with data columns set to NaN for problematic sequences while preserving
    the original row structure.
    """
    df["consecutive_60"] = df["max_consecutive_zeros_or_ones"] == CONSECUTIVE_60_MINUTES
    df["group"] = (df["consecutive_60"] != df["consecutive_60"].shift()).cumsum()
    group_sizes = df[df["consecutive_60"]].groupby("group").size()
    large_true_groups = group_sizes[group_sizes >= n].index

    mask = df["group"].isin(large_true_groups)

    if mask.any():
        # Set data columns to NaN for rows in large groups
        df.loc[mask, column_to_replace_with_nan] = np.nan

    # Remove the temporary columns
    df = df.drop(columns=["consecutive_60", "group"])

    return df.reset_index(drop=True).set_index("hour_of_date")


def max_consecutive_true_streak(sr: pd.Series) -> int:
    """Compute the maximum consecutive True values in a pandas series."""
    return (
        max((sum(1 for _ in group) if key else 0) for key, group in itertools.groupby(sr))
        if (len(sr) > 0)
        else 0
    )


def max_consecutive_zeros(series: pd.Series) -> int:
    """Compute the maximum consecutive 0 values in a pandas series."""
    sr = series == 0
    return max_consecutive_true_streak(sr)


def max_consecutive_zeros_or_ones(series: pd.Series) -> int:
    """Compute the maximum consecutive 0 or 1 values in a pandas series."""
    sr = series.isin([0, 1])
    return max_consecutive_true_streak(sr)


def aggregate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fcd by hour."""
    df["hour_of_date"] = df.index.floor("h")
    df = (
        df.groupby(["hour_of_date"])
        .agg(
            fcd_mean=("fcd", "mean"),
            max_consecutive_zeros=("fcd", max_consecutive_zeros),
            max_consecutive_zeros_or_ones=("fcd", max_consecutive_zeros_or_ones),
        )
        .reset_index()
    )
    return df


def aggregate_hour_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by hour a day of week and calculate profile values."""
    df = df.groupby(["day_of_week", "hour_of_day"], as_index=False).agg(
        fcd_mean_median=("fcd_mean", "median"),
        max_consecutive_zeros_q95=(
            "max_consecutive_zeros",
            quantile_95,
        ),
        max_consecutive_zeros_or_ones_q95=(
            "max_consecutive_zeros_or_ones",
            quantile_95,
        ),
        number_of_hours=("fcd_mean", "count"),
    )

    return df.round(1)


def filter_profile_low_availability(df: pd.DataFrame, columns: list, n: int) -> pd.DataFrame:
    """Filter data with low availability by setting the values to nan."""
    df_copy = df.copy()
    mask = df_copy["number_of_hours"] < n
    df_copy.loc[mask, columns] = np.nan
    return df_copy


def replace_nans_with_none(df: pd.DataFrame):
    """Replace nans with none, so we can store it in json files."""
    return df.replace({np.nan: None})


def map_day_of_week_to_name(df: pd.DataFrame) -> pd.DataFrame:
    """Map numerical day of week values to corresponding day names.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a 'day_of_week' column with numerical values

    Returns
    -------
    pd.DataFrame
        DataFrame with 'day_of_week' column mapped to day names using DAYS_OF_WEEK mapping
    """
    df["day_of_week"] = df["day_of_week"].map(DAYS_OF_WEEK)
    return df


def fill_missing_values_with_values(profile: pd.DataFrame) -> pd.DataFrame:
    """
    Replace missing values in profile columns with predefined constants.

    Parameters
    ----------
    profile : pd.DataFrame
        DataFrame containing the profile data with columns to be filled

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values replaced
    """
    # Convert to float first, then fill NA values
    profile["max_consecutive_zeros_or_ones_q95"] = (
        profile["max_consecutive_zeros_or_ones_q95"]
        .astype(float)
        .fillna(MAX_CONSECUTIVE_ZEROS_OR_ONES_Q95_REPLACEMENT_VALUE)
    )

    profile["max_consecutive_zeros_q95"] = (
        profile["max_consecutive_zeros_q95"]
        .astype(float)
        .fillna(MAX_CONSECUTIVE_ZEROS_OR_ONES_Q95_REPLACEMENT_VALUE)
    )

    profile["fcd_mean_median"] = (
        profile["fcd_mean_median"].astype(float).fillna(FCD_MEAN_MEDIAN_MISSING_REPLACEMENT_VALUE)
    )

    return profile


def does_profile_has_enough_data(profile: pd.DataFrame) -> None:
    """
    Evaluates if the profile contains sufficient data to be considered useful.

    This function calculates the ratio of segments with acceptable data coverage
    by counting rows where 'max_consecutive_zeros_q95' is below
    MAX_ACCEPTABLE_PERIOD_WITH_NO_TRAFFIC_DATA and dividing by the number
    of unique target segments.
    The profile has enough data if this ratio exceeds THRESHOLD_OF_USEFUL_DATA_PROFILE.

    Parameters
    ----------
    profile : pd.DataFrame
        DataFrame containing the profile data with 'max_consecutive_zeros_q95'
        and 'target_segment' columns

    Returns
    -------
    bool
        True if the average ratio of rows with acceptable consecutive zeros
        is greater than THRESHOLD_OF_USEFUL_DATA_PROFILE, False otherwise
    """

    # Count the number of values below MAX_ACCEPTABLE_PERIOD_WITH_NO_TRAFFIC_DATA
    # in max_consecutive_zeros_q95 column
    # and sum them up (TRUES are counted as 1)
    total_count_consecutive_zeros_below_max = (
        profile["max_consecutive_zeros_q95"] < MAX_ACCEPTABLE_CONSECUTIVE_ZEROS_Q95
    ).sum()

    # Return True if the average is greater than the minimum threshold
    if total_count_consecutive_zeros_below_max < THRESHOLD_OF_USEFUL_DATA_PROFILE:
        raise IDEAError("Not enough fcd input data for creating profile.")


def verify_start_and_end_time(df: pd.DataFrame, start: dt.datetime | None, end: dt.datetime | None):
    """
    Verifies that both start and end times are either provided together or not at all,
    checks that the difference between start and end is exactly one year,
    and ensures that start is not after the earliest index and end is not before the latest index.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a DateTimeIndex to validate against.
    start : datetime.datetime or None
        The start time for validation.
    end : datetime.datetime or None
        The end time for validation.

    Raises
    ------
    IDEAError
        If only one of start or end is provided.
        If the difference between start and end is not exactly one year.
        If start is after the earliest timestamp in df.
        If end is before the latest timestamp in df.
    """
    if start is None and end is None:
        return

    if start is None or end is None:
        raise IDEAError("Both start and end times must be provided together.")

    # Check if the time span is exactly 1 year
    try:
        expected_end = start.replace(year=start.year + 1)
    except ValueError:
        # Handle February 29 → 28 case
        expected_end = start + (dt.datetime(start.year + 1, 3, 1) - dt.datetime(start.year, 3, 1))

    if end != expected_end:
        raise IDEAError(
            f"Start and end must be exactly 1 year apart."
            f" Expected end: {expected_end}, got: {end}"
        )

    if start > df.index.min():
        raise IDEAError(
            f"Start time: {start} can not be after the earliest"
            f" timestamp in the data: {df.index.min()}"
        )

    if end < df.index.max():
        raise IDEAError(
            f"End time: {end} can not be before the latest"
            f" timestamp in the data: {df.index.max()}"
        )
