import numpy as np
import pandas as pd

from idea.constants import (
    CLOSED_LIMIT,
    COV_DROP_LIMIT,
    COV_HIGH,
    COV_THRESHOLD_ZEROS_OR_ONE_VALUE,
    MAX_PROFILE_VALUE,
    MINIMUM_PROFILE_VALUE,
    OPEN_LIMT,
)
from idea.exceptions import IDEAError


def update_counter(condition: bool, prev_counter: int) -> int:
    """
    Update a counter based on a boolean condition.

    Parameters
    ----------
    condition : bool
        If True, increment the counter; otherwise reset it.
    prev_counter : int
        The previous counter value.

    Returns
    -------
    int
        The updated counter value.
    """
    return prev_counter + 1 if condition else 0


def update_no_coverage_counters(fcd: int, prev_0: int, prev_1: int) -> tuple[int, int]:
    """
    Update counters for minutes with no or low coverage based on the FCD value.

    Parameters
    ----------
    fcd : int
        Floating car data value, expected between 0 and 10.
    prev_0 : int
        Previous count of consecutive minutes with FCD == 0.
    prev_1 : int
        Previous count of consecutive minutes with FCD in (0, 1).

    Returns
    -------
    tuple[int, int]
        Updated counters for FCD == 0 and FCD in (0, 1) respectively.
    """
    if not (0 <= fcd <= 10):
        raise IDEAError(f"fcd must be between 0 and 10. Got: {fcd}")
    updated_0 = update_counter(fcd == 0, prev_0)
    updated_1 = update_counter(fcd in (0, 1), prev_1)
    return updated_0, updated_1


def calculate_minutes_no_coverage(validation_df: pd.DataFrame) -> pd.DataFrame:
    """
    Loop through the validation DataFrame minute-by-minute and compute the consecutive
    counters for no coverage using FCD values.

    Two counters are maintained:
      - consecutive_zeros: counts minutes with fcd == 0.
      - consecutive_low: counts minutes with fcd in (0, 1).

    Missing values (NaN) reset both counters.

    Parameters
    ----------
    validation_df : pd.DataFrame
        DataFrame with a datetime index and a column 'fcd' containing FCD values.

    Returns
    -------
    pd.DataFrame
        A copy of validation_df with added columns:
            - 'consecutive_zeros': current consecutive count for fcd == 0.
            - 'consecutive_low': current consecutive count for fcd in (0, 1).
    """
    consecutive_zeros = []
    consecutive_low = []
    counter_zeros = 0
    counter_low = 0

    # Loop through each minute (row) in the DataFrame
    for _, row in validation_df.iterrows():
        fcd_value = row["fcd"]
        if pd.isna(fcd_value):
            # Reset counters if data is missing
            counter_zeros, counter_low = 0, 0
        else:
            # Update counters using the provided function
            counter_zeros, counter_low = update_no_coverage_counters(
                fcd_value, counter_zeros, counter_low
            )
        consecutive_zeros.append(counter_zeros)
        consecutive_low.append(counter_low)

    df = validation_df.copy()
    df["consecutive_zeros"] = consecutive_zeros
    df["consecutive_low"] = consecutive_low
    return df


def match_no_coverage_profile(
    df_with_coverage: pd.DataFrame, profile_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge the no coverage counters with the profile thresholds and flag deviations.

    The profile is expected to include the columns:
      - 'day_of_week'
      - 'hour_of_day'
      - 'max_consecutive_zeros_q95'
      - 'max_consecutive_zeros_or_ones_q95'

    For each record in df_with_coverage, this function adds:
      - 'day_of_week' and 'hour' extracted from the timestamp.
      - Flags 'zeros_within_threshold' indicating whether the current consecutive
        zeros count is below or equal to the profile threshold.
      - Flags 'zeros_or_ones_within_threshold' indicating whether the current consecutive
        low count is below or equal to the profile threshold.

    Parameters
    ----------
    df_with_coverage : pd.DataFrame
        DataFrame with FCD values, and the calculated 'consecutive_zeros'
        and 'consecutive_low' columns.
    profile_df : pd.DataFrame
        Profile DataFrame containing the required threshold columns.

    Returns
    -------
    pd.DataFrame
        DataFrame merged with profile data and additional flag columns:
        - 'zeros_within_threshold'
        - 'zeros_or_ones_within_threshold'
    """
    df = df_with_coverage.copy()
    # Extract day_of_week and hour to merge with profile data
    df["day_of_week"] = df.index.day_name()
    df["hour_of_day"] = df.index.hour
    df.index.name = "time"
    df = df.reset_index()

    # Merge with the profile on day_of_week and hour.
    # Assumes profile_df already contains the relevant threshold columns.
    merged = df.merge(
        profile_df[
            [
                "day_of_week",
                "hour_of_day",
                "max_consecutive_zeros_q95",
                "max_consecutive_zeros_or_ones_q95",
                "fcd_mean_median",
            ]
        ],
        on=["day_of_week", "hour_of_day"],
        how="left",
    )
    return merged


def determine_coverage_profile_value(
    row: pd.Series, previous_row: pd.Series, cov_threshold_zeros_or_one_values: float
) -> tuple[float, float, float]:
    """
    Determines coverage-related values based on whether the mean/median FCD is
    below a specified threshold.

    Parameters
    ----------
    row : pd.Series
        The current row containing coverage metrics.
    previous_row : pd.Series
        The previous row containing historical coverage metrics.
    cov_threshold_zeros_or_one_values : float
        Threshold below which data is considered to have zero coverage.

    Returns
    -------
    min_no_cov : float
        The number of consecutive zero or low coverage intervals in the current row.
    previous_min_no_cov : float
        The number of consecutive zero or low coverage intervals in the previous row.
    profile_value : float
        The Q95 value for max consecutive zeros or low coverage, from the previous row.
    """
    if row.fcd_mean_median < cov_threshold_zeros_or_one_values:
        attr = "consecutive_zeros"
        profile_attr = "max_consecutive_zeros_q95"
    else:
        attr = "consecutive_low"
        profile_attr = "max_consecutive_zeros_or_ones_q95"

    min_no_cov = getattr(row, attr)
    previous_min_no_cov = getattr(previous_row, attr)
    profile_value = getattr(previous_row, profile_attr)

    return min_no_cov, previous_min_no_cov, profile_value


def calculate_running_mean(
    profile_value: float, prev_running_mean: float, cov_weight: float, res: float
) -> float:
    """
    Calculates the updated running mean using a weighted average.

    Parameters
    ----------
    profile_value : float
        The current profile value (weight for the previous running mean).
    prev_running_mean : float
        The previously computed running mean.
    cov_weight : float
        The weight for the new result `res`.
    res : float
        The new value to incorporate into the running mean.

    Returns
    -------
    float
        The updated running mean.
    """
    total_weight = profile_value + cov_weight
    weighted_sum = (profile_value * prev_running_mean) + (cov_weight * res)
    return weighted_sum / total_weight


def handle_profile_value(profile_value: float | None) -> float:
    """Ensure the profile value is within allowed bounds and not NaN."""
    if np.isnan(profile_value):
        return 60  # Default value when profile_value is NaN
    return max(profile_value, MINIMUM_PROFILE_VALUE)


def sanitize_cov_values(min_no_cov: float, prev_min_no_cov: float) -> tuple[float, float]:
    """Ensure no NaNs in coverage values."""
    return (
        0 if np.isnan(min_no_cov) else min_no_cov,
        0 if np.isnan(prev_min_no_cov) else prev_min_no_cov,
    )


def calculate_running_mean_based_on_conditions(
    min_no_cov: float,
    prev_min_no_cov: float,
    profile_value: float,
    coverage: float,
    prev_running_mean: float,
    coverage_profile_value: float,
) -> float:
    """Update the running mean based on coverage behavior."""
    val = min_no_cov
    bound = profile_value

    if (profile_value > MAX_PROFILE_VALUE) and (min_no_cov > profile_value):
        return prev_running_mean  # No update needed

    elif (min_no_cov == 0) and (prev_min_no_cov < profile_value):
        return calculate_running_mean(
            profile_value, prev_running_mean, coverage, prev_min_no_cov / (2 * profile_value)
        )

    elif min_no_cov != 0:
        cov_weight = max(1, (val / (2 * profile_value)) ** 2)
        res = max(prev_running_mean, min(1, val / (2 * bound)))
        return calculate_running_mean(profile_value, prev_running_mean, cov_weight, res)

    elif (coverage_profile_value - coverage) > COV_DROP_LIMIT:
        return calculate_running_mean(MINIMUM_PROFILE_VALUE, prev_running_mean, 10, 1)

    elif coverage > COV_HIGH:
        return calculate_running_mean(MINIMUM_PROFILE_VALUE, prev_running_mean, coverage, 0)

    return prev_running_mean  # No change


def determine_road_status_by_minute(df_matched_profile: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the road status per minute using a running mean based on profile coverage.

    Parameters
    ----------
    df_matched_profile : pd.DataFrame
        Input DataFrame with profile and coverage columns.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with running mean and SEGMENT_CLOSURE_STATUS.
    """
    prev_running_mean = 0.5
    running_means = []
    previous_row = None

    for _, row in df_matched_profile.iterrows():
        if previous_row is None:
            running_means.append(prev_running_mean)
            previous_row = row.copy()
            continue

        coverage = row.fcd
        min_no_cov, prev_min_no_cov, profile_value = determine_coverage_profile_value(
            row, previous_row, COV_THRESHOLD_ZEROS_OR_ONE_VALUE
        )

        profile_value = handle_profile_value(profile_value)
        min_no_cov, prev_min_no_cov = sanitize_cov_values(min_no_cov, prev_min_no_cov)
        coverage_profile_value = row.fcd_mean_median

        prev_running_mean = calculate_running_mean_based_on_conditions(
            min_no_cov,
            prev_min_no_cov,
            profile_value,
            coverage,
            prev_running_mean,
            coverage_profile_value,
        )

        running_means.append(prev_running_mean)
        previous_row = row.copy()

    df_matched_profile["running_mean"] = running_means
    df_matched_profile = set_segment_closure_status(df_matched_profile)

    return df_matched_profile


def set_segment_closure_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sets the SEGMENT_CLOSURE_STATUS column based on running mean thresholds.

    Parameters
    ----------
    df : pd.DataFrame with the running mean column.

    Returns
    -------
    pd.DataFrame
        Updated DataFrame with SEGMENT_CLOSURE_STATUS.
    """
    conditions = [
        df.running_mean < OPEN_LIMT,
        df.running_mean > CLOSED_LIMIT,
    ]
    selections = ["open", "closed"]
    df["segment_closure_status"] = np.select(conditions, selections, default="undetermined")
    return df
