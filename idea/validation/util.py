import math

import numpy as np
import pandas as pd

from idea.constants import (
    CLOSED_LIMIT,
    COV_DROP_LIMIT,
    COV_HIGH,
    COV_THRESHOLD_ZEROS_OR_ONE_VALUE,
    DECAY_PARAM,
    K_MAX,
    K_START,
    MAX_PROFILE_VALUE,
    MINIMUM_PROFILE_VALUE,
    OPEN_LIMIT,
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


def update_k(
    k: float,
    coverage: float,
    current_minutes_no_low: float,
    previous_minutes_no_low: float,
    decay_window: float,
) -> float:
    """
    Update the momentum decay parameter k for the current minute.

    Three cases:
      - No vehicles (coverage == 0): k resets to K_START.
      - Event now and previous event within decay window: k doubles, capped at K_MAX.
      - Otherwise: k is unchanged.

    Parameters
    ----------
    k : float
        Current momentum decay parameter from the previous minute.
    coverage : float
        Current FCD value (number of vehicles observed).
    current_minutes_no_low : float
        Consecutive minutes with FCD in {0, 1} up to the current minute.
        A value of 0 means an event (2+ vehicles) was detected this minute.
    previous_minutes_no_low : float
        Consecutive minutes with FCD in {0, 1} up to the previous minute.
    decay_window : float
        Time window within which a previous event is considered recent.

    Returns
    -------
    float
        Updated k value.
    """
    if coverage == 0:
        return K_START
    if current_minutes_no_low == 0 and previous_minutes_no_low <= decay_window:
        return min(K_MAX, k * 2)
    return k


def apply_momentum(
    coverage: float,
    coverage_profile_value: float,
    current_minutes_no_low: float,
    previous_minutes_no_low: float,
    profile_cov_ones: float,
    k: float,
    running_mean: float,
) -> tuple[float, float]:
    """
    Apply momentum scaling to the running mean during low coverage periods.

    During low coverage periods, a decaying momentum factor (alpha) is applied to
    accelerate reopening detection when vehicles are observed. The decay parameter k
    is boosted when multiple vehicle events occur within the decay window, causing
    faster reduction of the closure probability.

    Momentum is only applied when:
      - coverage is below COV_HIGH (low coverage condition), and
      - the drop from the historical mean is not a sudden drop.

    Parameters
    ----------
    coverage : float
        Current FCD value (number of vehicles observed).
    coverage_profile_value : float
        Historical mean/median FCD for the current time slot.
    current_minutes_no_low : float
        Consecutive minutes with FCD in {0, 1} up to the current minute.
    previous_minutes_no_low : float
        Consecutive minutes with FCD in {0, 1} up to the previous minute.
    profile_cov_ones : float
        Q95 of max consecutive low-coverage minutes from the profile. Used to
        determine the decay window. NaN and values below MINIMUM_PROFILE_VALUE
        are clamped to MINIMUM_PROFILE_VALUE.
    k : float
        Current momentum decay parameter carried over from the previous minute.
    running_mean : float
        Running mean value to scale.

    Returns
    -------
    tuple[float, float]
        Updated (running_mean, k). If momentum conditions are not met, both
        values are returned unchanged.
    """
    is_low_coverage = coverage < COV_HIGH
    is_not_sudden_drop = (coverage_profile_value - coverage) <= COV_DROP_LIMIT

    if not (is_low_coverage and is_not_sudden_drop):
        return running_mean, k

    if np.isnan(profile_cov_ones) or profile_cov_ones < MINIMUM_PROFILE_VALUE:
        profile_cov_ones = float(MINIMUM_PROFILE_VALUE)

    decay_window = DECAY_PARAM * profile_cov_ones

    k = update_k(k, coverage, current_minutes_no_low, previous_minutes_no_low, decay_window)

    echo = 0.0
    if decay_window > 0:
        echo = max(0.0, (decay_window - current_minutes_no_low) / decay_window)

    alpha = math.exp(-k * echo)
    return running_mean * alpha, k


def determine_road_status_by_minute(df_matched_profile: pd.DataFrame) -> pd.DataFrame:
    """
    Determines the road status per minute using a running mean based on profile coverage.

    After the base running mean is computed, momentum scaling is applied via
    apply_momentum to accelerate reopening detection during low coverage periods.

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
    k = K_START
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

        prev_running_mean, k = apply_momentum(
            coverage,
            coverage_profile_value,
            row.consecutive_low,
            previous_row.consecutive_low,
            previous_row.max_consecutive_zeros_or_ones_q95,
            k,
            prev_running_mean,
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
        df.running_mean < OPEN_LIMIT,
        df.running_mean > CLOSED_LIMIT,
    ]
    selections = ["open", "closed"]
    df["segment_closure_status"] = np.select(conditions, selections, default="undetermined")
    return df
