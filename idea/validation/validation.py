import pandas as pd

from idea.validation.util import (
    calculate_minutes_no_coverage,
    determine_road_status_by_minute,
    match_no_coverage_profile,
)


def validate_roadwork(fcd_during_roadwork: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    """
    Validate roadwork periods using Floating Car Data (FCD) and a reference profile.

    This function compares observed FCD during a roadwork period to a reference profile,
    to determine whether traffic coverage significantly deviates from expected values.
    It can be used to detect whether roadworks likely caused disruptions based on FCD coverage.

    Parameters
    ----------
    fcd_during_roadwork : pd.DataFrame with observed FCD during the roadwork period.
        It must have a datetime index with minute-level frequency and a column named 'fcd', e.g.:

        datetime              | fcd
        ----------------------+------
        2023-01-01 00:00:00   | 5.2
        2023-01-01 00:01:00   | NaN
        2023-01-01 00:02:00   | 3.8

    profile : pd.DataFrame
        A reference profile DataFrame containing expected FCD values per day of week and hour of
        day. Typically, includes columns like `day_of_week`, `hour_of_day`, and expected FCD
        statistics.

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per minute in the `fcd_during_roadwork` input, annotated with:
        - deviation metrics compared to the profile
        - a classification indicating whether the situation is considered "open" or "closed"

    Notes
    -----
    This function is useful in real-time or historical validations of traffic disruptions caused
    by roadworks.
    """
    df = calculate_minutes_no_coverage(fcd_during_roadwork)
    df = match_no_coverage_profile(df, profile)
    df = determine_road_status_by_minute(df)
    return df
