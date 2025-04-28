import datetime as dt

import pandas as pd

from idea.constants import MINIMUM_WEEKS_INPUT_FOR_PROFILE, PROFILE_COLUMNS
from idea.profile import util as idea_util


def calculate_profile(
    df: pd.DataFrame,
    start: dt.datetime,
    end: dt.datetime,
    minimum_weeks_required: int = MINIMUM_WEEKS_INPUT_FOR_PROFILE,
) -> pd.DataFrame:
    """
    Calculate a profile based on Floating Car Data (FCD) between a given start and end time.

    This function processes the input DataFrame by performing a series of transformations
    such as interpolating missing minutes, aggregating data, and filtering based on available data,
    ultimately returning a profile DataFrame suitable for validating roadworks in real-time systems.

    An example of how the fcd input should look:
    datetime              | fcd
    ----------------------+------
    2023-01-01 00:00:00   | 5.2
    2023-01-01 00:01:00   | NaN
    2023-01-01 00:02:00   | 3.8
    ...                   | ...

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing FCD. It should have a datetime index with minute-level frequency
        and a column named 'fcd'. For example:
    start : datetime.datetime
        The start datetime for the profile generation.
    end : datetime.datetime
        The end datetime for the profile generation.
    minimum_weeks_required : int
        The minimum number of weeks required for a profile generation.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:

        - `day_of_week` : str
            Day of the week (e.g. "Monday").
        - `hour_of_day` : int
            Hour of the day (0-23).
        - `fcd_mean_median` : float
            Median value of FCD (floating car data) for the hour.
        - `max_consecutive_zeros_q95` : float
            95th percentile of the longest sequence of consecutive zeros in the time series.
        - `max_consecutive_zeros_or_ones_q95` : float
            95th percentile of the longest sequence of consecutive zeros or ones.
        - `number_of_hours` : int
            Number of hours observed for that hour-of-day/day-of-week combination.

    Example
    -------
        day_of_week  hour_of_day  fcd_mean_median  max_consecutive_zeros_q95  \
            Monday             0              4.4                         3.0
            Monday             1              4.3                         3.0
            Monday             2              4.4                         3.0
            Monday             3              4.3                         3.0
            Monday             4              4.3                         3.0

        max_consecutive_zeros_or_ones_q95  number_of_hours
                                       4.0               53
                                       4.0               53
                                       3.4               53
                                       4.0               53
                                       4.0               53

    """
    idea_util.verify_start_and_end_time(df, start, end)
    df = idea_util.interpolate_missing_minutes(df, start, end)
    df = idea_util.fill_nan_columns_with_zeros(df, column_subset=["fcd"])
    df = idea_util.aggregate_by_hour(df)
    df = idea_util.filter_max_consecutive_60(df)
    df = idea_util.add_periods(df)
    df = idea_util.aggregate_hour_of_week(df)
    df = idea_util.filter_profile_low_availability(df, PROFILE_COLUMNS, minimum_weeks_required)
    df = idea_util.replace_nans_with_none(df)
    df = idea_util.map_day_of_week_to_name(df)
    idea_util.does_profile_has_enough_data(df)
    df = idea_util.fill_missing_values_with_values(df)
    return df
