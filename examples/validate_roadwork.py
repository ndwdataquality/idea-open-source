import numpy as np
import pandas as pd
from idea.cli_overrides import parse_minimum_weeks_required_for_profile
from idea.profile.profile import calculate_profile
from idea.validation.validation import validate_roadwork


def generate_minute_data(year: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate one year of minute-level FCD (Floating Car Data) values.

    The 'fcd' values range from 0.0 to 10.0 (rounded to 1 decimal), with
    approximately 5% missing values (NaN). The values are reproducible
    using a fixed random seed.

    Parameters
    ----------
    year : int
        The year for which the data should be generated.
    seed : int, optional
        Seed for reproducibility, by default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with a UTC datetime index and one column `fcd`.
    """
    np.random.seed(seed)
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp(f"{year+1}-01-01 00:00:00", tz="UTC") - pd.Timedelta(minutes=1)
    time_index = pd.date_range(start=start, end=end, freq="min")
    fcd_values = np.round(np.random.uniform(0, 10, size=len(time_index)), 0)
    fcd_values[0] = 0.0
    fcd_values[-1] = 10.0
    nan_mask = np.random.rand(len(time_index)) < 0.05
    fcd_values[nan_mask] = np.nan
    return pd.DataFrame({"fcd": fcd_values}, index=time_index)


def generate_validation_data(
    start_time: pd.Timestamp, end_time: pd.Timestamp, seed: int = 123
) -> pd.DataFrame:
    """
    Generate synthetic FCD data for validation.

    Parameters
    ----------
    start_time : pd.Timestamp
        UTC start timestamp for the validation data.
    end_time : pd.Timestamp
        UTC start timestamp for the validation data.
    seed : int, optional
        Seed for reproducibility, by default 123.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime index and column `fcd`.
    """
    np.random.seed(seed)
    time_index = pd.date_range(start=start_time, end=end_time, freq="min", tz="UTC")
    fcd_values = np.round(np.random.uniform(0, 2, size=len(time_index)), 0)
    return pd.DataFrame({"fcd": fcd_values}, index=time_index)


if __name__ == "__main__":
    minimum_weeks_required = parse_minimum_weeks_required_for_profile()

    # Create profile for single segment
    df_minute = generate_minute_data(2024)
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")
    profile = calculate_profile(df_minute, start, end, minimum_weeks_required)

    # Create fcd during validation period (roadworks).
    validation_start = pd.Timestamp("2025-03-15 12:00:00", tz="UTC")
    validation_end = pd.Timestamp("2025-03-15 15:00:00", tz="UTC")
    fcd_during_roadwork = generate_validation_data(validation_start, validation_end)

    validate_roadwork(fcd_during_roadwork, profile)
