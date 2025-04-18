import numpy as np
import pandas as pd
from idea.profile.profile import calculate_profile


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
        Seed for reproducibility of random values, by default 42.

    Returns
    -------
    pd.DataFrame
        DataFrame with a UTC datetime index and one column `fcd`.

    Example
    -------
    >>> df = generate_minute_data(2024)
    >>> df.head()
                               fcd
    2024-01-01 00:00:00+00:00  0.0
    2024-01-01 00:01:00+00:00  2.0
    2024-01-01 00:02:00+00:00  NaN
    ...
    """
    np.random.seed(seed)

    # Define full-year datetime range with 1-minute frequency
    start = pd.Timestamp(f"{year}-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp(f"{year + 1}-01-01 00:00:00", tz="UTC") - pd.Timedelta(minutes=1)
    time_index = pd.date_range(start=start, end=end, freq="min")

    # Generate random values between 0 and 10 (rounded to 1 decimal)
    fcd_values = np.round(np.random.uniform(0, 10, size=len(time_index)), 0)

    # Ensure inclusion of edge values
    fcd_values[0] = 0.0
    fcd_values[-1] = 10.0

    # Add ~5% NaN values randomly
    nan_mask = np.random.rand(len(time_index)) < 0.05
    fcd_values[nan_mask] = np.nan

    return pd.DataFrame({"fcd": fcd_values}, index=time_index)


if __name__ == "__main__":
    # Generate synthetic minute-level FCD data for 2024
    df_minute = generate_minute_data(2024)

    # Define the profile range
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    # Calculate the profile
    profile = calculate_profile(df_minute, start, end)

    # Print (or optionally save) profile
    print(profile.head())
