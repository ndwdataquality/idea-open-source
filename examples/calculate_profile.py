import pandas as pd
from idea.cli_overrides import parse_minimum_weeks_required_for_profile
from idea.profile.profile import calculate_profile

from examples.util.util import generate_minute_data

if __name__ == "__main__":
    minimum_weeks_required = parse_minimum_weeks_required_for_profile()

    # Generate synthetic minute-level FCD data for 2024
    df_minute = generate_minute_data(2024)

    # Define the profile range
    start = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-01-01 00:00:00", tz="UTC")

    # Calculate the profile
    profile = calculate_profile(
        df_minute, start, end, minimum_weeks_required=minimum_weeks_required
    )

    # Print (or optionally save) profile
    print(profile.head())
