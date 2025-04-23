import pandas as pd
from idea.cli_overrides import parse_minimum_weeks_required_for_profile
from idea.profile.profile import calculate_profile
from idea.validation.validation import validate_roadwork

from examples.util.util import generate_minute_data, generate_validation_data

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
