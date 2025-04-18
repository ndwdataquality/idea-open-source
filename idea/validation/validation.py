import pandas as pd

from idea.validation.util import (
    calculate_minutes_no_coverage,
    determine_road_status_by_minute,
    match_no_coverage_profile,
)


def validate_roadwork(fcd_during_roadwork: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    df = calculate_minutes_no_coverage(fcd_during_roadwork)
    df = match_no_coverage_profile(df, profile)
    df = determine_road_status_by_minute(df)
    return df
