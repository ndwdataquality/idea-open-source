MINIMUM_HOURS_NO_TRAFFIC_FOR_PROFILE = 5
CONSECUTIVE_60_MINUTES = 60
MINIMUM_WEEKS_INPUT_FOR_PROFILE = 10
MAX_CONSECUTIVE_ZEROS_OR_ONES_Q95_REPLACEMENT_VALUE = 60
FCD_MEAN_MEDIAN_MISSING_REPLACEMENT_VALUE = 0
MAX_ACCEPTABLE_CONSECUTIVE_ZEROS_Q95 = 35
THRESHOLD_OF_USEFUL_DATA_PROFILE = 30
COLUMNS_TO_REPLACE_VALUES_WITH_NAN = [
    "cov_5_mean",
    "speed_mean",
    "max_consecutive_zeros",
    "max_consecutive_zeros_or_ones",
]
PROFILE_COLUMNS = [
    "fcd_mean_median",
    "max_consecutive_zeros_q95",
    "max_consecutive_zeros_or_ones_q95",
]

DAYS_OF_WEEK = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
    6: "Sunday",
}


MAX_PROFILE_VALUE = 54
COV_DROP_LIMIT = 8
COV_HIGH = 6
MINIMUM_PROFILE_VALUE = 5
COV_THRESHOLD_ZEROS_OR_ONE_VALUE = 3
CLOSED_LIMIT = 0.75
OPEN_LIMT = 0.15
