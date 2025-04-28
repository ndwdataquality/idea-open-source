import argparse

from idea.constants import (
    MINIMUM_WEEKS_INPUT_FOR_PROFILE,
)


def parse_minimum_weeks_required_for_profile(args=None):
    """Allows for parsing the minimum weeks required for a profile using command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minimum-weeks-required-for-profile",
        type=int,
        required=False,
        help="Value for minimum_weeks_required_for_profile",
    )

    parsed = parser.parse_args(args)
    return (
        parsed.minimum_weeks_required_for_profile
        if parsed.minimum_weeks_required_for_profile is not None
        else MINIMUM_WEEKS_INPUT_FOR_PROFILE
    )
