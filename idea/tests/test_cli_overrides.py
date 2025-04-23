from unittest import TestCase

from idea.cli_overrides import parse_minimum_weeks_required_for_profile
from idea.constants import (
    MINIMUM_WEEKS_INPUT_FOR_PROFILE,
)


class TestIdeaUtil(TestCase):
    def test_default_parse_minimum_weeks_required_for_profile(self):
        result = parse_minimum_weeks_required_for_profile([])
        self.assertEqual(result, MINIMUM_WEEKS_INPUT_FOR_PROFILE)

    def test_argument_is_parsed_correctly(self):
        result = parse_minimum_weeks_required_for_profile(
            ["--minimum-weeks-required-for-profile", "6"]
        )
        self.assertEqual(result, 6)

    def test_invalid_argument_type(self):
        with self.assertRaises(SystemExit):
            parse_minimum_weeks_required_for_profile(
                ["--minimum-weeks-required-for-profile", "not-a-number"]
            )
