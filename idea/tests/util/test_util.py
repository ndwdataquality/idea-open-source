from unittest import TestCase

from idea.util import util


class TestUtil(TestCase):
    def test_flatten_valid_cases(self):
        test_cases = [
            ([], []),
            ([1, 2, 3], [1, 2, 3]),
            ([[1, 2], [3, 4]], [1, 2, 3, 4]),
            ([1, [2, 3], 4], [1, 2, 3, 4]),
            ([1, [2, [3, [4]]]], [1, 2, 3, 4]),
            ([[[[[5]]]]], [5]),
            (["a", ["b", ["c"]]], ["a", "b", "c"]),
            ([[1], 2, [[3, [4]]]], [1, 2, 3, 4]),
        ]

        for input_list, expected_output in test_cases:
            with self.subTest(input=input_list):
                # Act
                result = util.flatten(input_list)

                # Assert
                self.assertEqual(result, expected_output)
                self.assertIsInstance(result, list)

    def test_reverse_dictionary_valid_cases(self):
        test_cases = [
            ({}, {}),
            ({1: "a", 2: "b"}, {"a": 1, "b": 2}),
            ({"x": 10, "y": 20}, {10: "x", 20: "y"}),
            ({"a": 1, "b": 2, "c": 3}, {1: "a", 2: "b", 3: "c"}),
            ({True: "yes", False: "no"}, {"yes": True, "no": False}),
        ]

        for input_dict, expected_output in test_cases:
            with self.subTest(input=input_dict):
                # Act
                result = util.reverse_dictionary(input_dict)

                # Assert
                self.assertEqual(result, expected_output)
                self.assertIsInstance(result, dict)

    def test_reverse_dictionary_value_collision(self):
        # Arrange
        input_dict = {"a": 1, "b": 1}

        # Act
        result = util.reverse_dictionary(input_dict)

        # Assert
        # Only one entry survives the key collision
        self.assertEqual(set(result.keys()), {1})
        self.assertIn(result[1], {"a", "b"})
