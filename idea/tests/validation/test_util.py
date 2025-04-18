from unittest import TestCase

from idea.exceptions import IDEAError
from idea.validation.util import update_no_coverage_counters


class TestUtil(TestCase):
    def test_fcd_zero(self):
        # fcd = 0: Beide counters moeten verhoogd worden.
        self.assertEqual(update_no_coverage_counters(0, 0, 0), (1, 1))
        self.assertEqual(update_no_coverage_counters(0, 2, 3), (3, 4))

    def test_fcd_one(self):
        # fcd = 1: FCD==0-counter reset, FCD in (0, 1)-counter verhoogt.
        self.assertEqual(update_no_coverage_counters(1, 5, 2), (0, 3))

    def test_fcd_not_zero_or_one(self):
        # fcd != 0 en fcd != 1: Beide counters worden gereset.
        self.assertEqual(update_no_coverage_counters(2, 4, 6), (0, 0))
        self.assertEqual(update_no_coverage_counters(5, 1, 1), (0, 0))

    def test_sequence(self):
        # Test een sequentie van fcd-waarden: [0, 0, 1, 2].
        counters = (0, 0)
        for fcd in [0, 0, 1, 2]:
            counters = update_no_coverage_counters(fcd, *counters)
        self.assertEqual(counters, (0, 0))

    def test_fcd_out_of_range(self):
        # Test dat een fcd-waarde buiten 0-10 een ValueError veroorzaakt.
        with self.assertRaises(IDEAError):
            update_no_coverage_counters(11, 0, 0)
        with self.assertRaises(IDEAError):
            update_no_coverage_counters(-1, 0, 0)
