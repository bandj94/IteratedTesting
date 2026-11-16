import unittest

from report import column_total

DATA = 'name, Amount ,note\nwidget,10.5,ok\n"gadget, large",4,"has, comma"\nthing,,missing\n'


class ReportTests(unittest.TestCase):
    def test_total_with_quotes_and_spaces(self):
        self.assertAlmostEqual(column_total(DATA, "amount"), 14.5)

    def test_missing_column(self):
        with self.assertRaises(KeyError):
            column_total(DATA, "price")


if __name__ == "__main__":
    unittest.main()
