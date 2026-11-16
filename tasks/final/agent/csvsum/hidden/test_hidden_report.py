import unittest

from report import column_total


class HiddenReportTests(unittest.TestCase):
    def test_case_insensitive_header(self):
        self.assertAlmostEqual(column_total("a,b\n1,2\n3,4\n", "B"), 6.0)

    def test_header_only(self):
        self.assertEqual(column_total("a,b\n", "a"), 0.0)

    def test_windows_line_endings(self):
        self.assertAlmostEqual(column_total("x\r\n1\r\n2\r\n", "x"), 3.0)

    def test_blank_and_space_cells(self):
        self.assertAlmostEqual(column_total("x,y\n1,a\n  ,b\n\n2,c\n", "x"), 3.0)

    def test_returns_float(self):
        self.assertIsInstance(column_total("q\n2\n", "q"), float)


if __name__ == "__main__":
    unittest.main()
