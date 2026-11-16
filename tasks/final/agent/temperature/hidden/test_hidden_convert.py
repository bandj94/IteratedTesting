import unittest

from convert import c_to_f, f_to_c, k_to_c, parse_reading


class HiddenConvertTests(unittest.TestCase):
    def test_below_absolute_zero(self):
        with self.assertRaises(ValueError):
            k_to_c(-1)

    def test_k_to_c_value(self):
        self.assertEqual(k_to_c(300), 26.85)

    def test_bad_readings(self):
        for bad in ["abc", "12K", "", "C", "1.2.3F"]:
            with self.assertRaises(ValueError):
                parse_reading(bad)

    def test_whitespace_and_case(self):
        self.assertEqual(parse_reading(" 98.6 f "), 37.0)
        self.assertEqual(parse_reading("21.5C"), 21.5)

    def test_minus_forty(self):
        self.assertEqual(c_to_f(-40), -40.0)
        self.assertEqual(f_to_c(-40), -40.0)


if __name__ == "__main__":
    unittest.main()
