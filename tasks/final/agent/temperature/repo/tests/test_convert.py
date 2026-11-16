import unittest

from convert import c_to_f, f_to_c, k_to_c, parse_reading


class ConvertTests(unittest.TestCase):
    def test_c_to_f(self):
        self.assertEqual(c_to_f(100), 212.0)

    def test_f_to_c(self):
        self.assertEqual(f_to_c(212), 100.0)
        self.assertEqual(f_to_c(32), 0.0)

    def test_parse_lowercase_and_spaces(self):
        self.assertEqual(parse_reading("-3.2 c"), -3.2)
        self.assertEqual(parse_reading("70F"), 21.11)

    def test_k_to_c(self):
        self.assertEqual(k_to_c(273.15), 0.0)


if __name__ == "__main__":
    unittest.main()
