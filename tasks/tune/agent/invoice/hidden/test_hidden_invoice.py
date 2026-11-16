import unittest

from invoice import invoice_total, tax


class HiddenInvoiceTests(unittest.TestCase):
    def test_half_cent_rounds_up(self):
        self.assertEqual(tax(5, 10), 1)
        self.assertEqual(tax(25, 2), 1)
        self.assertEqual(tax(4, 10), 0)

    def test_zero_rate(self):
        self.assertEqual(tax(999, 0), 0)

    def test_discount_then_tax(self):
        self.assertEqual(invoice_total([("a", 1, 1000)], 10, discount_cents=250), 825)

    def test_discount_equal_to_subtotal(self):
        self.assertEqual(invoice_total([("a", 1, 1000)], 10, discount_cents=1000), 0)

    def test_large_amount_is_exact(self):
        self.assertEqual(tax(10**15 + 5, 10), 10**14 + 1)


if __name__ == "__main__":
    unittest.main()
