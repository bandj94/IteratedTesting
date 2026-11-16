import unittest

from invoice import invoice_total, subtotal, tax


class InvoiceTests(unittest.TestCase):
    def test_subtotal(self):
        self.assertEqual(subtotal([("pen", 3, 150), ("pad", 2, 425)]), 1300)

    def test_tax_rounds_to_nearest_cent(self):
        self.assertEqual(tax(1050, 10), 105)
        self.assertEqual(tax(15, 10), 2)      # 1.5 cents rounds up
        self.assertEqual(tax(14, 10), 1)      # 1.4 cents rounds down

    def test_discount_never_goes_below_zero(self):
        self.assertEqual(invoice_total([("pen", 1, 100)], 20, discount_cents=500), 0)

    def test_total_with_tax(self):
        self.assertEqual(invoice_total([("pen", 2, 333)], 8), 719)


if __name__ == "__main__":
    unittest.main()
