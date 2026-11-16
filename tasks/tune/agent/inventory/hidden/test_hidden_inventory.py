import unittest

from inventory import Inventory


class HiddenInventoryTests(unittest.TestCase):
    def test_low_stock_includes_zero(self):
        inv = Inventory()
        inv.add("Z", 2)
        inv.remove("Z", 2)
        self.assertEqual(inv.low_stock(1), ["Z"])

    def test_low_stock_excludes_unknown(self):
        self.assertEqual(Inventory().low_stock(10), [])

    def test_remove_unknown_raises(self):
        with self.assertRaises(ValueError):
            Inventory().remove("nope", 1)

    def test_threshold_is_strict(self):
        inv = Inventory()
        inv.add("A", 5)
        self.assertEqual(inv.low_stock(5), [])

    def test_remove_exact_amount(self):
        inv = Inventory()
        inv.add("A", 3)
        inv.remove("A", 3)
        self.assertEqual(inv.quantity("A"), 0)


if __name__ == "__main__":
    unittest.main()
