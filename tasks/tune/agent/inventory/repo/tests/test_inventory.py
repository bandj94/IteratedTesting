import unittest

from inventory import Inventory


class InventoryTests(unittest.TestCase):
    def test_add_and_quantity(self):
        inv = Inventory()
        inv.add("A1", 5)
        inv.add("A1", 2)
        self.assertEqual(inv.quantity("A1"), 7)

    def test_remove_too_much_raises(self):
        inv = Inventory()
        inv.add("A1", 1)
        with self.assertRaises(ValueError):
            inv.remove("A1", 2)
        self.assertEqual(inv.quantity("A1"), 1)

    def test_low_stock(self):
        inv = Inventory()
        inv.add("B2", 10)
        inv.add("A1", 3)
        inv.add("C3", 1)
        self.assertEqual(inv.low_stock(5), ["A1", "C3"])


if __name__ == "__main__":
    unittest.main()
