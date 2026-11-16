"""A tiny in-memory stock tracker."""


class Inventory:
    def __init__(self):
        self._stock = {}

    def add(self, sku, qty):
        """Add qty (> 0) units of sku."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        self._stock[sku] = self._stock.get(sku, 0) + qty

    def remove(self, sku, qty):
        """Remove qty (> 0) units of sku. Raise ValueError if there is not enough stock."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        self._stock[sku] = self._stock.get(sku, 0) - qty

    def quantity(self, sku):
        """Units in stock for sku (0 if it was never added)."""
        return self._stock.get(sku, 0)

    # TODO: low_stock(threshold) -> a sorted list of the SKUs that have been added at some point and
    # now have fewer than `threshold` units (SKUs at zero units are included).
