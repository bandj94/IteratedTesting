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
        if self._stock.get(sku, 0) < qty:
            raise ValueError("not enough stock")
        self._stock[sku] -= qty

    def quantity(self, sku):
        """Units in stock for sku (0 if it was never added)."""
        return self._stock.get(sku, 0)

    def low_stock(self, threshold):
        """Sorted SKUs that were added at some point and now have fewer than `threshold` units."""
        return sorted(sku for sku, qty in self._stock.items() if qty < threshold)
