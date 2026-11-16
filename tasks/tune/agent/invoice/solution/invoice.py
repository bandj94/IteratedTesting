"""Invoice calculations. All amounts are integer cents."""


def line_total(qty, unit_price_cents):
    """Price of one invoice line."""
    return qty * unit_price_cents


def subtotal(lines):
    """Sum of all lines. `lines` is a list of (description, qty, unit_price_cents)."""
    total = 0
    for _, qty, price in lines:
        total += line_total(qty, price)
    return total


def tax(amount_cents, rate_percent):
    """Tax on an amount, rounded to the nearest cent; exact halves round up."""
    return (amount_cents * rate_percent + 50) // 100


def invoice_total(lines, rate_percent, discount_cents=0):
    """Subtotal minus the discount (never below zero), plus tax on the discounted amount."""
    base = max(0, subtotal(lines) - discount_cents)
    return base + tax(base, rate_percent)
