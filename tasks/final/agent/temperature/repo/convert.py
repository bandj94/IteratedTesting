"""Temperature conversions. Every result is rounded to 2 decimal places."""


def c_to_f(c):
    return round(c * 9 / 5 + 32, 2)


def f_to_c(f):
    return round(f - 32 * 5 / 9, 2)


def parse_reading(text):
    """Parse a reading such as "21.5C", "70F" or "-3.2 c": a number, optional spaces, then the unit
    C or F in any case. Surrounding whitespace is ignored. Return the value in Celsius, rounded to
    2 decimals. Raise ValueError for anything else."""
    value, unit = float(text[:-1]), text[-1]
    if unit == "C":
        return round(value, 2)
    return f_to_c(value)


# TODO: add k_to_c(k): Kelvin to Celsius (k - 273.15), rounded to 2 decimals.
# Raise ValueError when k is below absolute zero (k < 0).
