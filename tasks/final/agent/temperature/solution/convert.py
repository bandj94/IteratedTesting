"""Temperature conversions. Every result is rounded to 2 decimal places."""
import re


def c_to_f(c):
    return round(c * 9 / 5 + 32, 2)


def f_to_c(f):
    return round((f - 32) * 5 / 9, 2)


def parse_reading(text):
    """Parse a reading such as "21.5C", "70F" or "-3.2 c"."""
    m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*([CcFf])\s*", text)
    if not m:
        raise ValueError(f"bad reading {text!r}")
    value = float(m.group(1))
    return round(value, 2) if m.group(2) in "Cc" else f_to_c(value)


def k_to_c(k):
    if k < 0:
        raise ValueError("below absolute zero")
    return round(k - 273.15, 2)
