"""Summaries for CSV exports."""
import csv
import io


def column_total(csv_text, column):
    """Sum the numbers in one column of CSV text and return a float."""
    rows = list(csv.reader(io.StringIO(csv_text)))
    if not rows:
        raise KeyError(column)
    header = [h.strip().lower() for h in rows[0]]
    key = column.strip().lower()
    if key not in header:
        raise KeyError(column)
    idx = header.index(key)
    total = 0.0
    for row in rows[1:]:
        if idx >= len(row):
            continue
        cell = row[idx].strip()
        if cell:
            total += float(cell)
    return total
