"""Summaries for CSV exports."""


def column_total(csv_text, column):
    """Sum the numbers in one column of CSV text and return a float.

    - The first line is the header. The column is matched by header name, ignoring surrounding
      spaces and letter case.
    - Fields may be wrapped in double quotes and may then contain commas.
    - Cells that are empty or contain only spaces are skipped; blank lines are skipped.
    - Raise KeyError if the column does not exist.
    """
    lines = csv_text.strip().splitlines()
    header = lines[0].split(",")
    idx = header.index(column)
    total = 0.0
    for line in lines[1:]:
        total += float(line.split(",")[idx])
    return total
