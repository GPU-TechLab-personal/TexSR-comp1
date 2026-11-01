"""Drawing: every sheet a report writes is one of three shapes.

  strip(cols, title, order)   columns side by side, a label above each, full
                              resolution (16-bit in, 8-bit out) - sheet_<map>.png
  grid(cells, rows, columns)  rows x columns of square cells, row labels on the
                              left, column labels on top - montage.png
  table(rows, columns)        one row per material, one column per method -
                              the overview

GRID and TABLE hold the paddings the two layouts use; callers that draw their
own variant read them from here so the sheets line up.
"""
import collections

import cv2
import numpy as np

from variants.util.report.images import to_display

FONT = cv2.FONT_HERSHEY_SIMPLEX

Style = collections.namedtuple("Style", "pad left head title")
GRID = Style(pad=6, left=96, head=34, title=32)
TABLE = Style(pad=4, left=210, head=30, title=34)


def fit(text, width, start=0.5, floor=0.3):
    """The largest font scale <= start at which `text` fits in `width` pixels."""
    s = start
    while cv2.getTextSize(text, FONT, s, 1)[0][0] > width and s > floor:
        s -= 0.02
    return s


def placeholder(size, note):
    """A grey "n/a" cell with up to a few lines saying what would fill it."""
    cell = np.full((size, size, 3), 28, np.uint8)
    cv2.putText(cell, "n/a", (size // 2 - 26, size // 2 - 6), FONT, 0.7,
                (110, 110, 110), 2, cv2.LINE_AA)
    for i, line in enumerate(note):
        w = cv2.getTextSize(line, FONT, 0.36, 1)[0][0]
        cv2.putText(cell, line, (max(4, size // 2 - w // 2), size // 2 + 20 + i * 15),
                    FONT, 0.36, (90, 90, 90), 1, cv2.LINE_AA)
    return cell


def strip(cols, title, order=None):
    """Contact sheet for eyeballing. 8-bit: cv2.putText will not draw on 16-bit."""
    tiles = []
    for name in (order or list(cols)):
        a = cols[name]
        a8 = (a >> 8).astype(np.uint8)
        if a8.ndim == 2:
            a8 = cv2.cvtColor(a8, cv2.COLOR_GRAY2BGR)
        bar = np.zeros((30, a8.shape[1], 3), np.uint8)
        cv2.putText(bar, name, (8, 21), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, a8]))
        tiles.append(np.full((tiles[-1].shape[0], 4, 3), 255, np.uint8))
    sheet = np.hstack(tiles[:-1])
    head = np.zeros((26, sheet.shape[1], 3), np.uint8)
    cv2.putText(head, title, (8, 18), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([head, sheet])


def grid(cells, rows, columns, size, title, notes=None, left=GRID.left):
    """Rows x columns of `size` cells.

    cells   {(row label, column): array or None}; None draws a placeholder with
            notes.get((row label, column)) as its text.
    rows    row labels, top to bottom; columns: column labels, left to right.
    """
    notes = notes or {}
    pad, head, top = GRID.pad, GRID.head, GRID.title
    w = left + len(columns) * (size + pad) + pad
    h = top + head + len(rows) * (size + pad) + pad
    sheet = np.full((h, w, 3), 18, np.uint8)

    cv2.putText(sheet, title, (10, 22), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    for c, col in enumerate(columns):
        x = left + c * (size + pad)
        cv2.putText(sheet, col, (x + 4, top + 22), FONT, 0.55,
                    (210, 210, 210), 1, cv2.LINE_AA)

    for r, row in enumerate(rows):
        y = top + head + r * (size + pad)
        cv2.putText(sheet, row, (10, y + size // 2), FONT, 0.62,
                    (210, 210, 210), 1, cv2.LINE_AA)
        for c, col in enumerate(columns):
            x = left + c * (size + pad)
            cell = to_display(cells.get((row, col)), size)
            if cell is None:
                cell = placeholder(size, notes.get((row, col), []))
            sheet[y:y + size, x:x + size] = cell
    return sheet


def table(rows, columns, size, title):
    """One row per material, one column per method.

    rows     [(row label, {column key: array or None})]
    columns  [(column label, column key)]; a missing cell draws "n/a".
    """
    pad, left, head, top = TABLE.pad, TABLE.left, TABLE.head, TABLE.title
    w = left + len(columns) * (size + pad) + pad
    h = top + head + len(rows) * (size + pad) + pad
    sheet = np.full((h, w, 3), 18, np.uint8)

    cv2.putText(sheet, title, (10, 24), FONT, 0.62, (255, 255, 255), 1, cv2.LINE_AA)

    for c, (lab, _) in enumerate(columns):
        x = left + c * (size + pad)
        cv2.putText(sheet, lab, (x + 3, top + 20), FONT,
                    fit(lab, size - 6, 0.5), (215, 215, 215), 1, cv2.LINE_AA)

    for r, (name, cells) in enumerate(rows):
        y = top + head + r * (size + pad)
        cv2.putText(sheet, name, (8, y + size // 2), FONT,
                    fit(name, left - 16, 0.46), (215, 215, 215), 1, cv2.LINE_AA)
        for c, (_, key) in enumerate(columns):
            x = left + c * (size + pad)
            cell = to_display(cells.get(key), size, drop_alpha=True)
            if cell is None:
                cell = np.full((size, size, 3), 30, np.uint8)
                cv2.putText(cell, "n/a", (size // 2 - 18, size // 2), FONT, 0.5,
                            (110, 110, 110), 1, cv2.LINE_AA)
            sheet[y:y + size, x:x + size] = cell
    return sheet
