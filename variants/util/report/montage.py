"""Grids and tables read from comparison sets (layout.py).

    montage(set_path, maps, rows, size, title)   one material: rows x maps
    overview(mat_dirs, map, columns, size, title) every material: materials x columns

Both only read <set>/<map>/<column>.png; nothing is recomputed, so a montage can
never drift from the per-map columns it shows.
"""
import os

from variants.util.report import layout, sheets
from variants.util.report.images import read_if


def read_cells(set_path, maps, rows):
    """{(row label, map): array or None} for rows = [(row label, column file stem)]."""
    cells = {}
    for m in maps:
        for row, column in rows:
            p = layout.cell_path(set_path, m, column)
            cells[(row, m)] = read_if(p) if os.path.exists(p) else None
    return cells


def montage(set_path, maps, rows, size, title, notes=None, left=sheets.GRID.left):
    """Maps across, rows (label, column) down; missing cells get notes[(row, map)]."""
    cells = read_cells(set_path, maps, rows)
    return sheets.grid(cells, [r for r, _ in rows], maps, size, title, notes, left)


def overview_rows(mat_dirs, map_name, columns, require="gt"):
    """[(material, {column: array or None})] from each material's newest set;
    materials without a `require` cell are left out."""
    rows = []
    for d in mat_dirs:
        s = layout.newest_set(d)
        if not s:
            continue
        cells = {}
        for _, key in columns:
            p = layout.cell_path(s, map_name, key)
            cells[key] = read_if(p) if os.path.exists(p) else None
        if require and cells.get(require) is None:
            continue
        rows.append((os.path.basename(d.rstrip("/")), cells))
    return rows


def overview(mat_dirs, map_name, columns, size, title, require="gt"):
    """(sheet, n rows) - None, 0 when no material has a set."""
    rows = overview_rows(mat_dirs, map_name, columns, require)
    if not rows:
        return None, 0
    return sheets.table(rows, columns, size, title), len(rows)
