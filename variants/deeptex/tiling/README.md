# deeptex/tiling — the paper's "scalable tiling", measured

| file | old file | kind |
|---|---|---|
| `deeptex_tiling_report.py` | `scripts/deeptex/deeptex_tiling_report.py` | moved (L0: AST-identical) |

## What changed against the old code (new file:line)

| new | was | change |
|---|---|---|
| `deeptex_tiling_report.py:63-65` | `deeptex_tiling_report.py:63-64` | `texsr_metrics` from `variants.util` instead of `sys.path.insert(0, "/workspace/scripts/common")` |

L3 regenerates `runs/tiling/tiling_report.json` with old and new: numerically identical
to the committed file, old and new byte-identical (`l3_reports.py tiling`).

## Path dependencies on other arms

- Inputs `runs/deeptex/tiling/{src,upstream,ours,bicubic}`: `upstream` is upstream's
  `test.py` (split_large_image), `ours` is `infer/deeptex_sr.py`, both with the paper
  checkpoint (`runs/deeptex/paper/checkpoints/esrgan_40k`).
- `runs/tiling/summary.md` is hand-written; this script writes only the JSON and the
  `sheet_*.png` / `seam_*.png` images.

## Kept, not fixed

- The script never creates `--out`, and `cv2.imwrite` into a missing folder returns
  False unchecked (`:186`, `:197`), so the sheets silently go missing and only the
  final `open(.../tiling_report.json)` raises. The L3 test creates the folder first.
