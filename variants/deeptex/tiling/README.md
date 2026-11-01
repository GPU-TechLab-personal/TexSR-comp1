# variants/deeptex/tiling/

The tiling study: does the paper's "scalable tiling" (upstream `test.py`,
`split_large_image` / `join_large_image`, 512-px tiles butt-joined) leave seams? The
128 -> 512 comparison never triggers it (it fires only above 512 px), so this is a
separate run on six held-out textures_db images large enough to tile (2048^2: `080`,
`081`; 1024^2: `217`, `231`, `275`, `293`), scored against bicubic and against
`infer/deeptex_sr.py`'s in-memory overlap tiling, both with the paper checkpoint.

## Contents

| file | what it does |
|---|---|
| `deeptex_tiling_report.py` | dataprep container (numpy + cv2): PSNR / SSIM of `bicubic`, `upstream`, `ours` against GT with `variants/util/texsr_metrics.full_reference`, and the seam ratio across the 512-px grid; writes `sheet_<stem>.png`, `seam_<stem>.png`, `tiling_report.json` |

## How it works

```
runs/deeptex/tiling/src/<stem>.png                      GT (normalised, holdout split)
  -> external/DeepTextures/test.py  (CWD runs/deeptex/tiling/upstream, ckpt placed as ./checkpoints/original)
         runs/deeptex/tiling/upstream/<stem>/HR_<stem>.png.png   (or HR_<stem>.png_0_0.png for a 1x1 grid)
  -> infer/deeptex_sr.py --make-lr bicubic --tile-size 128 --overlap 16 --save-bicubic ...
         runs/deeptex/tiling/ours/<stem>.png, runs/deeptex/tiling/bicubic/<stem>.png
  -> deeptex_tiling_report.py
         runs/deeptex/tiling/{sheet_<stem>.png, seam_<stem>.png, tiling_report.json}
```

The seam ratio is the mean absolute first difference across each interior 512-px
boundary divided by the same over every other position; ~1.0 means the join is
statistically ordinary. The number to read is `seam_vs_gt` = ratio(image) /
ratio(GT), which divides out texture that genuinely lines up with the 512 grid.
`seam_<stem>.png` is a close-up across x = 512 (upstream | ours | GT).

`runs/tiling/` holds the committed copy of the outputs and a hand-written `summary.md`
(Korean) with the results; the script does not write `summary.md`.

## Run

`test.py` hardcodes `./checkpoints/original`, so the paper checkpoint
(`runs/deeptex/paper/checkpoints/esrgan_40k`) is placed under that name in
`runs/deeptex/tiling/upstream/checkpoints/original` first; the submodule is not changed.

```
docker compose -f docker/compose.yaml run --rm \
  -w /workspace/runs/deeptex/tiling/upstream \
  -e PYTHONPATH=/workspace/external/DeepTextures \
  deeptex python /workspace/external/DeepTextures/test.py \
    --cfg_path /workspace/variants/deeptex/paper/esrgan_paper.yaml \
    --img_path /workspace/runs/deeptex/tiling/src/081.png

docker compose -f docker/compose.yaml run --rm deeptex \
  python /workspace/variants/deeptex/infer/deeptex_sr.py \
    --cfg /workspace/variants/deeptex/paper/esrgan_paper.yaml \
    --ckpt-dir /workspace/runs/deeptex/paper/checkpoints/esrgan_40k \
    --in-dir /workspace/runs/deeptex/tiling/src \
    --out-dir /workspace/runs/deeptex/tiling/ours \
    --make-lr bicubic --tile-size 128 --overlap 16 \
    --save-bicubic /workspace/runs/deeptex/tiling/bicubic

docker compose -f docker/compose.yaml run --rm dataprep \
  python variants/deeptex/tiling/deeptex_tiling_report.py
```

`test.py` runs once per image (`--img_path`). The report's defaults are the
`runs/deeptex/tiling/{src,upstream,ours,bicubic}` inputs and `--out
/workspace/runs/deeptex/tiling`; `--cell` (340) is the sheet cell size.

## Known behaviour and defects

- The script never creates `--out`, and `cv2.imwrite` into a missing folder returns
  False unchecked (`:186`, `:197`), so the sheets silently go missing and only the
  final `open(.../tiling_report.json)` raises. Create the folder first.
- The `ours` seam figure is not like-for-like: `deeptex_sr.py` tiles in LR space with
  an overlap, so its boundaries mostly miss the 512-px columns sampled. It only shows
  that nothing pathological happens there.
- `test.py`'s own printed PSNR / SSIM use `rgb2ycbcr` on BGR input (R and B luma
  weights swapped); the report recomputes everything from the PNGs.
- The commands in `runs/tiling/summary.md` still name `scripts/...` and
  `configs/deeptex/...` paths; the current ones are above.
