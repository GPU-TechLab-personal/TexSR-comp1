# DeepTextures arms vs MatUp on the 12 MatSynth evaluation materials

PSNR/SSIM at 8 bits against GT, after the same 8 px crop as runs/montages.

- `n`: the materials every column has a finite PSNR on. All means in a table are
  over this same set, so the rows are comparable.
- `=GT`: materials a column reproduces exactly (infinite PSNR), counted over all
  materials. Metallic is the case that needs it: 9 of the 12 GTs are all zero.
- `offset`: mean over materials of the largest per-channel mean offset vs GT, in
  8-bit levels. `PSNR -off` is PSNR after removing that offset. Both are
  DIAGNOSTICS: they show how much of a column's error is a constant shift (the
  DeepTextures recipe drifts in colour) rather than wrong structure.
- `|grad| %GT`: each material's gradient magnitude as a percentage of its GT's,
  averaged. 100 means as much detail as the GT; PSNR rewards blur, this does not.
- Head-to-head lines count materials where the first column wins; an exact
  reproduction of GT counts as the best PSNR.

Arms: `dt_matsynth_all` = /workspace/runs/deeptex/eval/matsynth_all, `dt_render_r0b` = /workspace/runs/deeptex/eval/render_r0b_40k, `dt_render_r0a` = /workspace/runs/deeptex/eval/render_r0a_40k, `dt_render_r1` = /workspace/runs/deeptex/eval/render_r1_40k.

## baseColor

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT |
|---|---|---|---|---|---|---|---|
| lr | 12 | 31.32 | 0 | 0.673 | 0.0 | 31.32 | 51% |
| bilinear | 12 | 31.93 | 0 | 0.674 | 0.1 | 31.94 | 40% |
| matup | 12 | 27.22 | 0 | 0.600 | 5.1 | 30.20 | 43% |
| dt_matsynth_all | 12 | 26.88 | 0 | 0.628 | 8.7 | 30.47 | 95% |
| dt_render_r0b | 12 | 25.89 | 0 | 0.617 | 10.3 | 29.91 | 94% |
| dt_render_r0a | 12 | 26.30 | 0 | 0.637 | 8.2 | 30.83 | 86% |
| dt_render_r1 | 12 | 30.21 | 0 | 0.662 | 2.5 | 31.39 | 79% |

- `dt_render_r1` beats `dt_render_r0a` on PSNR 8/12, SSIM 8/12
- `dt_matsynth_all` beats `matup` on PSNR 1/12, SSIM 5/12
- `dt_render_r0b` beats `matup` on PSNR 1/12, SSIM 4/12
- `dt_render_r0a` beats `matup` on PSNR 4/12, SSIM 8/12
- `dt_render_r1` beats `matup` on PSNR 8/12, SSIM 6/12

## normal

| column | n | PSNR | =GT | SSIM | ang. err ° | offset | PSNR -off | \|grad\| %GT |
|---|---|---|---|---|---|---|---|---|
| lr | 12 | 33.22 | 0 | 0.734 | 6.51 | 0.0 | 33.22 | 39% |
| bilinear | 12 | 33.41 | 0 | 0.733 | 6.28 | 0.1 | 33.42 | 33% |
| matup | 12 | 32.29 | 0 | 0.721 | 6.54 | 5.9 | 33.03 | 28% |
| dt_matsynth_all | 12 | 23.65 | 0 | 0.675 | 13.26 | 20.5 | 30.86 | 94% |
| dt_render_r0b | 12 | 27.07 | 0 | 0.677 | 9.64 | 9.8 | 31.18 | 104% |
| dt_render_r0a | 12 | 24.06 | 0 | 0.670 | 9.57 | 14.8 | 30.54 | 132% |
| dt_render_r1 | 12 | 32.61 | 0 | 0.753 | 6.09 | 1.9 | 33.28 | 66% |

- `dt_render_r1` beats `dt_render_r0a` on PSNR 12/12, SSIM 12/12, angle 12/12
- `dt_matsynth_all` beats `matup` on PSNR 0/12, SSIM 0/12, angle 0/12
- `dt_render_r0b` beats `matup` on PSNR 0/12, SSIM 0/12, angle 0/12
- `dt_render_r0a` beats `matup` on PSNR 0/12, SSIM 1/12, angle 0/12
- `dt_render_r1` beats `matup` on PSNR 8/12, SSIM 7/12, angle 6/12

## roughness

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT | ch. spread /255 |
|---|---|---|---|---|---|---|---|---|
| lr | 11 | 32.18 | 1 | 0.723 | 0.0 | 32.18 | 45% | - |
| bilinear | 11 | 32.52 | 1 | 0.720 | 0.1 | 32.52 | 34% | - |
| matup | 11 | 23.61 | 0 | 0.574 | 11.5 | 26.41 | 128% | - |
| dt_matsynth_all | 11 | 31.31 | 0 | 0.707 | 2.8 | 31.59 | 83% | 8.60 |
| dt_render_r0b | 11 | 31.81 | 0 | 0.702 | 0.7 | 31.84 | 77% | 12.38 |
| dt_render_r0a | 11 | 30.15 | 0 | 0.723 | 2.3 | 32.05 | 72% | 7.01 |
| dt_render_r1 | 11 | 30.23 | 0 | 0.690 | 1.6 | 31.43 | 84% | 5.72 |

- `dt_render_r1` beats `dt_render_r0a` on PSNR 3/12, SSIM 2/12
- `dt_matsynth_all` beats `matup` on PSNR 12/12, SSIM 9/12
- `dt_render_r0b` beats `matup` on PSNR 12/12, SSIM 9/12
- `dt_render_r0a` beats `matup` on PSNR 12/12, SSIM 11/12
- `dt_render_r1` beats `matup` on PSNR 12/12, SSIM 7/12

## metallic

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT | ch. spread /255 |
|---|---|---|---|---|---|---|---|---|
| lr | 3 | 19.37 | 9 | 0.583 | 0.0 | 19.37 | 49% | - |
| bilinear | 3 | 19.70 | 9 | 0.570 | 0.1 | 19.70 | 36% | - |
| matup | 3 | 19.27 | 3 | 0.566 | 4.0 | 19.40 | 36% | - |
| dt_matsynth_all | 3 | 19.37 | 9 | 0.605 | 5.2 | 19.57 | 68% | 10.84 |
| dt_render_r0b | 3 | 19.65 | 9 | 0.614 | 1.6 | 19.68 | 66% | 14.45 |
| dt_render_r0a | 3 | 19.53 | 9 | 0.530 | 3.1 | 19.62 | 70% | 7.86 |
| dt_render_r1 | 3 | 19.96 | 9 | 0.616 | 0.8 | 19.96 | 64% | 8.13 |

- `dt_render_r1` beats `dt_render_r0a` on PSNR 3/12, SSIM 2/12
- `dt_matsynth_all` beats `matup` on PSNR 7/12, SSIM 9/12
- `dt_render_r0b` beats `matup` on PSNR 8/12, SSIM 9/12
- `dt_render_r0a` beats `matup` on PSNR 7/12, SSIM 8/12
- `dt_render_r1` beats `matup` on PSNR 8/12, SSIM 9/12

## height

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT | ch. spread /255 |
|---|---|---|---|---|---|---|---|---|
| lr | 12 | 31.97 | 0 | 0.787 | 0.0 | 31.97 | 69% | - |
| bilinear | 12 | 34.66 | 0 | 0.832 | 0.0 | 34.66 | 56% | - |
| matup | 12 | 31.13 | 0 | 0.825 | 0.0 | 31.13 | 79% | - |
| dt_matsynth_all | 12 | 34.08 | 0 | 0.829 | 1.3 | 34.23 | 90% | 7.42 |
| dt_render_r0b | 12 | 33.35 | 0 | 0.830 | 1.4 | 34.33 | 86% | 10.43 |
| dt_render_r0a | 12 | 32.68 | 0 | 0.840 | 2.6 | 34.49 | 83% | 6.44 |
| dt_render_r1 | 12 | 33.27 | 0 | 0.828 | 1.8 | 34.32 | 91% | 4.15 |

- `dt_render_r1` beats `dt_render_r0a` on PSNR 9/12, SSIM 4/12
- `dt_matsynth_all` beats `matup` on PSNR 5/12, SSIM 5/12
- `dt_render_r0b` beats `matup` on PSNR 5/12, SSIM 4/12
- `dt_render_r0a` beats `matup` on PSNR 5/12, SSIM 6/12
- `dt_render_r1` beats `matup` on PSNR 5/12, SSIM 4/12

## render

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT |
|---|---|---|---|---|---|---|---|
| lr | 12 | 22.37 | 0 | 0.411 | 2.2 | 22.45 | 46% |
| bilinear | 12 | 27.30 | 0 | 0.612 | 3.4 | 27.51 | 34% |
| matup | 12 | 26.36 | 0 | 0.605 | 4.0 | 26.60 | 33% |
| dt_matsynth_all | 12 | 25.96 | 0 | 0.591 | 5.4 | 26.59 | 77% |
| dt_render_r0b | 12 | 24.92 | 0 | 0.558 | 9.8 | 26.30 | 79% |
| dt_render_r0a | 12 | 25.65 | 0 | 0.604 | 7.8 | 26.90 | 67% |
| dt_render_r1 | 12 | 26.49 | 0 | 0.599 | 4.8 | 26.83 | 74% |

- `dt_render_r1` beats `dt_render_r0a` on PSNR 7/12, SSIM 4/12
- `dt_matsynth_all` beats `matup` on PSNR 1/12, SSIM 3/12
- `dt_render_r0b` beats `matup` on PSNR 1/12, SSIM 2/12
- `dt_render_r0a` beats `matup` on PSNR 3/12, SSIM 6/12
- `dt_render_r1` beats `matup` on PSNR 3/12, SSIM 5/12
