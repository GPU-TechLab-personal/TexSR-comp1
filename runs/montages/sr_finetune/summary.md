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

Arms: `dt_matsynth_all` = /workspace/runs/deeptex/eval/matsynth_all_40k, `dt_render_r1` = /workspace/runs/deeptex/eval/render_r1_40k, `bsr_esrgan_zs` = /workspace/runs/sr_ft/eval/esrgan_zs, `kair_swinir_cls_zs` = /workspace/runs/sr_ft/eval/swinir_cls_zs, `kair_swinir_real_zs` = /workspace/runs/sr_ft/eval/swinir_real_zs, `kair_swinir_realL_zs` = /workspace/runs/sr_ft/eval/swinir_realL_zs.

## baseColor

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT |
|---|---|---|---|---|---|---|---|
| lr | 12 | 31.32 | 0 | 0.673 | 0.0 | 31.32 | 51% |
| bilinear | 12 | 31.93 | 0 | 0.674 | 0.1 | 31.94 | 40% |
| matup | 12 | 27.22 | 0 | 0.600 | 5.1 | 30.20 | 43% |
| dt_matsynth_all | 12 | 26.88 | 0 | 0.628 | 8.8 | 30.47 | 95% |
| dt_render_r1 | 12 | 30.21 | 0 | 0.662 | 2.5 | 31.39 | 79% |
| bsr_esrgan_zs | 12 | 26.15 | 0 | 0.454 | 1.8 | 26.34 | 192% |
| kair_swinir_cls_zs | 12 | 32.68 | 0 | 0.727 | 0.7 | 32.70 | 64% |
| kair_swinir_real_zs | 12 | 26.95 | 0 | 0.576 | 4.9 | 27.63 | 183% |
| kair_swinir_realL_zs | 12 | 27.54 | 0 | 0.604 | 3.3 | 28.17 | 155% |

- `kair_swinir_real_zs` beats `kair_swinir_realL_zs` on PSNR 4/12, SSIM 3/12
- `dt_matsynth_all` beats `matup` on PSNR 1/12, SSIM 5/12
- `dt_render_r1` beats `matup` on PSNR 8/12, SSIM 6/12
- `bsr_esrgan_zs` beats `matup` on PSNR 3/12, SSIM 2/12
- `kair_swinir_cls_zs` beats `matup` on PSNR 12/12, SSIM 12/12
- `kair_swinir_real_zs` beats `matup` on PSNR 1/12, SSIM 3/12
- `kair_swinir_realL_zs` beats `matup` on PSNR 2/12, SSIM 3/12

## normal

| column | n | PSNR | =GT | SSIM | ang. err ° | offset | PSNR -off | \|grad\| %GT |
|---|---|---|---|---|---|---|---|---|
| lr | 12 | 33.22 | 0 | 0.734 | 6.51 | 0.0 | 33.22 | 39% |
| bilinear | 12 | 33.41 | 0 | 0.733 | 6.28 | 0.1 | 33.42 | 33% |
| matup | 12 | 32.29 | 0 | 0.721 | 6.54 | 5.9 | 33.03 | 28% |
| dt_matsynth_all | 12 | 23.64 | 0 | 0.675 | 13.26 | 20.6 | 30.86 | 94% |
| dt_render_r1 | 12 | 32.61 | 0 | 0.753 | 6.09 | 1.9 | 33.28 | 66% |
| bsr_esrgan_zs | 12 | 28.45 | 0 | 0.614 | 9.56 | 3.4 | 29.92 | 191% |
| kair_swinir_cls_zs | 12 | 33.61 | 0 | 0.771 | 6.02 | 0.7 | 33.62 | 53% |
| kair_swinir_real_zs | 12 | 25.97 | 0 | 0.636 | 11.03 | 9.3 | 29.71 | 88% |
| kair_swinir_realL_zs | 12 | 26.72 | 0 | 0.629 | 11.05 | 8.0 | 29.59 | 77% |

- `kair_swinir_real_zs` beats `kair_swinir_realL_zs` on PSNR 4/12, SSIM 3/12, angle 4/12
- `dt_matsynth_all` beats `matup` on PSNR 0/12, SSIM 0/12, angle 0/12
- `dt_render_r1` beats `matup` on PSNR 8/12, SSIM 7/12, angle 6/12
- `bsr_esrgan_zs` beats `matup` on PSNR 0/12, SSIM 0/12, angle 0/12
- `kair_swinir_cls_zs` beats `matup` on PSNR 11/12, SSIM 11/12, angle 12/12
- `kair_swinir_real_zs` beats `matup` on PSNR 1/12, SSIM 0/12, angle 0/12
- `kair_swinir_realL_zs` beats `matup` on PSNR 1/12, SSIM 0/12, angle 0/12

## roughness

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT | ch. spread /255 |
|---|---|---|---|---|---|---|---|---|
| lr | 11 | 32.18 | 1 | 0.723 | 0.0 | 32.18 | 45% | - |
| bilinear | 11 | 32.52 | 1 | 0.720 | 0.1 | 32.52 | 34% | - |
| matup | 11 | 23.61 | 0 | 0.574 | 11.5 | 26.41 | 128% | - |
| dt_matsynth_all | 11 | 31.31 | 0 | 0.707 | 2.8 | 31.59 | 83% | 8.65 |
| dt_render_r1 | 11 | 30.23 | 0 | 0.690 | 1.6 | 31.43 | 84% | 5.72 |
| bsr_esrgan_zs | 11 | 28.19 | 0 | 0.561 | 0.8 | 28.26 | 154% | 5.96 |
| kair_swinir_cls_zs | 11 | 32.90 | 1 | 0.756 | 1.2 | 32.96 | 61% | 0.21 |
| kair_swinir_real_zs | 11 | 28.58 | 0 | 0.631 | 3.3 | 29.34 | 66% | 1.12 |
| kair_swinir_realL_zs | 11 | 29.75 | 0 | 0.671 | 2.3 | 30.30 | 53% | 1.00 |

- `kair_swinir_real_zs` beats `kair_swinir_realL_zs` on PSNR 2/12, SSIM 2/12
- `dt_matsynth_all` beats `matup` on PSNR 12/12, SSIM 9/12
- `dt_render_r1` beats `matup` on PSNR 12/12, SSIM 7/12
- `bsr_esrgan_zs` beats `matup` on PSNR 6/12, SSIM 5/12
- `kair_swinir_cls_zs` beats `matup` on PSNR 12/12, SSIM 11/12
- `kair_swinir_real_zs` beats `matup` on PSNR 9/12, SSIM 6/12
- `kair_swinir_realL_zs` beats `matup` on PSNR 11/12, SSIM 8/12

## metallic

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT | ch. spread /255 |
|---|---|---|---|---|---|---|---|---|
| lr | 3 | 19.37 | 9 | 0.583 | 0.0 | 19.37 | 49% | - |
| bilinear | 3 | 19.70 | 9 | 0.570 | 0.1 | 19.70 | 36% | - |
| matup | 3 | 19.27 | 3 | 0.566 | 4.0 | 19.40 | 36% | - |
| dt_matsynth_all | 3 | 19.37 | 9 | 0.605 | 5.2 | 19.57 | 68% | 10.88 |
| dt_render_r1 | 3 | 19.96 | 9 | 0.616 | 0.8 | 19.96 | 64% | 8.13 |
| bsr_esrgan_zs | 3 | 15.12 | 9 | 0.468 | 0.6 | 15.14 | 154% | 9.17 |
| kair_swinir_cls_zs | 3 | 19.63 | 9 | 0.648 | 2.3 | 19.66 | 68% | 0.57 |
| kair_swinir_real_zs | 3 | 18.14 | 9 | 0.538 | 7.9 | 18.43 | 70% | 1.31 |
| kair_swinir_realL_zs | 3 | 18.09 | 9 | 0.535 | 7.5 | 18.35 | 76% | 1.35 |

- `kair_swinir_real_zs` beats `kair_swinir_realL_zs` on PSNR 1/12, SSIM 1/12
- `dt_matsynth_all` beats `matup` on PSNR 7/12, SSIM 9/12
- `dt_render_r1` beats `matup` on PSNR 8/12, SSIM 9/12
- `bsr_esrgan_zs` beats `matup` on PSNR 6/12, SSIM 6/12
- `kair_swinir_cls_zs` beats `matup` on PSNR 8/12, SSIM 9/12
- `kair_swinir_real_zs` beats `matup` on PSNR 6/12, SSIM 7/12
- `kair_swinir_realL_zs` beats `matup` on PSNR 7/12, SSIM 8/12

## height

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT | ch. spread /255 |
|---|---|---|---|---|---|---|---|---|
| lr | 12 | 31.97 | 0 | 0.787 | 0.0 | 31.97 | 69% | - |
| bilinear | 12 | 34.66 | 0 | 0.832 | 0.0 | 34.66 | 56% | - |
| matup | 12 | 31.13 | 0 | 0.825 | 0.0 | 31.13 | 79% | - |
| dt_matsynth_all | 12 | 34.08 | 0 | 0.829 | 1.3 | 34.23 | 90% | 7.46 |
| dt_render_r1 | 12 | 33.27 | 0 | 0.828 | 1.8 | 34.32 | 91% | 4.15 |
| bsr_esrgan_zs | 12 | 31.96 | 0 | 0.714 | 0.5 | 32.00 | 142% | 6.54 |
| kair_swinir_cls_zs | 12 | 36.96 | 0 | 0.866 | 0.4 | 36.99 | 74% | 0.13 |
| kair_swinir_real_zs | 12 | 24.26 | 0 | 0.567 | 3.9 | 24.93 | 180% | 1.45 |
| kair_swinir_realL_zs | 12 | 26.23 | 0 | 0.656 | 3.0 | 26.75 | 130% | 1.01 |

- `kair_swinir_real_zs` beats `kair_swinir_realL_zs` on PSNR 1/12, SSIM 1/12
- `dt_matsynth_all` beats `matup` on PSNR 5/12, SSIM 5/12
- `dt_render_r1` beats `matup` on PSNR 5/12, SSIM 4/12
- `bsr_esrgan_zs` beats `matup` on PSNR 4/12, SSIM 2/12
- `kair_swinir_cls_zs` beats `matup` on PSNR 9/12, SSIM 11/12
- `kair_swinir_real_zs` beats `matup` on PSNR 0/12, SSIM 0/12
- `kair_swinir_realL_zs` beats `matup` on PSNR 0/12, SSIM 0/12

## render

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT |
|---|---|---|---|---|---|---|---|
| lr | 12 | 22.37 | 0 | 0.411 | 2.2 | 22.45 | 46% |
| bilinear | 12 | 27.30 | 0 | 0.612 | 3.4 | 27.51 | 34% |
| matup | 12 | 26.36 | 0 | 0.605 | 4.0 | 26.60 | 33% |
| dt_matsynth_all | 12 | 25.95 | 0 | 0.591 | 5.4 | 26.59 | 77% |
| dt_render_r1 | 12 | 26.49 | 0 | 0.599 | 4.8 | 26.83 | 74% |
| bsr_esrgan_zs | 12 | 21.04 | 0 | 0.322 | 3.3 | 21.08 | 210% |
| kair_swinir_cls_zs | 12 | 27.16 | 0 | 0.651 | 2.7 | 27.29 | 64% |
| kair_swinir_real_zs | 12 | 24.87 | 0 | 0.541 | 4.4 | 25.05 | 87% |
| kair_swinir_realL_zs | 12 | 25.27 | 0 | 0.563 | 3.9 | 25.42 | 84% |

- `kair_swinir_real_zs` beats `kair_swinir_realL_zs` on PSNR 3/12, SSIM 2/12
- `dt_matsynth_all` beats `matup` on PSNR 1/12, SSIM 3/12
- `dt_render_r1` beats `matup` on PSNR 3/12, SSIM 5/12
- `bsr_esrgan_zs` beats `matup` on PSNR 1/12, SSIM 0/12
- `kair_swinir_cls_zs` beats `matup` on PSNR 9/12, SSIM 11/12
- `kair_swinir_real_zs` beats `matup` on PSNR 1/12, SSIM 0/12
- `kair_swinir_realL_zs` beats `matup` on PSNR 2/12, SSIM 2/12
