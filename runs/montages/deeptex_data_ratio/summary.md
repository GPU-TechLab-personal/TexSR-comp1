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

Arms: `dt_textures_db` = /workspace/runs/deeptex/eval/paper_allmaps, `dt_matsynth_all` = /workspace/runs/deeptex/eval/matsynth_all, `dt_matsynth_chratio` = /workspace/runs/deeptex/eval/matsynth_chratio, `dt_matsynth_basecolor` = /workspace/runs/deeptex/eval/matsynth_basecolor.

## baseColor

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT |
|---|---|---|---|---|---|---|---|
| lr | 12 | 31.32 | 0 | 0.673 | 0.0 | 31.32 | 51% |
| bilinear | 12 | 31.93 | 0 | 0.674 | 0.1 | 31.94 | 40% |
| matup | 12 | 27.22 | 0 | 0.600 | 5.1 | 30.20 | 43% |
| dt_textures_db | 12 | 25.58 | 0 | 0.580 | 5.7 | 28.79 | 112% |
| dt_matsynth_all | 12 | 26.88 | 0 | 0.628 | 8.7 | 30.47 | 95% |
| dt_matsynth_chratio | 12 | 26.85 | 0 | 0.604 | 6.4 | 30.22 | 98% |
| dt_matsynth_basecolor | 12 | 26.92 | 0 | 0.566 | 6.6 | 29.62 | 111% |

- `dt_matsynth_basecolor` beats `dt_matsynth_chratio` on PSNR 2/12, SSIM 3/12
- `dt_textures_db` beats `matup` on PSNR 2/12, SSIM 3/12
- `dt_matsynth_all` beats `matup` on PSNR 1/12, SSIM 5/12
- `dt_matsynth_chratio` beats `matup` on PSNR 2/12, SSIM 4/12
- `dt_matsynth_basecolor` beats `matup` on PSNR 2/12, SSIM 2/12

## normal

| column | n | PSNR | =GT | SSIM | ang. err ° | offset | PSNR -off | \|grad\| %GT |
|---|---|---|---|---|---|---|---|---|
| lr | 12 | 33.22 | 0 | 0.734 | 6.51 | 0.0 | 33.22 | 39% |
| bilinear | 12 | 33.41 | 0 | 0.733 | 6.28 | 0.1 | 33.42 | 33% |
| matup | 12 | 32.29 | 0 | 0.721 | 6.54 | 5.9 | 33.03 | 28% |
| dt_textures_db | 12 | 19.00 | 0 | 0.538 | 20.77 | 31.7 | 26.93 | 240% |
| dt_matsynth_all | 12 | 23.65 | 0 | 0.675 | 13.26 | 20.5 | 30.86 | 94% |
| dt_matsynth_chratio | 12 | 26.58 | 0 | 0.681 | 9.55 | 9.2 | 30.60 | 113% |
| dt_matsynth_basecolor | 12 | 27.56 | 0 | 0.674 | 8.12 | 5.2 | 31.33 | 111% |

- `dt_matsynth_basecolor` beats `dt_matsynth_chratio` on PSNR 9/12, SSIM 5/12, angle 11/12
- `dt_textures_db` beats `matup` on PSNR 0/12, SSIM 0/12, angle 0/12
- `dt_matsynth_all` beats `matup` on PSNR 0/12, SSIM 0/12, angle 0/12
- `dt_matsynth_chratio` beats `matup` on PSNR 0/12, SSIM 2/12, angle 0/12
- `dt_matsynth_basecolor` beats `matup` on PSNR 1/12, SSIM 0/12, angle 0/12

## roughness

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT | ch. spread /255 |
|---|---|---|---|---|---|---|---|---|
| lr | 11 | 32.18 | 1 | 0.723 | 0.0 | 32.18 | 45% | - |
| bilinear | 11 | 32.52 | 1 | 0.720 | 0.1 | 32.52 | 34% | - |
| matup | 11 | 23.61 | 0 | 0.574 | 11.5 | 26.41 | 128% | - |
| dt_textures_db | 11 | 27.68 | 0 | 0.671 | 4.0 | 30.75 | 87% | 6.78 |
| dt_matsynth_all | 11 | 31.31 | 0 | 0.707 | 2.8 | 31.59 | 83% | 8.60 |
| dt_matsynth_chratio | 11 | 29.25 | 0 | 0.673 | 3.1 | 31.18 | 87% | 10.79 |
| dt_matsynth_basecolor | 11 | 28.71 | 0 | 0.592 | 3.2 | 29.33 | 121% | 7.76 |

- `dt_matsynth_basecolor` beats `dt_matsynth_chratio` on PSNR 4/12, SSIM 1/12
- `dt_textures_db` beats `matup` on PSNR 8/12, SSIM 7/12
- `dt_matsynth_all` beats `matup` on PSNR 12/12, SSIM 9/12
- `dt_matsynth_chratio` beats `matup` on PSNR 9/12, SSIM 7/12
- `dt_matsynth_basecolor` beats `matup` on PSNR 6/12, SSIM 6/12

## metallic

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT | ch. spread /255 |
|---|---|---|---|---|---|---|---|---|
| lr | 3 | 19.37 | 9 | 0.583 | 0.0 | 19.37 | 49% | - |
| bilinear | 3 | 19.70 | 9 | 0.570 | 0.1 | 19.70 | 36% | - |
| matup | 3 | 19.27 | 3 | 0.566 | 4.0 | 19.40 | 36% | - |
| dt_textures_db | 3 | 18.71 | 0 | 0.539 | 1.4 | 18.73 | 84% | 10.26 |
| dt_matsynth_all | 3 | 19.37 | 9 | 0.605 | 5.2 | 19.57 | 68% | 10.84 |
| dt_matsynth_chratio | 3 | 19.61 | 9 | 0.587 | 1.0 | 19.62 | 69% | 15.06 |
| dt_matsynth_basecolor | 3 | 19.07 | 0 | 0.540 | 1.6 | 19.10 | 77% | 10.07 |

- `dt_matsynth_basecolor` beats `dt_matsynth_chratio` on PSNR 0/12, SSIM 0/12
- `dt_textures_db` beats `matup` on PSNR 3/12, SSIM 3/12
- `dt_matsynth_all` beats `matup` on PSNR 7/12, SSIM 9/12
- `dt_matsynth_chratio` beats `matup` on PSNR 8/12, SSIM 8/12
- `dt_matsynth_basecolor` beats `matup` on PSNR 3/12, SSIM 3/12

## height

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT | ch. spread /255 |
|---|---|---|---|---|---|---|---|---|
| lr | 12 | 31.97 | 0 | 0.787 | 0.0 | 31.97 | 69% | - |
| bilinear | 12 | 34.66 | 0 | 0.832 | 0.0 | 34.66 | 56% | - |
| matup | 12 | 31.13 | 0 | 0.825 | 0.0 | 31.13 | 79% | - |
| dt_textures_db | 12 | 29.52 | 0 | 0.798 | 4.0 | 32.56 | 96% | 6.14 |
| dt_matsynth_all | 12 | 34.08 | 0 | 0.829 | 1.3 | 34.23 | 90% | 7.42 |
| dt_matsynth_chratio | 12 | 32.11 | 0 | 0.809 | 2.3 | 33.50 | 95% | 10.86 |
| dt_matsynth_basecolor | 12 | 31.77 | 0 | 0.754 | 2.3 | 32.09 | 111% | 7.16 |

- `dt_matsynth_basecolor` beats `dt_matsynth_chratio` on PSNR 4/12, SSIM 0/12
- `dt_textures_db` beats `matup` on PSNR 4/12, SSIM 2/12
- `dt_matsynth_all` beats `matup` on PSNR 5/12, SSIM 5/12
- `dt_matsynth_chratio` beats `matup` on PSNR 5/12, SSIM 3/12
- `dt_matsynth_basecolor` beats `matup` on PSNR 5/12, SSIM 2/12

## render

| column | n | PSNR | =GT | SSIM | offset | PSNR -off | \|grad\| %GT |
|---|---|---|---|---|---|---|---|
| lr | 12 | 22.37 | 0 | 0.411 | 2.2 | 22.45 | 46% |
| bilinear | 12 | 27.30 | 0 | 0.612 | 3.4 | 27.51 | 34% |
| matup | 12 | 26.36 | 0 | 0.605 | 4.0 | 26.60 | 33% |
| dt_textures_db | 12 | 25.03 | 0 | 0.524 | 4.7 | 25.43 | 99% |
| dt_matsynth_all | 12 | 25.96 | 0 | 0.591 | 5.4 | 26.59 | 77% |
| dt_matsynth_chratio | 12 | 25.50 | 0 | 0.555 | 4.9 | 26.07 | 85% |
| dt_matsynth_basecolor | 12 | 24.68 | 0 | 0.517 | 8.4 | 25.34 | 98% |

- `dt_matsynth_basecolor` beats `dt_matsynth_chratio` on PSNR 2/12, SSIM 2/12
- `dt_textures_db` beats `matup` on PSNR 1/12, SSIM 0/12
- `dt_matsynth_all` beats `matup` on PSNR 1/12, SSIM 3/12
- `dt_matsynth_chratio` beats `matup` on PSNR 1/12, SSIM 0/12
- `dt_matsynth_basecolor` beats `matup` on PSNR 1/12, SSIM 1/12
