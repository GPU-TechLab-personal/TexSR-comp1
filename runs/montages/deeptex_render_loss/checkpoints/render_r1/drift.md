# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/matsynth/baseColor` (seed 7);
eval = 12 materials' `baseColor`. Checkpoints from `/workspace/runs/deeptex/render_r1/checkpoints_all/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-1 | 5000 | +0.08 | +4.82 | -13.49 | -2.86 | +11.52 | 13.59 | +1.24 | +5.34 | -12.75 | -2.06 | +11.10 | 13.40 |
| ckpt-2 | 10000 | -1.80 | -2.60 | -3.04 | -2.48 | -0.18 | 3.63 | -2.53 | -3.86 | -3.33 | -3.24 | -0.93 | 4.40 |
| ckpt-3 | 15000 | -1.38 | -2.34 | +3.75 | +0.01 | -3.53 | 4.40 | -1.61 | -2.70 | +3.44 | -0.29 | -3.61 | 4.03 |
| ckpt-4 | 20000 | +1.18 | +0.90 | +0.44 | +0.84 | +0.08 | 2.16 | +0.65 | +0.66 | +0.15 | +0.49 | +0.26 | 2.19 |
| ckpt-5 | 25000 | -2.62 | -1.50 | -5.67 | -3.26 | +2.65 | 5.68 | -2.77 | -2.08 | -6.01 | -3.62 | +2.31 | 6.37 |
| ckpt-6 | 30000 | -0.75 | -1.02 | +1.68 | -0.03 | -1.48 | 2.52 | -1.16 | -1.53 | +2.38 | -0.11 | -2.14 | 2.73 |
| ckpt-7 | 35000 | +2.92 | +2.52 | -3.54 | +0.63 | +2.83 | 4.44 | +2.83 | +2.02 | -3.95 | +0.30 | +2.58 | 4.24 |
| ckpt-8 | 40000 | -0.91 | +0.22 | +2.41 | +0.58 | -0.53 | 2.78 | -1.00 | -0.10 | +2.45 | +0.45 | -0.83 | 2.50 |
