# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/matsynth/baseColor` (seed 7);
eval = 12 materials' `baseColor`. Checkpoints from `/workspace/runs/deeptex/matsynth_chratio/checkpoints_all/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-1 | 5000 | +13.53 | +1.99 | +7.56 | +7.69 | -8.56 | 16.49 | +16.24 | +1.54 | +6.75 | +8.18 | -9.95 | 16.52 |
| ckpt-2 | 10000 | +0.04 | +4.39 | +23.92 | +9.45 | -7.59 | 23.92 | +1.47 | +5.21 | +29.42 | +12.04 | -10.23 | 29.42 |
| ckpt-3 | 15000 | -7.83 | +3.27 | -8.02 | -4.19 | +11.20 | 9.72 | -7.04 | +4.07 | -8.24 | -3.73 | +11.71 | 10.50 |
| ckpt-4 | 20000 | -3.64 | -4.99 | -15.43 | -8.02 | +4.54 | 15.71 | -3.37 | -6.41 | -16.26 | -8.68 | +3.41 | 16.61 |
| ckpt-5 | 25000 | -13.22 | +1.97 | +15.70 | +1.48 | +0.73 | 16.93 | -13.22 | +2.03 | +19.54 | +2.78 | -1.14 | 19.86 |
| ckpt-6 | 30000 | -2.83 | -3.36 | -2.86 | -3.02 | -0.51 | 5.73 | -1.43 | -3.89 | -2.72 | -2.68 | -1.82 | 6.61 |
| ckpt-7 | 35000 | -5.88 | -4.80 | +6.88 | -1.26 | -5.30 | 8.58 | -4.82 | -5.19 | +8.13 | -0.63 | -6.84 | 9.14 |
| ckpt-8 | 40000 | -3.89 | +1.31 | +3.14 | +0.19 | +1.69 | 5.33 | -4.44 | +0.17 | +3.06 | -0.40 | +0.86 | 6.38 |
