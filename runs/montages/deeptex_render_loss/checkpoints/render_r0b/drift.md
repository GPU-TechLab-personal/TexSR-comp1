# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/matsynth/baseColor` (seed 7);
eval = 12 materials' `baseColor`. Checkpoints from `/workspace/runs/deeptex/render_r0b/checkpoints_all/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-1 | 5000 | +12.75 | +1.20 | +26.38 | +13.45 | -18.37 | 26.39 | +17.18 | +0.90 | +31.42 | +16.50 | -23.40 | 31.42 |
| ckpt-2 | 10000 | -4.48 | +2.11 | +2.61 | +0.08 | +3.04 | 5.82 | -2.86 | +2.83 | +5.14 | +1.70 | +1.69 | 8.00 |
| ckpt-3 | 15000 | -11.60 | -12.79 | -15.34 | -13.24 | +0.68 | 17.90 | -10.74 | -15.04 | -16.26 | -14.02 | -1.54 | 19.04 |
| ckpt-4 | 20000 | -11.83 | -10.12 | -7.94 | -9.97 | -0.23 | 13.90 | -12.93 | -11.34 | -7.27 | -10.51 | -1.24 | 14.68 |
| ckpt-5 | 25000 | -1.20 | -0.19 | +2.62 | +0.41 | -0.89 | 5.55 | -0.78 | -1.21 | +4.79 | +0.94 | -3.21 | 7.41 |
| ckpt-6 | 30000 | -1.49 | +0.20 | +4.39 | +1.03 | -1.25 | 5.28 | +0.51 | -0.16 | +5.74 | +2.03 | -3.28 | 6.25 |
| ckpt-7 | 35000 | -0.69 | -1.46 | -3.45 | -1.87 | +0.61 | 4.23 | +1.71 | -1.00 | -1.27 | -0.19 | -1.21 | 4.73 |
| ckpt-8 | 40000 | +1.15 | -2.76 | -8.45 | -3.35 | +0.89 | 8.77 | +2.62 | -2.83 | -9.79 | -3.33 | +0.76 | 10.34 |
