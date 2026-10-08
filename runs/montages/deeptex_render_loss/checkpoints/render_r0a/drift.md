# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/matsynth/baseColor` (seed 7);
eval = 12 materials' `baseColor`. Checkpoints from `/workspace/runs/deeptex/render_r0a/checkpoints_all/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-1 | 5000 | +15.33 | -9.53 | +11.31 | +5.70 | -22.84 | 20.04 | +18.46 | -10.65 | +15.18 | +7.66 | -27.47 | 22.69 |
| ckpt-2 | 10000 | -19.10 | -8.08 | -5.59 | -10.92 | +4.26 | 19.19 | -19.98 | -8.46 | -3.47 | -10.64 | +3.27 | 21.21 |
| ckpt-3 | 15000 | -3.55 | -18.37 | -3.47 | -8.46 | -14.86 | 19.36 | -1.31 | -22.13 | -1.87 | -8.44 | -20.54 | 22.48 |
| ckpt-4 | 20000 | -7.88 | -7.51 | -18.95 | -11.45 | +5.91 | 19.23 | -8.13 | -9.79 | -19.94 | -12.62 | +4.24 | 21.76 |
| ckpt-5 | 25000 | -1.48 | -4.23 | -14.26 | -6.66 | +3.64 | 14.26 | -0.45 | -5.79 | -14.18 | -6.81 | +1.53 | 16.17 |
| ckpt-6 | 30000 | -7.91 | -2.13 | -3.58 | -4.54 | +3.62 | 8.19 | -9.43 | -4.30 | -4.19 | -5.97 | +2.50 | 9.51 |
| ckpt-7 | 35000 | +2.26 | -1.17 | +1.32 | +0.80 | -2.96 | 4.58 | +4.04 | -2.15 | +2.94 | +1.61 | -5.65 | 6.04 |
| ckpt-8 | 40000 | +2.65 | -1.81 | -5.66 | -1.60 | -0.30 | 6.40 | +4.08 | -2.16 | -3.43 | -0.50 | -2.48 | 8.08 |
