# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/matsynth/baseColor` (seed 7);
eval = 12 materials' `baseColor`. Checkpoints from `/workspace/runs/deeptex/matsynth_basecolor/checkpoints_all/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-1 | 5000 | +15.28 | +0.49 | -0.61 | +5.05 | -6.85 | 15.48 | +17.67 | -0.55 | -2.24 | +4.96 | -8.26 | 17.67 |
| ckpt-2 | 10000 | +5.76 | -8.97 | -0.02 | -1.07 | -11.84 | 9.85 | +8.15 | -10.02 | +0.86 | -0.34 | -14.53 | 10.85 |
| ckpt-3 | 15000 | +1.24 | -0.87 | +0.05 | +0.14 | -1.52 | 4.36 | +1.37 | -2.13 | +0.45 | -0.11 | -3.04 | 5.18 |
| ckpt-4 | 20000 | -3.49 | +5.98 | +0.31 | +0.93 | +7.58 | 7.01 | -4.53 | +5.71 | +0.21 | +0.47 | +7.87 | 7.30 |
| ckpt-5 | 25000 | -4.66 | -1.16 | +9.50 | +1.23 | -3.58 | 9.89 | -5.36 | -1.69 | +9.69 | +0.88 | -3.85 | 10.01 |
| ckpt-6 | 30000 | -5.55 | -5.26 | -6.15 | -5.65 | +0.58 | 7.58 | -6.37 | -7.37 | -8.36 | -7.36 | -0.00 | 9.44 |
| ckpt-7 | 35000 | -1.65 | -4.51 | -1.09 | -2.41 | -3.14 | 5.18 | -1.67 | -5.07 | -1.94 | -2.89 | -3.26 | 6.44 |
| ckpt-8 | 40000 | -3.20 | +0.49 | +4.26 | +0.52 | -0.04 | 5.29 | -2.62 | +2.18 | +5.67 | +1.74 | +0.65 | 6.57 |
