# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/matsynth/normal` (seed 7);
eval = 12 materials' `normal`. Checkpoints from `/workspace/runs/deeptex/render_r0b/checkpoints_all/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-1 | 5000 | -3.77 | -6.74 | +6.21 | -1.43 | -7.96 | 9.55 | -1.15 | -5.68 | +8.39 | +0.52 | -9.30 | 11.35 |
| ckpt-2 | 10000 | -14.16 | -6.52 | -13.33 | -11.34 | +7.22 | 14.22 | -13.58 | -6.07 | -13.09 | -10.91 | +7.27 | 13.78 |
| ckpt-3 | 15000 | -19.67 | -16.38 | -41.95 | -26.00 | +14.43 | 41.95 | -18.95 | -17.39 | -42.41 | -26.25 | +13.29 | 42.41 |
| ckpt-4 | 20000 | -1.78 | -26.41 | -11.47 | -13.22 | -19.79 | 26.41 | -2.02 | -27.01 | -11.76 | -13.60 | -20.12 | 27.01 |
| ckpt-5 | 25000 | -0.61 | -7.58 | -21.96 | -10.05 | +3.70 | 21.96 | -0.26 | -7.89 | -22.04 | -10.06 | +3.26 | 22.04 |
| ckpt-6 | 30000 | +0.30 | +3.94 | +3.80 | +2.68 | +1.89 | 5.03 | +0.84 | +3.70 | +4.80 | +3.11 | +0.88 | 5.25 |
| ckpt-7 | 35000 | -11.54 | -5.09 | -16.42 | -11.02 | +8.89 | 16.42 | -10.58 | -4.69 | -16.36 | -10.54 | +8.78 | 16.36 |
| ckpt-8 | 40000 | +8.84 | -4.76 | -0.19 | +1.30 | -9.09 | 8.94 | +9.07 | -5.23 | -1.28 | +0.85 | -9.12 | 9.59 |
