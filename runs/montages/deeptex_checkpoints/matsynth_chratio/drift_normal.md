# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/matsynth/normal` (seed 7);
eval = 12 materials' `normal`. Checkpoints from `/workspace/runs/deeptex/matsynth_chratio/checkpoints_all/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-1 | 5000 | +16.58 | -6.55 | +1.91 | +3.98 | -15.79 | 16.58 | +18.21 | -5.41 | +1.34 | +4.71 | -15.18 | 18.21 |
| ckpt-2 | 10000 | -2.16 | +6.04 | +6.04 | +3.31 | +4.10 | 8.15 | -1.46 | +6.23 | +7.53 | +4.10 | +3.20 | 9.11 |
| ckpt-3 | 15000 | -12.88 | -6.91 | -14.44 | -11.41 | +6.75 | 14.79 | -12.43 | -7.09 | -15.21 | -11.58 | +6.72 | 15.21 |
| ckpt-4 | 20000 | -11.53 | -4.54 | -25.97 | -14.01 | +14.21 | 25.97 | -11.47 | -5.01 | -27.69 | -14.73 | +14.57 | 27.69 |
| ckpt-5 | 25000 | -22.88 | -0.82 | -2.18 | -8.63 | +11.71 | 22.88 | -22.30 | -0.70 | -1.06 | -8.02 | +10.99 | 22.30 |
| ckpt-6 | 30000 | -4.43 | -10.28 | -3.36 | -6.03 | -6.39 | 10.28 | -3.92 | -10.26 | -3.61 | -5.93 | -6.49 | 10.26 |
| ckpt-7 | 35000 | -7.27 | -9.70 | +0.97 | -5.34 | -6.55 | 9.83 | -6.96 | -9.82 | +1.94 | -4.95 | -7.31 | 10.03 |
| ckpt-8 | 40000 | -9.08 | +5.19 | +2.28 | -0.54 | +8.60 | 9.18 | -9.11 | +4.44 | +2.63 | -0.68 | +7.68 | 9.11 |
