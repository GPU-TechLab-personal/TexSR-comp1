# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/matsynth/normal` (seed 7);
eval = 12 materials' `normal`. Checkpoints from `/workspace/runs/deeptex/render_r1/checkpoints_all/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-1 | 5000 | -16.49 | -12.52 | -9.35 | -12.78 | +0.40 | 16.65 | -17.08 | -12.98 | -10.84 | -13.63 | +0.98 | 17.47 |
| ckpt-2 | 10000 | -2.11 | -2.01 | +3.34 | -0.26 | -2.62 | 3.94 | -2.55 | -2.53 | +3.15 | -0.64 | -2.82 | 3.96 |
| ckpt-3 | 15000 | +1.20 | +0.27 | -3.73 | -0.75 | +1.53 | 3.75 | +1.03 | +0.22 | -4.03 | -0.92 | +1.72 | 4.03 |
| ckpt-4 | 20000 | +4.17 | +3.47 | +4.28 | +3.97 | -0.75 | 5.65 | +3.96 | +3.76 | +4.85 | +4.19 | -0.65 | 5.66 |
| ckpt-5 | 25000 | -1.91 | +1.31 | -8.34 | -2.98 | +6.43 | 8.34 | -2.04 | +1.21 | -9.28 | -3.37 | +6.87 | 9.28 |
| ckpt-6 | 30000 | +0.58 | -1.83 | +1.28 | +0.01 | -2.76 | 2.36 | +0.18 | -1.83 | +2.05 | +0.13 | -2.94 | 2.78 |
| ckpt-7 | 35000 | +4.04 | +3.82 | +1.14 | +3.00 | +1.23 | 4.27 | +3.57 | +3.82 | +1.21 | +2.86 | +1.43 | 4.19 |
| ckpt-8 | 40000 | +0.95 | +0.18 | +1.97 | +1.03 | -1.28 | 2.13 | +1.00 | +0.40 | +2.01 | +1.14 | -1.11 | 2.13 |
