# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/matsynth/baseColor` (seed 7);
eval = 12 materials' `baseColor`. Checkpoints from `/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-6 | 30000 | -8.01 | -3.06 | -4.81 | -5.29 | +3.35 | 8.84 | -8.20 | -3.68 | -5.41 | -5.76 | +3.13 | 10.88 |
| ckpt-7 | 35000 | +2.92 | +5.51 | -1.61 | +2.27 | +4.86 | 6.10 | +2.39 | +4.41 | -2.14 | +1.56 | +4.28 | 5.84 |
| ckpt-8 | 40000 | -7.17 | +1.00 | -3.12 | -3.10 | +6.14 | 7.77 | -6.89 | +0.84 | -3.48 | -3.18 | +6.02 | 8.64 |
