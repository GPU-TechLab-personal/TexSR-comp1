# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/matsynth/normal` (seed 7);
eval = 12 materials' `normal`. Checkpoints from `/workspace/runs/deeptex/matsynth_all/checkpoints/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-6 | 30000 | -5.66 | -8.21 | +0.53 | -4.45 | -5.64 | 8.37 | -5.29 | -7.99 | +0.59 | -4.23 | -5.64 | 8.18 |
| ckpt-7 | 35000 | +15.49 | +17.42 | +1.98 | +11.63 | +8.69 | 17.48 | +16.48 | +17.69 | +2.17 | +12.11 | +8.36 | 17.96 |
| ckpt-8 | 40000 | -20.32 | -7.25 | -5.95 | -11.17 | +5.89 | 20.32 | -20.37 | -7.69 | -6.92 | -11.66 | +5.96 | 20.37 |
