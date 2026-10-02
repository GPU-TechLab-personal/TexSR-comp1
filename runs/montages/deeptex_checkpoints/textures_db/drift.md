# Colour drift per checkpoint

mean(SR) - mean(HR) per channel, 8-bit levels; cast = G - (R+B)/2; max|off| = mean
over images of the largest channel offset. train = 60 seeded pairs from `/workspace/runs/deeptex/datasets/textures_db` (seed 7);
eval = 12 materials' `baseColor`. Checkpoints from `/workspace/runs/deeptex/paper/checkpoints/esrgan_40k`.

| checkpoint | step | train R | G | B | brightness | cast | max\|off\| | eval R | G | B | brightness | cast | max\|off\| |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ckpt-6 | 30000 | +3.20 | +1.53 | -1.97 | +0.92 | +0.91 | 5.12 | +4.77 | +1.20 | +0.07 | +2.01 | -1.22 | 7.14 |
| ckpt-7 | 35000 | +1.18 | +5.82 | +9.22 | +5.40 | +0.62 | 9.69 | -1.08 | +3.96 | +10.88 | +4.59 | -0.94 | 10.97 |
| ckpt-8 | 40000 | -0.09 | +1.44 | +0.39 | +0.58 | +1.29 | 4.24 | -2.70 | -1.28 | +0.31 | -1.23 | -0.08 | 5.54 |
