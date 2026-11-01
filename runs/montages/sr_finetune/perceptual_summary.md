# Perceptual metrics on renders of the predicted SVBRDF: 12 MatSynth evaluation materials

LPIPS and FLIP between the MatUp-renderer render of each column's predicted maps and
the render of the GT maps - the renders svbrdf_render_eval.py scores with PSNR / l2,1
(render_summary.md). Mean errors reward blur; these two charge lost texture and edges,
so they separate "R1 removed noise" (R1 <= R0a here) from "R1 removed detail GT has"
(R1 > R0a here). bilinear is the blur reference. **Lower is better everywhere.**

- **render LPIPS**: LPIPS-alex v0.1 per light, mean over the `heldout` set (17 lights: EVAL_LIGHT 67 deg, #1 10 deg, #2 17 deg, #3 9 deg, #4 13 deg, #5 16 deg, #6 20 deg, #7 24 deg, #8 28 deg, #9 32 deg, #10 36 deg, #11 41 deg, #12 46 deg, #13 51 deg, #14 58 deg, #15 65 deg, #16 76 deg).
- **render FLIP**: mean FLIP (LDR, default parameters) per light, mean over the `sheet` set (4 lights: EVAL_LIGHT 67 deg, #4 13 deg, #10 36 deg, #16 76 deg).
- **EVAL_LIGHT LPIPS / FLIP**: EVAL_LIGHT alone - the light render_summary.md's
  `|grad| %GT` is measured under.
- **baseColor LPIPS**: LPIPS-alex of the stored sRGB baseColor map (no lighting).
- Renders: display space (Reinhard + gamma, clipped to [0,1]), float, cropped by
  MatUp's border exactly as in render_summary.md.
- **off** = offset-removed maps (P1's variant: each map's per-channel mean error vs
  GT removed in the stored encoding); **raw** = the maps as delivered.
- GT-invalid texels (GT normal fails training's rule) take the GT render in every
  prediction before LPIPS / FLIP, the same for every column (masked fraction per
  material at the end). baseColor is not masked.
- `n`: materials every column was scored on; every mean is over that set
  (9 non-metal = GT metallic all zero, 3 metal).

Columns: `bilinear` = bilinear, `matup` = MatUp, `dt_matsynth_all_40k` = DeepTextures (matsynth_all) 40k (/workspace/runs/deeptex/eval/matsynth_all_40k), `dt_render_r1_40k` = DeepTextures + render loss (R1) 40k (/workspace/runs/deeptex/eval/render_r1_40k), `bsr_esrgan_zs` = ESRGAN official (zero-shot) (/workspace/runs/sr_ft/eval/esrgan_zs), `kair_swinir_cls_zs` = SwinIR-M classical (zero-shot) (/workspace/runs/sr_ft/eval/swinir_cls_zs), `kair_swinir_real_zs` = SwinIR-M real (zero-shot) (/workspace/runs/sr_ft/eval/swinir_real_zs), `kair_swinir_realL_zs` = SwinIR-L real (zero-shot) (/workspace/runs/sr_ft/eval/swinir_realL_zs).

## off - offset-removed maps

| column | n | render LPIPS | render FLIP | EVAL_LIGHT LPIPS | EVAL_LIGHT FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|
| bilinear | 12 | 0.5581 | 0.1141 | 0.5577 | 0.1112 | 0.5040 |
| MatUp | 12 | 0.5458 | 0.1373 | 0.5413 | 0.1354 | 0.4887 |
| DeepTextures (matsynth_all) 40k | 12 | 0.2969 | 0.1236 | 0.2718 | 0.1187 | 0.2241 |
| DeepTextures + render loss (R1) 40k | 12 | 0.2933 | 0.1037 | 0.2730 | 0.1010 | 0.2281 |
| ESRGAN official (zero-shot) | 12 | 0.3994 | 0.1303 | 0.4089 | 0.1280 | 0.4036 |
| SwinIR-M classical (zero-shot) | 12 | 0.4128 | 0.0987 | 0.3977 | 0.0940 | 0.3611 |
| SwinIR-M real (zero-shot) | 12 | 0.4444 | 0.1824 | 0.4135 | 0.1709 | 0.4037 |
| SwinIR-L real (zero-shot) | 12 | 0.4357 | 0.1832 | 0.3928 | 0.1737 | 0.3707 |

Non-metal / metal split:

| column | non-metal n | render LPIPS | render FLIP | baseColor LPIPS | metal n | render LPIPS | render FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 0.5393 | 0.1195 | 0.4665 | 3 | 0.6147 | 0.0979 | 0.6162 |
| MatUp | 9 | 0.5346 | 0.1412 | 0.4433 | 3 | 0.5794 | 0.1255 | 0.6248 |
| DeepTextures (matsynth_all) 40k | 9 | 0.3011 | 0.1275 | 0.2064 | 3 | 0.2842 | 0.1120 | 0.2773 |
| DeepTextures + render loss (R1) 40k | 9 | 0.2925 | 0.1069 | 0.2076 | 3 | 0.2957 | 0.0942 | 0.2897 |
| ESRGAN official (zero-shot) | 9 | 0.3375 | 0.1314 | 0.3756 | 3 | 0.5851 | 0.1267 | 0.4878 |
| SwinIR-M classical (zero-shot) | 9 | 0.3962 | 0.1024 | 0.3192 | 3 | 0.4629 | 0.0878 | 0.4867 |
| SwinIR-M real (zero-shot) | 9 | 0.4718 | 0.1967 | 0.4296 | 3 | 0.3621 | 0.1396 | 0.3262 |
| SwinIR-L real (zero-shot) | 9 | 0.4685 | 0.2000 | 0.3709 | 3 | 0.3375 | 0.1326 | 0.3700 |

## raw - raw maps

| column | n | render LPIPS | render FLIP | EVAL_LIGHT LPIPS | EVAL_LIGHT FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|
| bilinear | 12 | 0.5612 | 0.1154 | 0.5616 | 0.1135 | 0.5040 |
| MatUp | 12 | 0.5485 | 0.1455 | 0.5426 | 0.1453 | 0.4940 |
| DeepTextures (matsynth_all) 40k | 12 | 0.3425 | 0.2724 | 0.3093 | 0.2443 | 0.2472 |
| DeepTextures + render loss (R1) 40k | 12 | 0.2948 | 0.1184 | 0.2744 | 0.1186 | 0.2288 |
| ESRGAN official (zero-shot) | 12 | 0.4009 | 0.1459 | 0.4118 | 0.1442 | 0.4059 |
| SwinIR-M classical (zero-shot) | 12 | 0.4139 | 0.0984 | 0.3987 | 0.0938 | 0.3609 |
| SwinIR-M real (zero-shot) | 12 | 0.4493 | 0.2191 | 0.4201 | 0.2015 | 0.4052 |
| SwinIR-L real (zero-shot) | 12 | 0.4380 | 0.2062 | 0.3979 | 0.1939 | 0.3713 |

Non-metal / metal split:

| column | non-metal n | render LPIPS | render FLIP | baseColor LPIPS | metal n | render LPIPS | render FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 0.5433 | 0.1212 | 0.4666 | 3 | 0.6148 | 0.0979 | 0.6163 |
| MatUp | 9 | 0.5388 | 0.1532 | 0.4515 | 3 | 0.5776 | 0.1226 | 0.6213 |
| DeepTextures (matsynth_all) 40k | 9 | 0.3399 | 0.2654 | 0.2260 | 3 | 0.3505 | 0.2936 | 0.3110 |
| DeepTextures + render loss (R1) 40k | 9 | 0.2939 | 0.1205 | 0.2086 | 3 | 0.2975 | 0.1122 | 0.2895 |
| ESRGAN official (zero-shot) | 9 | 0.3392 | 0.1454 | 0.3788 | 3 | 0.5860 | 0.1475 | 0.4873 |
| SwinIR-M classical (zero-shot) | 9 | 0.3973 | 0.1023 | 0.3189 | 3 | 0.4640 | 0.0869 | 0.4868 |
| SwinIR-M real (zero-shot) | 9 | 0.4763 | 0.2235 | 0.4306 | 3 | 0.3682 | 0.2058 | 0.3292 |
| SwinIR-L real (zero-shot) | 9 | 0.4717 | 0.2204 | 0.3721 | 3 | 0.3371 | 0.1636 | 0.3692 |

## Head-to-head

Cells `k/n`: materials where the FIRST column is strictly LOWER (better). Mean diff
= first - second over the same materials; negative means the first is better.
Pairs as in render_summary.md (svbrdf_render_eval.head_to_head_pairs).

| first | second | n | render LPIPS off | render LPIPS raw | render FLIP off | render FLIP raw | baseColor LPIPS off | baseColor LPIPS raw | non-metal: render LPIPS off | non-metal: render FLIP off | mean diff render LPIPS off | mean diff render FLIP off | mean diff baseColor LPIPS off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SwinIR-M real (zero-shot) | SwinIR-L real (zero-shot) | 12 | 3/12 | 3/12 | 4/12 | 4/12 | 4/12 | 4/12 | 3/9 | 4/9 | +0.0086 | -0.0007 | +0.0331 |
| DeepTextures + render loss (R1) 40k | DeepTextures (matsynth_all) 40k | 12 | 7/12 | 9/12 | 12/12 | 12/12 | 5/12 | 8/12 | 5/9 | 9/9 | -0.0036 | -0.0199 | +0.0040 |
| DeepTextures (matsynth_all) 40k | bilinear | 12 | 12/12 | 12/12 | 3/12 | 0/12 | 12/12 | 11/12 | 9/9 | 3/9 | -0.2612 | +0.0095 | -0.2799 |
| DeepTextures (matsynth_all) 40k | MatUp | 12 | 12/12 | 12/12 | 10/12 | 0/12 | 12/12 | 11/12 | 9/9 | 7/9 | -0.2489 | -0.0137 | -0.2646 |
| DeepTextures + render loss (R1) 40k | bilinear | 12 | 12/12 | 12/12 | 10/12 | 3/12 | 12/12 | 12/12 | 9/9 | 8/9 | -0.2648 | -0.0104 | -0.2758 |
| DeepTextures + render loss (R1) 40k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 11/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.2525 | -0.0336 | -0.2606 |
| ESRGAN official (zero-shot) | bilinear | 12 | 11/12 | 11/12 | 1/12 | 1/12 | 9/12 | 9/12 | 9/9 | 1/9 | -0.1587 | +0.0162 | -0.1003 |
| ESRGAN official (zero-shot) | MatUp | 12 | 9/12 | 9/12 | 8/12 | 6/12 | 9/12 | 9/12 | 8/9 | 7/9 | -0.1464 | -0.0070 | -0.0851 |
| SwinIR-M classical (zero-shot) | bilinear | 12 | 12/12 | 12/12 | 11/12 | 11/12 | 12/12 | 12/12 | 9/9 | 8/9 | -0.1453 | -0.0154 | -0.1429 |
| SwinIR-M classical (zero-shot) | MatUp | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.1330 | -0.0386 | -0.1276 |
| SwinIR-M real (zero-shot) | bilinear | 12 | 11/12 | 11/12 | 0/12 | 0/12 | 9/12 | 9/12 | 8/9 | 0/9 | -0.1138 | +0.0683 | -0.1002 |
| SwinIR-M real (zero-shot) | MatUp | 12 | 11/12 | 11/12 | 0/12 | 1/12 | 9/12 | 10/12 | 8/9 | 0/9 | -0.1014 | +0.0451 | -0.0850 |
| SwinIR-L real (zero-shot) | bilinear | 12 | 11/12 | 11/12 | 0/12 | 0/12 | 10/12 | 10/12 | 8/9 | 0/9 | -0.1224 | +0.0691 | -0.1333 |
| SwinIR-L real (zero-shot) | MatUp | 12 | 11/12 | 11/12 | 1/12 | 0/12 | 9/12 | 10/12 | 8/9 | 0/9 | -0.1101 | +0.0459 | -0.1180 |

## Per checkpoint step

Mean over the `n` materials at each step, the mean over steps, and the
within-run range (max - min over steps).

### render LPIPS (off) (n=12)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 0.2969 | 0.2969 | - |
| DeepTextures + render loss (R1) | 0.2933 | 0.2933 | - |

### render LPIPS (raw) (n=12)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 0.3425 | 0.3425 | - |
| DeepTextures + render loss (R1) | 0.2948 | 0.2948 | - |

### render FLIP (off) (n=12)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 0.1236 | 0.1236 | - |
| DeepTextures + render loss (R1) | 0.1037 | 0.1037 | - |

### render FLIP (raw) (n=12)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 0.2724 | 0.2724 | - |
| DeepTextures + render loss (R1) | 0.1184 | 0.1184 | - |

### baseColor LPIPS (off) (n=12)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 0.2241 | 0.2241 | - |
| DeepTextures + render loss (R1) | 0.2281 | 0.2281 | - |

### baseColor LPIPS (raw) (n=12)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 0.2472 | 0.2472 | - |
| DeepTextures + render loss (R1) | 0.2288 | 0.2288 | - |

### render LPIPS (off), non-metal (n=9)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 0.3011 | 0.3011 | - |
| DeepTextures + render loss (R1) | 0.2925 | 0.2925 | - |

### render FLIP (off), non-metal (n=9)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 0.1275 | 0.1275 | - |
| DeepTextures + render loss (R1) | 0.1069 | 0.1069 | - |

## Checks

Metric self-check on 0559_brick_uneven_stones: LPIPS(gt, gt) max 0.00e+00, FLIP(gt, gt) max 0.00e+00, baseColor LPIPS(gt, gt) 0.00e+00 (bound 1e-06); GT vs 1 - GT under EVAL_LIGHT: LPIPS 0.341, FLIP 0.566 (must exceed 0.05); LPIPS input range: |x * 2 - 1 here - lpips normalize=True| on that pair 0.00e+00 (bound 1e-06).

| material | masked fraction | self-check GT (LSB) | bilinear | MatUp | missing |
|---|---|---|---|---|---|
| 0559_brick_uneven_stones | 0.0000 | 1.00 | 1.00 | 9.25 | - |
| 0844_syntetic_foam | 0.0004 | 1.00 | 1.00 | 16.64 | - |
| 1587_cobblestones | 0.0000 | 1.00 | 1.00 | 7.41 | - |
| 2119_corten_steel | 0.0000 | 1.00 | 1.00 | 1.63 | - |
| acg_concrete_026 | 0.0000 | 1.00 | 1.00 | 2.96 | - |
| acg_metal_026 | 0.0000 | 1.00 | 1.00 | 5.88 | - |
| acg_plastic_009 | 0.0000 | 1.00 | 1.00 | 8.54 | - |
| acg_roofing_tiles_006 | 0.0000 | 1.00 | 1.00 | 21.39 | - |
| acg_wood_floor_060 | 0.0000 | 1.00 | 1.00 | 7.02 | - |
| tc_metal_029 | 0.0000 | 1.00 | 1.00 | 6.76 | - |
| th_brick_wall_005 | 0.0000 | 1.00 | 1.00 | 19.01 | - |
| th_brown_mud_leaves_001 | 0.0000 | 1.00 | 1.00 | 2.67 | - |
