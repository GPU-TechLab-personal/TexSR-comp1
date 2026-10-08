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

Columns: `bilinear` = bilinear, `matup` = MatUp, `dt_matsynth_all` = DeepTextures (matsynth_all) (/workspace/runs/deeptex/eval/matsynth_all), `dt_render_r0a_30k` = DeepTextures grouped (R0a) 30k (/workspace/runs/deeptex/eval/render_r0a_30k), `dt_render_r0a_35k` = DeepTextures grouped (R0a) 35k (/workspace/runs/deeptex/eval/render_r0a_35k), `dt_render_r0a_40k` = DeepTextures grouped (R0a) 40k (/workspace/runs/deeptex/eval/render_r0a_40k), `dt_render_r0b_30k` = DeepTextures grouped (R0b) 30k (/workspace/runs/deeptex/eval/render_r0b_30k), `dt_render_r0b_35k` = DeepTextures grouped (R0b) 35k (/workspace/runs/deeptex/eval/render_r0b_35k), `dt_render_r0b_40k` = DeepTextures grouped (R0b) 40k (/workspace/runs/deeptex/eval/render_r0b_40k), `dt_render_r1_30k` = DeepTextures + render loss (R1) 30k (/workspace/runs/deeptex/eval/render_r1_30k), `dt_render_r1_35k` = DeepTextures + render loss (R1) 35k (/workspace/runs/deeptex/eval/render_r1_35k), `dt_render_r1_40k` = DeepTextures + render loss (R1) 40k (/workspace/runs/deeptex/eval/render_r1_40k).

## off - offset-removed maps

| column | n | render LPIPS | render FLIP | EVAL_LIGHT LPIPS | EVAL_LIGHT FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|
| bilinear | 12 | 0.5581 | 0.1141 | 0.5577 | 0.1112 | 0.5040 |
| MatUp | 12 | 0.5458 | 0.1373 | 0.5413 | 0.1354 | 0.4887 |
| DeepTextures (matsynth_all) | 12 | 0.2968 | 0.1236 | 0.2718 | 0.1187 | 0.2241 |
| DeepTextures grouped (R0a) 30k | 12 | 0.3054 | 0.1331 | 0.2955 | 0.1302 | 0.2442 |
| DeepTextures grouped (R0a) 35k | 12 | 0.3742 | 0.1378 | 0.3533 | 0.1319 | 0.2565 |
| DeepTextures grouped (R0a) 40k | 12 | 0.2793 | 0.1274 | 0.2737 | 0.1207 | 0.2520 |
| DeepTextures grouped (R0b) 30k | 12 | 0.3098 | 0.1222 | 0.2961 | 0.1195 | 0.2843 |
| DeepTextures grouped (R0b) 35k | 12 | 0.2736 | 0.1225 | 0.2580 | 0.1180 | 0.2435 |
| DeepTextures grouped (R0b) 40k | 12 | 0.2560 | 0.1237 | 0.2432 | 0.1211 | 0.2312 |
| DeepTextures + render loss (R1) 30k | 12 | 0.3510 | 0.1042 | 0.3388 | 0.1011 | 0.3116 |
| DeepTextures + render loss (R1) 35k | 12 | 0.3345 | 0.1076 | 0.3181 | 0.1046 | 0.2846 |
| DeepTextures + render loss (R1) 40k | 12 | 0.2933 | 0.1037 | 0.2730 | 0.1010 | 0.2281 |

Non-metal / metal split:

| column | non-metal n | render LPIPS | render FLIP | baseColor LPIPS | metal n | render LPIPS | render FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 0.5393 | 0.1195 | 0.4665 | 3 | 0.6147 | 0.0979 | 0.6162 |
| MatUp | 9 | 0.5346 | 0.1412 | 0.4433 | 3 | 0.5794 | 0.1255 | 0.6248 |
| DeepTextures (matsynth_all) | 9 | 0.3011 | 0.1275 | 0.2064 | 3 | 0.2842 | 0.1120 | 0.2772 |
| DeepTextures grouped (R0a) 30k | 9 | 0.2934 | 0.1370 | 0.2270 | 3 | 0.3412 | 0.1216 | 0.2959 |
| DeepTextures grouped (R0a) 35k | 9 | 0.3521 | 0.1419 | 0.2311 | 3 | 0.4405 | 0.1255 | 0.3326 |
| DeepTextures grouped (R0a) 40k | 9 | 0.2715 | 0.1324 | 0.2303 | 3 | 0.3026 | 0.1122 | 0.3168 |
| DeepTextures grouped (R0b) 30k | 9 | 0.3025 | 0.1280 | 0.2569 | 3 | 0.3317 | 0.1049 | 0.3663 |
| DeepTextures grouped (R0b) 35k | 9 | 0.2685 | 0.1274 | 0.2218 | 3 | 0.2889 | 0.1079 | 0.3087 |
| DeepTextures grouped (R0b) 40k | 9 | 0.2494 | 0.1286 | 0.2134 | 3 | 0.2759 | 0.1091 | 0.2845 |
| DeepTextures + render loss (R1) 30k | 9 | 0.3399 | 0.1073 | 0.2809 | 3 | 0.3840 | 0.0950 | 0.4036 |
| DeepTextures + render loss (R1) 35k | 9 | 0.3250 | 0.1112 | 0.2556 | 3 | 0.3631 | 0.0968 | 0.3715 |
| DeepTextures + render loss (R1) 40k | 9 | 0.2925 | 0.1069 | 0.2076 | 3 | 0.2957 | 0.0942 | 0.2897 |

## raw - raw maps

| column | n | render LPIPS | render FLIP | EVAL_LIGHT LPIPS | EVAL_LIGHT FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|
| bilinear | 12 | 0.5612 | 0.1154 | 0.5616 | 0.1135 | 0.5040 |
| MatUp | 12 | 0.5485 | 0.1455 | 0.5426 | 0.1453 | 0.4940 |
| DeepTextures (matsynth_all) | 12 | 0.3424 | 0.2721 | 0.3090 | 0.2441 | 0.2471 |
| DeepTextures grouped (R0a) 30k | 12 | 0.3137 | 0.1873 | 0.3042 | 0.1890 | 0.2548 |
| DeepTextures grouped (R0a) 35k | 12 | 0.4142 | 0.2982 | 0.3669 | 0.2561 | 0.2734 |
| DeepTextures grouped (R0a) 40k | 12 | 0.2894 | 0.1856 | 0.2850 | 0.1813 | 0.2620 |
| DeepTextures grouped (R0b) 30k | 12 | 0.3170 | 0.1505 | 0.3038 | 0.1519 | 0.2990 |
| DeepTextures grouped (R0b) 35k | 12 | 0.3005 | 0.2107 | 0.2851 | 0.1960 | 0.2523 |
| DeepTextures grouped (R0b) 40k | 12 | 0.2720 | 0.2155 | 0.2606 | 0.2065 | 0.2454 |
| DeepTextures + render loss (R1) 30k | 12 | 0.3540 | 0.1267 | 0.3420 | 0.1241 | 0.3126 |
| DeepTextures + render loss (R1) 35k | 12 | 0.3403 | 0.1608 | 0.3249 | 0.1620 | 0.2891 |
| DeepTextures + render loss (R1) 40k | 12 | 0.2948 | 0.1184 | 0.2744 | 0.1186 | 0.2288 |

Non-metal / metal split:

| column | non-metal n | render LPIPS | render FLIP | baseColor LPIPS | metal n | render LPIPS | render FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 0.5433 | 0.1212 | 0.4666 | 3 | 0.6148 | 0.0979 | 0.6163 |
| MatUp | 9 | 0.5388 | 0.1532 | 0.4515 | 3 | 0.5776 | 0.1226 | 0.6213 |
| DeepTextures (matsynth_all) | 9 | 0.3398 | 0.2651 | 0.2259 | 3 | 0.3502 | 0.2933 | 0.3108 |
| DeepTextures grouped (R0a) 30k | 9 | 0.2992 | 0.1917 | 0.2367 | 3 | 0.3572 | 0.1742 | 0.3089 |
| DeepTextures grouped (R0a) 35k | 9 | 0.3886 | 0.2841 | 0.2546 | 3 | 0.4910 | 0.3405 | 0.3297 |
| DeepTextures grouped (R0a) 40k | 9 | 0.2814 | 0.1830 | 0.2439 | 3 | 0.3134 | 0.1934 | 0.3162 |
| DeepTextures grouped (R0b) 30k | 9 | 0.3114 | 0.1576 | 0.2770 | 3 | 0.3338 | 0.1291 | 0.3652 |
| DeepTextures grouped (R0b) 35k | 9 | 0.2895 | 0.1987 | 0.2332 | 3 | 0.3335 | 0.2466 | 0.3099 |
| DeepTextures grouped (R0b) 40k | 9 | 0.2658 | 0.2116 | 0.2294 | 3 | 0.2906 | 0.2269 | 0.2934 |
| DeepTextures + render loss (R1) 30k | 9 | 0.3435 | 0.1253 | 0.2833 | 3 | 0.3855 | 0.1310 | 0.4004 |
| DeepTextures + render loss (R1) 35k | 9 | 0.3300 | 0.1567 | 0.2579 | 3 | 0.3714 | 0.1730 | 0.3827 |
| DeepTextures + render loss (R1) 40k | 9 | 0.2939 | 0.1205 | 0.2086 | 3 | 0.2975 | 0.1122 | 0.2895 |

## Head-to-head

Cells `k/n`: materials where the FIRST column is strictly LOWER (better). Mean diff
= first - second over the same materials; negative means the first is better.
Pairs as in render_summary.md (svbrdf_render_eval.head_to_head_pairs).

| first | second | n | render LPIPS off | render LPIPS raw | render FLIP off | render FLIP raw | baseColor LPIPS off | baseColor LPIPS raw | non-metal: render LPIPS off | non-metal: render FLIP off | mean diff render LPIPS off | mean diff render FLIP off | mean diff baseColor LPIPS off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DeepTextures + render loss (R1) 30k | DeepTextures grouped (R0a) 30k | 12 | 4/12 | 5/12 | 12/12 | 12/12 | 3/12 | 3/12 | 3/9 | 9/9 | +0.0456 | -0.0289 | +0.0673 |
| DeepTextures + render loss (R1) 30k | DeepTextures grouped (R0b) 30k | 12 | 3/12 | 5/12 | 12/12 | 8/12 | 3/12 | 4/12 | 3/9 | 9/9 | +0.0411 | -0.0180 | +0.0273 |
| DeepTextures grouped (R0a) 30k | DeepTextures grouped (R0b) 30k | 12 | 7/12 | 7/12 | 2/12 | 0/12 | 10/12 | 11/12 | 6/9 | 2/9 | -0.0045 | +0.0109 | -0.0400 |
| DeepTextures + render loss (R1) 35k | DeepTextures grouped (R0a) 35k | 12 | 6/12 | 7/12 | 12/12 | 12/12 | 3/12 | 4/12 | 4/9 | 9/9 | -0.0397 | -0.0301 | +0.0281 |
| DeepTextures + render loss (R1) 35k | DeepTextures grouped (R0b) 35k | 12 | 4/12 | 5/12 | 12/12 | 12/12 | 3/12 | 3/12 | 4/9 | 9/9 | +0.0609 | -0.0149 | +0.0411 |
| DeepTextures grouped (R0a) 35k | DeepTextures grouped (R0b) 35k | 12 | 1/12 | 0/12 | 0/12 | 0/12 | 6/12 | 4/12 | 1/9 | 0/9 | +0.1006 | +0.0152 | +0.0130 |
| DeepTextures + render loss (R1) 40k | DeepTextures grouped (R0a) 40k | 12 | 6/12 | 6/12 | 12/12 | 12/12 | 10/12 | 10/12 | 4/9 | 9/9 | +0.0140 | -0.0236 | -0.0238 |
| DeepTextures + render loss (R1) 40k | DeepTextures grouped (R0b) 40k | 12 | 4/12 | 6/12 | 12/12 | 12/12 | 6/12 | 7/12 | 3/9 | 9/9 | +0.0373 | -0.0200 | -0.0031 |
| DeepTextures grouped (R0a) 40k | DeepTextures grouped (R0b) 40k | 12 | 3/12 | 5/12 | 3/12 | 10/12 | 3/12 | 3/12 | 3/9 | 2/9 | +0.0233 | +0.0037 | +0.0208 |
| DeepTextures grouped (R0a) 30k | DeepTextures (matsynth_all) | 12 | 6/12 | 8/12 | 1/12 | 12/12 | 4/12 | 6/12 | 6/9 | 1/9 | +0.0085 | +0.0095 | +0.0202 |
| DeepTextures grouped (R0a) 35k | DeepTextures (matsynth_all) | 12 | 4/12 | 6/12 | 0/12 | 4/12 | 1/12 | 3/12 | 4/9 | 0/9 | +0.0774 | +0.0142 | +0.0324 |
| DeepTextures grouped (R0a) 40k | DeepTextures (matsynth_all) | 12 | 5/12 | 12/12 | 3/12 | 12/12 | 1/12 | 3/12 | 5/9 | 1/9 | -0.0176 | +0.0038 | +0.0279 |
| DeepTextures grouped (R0b) 30k | DeepTextures (matsynth_all) | 12 | 5/12 | 10/12 | 8/12 | 12/12 | 0/12 | 1/12 | 4/9 | 5/9 | +0.0130 | -0.0014 | +0.0602 |
| DeepTextures grouped (R0b) 35k | DeepTextures (matsynth_all) | 12 | 8/12 | 10/12 | 7/12 | 11/12 | 2/12 | 6/12 | 7/9 | 5/9 | -0.0233 | -0.0011 | +0.0194 |
| DeepTextures grouped (R0b) 40k | DeepTextures (matsynth_all) | 12 | 8/12 | 11/12 | 7/12 | 12/12 | 6/12 | 8/12 | 7/9 | 4/9 | -0.0408 | +0.0001 | +0.0071 |
| DeepTextures + render loss (R1) 30k | DeepTextures (matsynth_all) | 12 | 4/12 | 6/12 | 12/12 | 12/12 | 2/12 | 3/12 | 4/9 | 9/9 | +0.0541 | -0.0194 | +0.0875 |
| DeepTextures + render loss (R1) 35k | DeepTextures (matsynth_all) | 12 | 3/12 | 8/12 | 12/12 | 12/12 | 2/12 | 2/12 | 3/9 | 9/9 | +0.0376 | -0.0160 | +0.0605 |
| DeepTextures + render loss (R1) 40k | DeepTextures (matsynth_all) | 12 | 7/12 | 9/12 | 12/12 | 12/12 | 5/12 | 8/12 | 5/9 | 9/9 | -0.0036 | -0.0199 | +0.0040 |
| DeepTextures (matsynth_all) | bilinear | 12 | 12/12 | 12/12 | 3/12 | 0/12 | 12/12 | 11/12 | 9/9 | 3/9 | -0.2613 | +0.0095 | -0.2799 |
| DeepTextures (matsynth_all) | MatUp | 12 | 12/12 | 12/12 | 10/12 | 0/12 | 12/12 | 11/12 | 9/9 | 7/9 | -0.2490 | -0.0137 | -0.2646 |
| DeepTextures grouped (R0a) 30k | bilinear | 12 | 12/12 | 12/12 | 1/12 | 0/12 | 11/12 | 11/12 | 9/9 | 1/9 | -0.2527 | +0.0190 | -0.2597 |
| DeepTextures grouped (R0a) 30k | MatUp | 12 | 11/12 | 11/12 | 7/12 | 2/12 | 11/12 | 11/12 | 8/9 | 5/9 | -0.2404 | -0.0042 | -0.2445 |
| DeepTextures grouped (R0a) 35k | bilinear | 12 | 10/12 | 10/12 | 0/12 | 0/12 | 11/12 | 11/12 | 7/9 | 0/9 | -0.1839 | +0.0237 | -0.2475 |
| DeepTextures grouped (R0a) 35k | MatUp | 12 | 10/12 | 9/12 | 5/12 | 0/12 | 11/12 | 11/12 | 7/9 | 3/9 | -0.1716 | +0.0005 | -0.2322 |
| DeepTextures grouped (R0a) 40k | bilinear | 12 | 12/12 | 12/12 | 1/12 | 0/12 | 12/12 | 11/12 | 9/9 | 1/9 | -0.2788 | +0.0133 | -0.2520 |
| DeepTextures grouped (R0a) 40k | MatUp | 12 | 12/12 | 12/12 | 8/12 | 2/12 | 12/12 | 11/12 | 9/9 | 5/9 | -0.2665 | -0.0099 | -0.2367 |
| DeepTextures grouped (R0b) 30k | bilinear | 12 | 12/12 | 12/12 | 3/12 | 1/12 | 11/12 | 11/12 | 9/9 | 3/9 | -0.2483 | +0.0081 | -0.2197 |
| DeepTextures grouped (R0b) 30k | MatUp | 12 | 12/12 | 12/12 | 9/12 | 3/12 | 11/12 | 11/12 | 9/9 | 6/9 | -0.2360 | -0.0151 | -0.2044 |
| DeepTextures grouped (R0b) 35k | bilinear | 12 | 12/12 | 12/12 | 3/12 | 0/12 | 12/12 | 11/12 | 9/9 | 3/9 | -0.2845 | +0.0084 | -0.2605 |
| DeepTextures grouped (R0b) 35k | MatUp | 12 | 12/12 | 11/12 | 10/12 | 2/12 | 12/12 | 11/12 | 9/9 | 7/9 | -0.2722 | -0.0148 | -0.2452 |
| DeepTextures grouped (R0b) 40k | bilinear | 12 | 12/12 | 12/12 | 2/12 | 0/12 | 12/12 | 12/12 | 9/9 | 2/9 | -0.3021 | +0.0096 | -0.2728 |
| DeepTextures grouped (R0b) 40k | MatUp | 12 | 12/12 | 11/12 | 10/12 | 0/12 | 11/12 | 11/12 | 9/9 | 7/9 | -0.2898 | -0.0136 | -0.2575 |
| DeepTextures + render loss (R1) 30k | bilinear | 12 | 12/12 | 12/12 | 11/12 | 3/12 | 12/12 | 12/12 | 9/9 | 8/9 | -0.2072 | -0.0099 | -0.1924 |
| DeepTextures + render loss (R1) 30k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 7/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.1948 | -0.0331 | -0.1771 |
| DeepTextures + render loss (R1) 35k | bilinear | 12 | 12/12 | 12/12 | 7/12 | 1/12 | 12/12 | 12/12 | 9/9 | 6/9 | -0.2236 | -0.0065 | -0.2194 |
| DeepTextures + render loss (R1) 35k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 4/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.2113 | -0.0297 | -0.2041 |
| DeepTextures + render loss (R1) 40k | bilinear | 12 | 12/12 | 12/12 | 10/12 | 3/12 | 12/12 | 12/12 | 9/9 | 8/9 | -0.2648 | -0.0104 | -0.2758 |
| DeepTextures + render loss (R1) 40k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 11/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.2525 | -0.0336 | -0.2606 |

## Per checkpoint step

Mean over the `n` materials at each step, the mean over steps, and the
within-run range (max - min over steps).

### render LPIPS (off) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 0.3054 | 0.3742 | 0.2793 | 0.3196 | 0.0949 |
| DeepTextures grouped (R0b) | 0.3098 | 0.2736 | 0.2560 | 0.2798 | 0.0538 |
| DeepTextures + render loss (R1) | 0.3510 | 0.3345 | 0.2933 | 0.3262 | 0.0577 |

### render LPIPS (raw) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 0.3137 | 0.4142 | 0.2894 | 0.3391 | 0.1248 |
| DeepTextures grouped (R0b) | 0.3170 | 0.3005 | 0.2720 | 0.2965 | 0.0450 |
| DeepTextures + render loss (R1) | 0.3540 | 0.3403 | 0.2948 | 0.3297 | 0.0592 |

### render FLIP (off) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 0.1331 | 0.1378 | 0.1274 | 0.1327 | 0.0104 |
| DeepTextures grouped (R0b) | 0.1222 | 0.1225 | 0.1237 | 0.1228 | 0.0015 |
| DeepTextures + render loss (R1) | 0.1042 | 0.1076 | 0.1037 | 0.1052 | 0.0039 |

### render FLIP (raw) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 0.1873 | 0.2982 | 0.1856 | 0.2237 | 0.1126 |
| DeepTextures grouped (R0b) | 0.1505 | 0.2107 | 0.2155 | 0.1922 | 0.0650 |
| DeepTextures + render loss (R1) | 0.1267 | 0.1608 | 0.1184 | 0.1353 | 0.0424 |

### baseColor LPIPS (off) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 0.2442 | 0.2565 | 0.2520 | 0.2509 | 0.0122 |
| DeepTextures grouped (R0b) | 0.2843 | 0.2435 | 0.2312 | 0.2530 | 0.0531 |
| DeepTextures + render loss (R1) | 0.3116 | 0.2846 | 0.2281 | 0.2748 | 0.0834 |

### baseColor LPIPS (raw) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 0.2548 | 0.2734 | 0.2620 | 0.2634 | 0.0186 |
| DeepTextures grouped (R0b) | 0.2990 | 0.2523 | 0.2454 | 0.2656 | 0.0536 |
| DeepTextures + render loss (R1) | 0.3126 | 0.2891 | 0.2288 | 0.2768 | 0.0837 |

### render LPIPS (off), non-metal (n=9)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 0.2934 | 0.3521 | 0.2715 | 0.3057 | 0.0806 |
| DeepTextures grouped (R0b) | 0.3025 | 0.2685 | 0.2494 | 0.2735 | 0.0531 |
| DeepTextures + render loss (R1) | 0.3399 | 0.3250 | 0.2925 | 0.3191 | 0.0475 |

### render FLIP (off), non-metal (n=9)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 0.1370 | 0.1419 | 0.1324 | 0.1371 | 0.0095 |
| DeepTextures grouped (R0b) | 0.1280 | 0.1274 | 0.1286 | 0.1280 | 0.0012 |
| DeepTextures + render loss (R1) | 0.1073 | 0.1112 | 0.1069 | 0.1085 | 0.0043 |

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
