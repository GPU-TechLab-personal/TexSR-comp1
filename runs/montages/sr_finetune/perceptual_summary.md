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

Columns: `bilinear` = bilinear, `matup` = MatUp, `dt_matsynth_all_40k` = DeepTextures (matsynth_all) 40k (/workspace/runs/deeptex/eval/matsynth_all_40k), `dt_render_r1_40k` = DeepTextures + render loss (R1) 40k (/workspace/runs/deeptex/eval/render_r1_40k), `bsr_esrgan_zs` = ESRGAN official (zero-shot) (/workspace/runs/sr_ft/eval/esrgan_zs), `bsr_esrgan_ft_30k` = ESRGAN official (fine-tuned) 30k (/workspace/runs/sr_ft/eval/esrgan_ft_30k), `bsr_esrgan_ft_35k` = ESRGAN official (fine-tuned) 35k (/workspace/runs/sr_ft/eval/esrgan_ft_35k), `bsr_esrgan_ft_40k` = ESRGAN official (fine-tuned) 40k (/workspace/runs/sr_ft/eval/esrgan_ft_40k), `kair_swinir_cls_zs` = SwinIR-M classical (zero-shot) (/workspace/runs/sr_ft/eval/swinir_cls_zs), `kair_swinir_cls_ft_30k` = SwinIR-M classical (fine-tuned) 30k (/workspace/runs/sr_ft/eval/swinir_cls_ft_30k), `kair_swinir_cls_ft_35k` = SwinIR-M classical (fine-tuned) 35k (/workspace/runs/sr_ft/eval/swinir_cls_ft_35k), `kair_swinir_cls_ft_40k` = SwinIR-M classical (fine-tuned) 40k (/workspace/runs/sr_ft/eval/swinir_cls_ft_40k), `kair_swinir_real_zs` = SwinIR-M real (zero-shot) (/workspace/runs/sr_ft/eval/swinir_real_zs), `kair_swinir_real_ft_30k` = SwinIR-M real (fine-tuned) 30k (/workspace/runs/sr_ft/eval/swinir_real_ft_30k), `kair_swinir_real_ft_35k` = SwinIR-M real (fine-tuned) 35k (/workspace/runs/sr_ft/eval/swinir_real_ft_35k), `kair_swinir_real_ft_40k` = SwinIR-M real (fine-tuned) 40k (/workspace/runs/sr_ft/eval/swinir_real_ft_40k), `kair_swinir_realL_zs` = SwinIR-L real (zero-shot) (/workspace/runs/sr_ft/eval/swinir_realL_zs).

## off - offset-removed maps

| column | n | render LPIPS | render FLIP | EVAL_LIGHT LPIPS | EVAL_LIGHT FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|
| bilinear | 12 | 0.5581 | 0.1141 | 0.5577 | 0.1112 | 0.5040 |
| MatUp | 12 | 0.5458 | 0.1373 | 0.5413 | 0.1354 | 0.4887 |
| DeepTextures (matsynth_all) 40k | 12 | 0.2969 | 0.1236 | 0.2718 | 0.1187 | 0.2241 |
| DeepTextures + render loss (R1) 40k | 12 | 0.2933 | 0.1037 | 0.2730 | 0.1010 | 0.2281 |
| ESRGAN official (zero-shot) | 12 | 0.3994 | 0.1303 | 0.4089 | 0.1280 | 0.4036 |
| ESRGAN official (fine-tuned) 30k | 12 | 0.2150 | 0.1102 | 0.2037 | 0.1066 | 0.1554 |
| ESRGAN official (fine-tuned) 35k | 12 | 0.2197 | 0.1122 | 0.2104 | 0.1084 | 0.1594 |
| ESRGAN official (fine-tuned) 40k | 12 | 0.2159 | 0.1146 | 0.2077 | 0.1110 | 0.1540 |
| SwinIR-M classical (zero-shot) | 12 | 0.4128 | 0.0987 | 0.3977 | 0.0940 | 0.3611 |
| SwinIR-M classical (fine-tuned) 30k | 12 | 0.4151 | 0.0871 | 0.4116 | 0.0832 | 0.3920 |
| SwinIR-M classical (fine-tuned) 35k | 12 | 0.4116 | 0.0868 | 0.4077 | 0.0829 | 0.3897 |
| SwinIR-M classical (fine-tuned) 40k | 12 | 0.4122 | 0.0868 | 0.4085 | 0.0829 | 0.3901 |
| SwinIR-M real (zero-shot) | 12 | 0.4444 | 0.1824 | 0.4135 | 0.1709 | 0.4037 |
| SwinIR-M real (fine-tuned) 30k | 12 | 0.3847 | 0.1612 | 0.3595 | 0.1578 | 0.3532 |
| SwinIR-M real (fine-tuned) 35k | 12 | 0.3794 | 0.1616 | 0.3536 | 0.1582 | 0.3555 |
| SwinIR-M real (fine-tuned) 40k | 12 | 0.3784 | 0.1619 | 0.3578 | 0.1586 | 0.3570 |
| SwinIR-L real (zero-shot) | 12 | 0.4357 | 0.1832 | 0.3928 | 0.1737 | 0.3707 |

Non-metal / metal split:

| column | non-metal n | render LPIPS | render FLIP | baseColor LPIPS | metal n | render LPIPS | render FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 0.5393 | 0.1195 | 0.4665 | 3 | 0.6147 | 0.0979 | 0.6162 |
| MatUp | 9 | 0.5346 | 0.1412 | 0.4433 | 3 | 0.5794 | 0.1255 | 0.6248 |
| DeepTextures (matsynth_all) 40k | 9 | 0.3011 | 0.1275 | 0.2064 | 3 | 0.2842 | 0.1120 | 0.2773 |
| DeepTextures + render loss (R1) 40k | 9 | 0.2925 | 0.1069 | 0.2076 | 3 | 0.2957 | 0.0942 | 0.2897 |
| ESRGAN official (zero-shot) | 9 | 0.3375 | 0.1314 | 0.3756 | 3 | 0.5851 | 0.1267 | 0.4878 |
| ESRGAN official (fine-tuned) 30k | 9 | 0.2014 | 0.1132 | 0.1425 | 3 | 0.2556 | 0.1013 | 0.1941 |
| ESRGAN official (fine-tuned) 35k | 9 | 0.1997 | 0.1151 | 0.1435 | 3 | 0.2796 | 0.1034 | 0.2073 |
| ESRGAN official (fine-tuned) 40k | 9 | 0.1987 | 0.1178 | 0.1374 | 3 | 0.2675 | 0.1050 | 0.2039 |
| SwinIR-M classical (zero-shot) | 9 | 0.3962 | 0.1024 | 0.3192 | 3 | 0.4629 | 0.0878 | 0.4867 |
| SwinIR-M classical (fine-tuned) 30k | 9 | 0.3845 | 0.0903 | 0.3432 | 3 | 0.5071 | 0.0775 | 0.5383 |
| SwinIR-M classical (fine-tuned) 35k | 9 | 0.3809 | 0.0898 | 0.3415 | 3 | 0.5035 | 0.0777 | 0.5343 |
| SwinIR-M classical (fine-tuned) 40k | 9 | 0.3810 | 0.0898 | 0.3421 | 3 | 0.5057 | 0.0777 | 0.5342 |
| SwinIR-M real (zero-shot) | 9 | 0.4718 | 0.1967 | 0.4296 | 3 | 0.3621 | 0.1396 | 0.3262 |
| SwinIR-M real (fine-tuned) 30k | 9 | 0.3880 | 0.1689 | 0.3566 | 3 | 0.3750 | 0.1381 | 0.3431 |
| SwinIR-M real (fine-tuned) 35k | 9 | 0.3811 | 0.1694 | 0.3610 | 3 | 0.3743 | 0.1382 | 0.3388 |
| SwinIR-M real (fine-tuned) 40k | 9 | 0.3814 | 0.1705 | 0.3629 | 3 | 0.3695 | 0.1362 | 0.3394 |
| SwinIR-L real (zero-shot) | 9 | 0.4685 | 0.2000 | 0.3709 | 3 | 0.3375 | 0.1326 | 0.3700 |

## raw - raw maps

| column | n | render LPIPS | render FLIP | EVAL_LIGHT LPIPS | EVAL_LIGHT FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|
| bilinear | 12 | 0.5612 | 0.1154 | 0.5616 | 0.1135 | 0.5040 |
| MatUp | 12 | 0.5485 | 0.1455 | 0.5426 | 0.1453 | 0.4940 |
| DeepTextures (matsynth_all) 40k | 12 | 0.3425 | 0.2724 | 0.3093 | 0.2443 | 0.2472 |
| DeepTextures + render loss (R1) 40k | 12 | 0.2948 | 0.1184 | 0.2744 | 0.1186 | 0.2288 |
| ESRGAN official (zero-shot) | 12 | 0.4009 | 0.1459 | 0.4118 | 0.1442 | 0.4059 |
| ESRGAN official (fine-tuned) 30k | 12 | 0.2165 | 0.1188 | 0.2057 | 0.1163 | 0.1573 |
| ESRGAN official (fine-tuned) 35k | 12 | 0.2217 | 0.1195 | 0.2131 | 0.1169 | 0.1617 |
| ESRGAN official (fine-tuned) 40k | 12 | 0.2175 | 0.1205 | 0.2099 | 0.1178 | 0.1554 |
| SwinIR-M classical (zero-shot) | 12 | 0.4139 | 0.0984 | 0.3987 | 0.0938 | 0.3609 |
| SwinIR-M classical (fine-tuned) 30k | 12 | 0.4167 | 0.0868 | 0.4132 | 0.0831 | 0.3919 |
| SwinIR-M classical (fine-tuned) 35k | 12 | 0.4131 | 0.0864 | 0.4092 | 0.0827 | 0.3895 |
| SwinIR-M classical (fine-tuned) 40k | 12 | 0.4137 | 0.0865 | 0.4101 | 0.0829 | 0.3900 |
| SwinIR-M real (zero-shot) | 12 | 0.4493 | 0.2191 | 0.4201 | 0.2015 | 0.4052 |
| SwinIR-M real (fine-tuned) 30k | 12 | 0.3891 | 0.1720 | 0.3642 | 0.1699 | 0.3574 |
| SwinIR-M real (fine-tuned) 35k | 12 | 0.3830 | 0.1681 | 0.3574 | 0.1658 | 0.3592 |
| SwinIR-M real (fine-tuned) 40k | 12 | 0.3824 | 0.1686 | 0.3618 | 0.1660 | 0.3600 |
| SwinIR-L real (zero-shot) | 12 | 0.4380 | 0.2062 | 0.3979 | 0.1939 | 0.3713 |

Non-metal / metal split:

| column | non-metal n | render LPIPS | render FLIP | baseColor LPIPS | metal n | render LPIPS | render FLIP | baseColor LPIPS |
|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 0.5433 | 0.1212 | 0.4666 | 3 | 0.6148 | 0.0979 | 0.6163 |
| MatUp | 9 | 0.5388 | 0.1532 | 0.4515 | 3 | 0.5776 | 0.1226 | 0.6213 |
| DeepTextures (matsynth_all) 40k | 9 | 0.3399 | 0.2654 | 0.2260 | 3 | 0.3505 | 0.2936 | 0.3110 |
| DeepTextures + render loss (R1) 40k | 9 | 0.2939 | 0.1205 | 0.2086 | 3 | 0.2975 | 0.1122 | 0.2895 |
| ESRGAN official (zero-shot) | 9 | 0.3392 | 0.1454 | 0.3788 | 3 | 0.5860 | 0.1475 | 0.4873 |
| ESRGAN official (fine-tuned) 30k | 9 | 0.2029 | 0.1200 | 0.1441 | 3 | 0.2574 | 0.1154 | 0.1969 |
| ESRGAN official (fine-tuned) 35k | 9 | 0.2017 | 0.1226 | 0.1456 | 3 | 0.2817 | 0.1103 | 0.2099 |
| ESRGAN official (fine-tuned) 40k | 9 | 0.2003 | 0.1241 | 0.1386 | 3 | 0.2692 | 0.1096 | 0.2059 |
| SwinIR-M classical (zero-shot) | 9 | 0.3973 | 0.1023 | 0.3189 | 3 | 0.4640 | 0.0869 | 0.4868 |
| SwinIR-M classical (fine-tuned) 30k | 9 | 0.3864 | 0.0900 | 0.3431 | 3 | 0.5077 | 0.0770 | 0.5383 |
| SwinIR-M classical (fine-tuned) 35k | 9 | 0.3828 | 0.0895 | 0.3413 | 3 | 0.5042 | 0.0771 | 0.5342 |
| SwinIR-M classical (fine-tuned) 40k | 9 | 0.3829 | 0.0897 | 0.3419 | 3 | 0.5063 | 0.0772 | 0.5341 |
| SwinIR-M real (zero-shot) | 9 | 0.4763 | 0.2235 | 0.4306 | 3 | 0.3682 | 0.2058 | 0.3292 |
| SwinIR-M real (fine-tuned) 30k | 9 | 0.3904 | 0.1797 | 0.3585 | 3 | 0.3850 | 0.1488 | 0.3542 |
| SwinIR-M real (fine-tuned) 35k | 9 | 0.3836 | 0.1755 | 0.3632 | 3 | 0.3813 | 0.1457 | 0.3470 |
| SwinIR-M real (fine-tuned) 40k | 9 | 0.3838 | 0.1760 | 0.3640 | 3 | 0.3779 | 0.1464 | 0.3478 |
| SwinIR-L real (zero-shot) | 9 | 0.4717 | 0.2204 | 0.3721 | 3 | 0.3371 | 0.1636 | 0.3692 |

## Head-to-head

Cells `k/n`: materials where the FIRST column is strictly LOWER (better). Mean diff
= first - second over the same materials; negative means the first is better.
Pairs as in render_summary.md (svbrdf_render_eval.head_to_head_pairs).

| first | second | n | render LPIPS off | render LPIPS raw | render FLIP off | render FLIP raw | baseColor LPIPS off | baseColor LPIPS raw | non-metal: render LPIPS off | non-metal: render FLIP off | mean diff render LPIPS off | mean diff render FLIP off | mean diff baseColor LPIPS off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SwinIR-M real (zero-shot) | SwinIR-L real (zero-shot) | 12 | 3/12 | 3/12 | 4/12 | 4/12 | 4/12 | 4/12 | 3/9 | 4/9 | +0.0086 | -0.0007 | +0.0331 |
| ESRGAN official (fine-tuned) 40k | DeepTextures (matsynth_all) 40k | 12 | 11/12 | 12/12 | 10/12 | 12/12 | 12/12 | 12/12 | 9/9 | 7/9 | -0.0810 | -0.0090 | -0.0701 |
| SwinIR-M classical (fine-tuned) 40k | DeepTextures (matsynth_all) 40k | 12 | 2/12 | 3/12 | 12/12 | 12/12 | 2/12 | 4/12 | 2/9 | 9/9 | +0.1153 | -0.0369 | +0.1661 |
| SwinIR-M real (fine-tuned) 40k | DeepTextures (matsynth_all) 40k | 12 | 1/12 | 2/12 | 0/12 | 12/12 | 0/12 | 2/12 | 1/9 | 0/9 | +0.0816 | +0.0383 | +0.1329 |
| SwinIR-M real (fine-tuned) 40k | DeepTextures + render loss (R1) 40k | 12 | 0/12 | 0/12 | 0/12 | 1/12 | 0/12 | 0/12 | 0/9 | 0/9 | +0.0852 | +0.0582 | +0.1289 |
| DeepTextures + render loss (R1) 40k | DeepTextures (matsynth_all) 40k | 12 | 7/12 | 9/12 | 12/12 | 12/12 | 5/12 | 8/12 | 5/9 | 9/9 | -0.0036 | -0.0199 | +0.0040 |
| ESRGAN official (fine-tuned) 30k | ESRGAN official (zero-shot) | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.1845 | -0.0201 | -0.2482 |
| ESRGAN official (fine-tuned) 35k | ESRGAN official (zero-shot) | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.1797 | -0.0181 | -0.2442 |
| ESRGAN official (fine-tuned) 40k | ESRGAN official (zero-shot) | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.1835 | -0.0157 | -0.2496 |
| SwinIR-M classical (fine-tuned) 30k | SwinIR-M classical (zero-shot) | 12 | 6/12 | 6/12 | 12/12 | 12/12 | 2/12 | 2/12 | 6/9 | 9/9 | +0.0023 | -0.0117 | +0.0309 |
| SwinIR-M classical (fine-tuned) 35k | SwinIR-M classical (zero-shot) | 12 | 6/12 | 6/12 | 12/12 | 12/12 | 2/12 | 2/12 | 6/9 | 9/9 | -0.0013 | -0.0120 | +0.0286 |
| SwinIR-M classical (fine-tuned) 40k | SwinIR-M classical (zero-shot) | 12 | 6/12 | 6/12 | 12/12 | 12/12 | 2/12 | 2/12 | 6/9 | 9/9 | -0.0006 | -0.0120 | +0.0290 |
| SwinIR-M real (fine-tuned) 30k | SwinIR-M real (zero-shot) | 12 | 10/12 | 10/12 | 10/12 | 12/12 | 8/12 | 8/12 | 9/9 | 8/9 | -0.0596 | -0.0213 | -0.0505 |
| SwinIR-M real (fine-tuned) 35k | SwinIR-M real (zero-shot) | 12 | 10/12 | 10/12 | 10/12 | 12/12 | 8/12 | 8/12 | 9/9 | 8/9 | -0.0650 | -0.0208 | -0.0482 |
| SwinIR-M real (fine-tuned) 40k | SwinIR-M real (zero-shot) | 12 | 10/12 | 10/12 | 10/12 | 11/12 | 8/12 | 8/12 | 9/9 | 8/9 | -0.0659 | -0.0205 | -0.0467 |
| DeepTextures (matsynth_all) 40k | bilinear | 12 | 12/12 | 12/12 | 3/12 | 0/12 | 12/12 | 11/12 | 9/9 | 3/9 | -0.2612 | +0.0095 | -0.2799 |
| DeepTextures (matsynth_all) 40k | MatUp | 12 | 12/12 | 12/12 | 10/12 | 0/12 | 12/12 | 11/12 | 9/9 | 7/9 | -0.2489 | -0.0137 | -0.2646 |
| DeepTextures + render loss (R1) 40k | bilinear | 12 | 12/12 | 12/12 | 10/12 | 3/12 | 12/12 | 12/12 | 9/9 | 8/9 | -0.2648 | -0.0104 | -0.2758 |
| DeepTextures + render loss (R1) 40k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 11/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.2525 | -0.0336 | -0.2606 |
| ESRGAN official (zero-shot) | bilinear | 12 | 11/12 | 11/12 | 1/12 | 1/12 | 9/12 | 9/12 | 9/9 | 1/9 | -0.1587 | +0.0162 | -0.1003 |
| ESRGAN official (zero-shot) | MatUp | 12 | 9/12 | 9/12 | 8/12 | 6/12 | 9/12 | 9/12 | 8/9 | 7/9 | -0.1464 | -0.0070 | -0.0851 |
| ESRGAN official (fine-tuned) 30k | bilinear | 12 | 12/12 | 12/12 | 7/12 | 3/12 | 12/12 | 12/12 | 9/9 | 6/9 | -0.3432 | -0.0039 | -0.3486 |
| ESRGAN official (fine-tuned) 30k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.3309 | -0.0271 | -0.3333 |
| ESRGAN official (fine-tuned) 35k | bilinear | 12 | 12/12 | 12/12 | 5/12 | 4/12 | 12/12 | 12/12 | 9/9 | 5/9 | -0.3384 | -0.0019 | -0.3445 |
| ESRGAN official (fine-tuned) 35k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.3261 | -0.0251 | -0.3293 |
| ESRGAN official (fine-tuned) 40k | bilinear | 12 | 12/12 | 12/12 | 3/12 | 2/12 | 12/12 | 12/12 | 9/9 | 3/9 | -0.3422 | +0.0005 | -0.3500 |
| ESRGAN official (fine-tuned) 40k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.3299 | -0.0227 | -0.3347 |
| SwinIR-M classical (zero-shot) | bilinear | 12 | 12/12 | 12/12 | 11/12 | 11/12 | 12/12 | 12/12 | 9/9 | 8/9 | -0.1453 | -0.0154 | -0.1429 |
| SwinIR-M classical (zero-shot) | MatUp | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.1330 | -0.0386 | -0.1276 |
| SwinIR-M classical (fine-tuned) 30k | bilinear | 12 | 12/12 | 12/12 | 11/12 | 11/12 | 12/12 | 12/12 | 9/9 | 8/9 | -0.1430 | -0.0270 | -0.1120 |
| SwinIR-M classical (fine-tuned) 30k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.1307 | -0.0502 | -0.0967 |
| SwinIR-M classical (fine-tuned) 35k | bilinear | 12 | 12/12 | 12/12 | 11/12 | 11/12 | 12/12 | 12/12 | 9/9 | 8/9 | -0.1465 | -0.0273 | -0.1143 |
| SwinIR-M classical (fine-tuned) 35k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.1342 | -0.0505 | -0.0990 |
| SwinIR-M classical (fine-tuned) 40k | bilinear | 12 | 12/12 | 12/12 | 11/12 | 11/12 | 12/12 | 12/12 | 9/9 | 8/9 | -0.1459 | -0.0273 | -0.1138 |
| SwinIR-M classical (fine-tuned) 40k | MatUp | 12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 12/12 | 9/9 | 9/9 | -0.1336 | -0.0505 | -0.0986 |
| SwinIR-M real (zero-shot) | bilinear | 12 | 11/12 | 11/12 | 0/12 | 0/12 | 9/12 | 9/12 | 8/9 | 0/9 | -0.1138 | +0.0683 | -0.1002 |
| SwinIR-M real (zero-shot) | MatUp | 12 | 11/12 | 11/12 | 0/12 | 1/12 | 9/12 | 10/12 | 8/9 | 0/9 | -0.1014 | +0.0451 | -0.0850 |
| SwinIR-M real (fine-tuned) 30k | bilinear | 12 | 12/12 | 12/12 | 0/12 | 0/12 | 10/12 | 10/12 | 9/9 | 0/9 | -0.1734 | +0.0471 | -0.1507 |
| SwinIR-M real (fine-tuned) 30k | MatUp | 12 | 12/12 | 12/12 | 1/12 | 1/12 | 11/12 | 11/12 | 9/9 | 0/9 | -0.1611 | +0.0239 | -0.1355 |
| SwinIR-M real (fine-tuned) 35k | bilinear | 12 | 12/12 | 12/12 | 0/12 | 0/12 | 10/12 | 10/12 | 9/9 | 0/9 | -0.1787 | +0.0475 | -0.1485 |
| SwinIR-M real (fine-tuned) 35k | MatUp | 12 | 12/12 | 12/12 | 2/12 | 2/12 | 11/12 | 11/12 | 9/9 | 1/9 | -0.1664 | +0.0243 | -0.1332 |
| SwinIR-M real (fine-tuned) 40k | bilinear | 12 | 12/12 | 12/12 | 0/12 | 0/12 | 10/12 | 11/12 | 9/9 | 0/9 | -0.1797 | +0.0478 | -0.1470 |
| SwinIR-M real (fine-tuned) 40k | MatUp | 12 | 12/12 | 12/12 | 2/12 | 2/12 | 11/12 | 11/12 | 9/9 | 1/9 | -0.1674 | +0.0246 | -0.1317 |
| SwinIR-L real (zero-shot) | bilinear | 12 | 11/12 | 11/12 | 0/12 | 0/12 | 10/12 | 10/12 | 8/9 | 0/9 | -0.1224 | +0.0691 | -0.1333 |
| SwinIR-L real (zero-shot) | MatUp | 12 | 11/12 | 11/12 | 1/12 | 0/12 | 9/12 | 10/12 | 8/9 | 0/9 | -0.1101 | +0.0459 | -0.1180 |

## Per checkpoint step

Mean over the `n` materials at each step, the mean over steps, and the
within-run range (max - min over steps).

### render LPIPS (off) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 0.2969 | 0.2969 | - |
| DeepTextures + render loss (R1) | - | - | 0.2933 | 0.2933 | - |
| ESRGAN official (fine-tuned) | 0.2150 | 0.2197 | 0.2159 | 0.2168 | 0.0047 |
| SwinIR-M classical (fine-tuned) | 0.4151 | 0.4116 | 0.4122 | 0.4130 | 0.0036 |
| SwinIR-M real (fine-tuned) | 0.3847 | 0.3794 | 0.3784 | 0.3809 | 0.0063 |

### render LPIPS (raw) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 0.3425 | 0.3425 | - |
| DeepTextures + render loss (R1) | - | - | 0.2948 | 0.2948 | - |
| ESRGAN official (fine-tuned) | 0.2165 | 0.2217 | 0.2175 | 0.2186 | 0.0051 |
| SwinIR-M classical (fine-tuned) | 0.4167 | 0.4131 | 0.4137 | 0.4145 | 0.0036 |
| SwinIR-M real (fine-tuned) | 0.3891 | 0.3830 | 0.3824 | 0.3848 | 0.0067 |

### render FLIP (off) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 0.1236 | 0.1236 | - |
| DeepTextures + render loss (R1) | - | - | 0.1037 | 0.1037 | - |
| ESRGAN official (fine-tuned) | 0.1102 | 0.1122 | 0.1146 | 0.1123 | 0.0044 |
| SwinIR-M classical (fine-tuned) | 0.0871 | 0.0868 | 0.0868 | 0.0869 | 0.0003 |
| SwinIR-M real (fine-tuned) | 0.1612 | 0.1616 | 0.1619 | 0.1616 | 0.0007 |

### render FLIP (raw) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 0.2724 | 0.2724 | - |
| DeepTextures + render loss (R1) | - | - | 0.1184 | 0.1184 | - |
| ESRGAN official (fine-tuned) | 0.1188 | 0.1195 | 0.1205 | 0.1196 | 0.0017 |
| SwinIR-M classical (fine-tuned) | 0.0868 | 0.0864 | 0.0865 | 0.0866 | 0.0003 |
| SwinIR-M real (fine-tuned) | 0.1720 | 0.1681 | 0.1686 | 0.1696 | 0.0039 |

### baseColor LPIPS (off) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 0.2241 | 0.2241 | - |
| DeepTextures + render loss (R1) | - | - | 0.2281 | 0.2281 | - |
| ESRGAN official (fine-tuned) | 0.1554 | 0.1594 | 0.1540 | 0.1563 | 0.0054 |
| SwinIR-M classical (fine-tuned) | 0.3920 | 0.3897 | 0.3901 | 0.3906 | 0.0023 |
| SwinIR-M real (fine-tuned) | 0.3532 | 0.3555 | 0.3570 | 0.3552 | 0.0038 |

### baseColor LPIPS (raw) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 0.2472 | 0.2472 | - |
| DeepTextures + render loss (R1) | - | - | 0.2288 | 0.2288 | - |
| ESRGAN official (fine-tuned) | 0.1573 | 0.1617 | 0.1554 | 0.1581 | 0.0063 |
| SwinIR-M classical (fine-tuned) | 0.3919 | 0.3895 | 0.3900 | 0.3905 | 0.0023 |
| SwinIR-M real (fine-tuned) | 0.3574 | 0.3592 | 0.3600 | 0.3588 | 0.0026 |

### render LPIPS (off), non-metal (n=9)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 0.3011 | 0.3011 | - |
| DeepTextures + render loss (R1) | - | - | 0.2925 | 0.2925 | - |
| ESRGAN official (fine-tuned) | 0.2014 | 0.1997 | 0.1987 | 0.1999 | 0.0027 |
| SwinIR-M classical (fine-tuned) | 0.3845 | 0.3809 | 0.3810 | 0.3822 | 0.0036 |
| SwinIR-M real (fine-tuned) | 0.3880 | 0.3811 | 0.3814 | 0.3835 | 0.0069 |

### render FLIP (off), non-metal (n=9)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 0.1275 | 0.1275 | - |
| DeepTextures + render loss (R1) | - | - | 0.1069 | 0.1069 | - |
| ESRGAN official (fine-tuned) | 0.1132 | 0.1151 | 0.1178 | 0.1153 | 0.0046 |
| SwinIR-M classical (fine-tuned) | 0.0903 | 0.0898 | 0.0898 | 0.0899 | 0.0005 |
| SwinIR-M real (fine-tuned) | 0.1689 | 0.1694 | 0.1705 | 0.1696 | 0.0016 |

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
