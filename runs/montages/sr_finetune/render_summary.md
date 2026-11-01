# Renders of the predicted SVBRDF: 12 MatSynth evaluation materials

Every column's five maps are decoded the way MatUp decodes a MatSynth folder and
rendered by MatUp's own torch renderer (UNGUARDED, as MatUp renders). Errors are
against the render of the GT maps, over texels whose GT normal passes training's
validity rule (|2v-1| >= 0.1, unit n_z >= 1e-3).

- **held-out** = EVAL_LIGHT + 1.4 x fib(16) turned 43.25 deg about z (17 lights, none within 2.89 deg of a training direction);
  **train-100** = MatUp's 1.4 x fib(100), the lights the render loss uses.
- **P1** = held-out PSNR / l2,1 on OFFSET-REMOVED maps (primary); **P1raw** = the
  same on the raw maps; **P2** = train-100 l2,1. Offset removal subtracts each map's
  per-channel mean error vs GT in the stored encoding (the normal renormalised to
  unit length on both sides first: only its direction renders), so P1 is what is
  left after a constant drift is forgiven.
- PSNR: MSE over texels x lights x RGB in display space ([0,1]). l2,1: MatUp's loss
  (mean over texel-lights of the RGB L2 norm). Lower l2,1 is better.
- `n`: materials every column was scored on; every mean below is over that set
  (9 non-metal = GT metallic all zero, 3 metal).
- bands: train-100 lights by elevation, grazing < 20 deg (34), high > 60 deg (13),
  mid otherwise (53).
- `|grad| %GT`: matup_postprocess.sharpness of the EVAL_LIGHT render as a % of the
  GT render's; PSNR rewards blur, this does not.
- self-check: EVAL_LIGHT renders of GT / bilinear / MatUp reproduce the committed
  render_gt / render_bilinear / render_opt PNGs (see the table at the end).

Columns: `bilinear` = bilinear, `matup` = MatUp, `dt_matsynth_all_40k` = DeepTextures (matsynth_all) 40k (/workspace/runs/deeptex/eval/matsynth_all_40k), `dt_render_r1_40k` = DeepTextures + render loss (R1) 40k (/workspace/runs/deeptex/eval/render_r1_40k), `bsr_esrgan_zs` = ESRGAN official (zero-shot) (/workspace/runs/sr_ft/eval/esrgan_zs), `kair_swinir_cls_zs` = SwinIR-M classical (zero-shot) (/workspace/runs/sr_ft/eval/swinir_cls_zs), `kair_swinir_real_zs` = SwinIR-M real (zero-shot) (/workspace/runs/sr_ft/eval/swinir_real_zs), `kair_swinir_realL_zs` = SwinIR-L real (zero-shot) (/workspace/runs/sr_ft/eval/swinir_realL_zs).

## P1 - offset-removed maps

| column | n | held-out PSNR | held-out l2,1 | train-100 l2,1 | train-100 PSNR | grazing l2,1 | mid l2,1 | high l2,1 | eval PSNR | eval PSNR 8-bit | normal ° | \|grad\| %GT | n_z<1e-3 | metallic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 12 | 26.64 | 0.0724 | 0.0733 | 26.61 | 0.0854 | 0.0687 | 0.0598 | 27.43 | 27.41 | 6.28 | 35% | 0 | 0.133 |
| MatUp | 12 | 26.18 | 0.0769 | 0.0777 | 26.14 | 0.0899 | 0.0733 | 0.0639 | 27.02 | 27.00 | 6.47 | 33% | 0 | 0.135 |
| DeepTextures (matsynth_all) 40k | 12 | 25.21 | 0.0819 | 0.0831 | 25.14 | 0.0978 | 0.0774 | 0.0680 | 26.16 | 26.15 | 7.35 | 82% | 2 | 0.132 |
| DeepTextures + render loss (R1) 40k | 12 | 26.51 | 0.0721 | 0.0729 | 26.47 | 0.0839 | 0.0688 | 0.0611 | 27.22 | 27.21 | 6.03 | 71% | 2 | 0.133 |
| ESRGAN official (zero-shot) | 12 | 21.91 | 0.1239 | 0.1248 | 21.90 | 0.1362 | 0.1213 | 0.1092 | 22.40 | 22.40 | 9.26 | 174% | 313 | 0.133 |
| SwinIR-M classical (zero-shot) | 12 | 26.87 | 0.0687 | 0.0697 | 26.83 | 0.0820 | 0.0650 | 0.0563 | 27.80 | 27.79 | 6.04 | 63% | 4 | 0.133 |
| SwinIR-M real (zero-shot) | 12 | 23.45 | 0.1016 | 0.1034 | 23.40 | 0.1250 | 0.0954 | 0.0797 | 24.51 | 24.50 | 9.61 | 70% | 339 | 0.133 |
| SwinIR-L real (zero-shot) | 12 | 23.62 | 0.1025 | 0.1042 | 23.59 | 0.1259 | 0.0960 | 0.0809 | 24.53 | 24.52 | 10.01 | 65% | 529 | 0.133 |

Non-metal / metal split:

| column | non-metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° | metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° |
|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 26.23 | 0.0769 | 0.0782 | 7.87 | 3 | 27.87 | 0.0588 | 0.0585 | 1.48 |
| MatUp | 9 | 25.86 | 0.0808 | 0.0821 | 8.12 | 3 | 27.15 | 0.0650 | 0.0647 | 1.52 |
| DeepTextures (matsynth_all) 40k | 9 | 24.77 | 0.0867 | 0.0883 | 9.07 | 3 | 26.52 | 0.0675 | 0.0674 | 2.18 |
| DeepTextures + render loss (R1) 40k | 9 | 26.21 | 0.0756 | 0.0768 | 7.52 | 3 | 27.40 | 0.0617 | 0.0613 | 1.58 |
| ESRGAN official (zero-shot) | 9 | 21.89 | 0.1258 | 0.1273 | 11.60 | 3 | 21.97 | 0.1184 | 0.1174 | 2.26 |
| SwinIR-M classical (zero-shot) | 9 | 26.47 | 0.0730 | 0.0744 | 7.55 | 3 | 28.07 | 0.0558 | 0.0556 | 1.49 |
| SwinIR-M real (zero-shot) | 9 | 22.81 | 0.1102 | 0.1128 | 12.19 | 3 | 25.37 | 0.0757 | 0.0753 | 1.88 |
| SwinIR-L real (zero-shot) | 9 | 22.86 | 0.1126 | 0.1150 | 12.75 | 3 | 25.91 | 0.0723 | 0.0719 | 1.80 |

Map-swap attribution - held-out l2,1 of the GT with one map replaced by the
column's (the full-prediction value last):

| column | baseColor | normal | roughness | metallic | height | all five |
|---|---|---|---|---|---|---|
| bilinear | 0.0368 | 0.0498 | 0.0054 | 0.0042 | 0.0001 | 0.0724 |
| MatUp | 0.0406 | 0.0511 | 0.0084 | 0.0060 | 0.0001 | 0.0769 |
| DeepTextures (matsynth_all) 40k | 0.0408 | 0.0585 | 0.0061 | 0.0041 | 0.0001 | 0.0819 |
| DeepTextures + render loss (R1) 40k | 0.0386 | 0.0480 | 0.0061 | 0.0040 | 0.0001 | 0.0721 |
| ESRGAN official (zero-shot) | 0.0742 | 0.0717 | 0.0092 | 0.0066 | 0.0001 | 0.1239 |
| SwinIR-M classical (zero-shot) | 0.0337 | 0.0479 | 0.0050 | 0.0040 | 0.0001 | 0.0687 |
| SwinIR-M real (zero-shot) | 0.0460 | 0.0742 | 0.0081 | 0.0055 | 0.0002 | 0.1016 |
| SwinIR-L real (zero-shot) | 0.0448 | 0.0764 | 0.0069 | 0.0055 | 0.0001 | 0.1025 |

## P1raw - raw maps

| column | n | held-out PSNR | held-out l2,1 | train-100 l2,1 | train-100 PSNR | grazing l2,1 | mid l2,1 | high l2,1 | eval PSNR | eval PSNR 8-bit | normal ° | \|grad\| %GT | n_z<1e-3 | metallic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 12 | 26.59 | 0.0729 | 0.0737 | 26.56 | 0.0857 | 0.0693 | 0.0604 | 27.32 | 27.30 | 6.33 | 34% | 0 | 0.133 |
| MatUp | 12 | 25.61 | 0.0787 | 0.0795 | 25.57 | 0.0908 | 0.0751 | 0.0675 | 26.37 | 26.36 | 6.54 | 33% | 0 | 0.141 |
| DeepTextures (matsynth_all) 40k | 12 | 19.84 | 0.1330 | 0.1360 | 19.63 | 0.1625 | 0.1227 | 0.1210 | 21.32 | 21.32 | 13.26 | 83% | 2 | 0.128 |
| DeepTextures + render loss (R1) 40k | 12 | 26.25 | 0.0734 | 0.0742 | 26.21 | 0.0849 | 0.0702 | 0.0628 | 26.92 | 26.91 | 6.09 | 70% | 1 | 0.134 |
| ESRGAN official (zero-shot) | 12 | 21.59 | 0.1257 | 0.1266 | 21.59 | 0.1375 | 0.1231 | 0.1124 | 22.05 | 22.04 | 9.56 | 175% | 337 | 0.133 |
| SwinIR-M classical (zero-shot) | 12 | 26.88 | 0.0685 | 0.0694 | 26.84 | 0.0818 | 0.0648 | 0.0561 | 27.81 | 27.80 | 6.02 | 62% | 3 | 0.133 |
| SwinIR-M real (zero-shot) | 12 | 21.65 | 0.1122 | 0.1144 | 21.53 | 0.1373 | 0.1044 | 0.0952 | 22.74 | 22.74 | 11.03 | 70% | 327 | 0.135 |
| SwinIR-L real (zero-shot) | 12 | 22.17 | 0.1098 | 0.1116 | 22.14 | 0.1348 | 0.1017 | 0.0907 | 23.19 | 23.18 | 11.05 | 66% | 528 | 0.135 |

Non-metal / metal split:

| column | non-metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° | metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° |
|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 26.18 | 0.0775 | 0.0787 | 7.95 | 3 | 27.81 | 0.0591 | 0.0588 | 1.48 |
| MatUp | 9 | 24.91 | 0.0844 | 0.0856 | 8.21 | 3 | 27.69 | 0.0614 | 0.0610 | 1.52 |
| DeepTextures (matsynth_all) 40k | 9 | 19.72 | 0.1317 | 0.1348 | 14.71 | 3 | 20.19 | 0.1370 | 0.1395 | 8.93 |
| DeepTextures + render loss (R1) 40k | 9 | 25.92 | 0.0769 | 0.0781 | 7.56 | 3 | 27.24 | 0.0629 | 0.0626 | 1.66 |
| ESRGAN official (zero-shot) | 9 | 21.54 | 0.1273 | 0.1288 | 11.86 | 3 | 21.77 | 0.1210 | 0.1201 | 2.66 |
| SwinIR-M classical (zero-shot) | 9 | 26.48 | 0.0728 | 0.0742 | 7.53 | 3 | 28.08 | 0.0555 | 0.0553 | 1.48 |
| SwinIR-M real (zero-shot) | 9 | 21.07 | 0.1183 | 0.1211 | 13.25 | 3 | 23.37 | 0.0939 | 0.0944 | 4.36 |
| SwinIR-L real (zero-shot) | 9 | 21.31 | 0.1193 | 0.1217 | 13.63 | 3 | 24.75 | 0.0813 | 0.0811 | 3.29 |

Map-swap attribution - held-out l2,1 of the GT with one map replaced by the
column's (the full-prediction value last):

| column | baseColor | normal | roughness | metallic | height | all five |
|---|---|---|---|---|---|---|
| bilinear | 0.0368 | 0.0501 | 0.0054 | 0.0042 | 0.0001 | 0.0729 |
| MatUp | 0.0469 | 0.0516 | 0.0129 | 0.0047 | 0.0001 | 0.0787 |
| DeepTextures (matsynth_all) 40k | 0.0498 | 0.1146 | 0.0064 | 0.0045 | 0.0001 | 0.1330 |
| DeepTextures + render loss (R1) 40k | 0.0402 | 0.0485 | 0.0066 | 0.0038 | 0.0001 | 0.0734 |
| ESRGAN official (zero-shot) | 0.0747 | 0.0746 | 0.0093 | 0.0066 | 0.0001 | 0.1257 |
| SwinIR-M classical (zero-shot) | 0.0337 | 0.0478 | 0.0050 | 0.0035 | 0.0000 | 0.0685 |
| SwinIR-M real (zero-shot) | 0.0484 | 0.0897 | 0.0082 | 0.0044 | 0.0002 | 0.1122 |
| SwinIR-L real (zero-shot) | 0.0462 | 0.0871 | 0.0071 | 0.0045 | 0.0001 | 0.1098 |

## Head-to-head

Materials where the first column is strictly better; mean paired difference
(first - second) over the same materials.

- SwinIR-M real (zero-shot) vs SwinIR-L real (zero-shot): P1 5/12 (non-metal 4/9), P1 l2,1 6/12, P1raw 6/12; mean diff P1 -0.168 dB, P1raw -0.524 dB, P1 l2,1 -0.00096
- DeepTextures + render loss (R1) 40k vs DeepTextures (matsynth_all) 40k: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.304 dB, P1raw +6.413 dB, P1 l2,1 -0.00986
- DeepTextures (matsynth_all) 40k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.435 dB, P1raw -6.753 dB, P1 l2,1 +0.00958
- DeepTextures (matsynth_all) 40k vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 1/12, P1raw 0/12; mean diff P1 -0.972 dB, P1raw -5.771 dB, P1 l2,1 +0.00508
- DeepTextures + render loss (R1) 40k vs bilinear: P1 4/12 (non-metal 4/9), P1 l2,1 5/12, P1raw 5/12; mean diff P1 -0.131 dB, P1raw -0.340 dB, P1 l2,1 -0.00028
- DeepTextures + render loss (R1) 40k vs MatUp: P1 8/12 (non-metal 7/9), P1 l2,1 10/12, P1raw 7/12; mean diff P1 +0.332 dB, P1raw +0.642 dB, P1 l2,1 -0.00478
- ESRGAN official (zero-shot) vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -4.734 dB, P1raw -4.994 dB, P1 l2,1 +0.05159
- ESRGAN official (zero-shot) vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 1/12; mean diff P1 -4.271 dB, P1raw -4.012 dB, P1 l2,1 +0.04709
- SwinIR-M classical (zero-shot) vs bilinear: P1 8/12 (non-metal 6/9), P1 l2,1 10/12, P1raw 10/12; mean diff P1 +0.229 dB, P1raw +0.293 dB, P1 l2,1 -0.00363
- SwinIR-M classical (zero-shot) vs MatUp: P1 10/12 (non-metal 8/9), P1 l2,1 12/12, P1raw 10/12; mean diff P1 +0.693 dB, P1raw +1.275 dB, P1 l2,1 -0.00813
- SwinIR-M real (zero-shot) vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -3.188 dB, P1raw -4.942 dB, P1 l2,1 +0.02920
- SwinIR-M real (zero-shot) vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -2.725 dB, P1raw -3.960 dB, P1 l2,1 +0.02470
- SwinIR-L real (zero-shot) vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -3.021 dB, P1raw -4.418 dB, P1 l2,1 +0.03017
- SwinIR-L real (zero-shot) vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -2.558 dB, P1raw -3.436 dB, P1 l2,1 +0.02566

## Per checkpoint step

Mean over the `n` materials at each step, the mean over steps, and the
within-run range (max - min over steps).

### P1 PSNR (off) (n=12)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 25.208 | 25.208 | - |
| DeepTextures + render loss (R1) | 26.512 | 26.512 | - |

### P1raw PSNR (raw) (n=12)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 19.835 | 19.835 | - |
| DeepTextures + render loss (R1) | 26.248 | 26.248 | - |

### P1 PSNR (off), non-metal (n=9)

| arm | 40k | mean | range |
|---|---|---|---|
| DeepTextures (matsynth_all) | 24.771 | 24.771 | - |
| DeepTextures + render loss (R1) | 26.215 | 26.215 | - |

## Self-check (max |ours - committed| in 16-bit LSB)

Tolerance: GT and bilinear 2.0 LSB (same code wrote them; PNG truncation), MatUp 32.8 LSB (render_opt.png came from MatUp's in-memory maps).

| material | GT | bilinear | MatUp |
|---|---|---|---|
| 0559_brick_uneven_stones | 1.00 | 1.00 | 9.25 |
| 0844_syntetic_foam | 1.00 | 1.00 | 16.64 |
| 1587_cobblestones | 1.00 | 1.00 | 7.41 |
| 2119_corten_steel | 1.00 | 1.00 | 1.63 |
| acg_concrete_026 | 1.00 | 1.00 | 2.96 |
| acg_metal_026 | 1.00 | 1.00 | 5.88 |
| acg_plastic_009 | 1.00 | 1.00 | 8.54 |
| acg_roofing_tiles_006 | 1.00 | 1.00 | 21.39 |
| acg_wood_floor_060 | 1.00 | 1.00 | 7.02 |
| tc_metal_029 | 1.00 | 1.00 | 6.76 |
| th_brick_wall_005 | 1.00 | 1.00 | 19.01 |
| th_brown_mud_leaves_001 | 1.00 | 1.00 | 2.67 |

Texels whose rendered normal had n_z == 0 exactly (MatUp's ONB is 0/0 there; given the smallest positive n_z): none.
