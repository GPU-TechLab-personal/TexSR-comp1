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

Columns: `bilinear` = bilinear, `matup` = MatUp, `dt_matsynth_all` = DeepTextures (matsynth_all) (/workspace/runs/deeptex/eval/matsynth_all), `dt_render_r0a_30k` = DeepTextures grouped (R0a) 30k (/workspace/runs/deeptex/eval/render_r0a_30k), `dt_render_r0a_35k` = DeepTextures grouped (R0a) 35k (/workspace/runs/deeptex/eval/render_r0a_35k), `dt_render_r0a_40k` = DeepTextures grouped (R0a) 40k (/workspace/runs/deeptex/eval/render_r0a_40k), `dt_render_r0b_30k` = DeepTextures grouped (R0b) 30k (/workspace/runs/deeptex/eval/render_r0b_30k), `dt_render_r0b_35k` = DeepTextures grouped (R0b) 35k (/workspace/runs/deeptex/eval/render_r0b_35k), `dt_render_r0b_40k` = DeepTextures grouped (R0b) 40k (/workspace/runs/deeptex/eval/render_r0b_40k), `dt_render_r1_30k` = DeepTextures + render loss (R1) 30k (/workspace/runs/deeptex/eval/render_r1_30k), `dt_render_r1_35k` = DeepTextures + render loss (R1) 35k (/workspace/runs/deeptex/eval/render_r1_35k), `dt_render_r1_40k` = DeepTextures + render loss (R1) 40k (/workspace/runs/deeptex/eval/render_r1_40k).

## P1 - offset-removed maps

| column | n | held-out PSNR | held-out l2,1 | train-100 l2,1 | train-100 PSNR | grazing l2,1 | mid l2,1 | high l2,1 | eval PSNR | eval PSNR 8-bit | normal ° | \|grad\| %GT | n_z<1e-3 | metallic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 12 | 26.64 | 0.0724 | 0.0733 | 26.61 | 0.0854 | 0.0687 | 0.0598 | 27.43 | 27.41 | 6.28 | 35% | 0 | 0.133 |
| MatUp | 12 | 26.18 | 0.0769 | 0.0777 | 26.14 | 0.0899 | 0.0733 | 0.0639 | 27.02 | 27.00 | 6.47 | 33% | 0 | 0.135 |
| DeepTextures (matsynth_all) | 12 | 25.21 | 0.0819 | 0.0831 | 25.14 | 0.0978 | 0.0774 | 0.0680 | 26.16 | 26.15 | 7.35 | 82% | 2 | 0.132 |
| DeepTextures grouped (R0a) 30k | 12 | 24.62 | 0.0887 | 0.0899 | 24.57 | 0.1058 | 0.0839 | 0.0722 | 25.53 | 25.52 | 8.15 | 88% | 23 | 0.132 |
| DeepTextures grouped (R0a) 35k | 12 | 24.14 | 0.0875 | 0.0888 | 24.05 | 0.1035 | 0.0827 | 0.0751 | 25.08 | 25.08 | 7.87 | 95% | 0 | 0.133 |
| DeepTextures grouped (R0a) 40k | 12 | 24.80 | 0.0867 | 0.0881 | 24.73 | 0.1060 | 0.0814 | 0.0683 | 26.00 | 26.00 | 8.23 | 85% | 19 | 0.133 |
| DeepTextures grouped (R0b) 30k | 12 | 25.47 | 0.0813 | 0.0825 | 25.41 | 0.0976 | 0.0768 | 0.0659 | 26.33 | 26.32 | 7.50 | 72% | 2 | 0.133 |
| DeepTextures grouped (R0b) 35k | 12 | 25.16 | 0.0840 | 0.0852 | 25.11 | 0.1003 | 0.0795 | 0.0687 | 26.21 | 26.20 | 7.51 | 82% | 4 | 0.133 |
| DeepTextures grouped (R0b) 40k | 12 | 25.17 | 0.0862 | 0.0874 | 25.11 | 0.1030 | 0.0816 | 0.0703 | 26.17 | 26.16 | 7.73 | 83% | 8 | 0.132 |
| DeepTextures + render loss (R1) 30k | 12 | 26.89 | 0.0699 | 0.0708 | 26.84 | 0.0828 | 0.0663 | 0.0579 | 27.78 | 27.76 | 6.08 | 60% | 0 | 0.133 |
| DeepTextures + render loss (R1) 35k | 12 | 26.70 | 0.0711 | 0.0720 | 26.65 | 0.0835 | 0.0677 | 0.0595 | 27.51 | 27.50 | 6.12 | 65% | 4 | 0.133 |
| DeepTextures + render loss (R1) 40k | 12 | 26.51 | 0.0721 | 0.0729 | 26.47 | 0.0839 | 0.0688 | 0.0611 | 27.22 | 27.21 | 6.03 | 71% | 2 | 0.133 |

Non-metal / metal split:

| column | non-metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° | metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° |
|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 26.23 | 0.0769 | 0.0782 | 7.87 | 3 | 27.87 | 0.0588 | 0.0585 | 1.48 |
| MatUp | 9 | 25.86 | 0.0808 | 0.0821 | 8.12 | 3 | 27.15 | 0.0650 | 0.0647 | 1.52 |
| DeepTextures (matsynth_all) | 9 | 24.77 | 0.0867 | 0.0883 | 9.07 | 3 | 26.52 | 0.0675 | 0.0674 | 2.18 |
| DeepTextures grouped (R0a) 30k | 9 | 24.07 | 0.0949 | 0.0966 | 10.16 | 3 | 26.28 | 0.0698 | 0.0695 | 2.14 |
| DeepTextures grouped (R0a) 35k | 9 | 23.68 | 0.0917 | 0.0934 | 9.57 | 3 | 25.52 | 0.0751 | 0.0750 | 2.77 |
| DeepTextures grouped (R0a) 40k | 9 | 24.16 | 0.0935 | 0.0954 | 10.24 | 3 | 26.74 | 0.0663 | 0.0662 | 2.18 |
| DeepTextures grouped (R0b) 30k | 9 | 24.84 | 0.0877 | 0.0893 | 9.38 | 3 | 27.35 | 0.0620 | 0.0618 | 1.86 |
| DeepTextures grouped (R0b) 35k | 9 | 24.68 | 0.0899 | 0.0915 | 9.34 | 3 | 26.63 | 0.0664 | 0.0662 | 2.02 |
| DeepTextures grouped (R0b) 40k | 9 | 24.67 | 0.0928 | 0.0944 | 9.65 | 3 | 26.66 | 0.0666 | 0.0663 | 1.98 |
| DeepTextures + render loss (R1) 30k | 9 | 26.51 | 0.0740 | 0.0753 | 7.57 | 3 | 28.05 | 0.0576 | 0.0574 | 1.61 |
| DeepTextures + render loss (R1) 35k | 9 | 26.33 | 0.0750 | 0.0763 | 7.63 | 3 | 27.80 | 0.0593 | 0.0591 | 1.59 |
| DeepTextures + render loss (R1) 40k | 9 | 26.21 | 0.0756 | 0.0768 | 7.52 | 3 | 27.40 | 0.0617 | 0.0613 | 1.58 |

Map-swap attribution - held-out l2,1 of the GT with one map replaced by the
column's (the full-prediction value last):

| column | baseColor | normal | roughness | metallic | height | all five |
|---|---|---|---|---|---|---|
| bilinear | 0.0368 | 0.0498 | 0.0054 | 0.0042 | 0.0001 | 0.0724 |
| MatUp | 0.0406 | 0.0511 | 0.0084 | 0.0060 | 0.0001 | 0.0769 |
| DeepTextures (matsynth_all) | 0.0408 | 0.0585 | 0.0061 | 0.0041 | 0.0001 | 0.0819 |
| DeepTextures grouped (R0a) 30k | 0.0430 | 0.0644 | 0.0066 | 0.0044 | 0.0001 | 0.0887 |
| DeepTextures grouped (R0a) 35k | 0.0432 | 0.0640 | 0.0064 | 0.0045 | 0.0001 | 0.0875 |
| DeepTextures grouped (R0a) 40k | 0.0392 | 0.0652 | 0.0056 | 0.0046 | 0.0001 | 0.0867 |
| DeepTextures grouped (R0b) 30k | 0.0392 | 0.0590 | 0.0057 | 0.0038 | 0.0001 | 0.0813 |
| DeepTextures grouped (R0b) 35k | 0.0423 | 0.0596 | 0.0059 | 0.0042 | 0.0001 | 0.0840 |
| DeepTextures grouped (R0b) 40k | 0.0434 | 0.0610 | 0.0058 | 0.0039 | 0.0001 | 0.0862 |
| DeepTextures + render loss (R1) 30k | 0.0356 | 0.0485 | 0.0053 | 0.0036 | 0.0001 | 0.0699 |
| DeepTextures + render loss (R1) 35k | 0.0370 | 0.0485 | 0.0057 | 0.0037 | 0.0001 | 0.0711 |
| DeepTextures + render loss (R1) 40k | 0.0386 | 0.0480 | 0.0061 | 0.0040 | 0.0001 | 0.0721 |

## P1raw - raw maps

| column | n | held-out PSNR | held-out l2,1 | train-100 l2,1 | train-100 PSNR | grazing l2,1 | mid l2,1 | high l2,1 | eval PSNR | eval PSNR 8-bit | normal ° | \|grad\| %GT | n_z<1e-3 | metallic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 12 | 26.59 | 0.0729 | 0.0737 | 26.56 | 0.0857 | 0.0693 | 0.0604 | 27.32 | 27.30 | 6.33 | 34% | 0 | 0.133 |
| MatUp | 12 | 25.61 | 0.0787 | 0.0795 | 25.57 | 0.0908 | 0.0751 | 0.0675 | 26.37 | 26.36 | 6.54 | 33% | 0 | 0.141 |
| DeepTextures (matsynth_all) | 12 | 19.84 | 0.1329 | 0.1359 | 19.63 | 0.1624 | 0.1227 | 0.1210 | 21.33 | 21.33 | 13.26 | 83% | 2 | 0.128 |
| DeepTextures grouped (R0a) 30k | 12 | 22.82 | 0.1022 | 0.1031 | 22.78 | 0.1165 | 0.0973 | 0.0916 | 23.69 | 23.68 | 9.06 | 90% | 22 | 0.127 |
| DeepTextures grouped (R0a) 35k | 12 | 18.97 | 0.1470 | 0.1513 | 18.71 | 0.1810 | 0.1364 | 0.1339 | 20.51 | 20.51 | 14.63 | 94% | 0 | 0.134 |
| DeepTextures grouped (R0a) 40k | 12 | 22.36 | 0.1037 | 0.1054 | 22.25 | 0.1223 | 0.0979 | 0.0920 | 23.44 | 23.44 | 9.57 | 84% | 18 | 0.134 |
| DeepTextures grouped (R0b) 30k | 12 | 24.30 | 0.0905 | 0.0915 | 24.24 | 0.1046 | 0.0865 | 0.0779 | 25.16 | 25.15 | 7.91 | 71% | 1 | 0.132 |
| DeepTextures grouped (R0b) 35k | 12 | 20.86 | 0.1181 | 0.1201 | 20.68 | 0.1410 | 0.1089 | 0.1113 | 22.25 | 22.24 | 11.11 | 83% | 4 | 0.133 |
| DeepTextures grouped (R0b) 40k | 12 | 22.32 | 0.1081 | 0.1098 | 22.19 | 0.1260 | 0.1027 | 0.0962 | 23.51 | 23.51 | 9.64 | 83% | 8 | 0.132 |
| DeepTextures + render loss (R1) 30k | 12 | 26.10 | 0.0729 | 0.0738 | 26.04 | 0.0855 | 0.0692 | 0.0621 | 26.90 | 26.89 | 6.37 | 60% | 0 | 0.132 |
| DeepTextures + render loss (R1) 35k | 12 | 24.91 | 0.0798 | 0.0808 | 24.84 | 0.0925 | 0.0758 | 0.0709 | 25.75 | 25.74 | 7.00 | 64% | 3 | 0.132 |
| DeepTextures + render loss (R1) 40k | 12 | 26.25 | 0.0734 | 0.0742 | 26.21 | 0.0849 | 0.0702 | 0.0628 | 26.92 | 26.91 | 6.09 | 70% | 1 | 0.134 |

Non-metal / metal split:

| column | non-metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° | metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° |
|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 26.18 | 0.0775 | 0.0787 | 7.95 | 3 | 27.81 | 0.0591 | 0.0588 | 1.48 |
| MatUp | 9 | 24.91 | 0.0844 | 0.0856 | 8.21 | 3 | 27.69 | 0.0614 | 0.0610 | 1.52 |
| DeepTextures (matsynth_all) | 9 | 19.72 | 0.1316 | 0.1348 | 14.70 | 3 | 20.19 | 0.1369 | 0.1394 | 8.93 |
| DeepTextures grouped (R0a) 30k | 9 | 22.25 | 0.1081 | 0.1093 | 10.87 | 3 | 24.54 | 0.0846 | 0.0844 | 3.64 |
| DeepTextures grouped (R0a) 35k | 9 | 19.18 | 0.1397 | 0.1440 | 15.56 | 3 | 18.31 | 0.1686 | 0.1731 | 11.84 |
| DeepTextures grouped (R0a) 40k | 9 | 21.90 | 0.1079 | 0.1097 | 11.11 | 3 | 23.74 | 0.0911 | 0.0924 | 4.94 |
| DeepTextures grouped (R0b) 30k | 9 | 23.53 | 0.0982 | 0.0996 | 9.76 | 3 | 26.60 | 0.0675 | 0.0673 | 2.35 |
| DeepTextures grouped (R0b) 35k | 9 | 20.95 | 0.1141 | 0.1162 | 11.96 | 3 | 20.56 | 0.1300 | 0.1320 | 8.53 |
| DeepTextures grouped (R0b) 40k | 9 | 21.79 | 0.1134 | 0.1152 | 11.36 | 3 | 23.90 | 0.0923 | 0.0936 | 4.49 |
| DeepTextures + render loss (R1) 30k | 9 | 25.65 | 0.0766 | 0.0779 | 7.80 | 3 | 27.47 | 0.0617 | 0.0617 | 2.09 |
| DeepTextures + render loss (R1) 35k | 9 | 24.56 | 0.0821 | 0.0833 | 8.34 | 3 | 25.97 | 0.0731 | 0.0732 | 2.97 |
| DeepTextures + render loss (R1) 40k | 9 | 25.92 | 0.0769 | 0.0781 | 7.56 | 3 | 27.24 | 0.0629 | 0.0626 | 1.66 |

Map-swap attribution - held-out l2,1 of the GT with one map replaced by the
column's (the full-prediction value last):

| column | baseColor | normal | roughness | metallic | height | all five |
|---|---|---|---|---|---|---|
| bilinear | 0.0368 | 0.0501 | 0.0054 | 0.0042 | 0.0001 | 0.0729 |
| MatUp | 0.0469 | 0.0516 | 0.0129 | 0.0047 | 0.0001 | 0.0787 |
| DeepTextures (matsynth_all) | 0.0498 | 0.1145 | 0.0064 | 0.0045 | 0.0001 | 0.1329 |
| DeepTextures grouped (R0a) 30k | 0.0562 | 0.0744 | 0.0068 | 0.0047 | 0.0001 | 0.1022 |
| DeepTextures grouped (R0a) 35k | 0.0507 | 0.1287 | 0.0064 | 0.0044 | 0.0001 | 0.1470 |
| DeepTextures grouped (R0a) 40k | 0.0499 | 0.0802 | 0.0064 | 0.0053 | 0.0001 | 0.1037 |
| DeepTextures grouped (R0b) 30k | 0.0487 | 0.0629 | 0.0068 | 0.0039 | 0.0001 | 0.0905 |
| DeepTextures grouped (R0b) 35k | 0.0474 | 0.0991 | 0.0062 | 0.0049 | 0.0001 | 0.1181 |
| DeepTextures grouped (R0b) 40k | 0.0564 | 0.0799 | 0.0058 | 0.0040 | 0.0001 | 0.1081 |
| DeepTextures + render loss (R1) 30k | 0.0377 | 0.0516 | 0.0056 | 0.0036 | 0.0001 | 0.0729 |
| DeepTextures + render loss (R1) 35k | 0.0407 | 0.0578 | 0.0065 | 0.0037 | 0.0001 | 0.0798 |
| DeepTextures + render loss (R1) 40k | 0.0402 | 0.0485 | 0.0066 | 0.0038 | 0.0001 | 0.0734 |

## Head-to-head

Materials where the first column is strictly better; mean paired difference
(first - second) over the same materials.

- DeepTextures + render loss (R1) 30k vs DeepTextures grouped (R0a) 30k: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +2.270 dB, P1raw +3.280 dB, P1 l2,1 -0.01878
- DeepTextures + render loss (R1) 30k vs DeepTextures grouped (R0b) 30k: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.424 dB, P1raw +1.806 dB, P1 l2,1 -0.01139
- DeepTextures grouped (R0a) 30k vs DeepTextures grouped (R0b) 30k: P1 1/12 (non-metal 1/9), P1 l2,1 1/12, P1raw 2/12; mean diff P1 -0.846 dB, P1raw -1.473 dB, P1 l2,1 +0.00739
- DeepTextures + render loss (R1) 35k vs DeepTextures grouped (R0a) 35k: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +2.556 dB, P1raw +5.944 dB, P1 l2,1 -0.01645
- DeepTextures + render loss (R1) 35k vs DeepTextures grouped (R0b) 35k: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.531 dB, P1raw +4.054 dB, P1 l2,1 -0.01293
- DeepTextures grouped (R0a) 35k vs DeepTextures grouped (R0b) 35k: P1 4/12 (non-metal 4/9), P1 l2,1 4/12, P1raw 0/12; mean diff P1 -1.024 dB, P1raw -1.891 dB, P1 l2,1 +0.00352
- DeepTextures + render loss (R1) 40k vs DeepTextures grouped (R0a) 40k: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.707 dB, P1raw +3.883 dB, P1 l2,1 -0.01464
- DeepTextures + render loss (R1) 40k vs DeepTextures grouped (R0b) 40k: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.343 dB, P1raw +3.931 dB, P1 l2,1 -0.01415
- DeepTextures grouped (R0a) 40k vs DeepTextures grouped (R0b) 40k: P1 3/12 (non-metal 2/9), P1 l2,1 4/12, P1raw 6/12; mean diff P1 -0.364 dB, P1raw +0.048 dB, P1 l2,1 +0.00049
- DeepTextures grouped (R0a) 30k vs DeepTextures (matsynth_all): P1 1/12 (non-metal 1/9), P1 l2,1 0/12, P1raw 12/12; mean diff P1 -0.586 dB, P1raw +2.985 dB, P1 l2,1 +0.00672
- DeepTextures grouped (R0a) 35k vs DeepTextures (matsynth_all): P1 1/12 (non-metal 1/9), P1 l2,1 0/12, P1raw 3/12; mean diff P1 -1.068 dB, P1raw -0.871 dB, P1 l2,1 +0.00561
- DeepTextures grouped (R0a) 40k vs DeepTextures (matsynth_all): P1 2/12 (non-metal 0/9), P1 l2,1 4/12, P1raw 12/12; mean diff P1 -0.404 dB, P1raw +2.527 dB, P1 l2,1 +0.00478
- DeepTextures grouped (R0b) 30k vs DeepTextures (matsynth_all): P1 6/12 (non-metal 3/9), P1 l2,1 6/12, P1raw 12/12; mean diff P1 +0.260 dB, P1raw +4.458 dB, P1 l2,1 -0.00067
- DeepTextures grouped (R0b) 35k vs DeepTextures (matsynth_all): P1 5/12 (non-metal 3/9), P1 l2,1 6/12, P1raw 12/12; mean diff P1 -0.044 dB, P1raw +1.020 dB, P1 l2,1 +0.00209
- DeepTextures grouped (R0b) 40k vs DeepTextures (matsynth_all): P1 6/12 (non-metal 3/9), P1 l2,1 4/12, P1raw 11/12; mean diff P1 -0.039 dB, P1raw +2.479 dB, P1 l2,1 +0.00429
- DeepTextures + render loss (R1) 30k vs DeepTextures (matsynth_all): P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.684 dB, P1raw +6.265 dB, P1 l2,1 -0.01206
- DeepTextures + render loss (R1) 35k vs DeepTextures (matsynth_all): P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.488 dB, P1raw +5.073 dB, P1 l2,1 -0.01084
- DeepTextures + render loss (R1) 40k vs DeepTextures (matsynth_all): P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.304 dB, P1raw +6.410 dB, P1 l2,1 -0.00986
- DeepTextures (matsynth_all) vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.435 dB, P1raw -6.750 dB, P1 l2,1 +0.00958
- DeepTextures (matsynth_all) vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 1/12, P1raw 0/12; mean diff P1 -0.972 dB, P1raw -5.768 dB, P1 l2,1 +0.00508
- DeepTextures grouped (R0a) 30k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -2.021 dB, P1raw -3.766 dB, P1 l2,1 +0.01629
- DeepTextures grouped (R0a) 30k vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.557 dB, P1raw -2.783 dB, P1 l2,1 +0.01179
- DeepTextures grouped (R0a) 35k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -2.503 dB, P1raw -7.621 dB, P1 l2,1 +0.01519
- DeepTextures grouped (R0a) 35k vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -2.039 dB, P1raw -6.639 dB, P1 l2,1 +0.01068
- DeepTextures grouped (R0a) 40k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.838 dB, P1raw -4.223 dB, P1 l2,1 +0.01436
- DeepTextures grouped (R0a) 40k vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 1/12, P1raw 0/12; mean diff P1 -1.375 dB, P1raw -3.241 dB, P1 l2,1 +0.00985
- DeepTextures grouped (R0b) 30k vs bilinear: P1 1/12 (non-metal 0/9), P1 l2,1 1/12, P1raw 1/12; mean diff P1 -1.175 dB, P1raw -2.292 dB, P1 l2,1 +0.00891
- DeepTextures grouped (R0b) 30k vs MatUp: P1 2/12 (non-metal 0/9), P1 l2,1 3/12, P1raw 1/12; mean diff P1 -0.711 dB, P1raw -1.310 dB, P1 l2,1 +0.00441
- DeepTextures grouped (R0b) 35k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.478 dB, P1raw -5.731 dB, P1 l2,1 +0.01167
- DeepTextures grouped (R0b) 35k vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 1/12, P1raw 0/12; mean diff P1 -1.015 dB, P1raw -4.749 dB, P1 l2,1 +0.00716
- DeepTextures grouped (R0b) 40k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.474 dB, P1raw -4.271 dB, P1 l2,1 +0.01387
- DeepTextures grouped (R0b) 40k vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 2/12, P1raw 0/12; mean diff P1 -1.011 dB, P1raw -3.289 dB, P1 l2,1 +0.00937
- DeepTextures + render loss (R1) 30k vs bilinear: P1 9/12 (non-metal 7/9), P1 l2,1 10/12, P1raw 7/12; mean diff P1 +0.249 dB, P1raw -0.486 dB, P1 l2,1 -0.00248
- DeepTextures + render loss (R1) 30k vs MatUp: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 9/12; mean diff P1 +0.712 dB, P1raw +0.496 dB, P1 l2,1 -0.00699
- DeepTextures + render loss (R1) 35k vs bilinear: P1 8/12 (non-metal 6/9), P1 l2,1 6/12, P1raw 3/12; mean diff P1 +0.053 dB, P1raw -1.677 dB, P1 l2,1 -0.00126
- DeepTextures + render loss (R1) 35k vs MatUp: P1 12/12 (non-metal 9/9), P1 l2,1 11/12, P1raw 5/12; mean diff P1 +0.516 dB, P1raw -0.695 dB, P1 l2,1 -0.00576
- DeepTextures + render loss (R1) 40k vs bilinear: P1 4/12 (non-metal 4/9), P1 l2,1 5/12, P1raw 5/12; mean diff P1 -0.131 dB, P1raw -0.340 dB, P1 l2,1 -0.00028
- DeepTextures + render loss (R1) 40k vs MatUp: P1 8/12 (non-metal 7/9), P1 l2,1 10/12, P1raw 7/12; mean diff P1 +0.332 dB, P1raw +0.642 dB, P1 l2,1 -0.00478

## Per checkpoint step

Mean over the `n` materials at each step, the mean over steps, and the
within-run range (max - min over steps).

### P1 PSNR (off) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 24.622 | 24.140 | 24.804 | 24.522 | 0.664 |
| DeepTextures grouped (R0b) | 25.468 | 25.164 | 25.169 | 25.267 | 0.304 |
| DeepTextures + render loss (R1) | 26.892 | 26.696 | 26.512 | 26.700 | 0.380 |

### P1raw PSNR (raw) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 22.823 | 18.967 | 22.365 | 21.385 | 3.856 |
| DeepTextures grouped (R0b) | 24.296 | 20.857 | 22.317 | 22.490 | 3.439 |
| DeepTextures + render loss (R1) | 26.102 | 24.911 | 26.248 | 25.754 | 1.337 |

### P1 PSNR (off), non-metal (n=9)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures grouped (R0a) | 24.071 | 23.681 | 24.160 | 23.970 | 0.479 |
| DeepTextures grouped (R0b) | 24.841 | 24.675 | 24.673 | 24.730 | 0.168 |
| DeepTextures + render loss (R1) | 26.505 | 26.328 | 26.215 | 26.349 | 0.290 |

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
