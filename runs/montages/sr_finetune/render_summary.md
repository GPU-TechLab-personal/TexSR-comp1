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

Columns: `bilinear` = bilinear, `matup` = MatUp, `dt_matsynth_all_40k` = DeepTextures (matsynth_all) 40k (/workspace/runs/deeptex/eval/matsynth_all_40k), `dt_render_r1_40k` = DeepTextures + render loss (R1) 40k (/workspace/runs/deeptex/eval/render_r1_40k), `bsr_esrgan_zs` = ESRGAN official (zero-shot) (/workspace/runs/sr_ft/eval/esrgan_zs), `bsr_esrgan_ft_30k` = ESRGAN official (fine-tuned) 30k (/workspace/runs/sr_ft/eval/esrgan_ft_30k), `bsr_esrgan_ft_35k` = ESRGAN official (fine-tuned) 35k (/workspace/runs/sr_ft/eval/esrgan_ft_35k), `bsr_esrgan_ft_40k` = ESRGAN official (fine-tuned) 40k (/workspace/runs/sr_ft/eval/esrgan_ft_40k), `kair_swinir_cls_zs` = SwinIR-M classical (zero-shot) (/workspace/runs/sr_ft/eval/swinir_cls_zs), `kair_swinir_cls_ft_30k` = SwinIR-M classical (fine-tuned) 30k (/workspace/runs/sr_ft/eval/swinir_cls_ft_30k), `kair_swinir_cls_ft_35k` = SwinIR-M classical (fine-tuned) 35k (/workspace/runs/sr_ft/eval/swinir_cls_ft_35k), `kair_swinir_cls_ft_40k` = SwinIR-M classical (fine-tuned) 40k (/workspace/runs/sr_ft/eval/swinir_cls_ft_40k), `kair_swinir_real_zs` = SwinIR-M real (zero-shot) (/workspace/runs/sr_ft/eval/swinir_real_zs), `kair_swinir_real_ft_30k` = SwinIR-M real (fine-tuned) 30k (/workspace/runs/sr_ft/eval/swinir_real_ft_30k), `kair_swinir_real_ft_35k` = SwinIR-M real (fine-tuned) 35k (/workspace/runs/sr_ft/eval/swinir_real_ft_35k), `kair_swinir_real_ft_40k` = SwinIR-M real (fine-tuned) 40k (/workspace/runs/sr_ft/eval/swinir_real_ft_40k), `kair_swinir_realL_zs` = SwinIR-L real (zero-shot) (/workspace/runs/sr_ft/eval/swinir_realL_zs).

## P1 - offset-removed maps

| column | n | held-out PSNR | held-out l2,1 | train-100 l2,1 | train-100 PSNR | grazing l2,1 | mid l2,1 | high l2,1 | eval PSNR | eval PSNR 8-bit | normal ° | \|grad\| %GT | n_z<1e-3 | metallic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 12 | 26.64 | 0.0724 | 0.0733 | 26.61 | 0.0854 | 0.0687 | 0.0598 | 27.43 | 27.41 | 6.28 | 35% | 0 | 0.133 |
| MatUp | 12 | 26.18 | 0.0769 | 0.0777 | 26.14 | 0.0899 | 0.0733 | 0.0639 | 27.02 | 27.00 | 6.47 | 33% | 0 | 0.135 |
| DeepTextures (matsynth_all) 40k | 12 | 25.21 | 0.0819 | 0.0831 | 25.14 | 0.0978 | 0.0774 | 0.0680 | 26.16 | 26.15 | 7.35 | 82% | 2 | 0.132 |
| DeepTextures + render loss (R1) 40k | 12 | 26.51 | 0.0721 | 0.0729 | 26.47 | 0.0839 | 0.0688 | 0.0611 | 27.22 | 27.21 | 6.03 | 71% | 2 | 0.133 |
| ESRGAN official (zero-shot) | 12 | 21.91 | 0.1239 | 0.1248 | 21.90 | 0.1362 | 0.1213 | 0.1092 | 22.40 | 22.40 | 9.26 | 174% | 313 | 0.133 |
| ESRGAN official (fine-tuned) 30k | 12 | 24.98 | 0.0908 | 0.0921 | 24.93 | 0.1081 | 0.0863 | 0.0736 | 25.92 | 25.91 | 7.93 | 99% | 3 | 0.133 |
| ESRGAN official (fine-tuned) 35k | 12 | 24.79 | 0.0933 | 0.0946 | 24.75 | 0.1111 | 0.0887 | 0.0750 | 25.77 | 25.76 | 8.19 | 102% | 1 | 0.133 |
| ESRGAN official (fine-tuned) 40k | 12 | 24.72 | 0.0950 | 0.0962 | 24.67 | 0.1132 | 0.0903 | 0.0760 | 25.65 | 25.64 | 8.39 | 104% | 1 | 0.133 |
| SwinIR-M classical (zero-shot) | 12 | 26.87 | 0.0687 | 0.0697 | 26.83 | 0.0820 | 0.0650 | 0.0563 | 27.80 | 27.79 | 6.04 | 63% | 4 | 0.133 |
| SwinIR-M classical (fine-tuned) 30k | 12 | 27.79 | 0.0628 | 0.0636 | 27.74 | 0.0745 | 0.0594 | 0.0520 | 28.67 | 28.66 | 5.39 | 54% | 0 | 0.133 |
| SwinIR-M classical (fine-tuned) 35k | 12 | 27.85 | 0.0626 | 0.0635 | 27.80 | 0.0744 | 0.0593 | 0.0518 | 28.75 | 28.73 | 5.37 | 54% | 0 | 0.133 |
| SwinIR-M classical (fine-tuned) 40k | 12 | 27.83 | 0.0626 | 0.0635 | 27.78 | 0.0744 | 0.0593 | 0.0518 | 28.72 | 28.70 | 5.37 | 54% | 0 | 0.133 |
| SwinIR-M real (zero-shot) | 12 | 23.45 | 0.1016 | 0.1034 | 23.40 | 0.1250 | 0.0954 | 0.0797 | 24.51 | 24.50 | 9.61 | 70% | 339 | 0.133 |
| SwinIR-M real (fine-tuned) 30k | 12 | 24.71 | 0.0898 | 0.0910 | 24.66 | 0.1058 | 0.0854 | 0.0749 | 25.51 | 25.50 | 7.94 | 66% | 40 | 0.133 |
| SwinIR-M real (fine-tuned) 35k | 12 | 24.59 | 0.0906 | 0.0918 | 24.54 | 0.1065 | 0.0863 | 0.0758 | 25.33 | 25.32 | 7.99 | 70% | 54 | 0.133 |
| SwinIR-M real (fine-tuned) 40k | 12 | 24.66 | 0.0903 | 0.0915 | 24.60 | 0.1066 | 0.0859 | 0.0752 | 25.38 | 25.37 | 8.06 | 67% | 51 | 0.133 |
| SwinIR-L real (zero-shot) | 12 | 23.62 | 0.1025 | 0.1042 | 23.59 | 0.1259 | 0.0960 | 0.0809 | 24.53 | 24.52 | 10.01 | 65% | 529 | 0.133 |

Non-metal / metal split:

| column | non-metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° | metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° |
|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 26.23 | 0.0769 | 0.0782 | 7.87 | 3 | 27.87 | 0.0588 | 0.0585 | 1.48 |
| MatUp | 9 | 25.86 | 0.0808 | 0.0821 | 8.12 | 3 | 27.15 | 0.0650 | 0.0647 | 1.52 |
| DeepTextures (matsynth_all) 40k | 9 | 24.77 | 0.0867 | 0.0883 | 9.07 | 3 | 26.52 | 0.0675 | 0.0674 | 2.18 |
| DeepTextures + render loss (R1) 40k | 9 | 26.21 | 0.0756 | 0.0768 | 7.52 | 3 | 27.40 | 0.0617 | 0.0613 | 1.58 |
| ESRGAN official (zero-shot) | 9 | 21.89 | 0.1258 | 0.1273 | 11.60 | 3 | 21.97 | 0.1184 | 0.1174 | 2.26 |
| ESRGAN official (fine-tuned) 30k | 9 | 24.73 | 0.0958 | 0.0975 | 9.91 | 3 | 25.71 | 0.0760 | 0.0756 | 1.98 |
| ESRGAN official (fine-tuned) 35k | 9 | 24.60 | 0.0981 | 0.0999 | 10.24 | 3 | 25.38 | 0.0788 | 0.0785 | 2.04 |
| ESRGAN official (fine-tuned) 40k | 9 | 24.51 | 0.1002 | 0.1020 | 10.51 | 3 | 25.34 | 0.0794 | 0.0790 | 2.04 |
| SwinIR-M classical (zero-shot) | 9 | 26.47 | 0.0730 | 0.0744 | 7.55 | 3 | 28.07 | 0.0558 | 0.0556 | 1.49 |
| SwinIR-M classical (fine-tuned) 30k | 9 | 27.51 | 0.0663 | 0.0675 | 6.71 | 3 | 28.65 | 0.0522 | 0.0520 | 1.43 |
| SwinIR-M classical (fine-tuned) 35k | 9 | 27.58 | 0.0661 | 0.0673 | 6.68 | 3 | 28.65 | 0.0522 | 0.0520 | 1.43 |
| SwinIR-M classical (fine-tuned) 40k | 9 | 27.56 | 0.0661 | 0.0673 | 6.68 | 3 | 28.65 | 0.0522 | 0.0520 | 1.43 |
| SwinIR-M real (zero-shot) | 9 | 22.81 | 0.1102 | 0.1128 | 12.19 | 3 | 25.37 | 0.0757 | 0.0753 | 1.88 |
| SwinIR-M real (fine-tuned) 30k | 9 | 24.43 | 0.0949 | 0.0967 | 10.01 | 3 | 25.54 | 0.0743 | 0.0738 | 1.75 |
| SwinIR-M real (fine-tuned) 35k | 9 | 24.37 | 0.0955 | 0.0973 | 10.07 | 3 | 25.27 | 0.0760 | 0.0756 | 1.77 |
| SwinIR-M real (fine-tuned) 40k | 9 | 24.37 | 0.0956 | 0.0974 | 10.15 | 3 | 25.52 | 0.0743 | 0.0738 | 1.78 |
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
| ESRGAN official (fine-tuned) 30k | 0.0458 | 0.0626 | 0.0066 | 0.0048 | 0.0001 | 0.0908 |
| ESRGAN official (fine-tuned) 35k | 0.0462 | 0.0649 | 0.0068 | 0.0050 | 0.0001 | 0.0933 |
| ESRGAN official (fine-tuned) 40k | 0.0468 | 0.0664 | 0.0068 | 0.0048 | 0.0001 | 0.0950 |
| SwinIR-M classical (zero-shot) | 0.0337 | 0.0479 | 0.0050 | 0.0040 | 0.0001 | 0.0687 |
| SwinIR-M classical (fine-tuned) 30k | 0.0319 | 0.0433 | 0.0048 | 0.0036 | 0.0000 | 0.0628 |
| SwinIR-M classical (fine-tuned) 35k | 0.0319 | 0.0432 | 0.0048 | 0.0036 | 0.0000 | 0.0626 |
| SwinIR-M classical (fine-tuned) 40k | 0.0319 | 0.0432 | 0.0048 | 0.0036 | 0.0000 | 0.0626 |
| SwinIR-M real (zero-shot) | 0.0460 | 0.0742 | 0.0081 | 0.0055 | 0.0002 | 0.1016 |
| SwinIR-M real (fine-tuned) 30k | 0.0458 | 0.0618 | 0.0075 | 0.0044 | 0.0001 | 0.0898 |
| SwinIR-M real (fine-tuned) 35k | 0.0464 | 0.0620 | 0.0076 | 0.0048 | 0.0001 | 0.0906 |
| SwinIR-M real (fine-tuned) 40k | 0.0457 | 0.0624 | 0.0075 | 0.0046 | 0.0001 | 0.0903 |
| SwinIR-L real (zero-shot) | 0.0448 | 0.0764 | 0.0069 | 0.0055 | 0.0001 | 0.1025 |

## P1raw - raw maps

| column | n | held-out PSNR | held-out l2,1 | train-100 l2,1 | train-100 PSNR | grazing l2,1 | mid l2,1 | high l2,1 | eval PSNR | eval PSNR 8-bit | normal ° | \|grad\| %GT | n_z<1e-3 | metallic |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 12 | 26.59 | 0.0729 | 0.0737 | 26.56 | 0.0857 | 0.0693 | 0.0604 | 27.32 | 27.30 | 6.33 | 34% | 0 | 0.133 |
| MatUp | 12 | 25.61 | 0.0787 | 0.0795 | 25.57 | 0.0908 | 0.0751 | 0.0675 | 26.37 | 26.36 | 6.54 | 33% | 0 | 0.141 |
| DeepTextures (matsynth_all) 40k | 12 | 19.84 | 0.1330 | 0.1360 | 19.63 | 0.1625 | 0.1227 | 0.1210 | 21.32 | 21.32 | 13.26 | 83% | 2 | 0.128 |
| DeepTextures + render loss (R1) 40k | 12 | 26.25 | 0.0734 | 0.0742 | 26.21 | 0.0849 | 0.0702 | 0.0628 | 26.92 | 26.91 | 6.09 | 70% | 1 | 0.134 |
| ESRGAN official (zero-shot) | 12 | 21.59 | 0.1257 | 0.1266 | 21.59 | 0.1375 | 0.1231 | 0.1124 | 22.05 | 22.04 | 9.56 | 175% | 337 | 0.133 |
| ESRGAN official (fine-tuned) 30k | 12 | 24.65 | 0.0921 | 0.0934 | 24.59 | 0.1095 | 0.0876 | 0.0746 | 25.68 | 25.67 | 7.99 | 98% | 2 | 0.134 |
| ESRGAN official (fine-tuned) 35k | 12 | 24.42 | 0.0947 | 0.0960 | 24.37 | 0.1126 | 0.0902 | 0.0762 | 25.43 | 25.43 | 8.26 | 103% | 1 | 0.135 |
| ESRGAN official (fine-tuned) 40k | 12 | 24.47 | 0.0961 | 0.0974 | 24.42 | 0.1144 | 0.0915 | 0.0769 | 25.44 | 25.43 | 8.46 | 104% | 1 | 0.134 |
| SwinIR-M classical (zero-shot) | 12 | 26.88 | 0.0685 | 0.0694 | 26.84 | 0.0818 | 0.0648 | 0.0561 | 27.81 | 27.80 | 6.02 | 62% | 3 | 0.133 |
| SwinIR-M classical (fine-tuned) 30k | 12 | 27.80 | 0.0627 | 0.0635 | 27.75 | 0.0744 | 0.0593 | 0.0519 | 28.68 | 28.66 | 5.39 | 53% | 0 | 0.133 |
| SwinIR-M classical (fine-tuned) 35k | 12 | 27.86 | 0.0625 | 0.0633 | 27.81 | 0.0743 | 0.0592 | 0.0517 | 28.75 | 28.73 | 5.36 | 54% | 0 | 0.133 |
| SwinIR-M classical (fine-tuned) 40k | 12 | 27.84 | 0.0625 | 0.0634 | 27.79 | 0.0743 | 0.0592 | 0.0517 | 28.72 | 28.71 | 5.37 | 54% | 0 | 0.133 |
| SwinIR-M real (zero-shot) | 12 | 21.65 | 0.1122 | 0.1144 | 21.53 | 0.1373 | 0.1044 | 0.0952 | 22.74 | 22.74 | 11.03 | 70% | 327 | 0.135 |
| SwinIR-M real (fine-tuned) 30k | 12 | 24.31 | 0.0913 | 0.0924 | 24.26 | 0.1070 | 0.0869 | 0.0767 | 25.14 | 25.13 | 8.06 | 66% | 84 | 0.135 |
| SwinIR-M real (fine-tuned) 35k | 12 | 24.32 | 0.0916 | 0.0928 | 24.28 | 0.1073 | 0.0874 | 0.0772 | 25.07 | 25.06 | 8.11 | 71% | 84 | 0.136 |
| SwinIR-M real (fine-tuned) 40k | 12 | 24.37 | 0.0912 | 0.0925 | 24.32 | 0.1073 | 0.0869 | 0.0766 | 25.10 | 25.09 | 8.19 | 68% | 68 | 0.132 |
| SwinIR-L real (zero-shot) | 12 | 22.17 | 0.1098 | 0.1116 | 22.14 | 0.1348 | 0.1017 | 0.0907 | 23.19 | 23.18 | 11.05 | 66% | 528 | 0.135 |

Non-metal / metal split:

| column | non-metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° | metal n | held-out PSNR | held-out l2,1 | train-100 l2,1 | normal ° |
|---|---|---|---|---|---|---|---|---|---|---|
| bilinear | 9 | 26.18 | 0.0775 | 0.0787 | 7.95 | 3 | 27.81 | 0.0591 | 0.0588 | 1.48 |
| MatUp | 9 | 24.91 | 0.0844 | 0.0856 | 8.21 | 3 | 27.69 | 0.0614 | 0.0610 | 1.52 |
| DeepTextures (matsynth_all) 40k | 9 | 19.72 | 0.1317 | 0.1348 | 14.71 | 3 | 20.19 | 0.1370 | 0.1395 | 8.93 |
| DeepTextures + render loss (R1) 40k | 9 | 25.92 | 0.0769 | 0.0781 | 7.56 | 3 | 27.24 | 0.0629 | 0.0626 | 1.66 |
| ESRGAN official (zero-shot) | 9 | 21.54 | 0.1273 | 0.1288 | 11.86 | 3 | 21.77 | 0.1210 | 0.1201 | 2.66 |
| ESRGAN official (fine-tuned) 30k | 9 | 24.34 | 0.0971 | 0.0989 | 9.95 | 3 | 25.57 | 0.0771 | 0.0768 | 2.10 |
| ESRGAN official (fine-tuned) 35k | 9 | 24.12 | 0.0999 | 0.1017 | 10.33 | 3 | 25.35 | 0.0790 | 0.0786 | 2.06 |
| ESRGAN official (fine-tuned) 40k | 9 | 24.19 | 0.1016 | 0.1034 | 10.60 | 3 | 25.33 | 0.0796 | 0.0792 | 2.04 |
| SwinIR-M classical (zero-shot) | 9 | 26.48 | 0.0728 | 0.0742 | 7.53 | 3 | 28.08 | 0.0555 | 0.0553 | 1.48 |
| SwinIR-M classical (fine-tuned) 30k | 9 | 27.52 | 0.0662 | 0.0674 | 6.71 | 3 | 28.66 | 0.0520 | 0.0518 | 1.43 |
| SwinIR-M classical (fine-tuned) 35k | 9 | 27.59 | 0.0660 | 0.0672 | 6.67 | 3 | 28.66 | 0.0521 | 0.0519 | 1.43 |
| SwinIR-M classical (fine-tuned) 40k | 9 | 27.57 | 0.0660 | 0.0672 | 6.68 | 3 | 28.66 | 0.0520 | 0.0518 | 1.43 |
| SwinIR-M real (zero-shot) | 9 | 21.07 | 0.1183 | 0.1211 | 13.25 | 3 | 23.37 | 0.0939 | 0.0944 | 4.36 |
| SwinIR-M real (fine-tuned) 30k | 9 | 24.00 | 0.0963 | 0.0980 | 10.12 | 3 | 25.25 | 0.0762 | 0.0757 | 1.88 |
| SwinIR-M real (fine-tuned) 35k | 9 | 24.07 | 0.0964 | 0.0982 | 10.19 | 3 | 25.07 | 0.0772 | 0.0767 | 1.89 |
| SwinIR-M real (fine-tuned) 40k | 9 | 24.09 | 0.0962 | 0.0980 | 10.28 | 3 | 25.23 | 0.0764 | 0.0759 | 1.92 |
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
| ESRGAN official (fine-tuned) 30k | 0.0473 | 0.0633 | 0.0067 | 0.0047 | 0.0001 | 0.0921 |
| ESRGAN official (fine-tuned) 35k | 0.0477 | 0.0655 | 0.0068 | 0.0047 | 0.0001 | 0.0947 |
| ESRGAN official (fine-tuned) 40k | 0.0479 | 0.0668 | 0.0068 | 0.0047 | 0.0001 | 0.0961 |
| SwinIR-M classical (zero-shot) | 0.0337 | 0.0478 | 0.0050 | 0.0035 | 0.0000 | 0.0685 |
| SwinIR-M classical (fine-tuned) 30k | 0.0318 | 0.0433 | 0.0048 | 0.0033 | 0.0000 | 0.0627 |
| SwinIR-M classical (fine-tuned) 35k | 0.0318 | 0.0431 | 0.0048 | 0.0033 | 0.0000 | 0.0625 |
| SwinIR-M classical (fine-tuned) 40k | 0.0318 | 0.0431 | 0.0048 | 0.0033 | 0.0000 | 0.0625 |
| SwinIR-M real (zero-shot) | 0.0484 | 0.0897 | 0.0082 | 0.0044 | 0.0002 | 0.1122 |
| SwinIR-M real (fine-tuned) 30k | 0.0481 | 0.0628 | 0.0079 | 0.0047 | 0.0001 | 0.0913 |
| SwinIR-M real (fine-tuned) 35k | 0.0482 | 0.0630 | 0.0080 | 0.0048 | 0.0001 | 0.0916 |
| SwinIR-M real (fine-tuned) 40k | 0.0470 | 0.0635 | 0.0080 | 0.0044 | 0.0001 | 0.0912 |
| SwinIR-L real (zero-shot) | 0.0462 | 0.0871 | 0.0071 | 0.0045 | 0.0001 | 0.1098 |

## Head-to-head

Materials where the first column is strictly better; mean paired difference
(first - second) over the same materials.

- SwinIR-M real (zero-shot) vs SwinIR-L real (zero-shot): P1 5/12 (non-metal 4/9), P1 l2,1 6/12, P1raw 6/12; mean diff P1 -0.168 dB, P1raw -0.524 dB, P1 l2,1 -0.00096
- ESRGAN official (fine-tuned) 40k vs DeepTextures (matsynth_all) 40k: P1 4/12 (non-metal 3/9), P1 l2,1 5/12, P1raw 9/12; mean diff P1 -0.492 dB, P1raw +4.639 dB, P1 l2,1 +0.01303
- SwinIR-M classical (fine-tuned) 40k vs DeepTextures (matsynth_all) 40k: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +2.624 dB, P1raw +8.004 dB, P1 l2,1 -0.01931
- SwinIR-M real (fine-tuned) 40k vs DeepTextures (matsynth_all) 40k: P1 3/12 (non-metal 2/9), P1 l2,1 3/12, P1raw 12/12; mean diff P1 -0.552 dB, P1raw +4.537 dB, P1 l2,1 +0.00832
- SwinIR-M real (fine-tuned) 40k vs DeepTextures + render loss (R1) 40k: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 1/12; mean diff P1 -1.856 dB, P1raw -1.876 dB, P1 l2,1 +0.01818
- DeepTextures + render loss (R1) 40k vs DeepTextures (matsynth_all) 40k: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.304 dB, P1raw +6.413 dB, P1 l2,1 -0.00986
- ESRGAN official (fine-tuned) 30k vs ESRGAN official (zero-shot): P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +3.068 dB, P1raw +3.055 dB, P1 l2,1 -0.03312
- ESRGAN official (fine-tuned) 35k vs ESRGAN official (zero-shot): P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +2.884 dB, P1raw +2.830 dB, P1 l2,1 -0.03063
- ESRGAN official (fine-tuned) 40k vs ESRGAN official (zero-shot): P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +2.807 dB, P1raw +2.880 dB, P1 l2,1 -0.02898
- SwinIR-M classical (fine-tuned) 30k vs SwinIR-M classical (zero-shot): P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +0.922 dB, P1raw +0.921 dB, P1 l2,1 -0.00596
- SwinIR-M classical (fine-tuned) 35k vs SwinIR-M classical (zero-shot): P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +0.977 dB, P1raw +0.977 dB, P1 l2,1 -0.00611
- SwinIR-M classical (fine-tuned) 40k vs SwinIR-M classical (zero-shot): P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +0.959 dB, P1raw +0.958 dB, P1 l2,1 -0.00610
- SwinIR-M real (fine-tuned) 30k vs SwinIR-M real (zero-shot): P1 9/12 (non-metal 7/9), P1 l2,1 9/12, P1raw 10/12; mean diff P1 +1.253 dB, P1raw +2.668 dB, P1 l2,1 -0.01179
- SwinIR-M real (fine-tuned) 35k vs SwinIR-M real (zero-shot): P1 9/12 (non-metal 7/9), P1 l2,1 9/12, P1raw 10/12; mean diff P1 +1.139 dB, P1raw +2.677 dB, P1 l2,1 -0.01093
- SwinIR-M real (fine-tuned) 40k vs SwinIR-M real (zero-shot): P1 9/12 (non-metal 7/9), P1 l2,1 9/12, P1raw 10/12; mean diff P1 +1.201 dB, P1raw +2.726 dB, P1 l2,1 -0.01130
- DeepTextures (matsynth_all) 40k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.435 dB, P1raw -6.753 dB, P1 l2,1 +0.00958
- DeepTextures (matsynth_all) 40k vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 1/12, P1raw 0/12; mean diff P1 -0.972 dB, P1raw -5.771 dB, P1 l2,1 +0.00508
- DeepTextures + render loss (R1) 40k vs bilinear: P1 4/12 (non-metal 4/9), P1 l2,1 5/12, P1raw 5/12; mean diff P1 -0.131 dB, P1raw -0.340 dB, P1 l2,1 -0.00028
- DeepTextures + render loss (R1) 40k vs MatUp: P1 8/12 (non-metal 7/9), P1 l2,1 10/12, P1raw 7/12; mean diff P1 +0.332 dB, P1raw +0.642 dB, P1 l2,1 -0.00478
- ESRGAN official (zero-shot) vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -4.734 dB, P1raw -4.994 dB, P1 l2,1 +0.05159
- ESRGAN official (zero-shot) vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 1/12; mean diff P1 -4.271 dB, P1raw -4.012 dB, P1 l2,1 +0.04709
- ESRGAN official (fine-tuned) 30k vs bilinear: P1 1/12 (non-metal 1/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.667 dB, P1raw -1.939 dB, P1 l2,1 +0.01847
- ESRGAN official (fine-tuned) 30k vs MatUp: P1 1/12 (non-metal 1/9), P1 l2,1 2/12, P1raw 1/12; mean diff P1 -1.204 dB, P1raw -0.957 dB, P1 l2,1 +0.01397
- ESRGAN official (fine-tuned) 35k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.850 dB, P1raw -2.164 dB, P1 l2,1 +0.02096
- ESRGAN official (fine-tuned) 35k vs MatUp: P1 1/12 (non-metal 1/9), P1 l2,1 2/12, P1raw 1/12; mean diff P1 -1.387 dB, P1raw -1.182 dB, P1 l2,1 +0.01646
- ESRGAN official (fine-tuned) 40k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.927 dB, P1raw -2.114 dB, P1 l2,1 +0.02261
- ESRGAN official (fine-tuned) 40k vs MatUp: P1 1/12 (non-metal 1/9), P1 l2,1 3/12, P1raw 1/12; mean diff P1 -1.464 dB, P1raw -1.132 dB, P1 l2,1 +0.01811
- SwinIR-M classical (zero-shot) vs bilinear: P1 8/12 (non-metal 6/9), P1 l2,1 10/12, P1raw 10/12; mean diff P1 +0.229 dB, P1raw +0.293 dB, P1 l2,1 -0.00363
- SwinIR-M classical (zero-shot) vs MatUp: P1 10/12 (non-metal 8/9), P1 l2,1 12/12, P1raw 10/12; mean diff P1 +0.693 dB, P1raw +1.275 dB, P1 l2,1 -0.00813
- SwinIR-M classical (fine-tuned) 30k vs bilinear: P1 11/12 (non-metal 8/9), P1 l2,1 11/12, P1raw 11/12; mean diff P1 +1.151 dB, P1raw +1.214 dB, P1 l2,1 -0.00959
- SwinIR-M classical (fine-tuned) 30k vs MatUp: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.614 dB, P1raw +2.196 dB, P1 l2,1 -0.01409
- SwinIR-M classical (fine-tuned) 35k vs bilinear: P1 11/12 (non-metal 8/9), P1 l2,1 11/12, P1raw 11/12; mean diff P1 +1.206 dB, P1raw +1.270 dB, P1 l2,1 -0.00974
- SwinIR-M classical (fine-tuned) 35k vs MatUp: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.669 dB, P1raw +2.252 dB, P1 l2,1 -0.01424
- SwinIR-M classical (fine-tuned) 40k vs bilinear: P1 11/12 (non-metal 8/9), P1 l2,1 11/12, P1raw 11/12; mean diff P1 +1.189 dB, P1raw +1.252 dB, P1 l2,1 -0.00973
- SwinIR-M classical (fine-tuned) 40k vs MatUp: P1 12/12 (non-metal 9/9), P1 l2,1 12/12, P1raw 12/12; mean diff P1 +1.652 dB, P1raw +2.234 dB, P1 l2,1 -0.01423
- SwinIR-M real (zero-shot) vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -3.188 dB, P1raw -4.942 dB, P1 l2,1 +0.02920
- SwinIR-M real (zero-shot) vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -2.725 dB, P1raw -3.960 dB, P1 l2,1 +0.02470
- SwinIR-M real (fine-tuned) 30k vs bilinear: P1 1/12 (non-metal 1/9), P1 l2,1 0/12, P1raw 1/12; mean diff P1 -1.936 dB, P1raw -2.274 dB, P1 l2,1 +0.01741
- SwinIR-M real (fine-tuned) 30k vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 1/12, P1raw 1/12; mean diff P1 -1.472 dB, P1raw -1.292 dB, P1 l2,1 +0.01291
- SwinIR-M real (fine-tuned) 35k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -2.049 dB, P1raw -2.265 dB, P1 l2,1 +0.01827
- SwinIR-M real (fine-tuned) 35k vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 1/12, P1raw 1/12; mean diff P1 -1.586 dB, P1raw -1.283 dB, P1 l2,1 +0.01377
- SwinIR-M real (fine-tuned) 40k vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -1.987 dB, P1raw -2.216 dB, P1 l2,1 +0.01790
- SwinIR-M real (fine-tuned) 40k vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 1/12, P1raw 1/12; mean diff P1 -1.524 dB, P1raw -1.234 dB, P1 l2,1 +0.01340
- SwinIR-L real (zero-shot) vs bilinear: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -3.021 dB, P1raw -4.418 dB, P1 l2,1 +0.03017
- SwinIR-L real (zero-shot) vs MatUp: P1 0/12 (non-metal 0/9), P1 l2,1 0/12, P1raw 0/12; mean diff P1 -2.558 dB, P1raw -3.436 dB, P1 l2,1 +0.02566

## Per checkpoint step

Mean over the `n` materials at each step, the mean over steps, and the
within-run range (max - min over steps).

### P1 PSNR (off) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 25.208 | 25.208 | - |
| DeepTextures + render loss (R1) | - | - | 26.512 | 26.512 | - |
| ESRGAN official (fine-tuned) | 24.976 | 24.793 | 24.716 | 24.828 | 0.260 |
| SwinIR-M classical (fine-tuned) | 27.794 | 27.849 | 27.831 | 27.825 | 0.055 |
| SwinIR-M real (fine-tuned) | 24.707 | 24.593 | 24.655 | 24.652 | 0.114 |

### P1raw PSNR (raw) (n=12)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 19.835 | 19.835 | - |
| DeepTextures + render loss (R1) | - | - | 26.248 | 26.248 | - |
| ESRGAN official (fine-tuned) | 24.649 | 24.425 | 24.474 | 24.516 | 0.224 |
| SwinIR-M classical (fine-tuned) | 27.803 | 27.858 | 27.840 | 27.834 | 0.056 |
| SwinIR-M real (fine-tuned) | 24.314 | 24.323 | 24.373 | 24.336 | 0.059 |

### P1 PSNR (off), non-metal (n=9)

| arm | 30k | 35k | 40k | mean | range |
|---|---|---|---|---|---|
| DeepTextures (matsynth_all) | - | - | 24.771 | 24.771 | - |
| DeepTextures + render loss (R1) | - | - | 26.215 | 26.215 | - |
| ESRGAN official (fine-tuned) | 24.730 | 24.598 | 24.507 | 24.612 | 0.224 |
| SwinIR-M classical (fine-tuned) | 27.508 | 27.582 | 27.558 | 27.550 | 0.074 |
| SwinIR-M real (fine-tuned) | 24.430 | 24.366 | 24.368 | 24.388 | 0.064 |

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
