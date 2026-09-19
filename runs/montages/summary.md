# MatUp vs bilinear

`matup / bilinear` on mean gradient magnitude, over the valid region.
bilinear is not just a baseline here - it is MatUp's own initialisation
(`dataLoader.py` resizes every map INTER_LINEAR x4 before optimising), so
**below 1.0 means the optimisation removed detail it started with.**

`n/a` marks a constant map with no gradient to divide by (metallic is all
zero on non-metals). `height` is reconstructed by scripts/matup/matup_height.py,
not produced by MatUp - it adds high frequency by construction, so its
column says nothing about MatUp's quality.

`<material>_deeptex.png` is a separate 5x2 sheet - LR / bilinear / MatUp /
DeepTex / GT over baseColor and render. DeepTextures is a colour model:
its released code has no notion of a map type, and the 300 training files
that ship with it are colour-dominated, so the other four maps are not
part of that comparison. See docs/DEEPTEX-DEVIATIONS.md 0.5-0.8.

| material | baseColor | normal | roughness | metallic | height |
|---|---|---|---|---|---|
| 0559_brick_uneven_stones | 0.879 | 0.923 | 0.968 | n/a | 1.424 |
| 0844_syntetic_foam | 1.248 | 0.912 | n/a | n/a | 1.645 |
| 1587_cobblestones | 0.951 | 0.906 | 4.092 | n/a | 1.207 |
| 2119_corten_steel | 1.097 | 0.891 | 10.606 | 0.978 | 1.102 |
| acg_concrete_026 | 0.709 | 0.739 | 0.894 | n/a | 1.254 |
| acg_metal_026 | 0.992 | 0.697 | 0.954 | 0.990 | 1.010 |
| acg_plastic_009 | 0.440 | 0.601 | 0.551 | n/a | 1.053 |
| acg_roofing_tiles_006 | 1.831 | 1.175 | 2.463 | n/a | 1.426 |
| acg_wood_floor_060 | 1.003 | 0.577 | 1.069 | n/a | 1.014 |
| tc_metal_029 | 1.018 | 0.801 | 1.027 | 0.979 | 1.501 |
| th_brick_wall_005 | 0.988 | 0.940 | 11.440 | n/a | 1.893 |
| th_brown_mud_leaves_001 | 0.594 | 0.683 | 0.547 | n/a | 1.489 |
