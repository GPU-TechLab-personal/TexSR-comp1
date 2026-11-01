# variants/sr_finetune/

The official ESRGAN (BasicSR v1.4.2) and SwinIR (KAIR training, the SwinIR release
for inference) fine-tuned on the MatSynth maps, and run zero-shot, as arms beside
DeepTextures, MatUp and bilinear. Results in `runs/montages/sr_finetune/`.

The official code is only imported: `external/BasicSR`, `external/KAIR`,
`external/SwinIR` are submodules at pinned commits (`util/paths.py` `PINS`), never
pip-installed, copied or edited. What runs around it is in this folder.

## Arms

| tag | code | starts from | data | losses (official) |
|---|---|---|---|---|
| `esrgan_zs` / `esrgan_ft` | BasicSR `ESRGANModel` | `ESRGAN_SRx4_DF2KOST_official` (`params`); D scratch | area pairs, gt 128, hflip/rot | L1 1e-2, VGG19 conv5_4 1, GAN 5e-3, EMA 0.999 |
| `swinir_cls_zs` / `swinir_cls_ft` | KAIR `main_train_psnr.py`, `plain` | `001_classicalSR_DF2K_s64w8_SwinIR-M_x4` (`params`) | area pairs, H 256 | L1, EMA 0.999 |
| `swinir_real_zs` / `swinir_real_ft` | KAIR `main_train_gan.py`, `gan` | `003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN` (`params_ema`); D scratch | HR only, BSRGAN degradation, H 256 | L1 1, VGG 1, GAN 0.1, EMA 0.999 |
| `swinir_realL_zs` | inference only | `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN` (MatUp's copy) | - | - |

All fine-tuned arms: 40k iterations, learning rate halved at 20k and 30k, a
checkpoint every 5k, no validation during training, the five maps 1:1.
Names, reference arms and comparisons: `variants/arms.json`.

## Contents

| file | what it does |
|---|---|
| `nets.py` | the official networks (constructor arguments copied from the official inference code), the weight files, and which weight each arm tag starts from / its trainer writes |
| `weights.py` | `--check`: every weight present, sha256, key -> `runs/sr_ft/weights/manifest.json`; `--kair-init`: the official weights as KAIR's `0_G.pth` / `0_E.pth` |
| `write_arm.py` | one arm (zero-shot or a checkpoint) -> the arm folder every evaluator reads, through `util/arm_output.py` |
| `prep_pairs.py` | hard-links the five maps' existing MatSynth pairs into `runs/sr_ft/pairs/{HR,LRx4}` (host) |
| `esrgan/esrgan_ft.yml`, `esrgan/train.py` | the official ESRGAN config with the changed lines marked; BasicSR's own `train_pipeline` |
| `swinir/swinir_cls_ft.json`, `swinir/swinir_real_ft.json`, `swinir/kair_train.py` | the official KAIR configs with the changed lines marked; KAIR's own `main_train_*.py` via runpy, plus the starting weights and an end |
| `drift.py` | colour drift per checkpoint: the torch driver of `util/report/drift.py` |
| `run.ps1` | `preflight`, `setup`, `prep`, `gate`, `train`, `infer`, `eval` |
| `gates/` | the checks below, each a json under `runs/sr_ft/gate/` |

`gates/` (`run.ps1 -Stage gate -Gate <name>`):

| gate | where | checks |
|---|---|---|
| `env` | host + srft | submodules at their pins and clean; torch 2.0.0, scipy 1.11.4 (`interp2d` callable, `scipy.finfo` present), CUDA; no installed package named `basicsr` / `models` / `utils` / `data`; every module the trainers use imported from its submodule; one BSRGAN degradation |
| `weights` | srft | sha256 (publisher's prefix where the name has one); strict load of every weight into its official network + a forward; `--kair-init` output strictly loaded by KAIR's own `load_network`; VGG19 |
| `parity` | srft | `write_arm.py`'s predict against the official inference scripts (`inference_esrgan.py`, `main_test_swinir.py`) on the evaluation inputs: within 1 LSB (measured: 0) |
| `registry` | host | `arms.json` with the SR arms answers every existing question (names, keys, pairs for every subset of the old arms) exactly as before; new entries consistent |
| `data` | dataprep | 14,520 pairs, HR / LRx4 name lists identical, 512 / 128 8-bit RGB, bytes = source, no evaluation material or blend of one |
| `zeroshot` | all | the whole eval chain on the four zero-shot arms |
| `smoke` | all | each trainer 200 iterations, then the same command to 300: resumes at 200, keeps the halved rate; speed, VRAM; its step-300 checkpoint through infer and eval (drift included), all under `runs/variants_verify/sr_ft_smoke/` |

## How it works

    MatSynth pairs (deeptex/matsynth) --prep_pairs--> runs/sr_ft/pairs/{HR,LRx4}
    official weights (nets.WEIGHTS)   --train-------> runs/sr_ft/basicsr/experiments/esrgan_ft/models/net_g_<N>.pth
                                                      runs/sr_ft/kair/swinir_{cls,real}_ft/models/<N>_E.pth
    weights / checkpoints             --write_arm---> runs/sr_ft/eval/<tag>[_<NN>k]/<material>/<map>.png + sr_manifest.json
    arm folders + DeepTextures refs   --eval--------> render / map / perceptual / SR-only / drift -> runs/montages/sr_finetune/

Every map goes in and out the way the DeepTextures arms do: `map_encode.to_rgb8`
(a 1-channel map replicated to three channels, 16-bit height reduced to 8 bit) and
`arm_output.save_sr(..., "mean")` (a 1-channel map is the mean of the three output
channels; their spread is `channel_spread` in the manifest). Training reads the
same encoded pairs. The evaluators are the existing ones, unmodified, given
`--arm TAG=DIR`.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File variants/sr_finetune/run.ps1 -Stage setup
powershell -ExecutionPolicy Bypass -File variants/sr_finetune/run.ps1 -Stage prep
powershell -ExecutionPolicy Bypass -File variants/sr_finetune/run.ps1 -Stage gate -Gate smoke
# training, one at a time (gate smoke on the RTX 3080 Ti, 2026-11-02: ~2.9 h, ~3.0 h, ~4.6 h)
powershell -ExecutionPolicy Bypass -File variants/sr_finetune/run.ps1 -Stage train -Arm esrgan
powershell -ExecutionPolicy Bypass -File variants/sr_finetune/run.ps1 -Stage train -Arm swinir_cls
powershell -ExecutionPolicy Bypass -File variants/sr_finetune/run.ps1 -Stage train -Arm swinir_real
# arm folders for 30k / 35k / 40k, then everything into runs/montages/sr_finetune/
powershell -ExecutionPolicy Bypass -File variants/sr_finetune/run.ps1 -Stage infer -Arms esrgan_ft,swinir_cls_ft,swinir_real_ft -Steps 30000,35000,40000
powershell -ExecutionPolicy Bypass -File variants/sr_finetune/run.ps1 -Stage eval
```

A stopped training run continues where it stopped when the same `-Stage train`
command is run again. `-Stage eval` needs the zero-shot folders (`-Stage infer`
with the default `-Arms` writes them; they are skipped when complete) and the
DeepTextures references `runs/deeptex/eval/matsynth_all_40k`, `render_r1_40k`.

## Verdict (fixed before training)

1. `*_ft` against `*_zs`; 2. `esrgan_ft` against DeepTextures `matsynth_all`;
3. classical against real-world, and against R1, MatUp, bilinear. Metrics: map
PSNR / SSIM (and after offset removal), normal angle, |grad| %GT, drift; render
P1, LPIPS / FLIP, baseColor LPIPS. "Better" needs all three: a difference of at
least tau = |R0a - R0b| for that metric (3-checkpoint mean); the same sign at 30k,
35k and 40k (dropped for zero-shot and DeepTextures, which have one checkpoint);
9 or more of the 12 materials.

## Known behaviour

Deviations from the official recipes (on purpose):

* LR is area-decimated (official: bicubic) - the evaluation LR is area.
* 40k iterations, one GPU. Batch: ESRGAN 16 (official); SwinIR 4, the official
  per-GPU batch (the official total is 4 x 8 GPUs). Batch 8 does not fit 12 GB:
  the card spills into shared memory and a step takes ~30 s.
* SwinIR classical fine-tunes at lr 1e-4: the official file's own note says the x4
  model was fine-tuned from x2 with the rate halved (its value 2e-4 is for x2 from
  scratch). `img_size` 64 / `H_size` 256: the released x4 model is s64, and its
  attention-mask buffers are in the state_dict at that size.
* ESRGAN starts from the released GAN generator (official: the authors' PSNR model).
  It has only `params`; `net_g_ema` takes the same (BasicSR falls back to `params`).
* SwinIR real-world: only `params_ema` was released, so G and E both start from it.
  D starts from scratch, and its optimizer state is not saved (official
  `D_optimizer_reuse: false`), so a resumed run restarts D's Adam.
* The real-world arm trains on BSRGAN-degraded HR and is evaluated on area LR (the
  official setting; a distribution gap by design).
* `manual_seed` 0 added to both KAIR configs (official: random).
* flip / rot augmentation ignores what a normal map's channels mean (as DeepTextures).
* 1-channel maps go in replicated and come out as the channel mean; height is 8-bit
  in and out while its GT is 16-bit. Stored normals are used as they are (not unit
  length: mean length 0.95-0.97). 2,439 of 2,904 metallic maps are constant.

Official-code behaviour this folder works around, without editing it:

* **BasicSR `--auto_resume` looks relative to the working directory**
  (`basicsr/train.py:71`), while it saves under `root_path`: from `/workspace` a
  resume silently starts over and archives the run. `esrgan/train.py` chdirs to the
  root first.
* **BasicSR's source tree has no `basicsr/version.py`** (`setup.py` writes it):
  `paths.use_basicsr()` registers the pinned version in `sys.modules`.
* **KAIR ignores `pretrained_netG`** (`main_train_*.py` call `find_last_checkpoint`
  without it): the official weights go in as `models/0_G.pth`, `0_E.pth`.
* **KAIR's loop never ends** (`for epoch in range(1000000)`): `kair_train.py` wraps
  `define_Model` so the model exits 0 after saving step >= `--total-iter`.
* **KAIR reads `test_loader` at every `checkpoint_test` step** even with no test
  set: `checkpoint_test` is 1e9.
* **KAIR strips everything after `//` on each line** of its json, inside strings too.
* **KAIR's BSRGAN degradation needs `scipy.interpolate.interp2d` and `scipy.finfo`**,
  both gone from current scipy and both failing only when called: the image pins
  scipy 1.11.4.
* **SwinIR's inference pads by a full window even when the size is already a
  multiple of 8** (`main_test_swinir.py:70-73`); `write_arm.py` does the same (other
  padding changes 31% of output values by up to 28 levels).
* **KAIR's `network_swinir.py` differs from the official one** (the x4 upsampler is
  asserted rather than branched); same parameters for these configurations, so a
  KAIR-trained state_dict loads into the official file strictly (gate weights).
