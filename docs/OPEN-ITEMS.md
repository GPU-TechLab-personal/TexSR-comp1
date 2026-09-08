# Open items

Decisions and unknowns that are deliberately unresolved. Each entry says what
blocks it, why it matters, and how to close it.

Last updated: 2026-09-08

---

## 1. `DAS_ROOT` — the MatSynth data path

**Status:** RESOLVED 2026-09-08 — `DAS_ROOT=U:/`.

The DAS is a multi-bay USB enclosure with three disks:

| Drive | Label | Size | Free | Contents |
|---|---|---|---|---|
| T: | NAS-1TB | 931.5 GB | 931.2 GB | empty |
| **U:** | **HITACHI-A** | **1863 GB** | **1581.9 GB** | **`MatSynth/`** |
| V: | HITACHI-B | 1863 GB | 1862.6 GB | empty |

`DAS_ROOT=U:/` rather than `U:/MatSynth`, so that `/data` has room for run
outputs and derived datasets beside the source data:

```
/data/MatSynth/data/*.parquet   the downloaded HF shards
/data/...                       anything we generate later
```

Pointing it at `U:/MatSynth` would leave nowhere inside `/data` to write.

**⚠️ The enclosure is not reliable.** On 2026-09-08 **T: and V: silently
stopped enumerating** — Explorer showed neither, `Get-Disk` reported only three
physical disks total, and the two bays were absent at the hardware level, not
merely unlettered. Ghost PnP entries under instance id `202000917003C` show the
same enclosure has held at least six different disks over time. Power-cycling
the enclosure brought both back with no data loss.

**Why this matters more than it looks:** U: is in that same enclosure and holds
the only copy of MatSynth here. Had this happened mid-training rather than
mid-session, the dataset would have vanished under a running job. Before any
long unattended run:

- confirm all three bays are present (`Get-Volume | Where-Object DriveLetter`)
- do not treat U: as durable storage for anything generated — results worth
  keeping belong on C: or pushed off-machine
- if it recurs, move MatSynth off the enclosure before the training round

**Windows note:** `DAS_ROOT` uses forward slashes (`U:/`), and the drive must be
shared under Docker Desktop → Settings → Resources → File sharing, otherwise
the bind mount silently becomes an empty directory.

**Verify:** `docker compose run --rm matup ls /data/MatSynth/data | head`

---

## 2. torchvision version in the MatUp base image

**Status:** RESOLVED 2026-09-08 — measured, deviation accepted.

**Measurement.** From the built image on the desktop:

```
docker compose run --rm matup python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
2.0.0 0.15.0 True
```

So `pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime` ships **torch 2.0.0** (exactly
upstream's pin) and **torchvision 0.15.0** — *not* the pinned 0.15.1. The claim
previously sitting in `docker/matup/requirements.txt`, that the base image
"already provides exactly these versions", was therefore half wrong; it has been
corrected in place.

**Verdict: keep 0.15.0, do not restore the pin.** Grepping the pinned sources
shows torchvision has no call site in this project:

- `matup_egsr` @ c74ec99 — no `torchvision` import anywhere in its own code
- `SwinIR` — no `torchvision` import either; `models/network_swinir.py:11` imports
  `timm`, which is what drags torchvision in transitively

So the only thing 0.15.0 vs 0.15.1 could affect is a module that gets imported
and never called. Against that, restoring `torchvision==0.15.1` would force a
reinstall from the `torch_stable` index — which is precisely the
differently-built-wheel risk (CPU-only build clobbering the CUDA one) that
commenting the pin out was meant to avoid. Trading a real risk for a nil one is
the wrong direction.

`torch.cuda.is_available()` returning `True` in the same call confirms the CUDA
build survived intact.

**Note for reproduction writeups:** the official PyTorch image for torch 2.0.0
does not carry the torchvision release that upstream pinned. If exact upstream
parity is ever required, it has to come from a different base image, not from a
pip reinstall on top of this one.

---

## 3. MatSynth `basecolor` colour space (sRGB vs linear)

**Status:** undocumented upstream; assumption in force.

Neither the [MatSynth dataset card](https://huggingface.co/datasets/gvecchio/MatSynth),
the CVPR paper, nor the project page states whether `basecolor` is stored
sRGB-encoded or linear. MatUp's loader **assumes sRGB** and applies
`srgb2linrgb()` on load.

**Why it matters:** if the assumption is wrong, every MatUp result is
photometrically off. Note the comparison itself stays valid — the same
assumption applies to both methods — but absolute numbers would be
misinterpreted.

**Practically:** the source sites (AmbientCG, PolyHaven, CGBookCase,
ShareTextures, TextureCan) conventionally store basecolor/diffuse as sRGB 8-bit
PNG and roughness/metallic/height as linear, so sRGB is very likely correct.

**To close:** when the first MatSynth shard is downloaded, inspect basecolor
histograms/medians against a known-sRGB reference and record the verdict here.

---

## 4. MatSynth train split size (documentation contradiction)

**Status:** contradictory upstream.

- Dataset card prose and project page: **train 3,980 / test 89**
- Card YAML `dataset_info.splits`: **train 5,700** / test 89

The YAML is auto-generated by `push_to_hub`, so 5,700 is more likely the real
row count on the Hub.

**Why it matters:** only for reporting ("we sampled 1,500 of N"). It does not
affect the pipeline.

**To close:** record `load_dataset(...).num_rows` at collection time and use the
measured value.

### The local copy is incomplete (train only)

Counted on `U:/MatSynth/data` on 2026-09-08:

| split | shards present | expected |
|---|---|---|
| test | **8** | 8 ✅ |
| train | **295** | 431 ❌ |

281 GB on disk. The train download stopped in the ~300 range — 287, 288, 291,
293–295, 298–300, 302 and then everything from 304 to 430 are missing. No
zero-byte files, so what did land is probably intact, though unverified.

**This does not block MatUp**, which is per-material test-time optimisation with
no training, and draws from the complete test split. It blocks only the
DeepTextures training round (#6), and it interacts with the count question
above: any "we sampled N of M" statement must be based on what was actually
read, not on the shard names.

**To close:** either finish the train download before the training round, or
decide to train on the 295 shards present and say so explicitly.

---

## 5. TensorFlow version for DeepTextures

**Status:** defaulted to 2.10.1, upper bound firm, lower bound unexplored.

The workable window is roughly **TF 2.5 – 2.10**:
- **Upper bound (firm):** from TF 2.11 the new Keras optimizers break
  `train_esrgan.py`'s `optimizer_G.lr(steps)`.
- **Lower bound (still unverified):** TF 2.5+ builds against CUDA 11.2, which
  supports Ampere (sm_86). TF 2.4 uses CUDA 11.0, which does not. This comes
  from TensorFlow's tested-build-configurations table, not from a build we ran.

**The chosen version is no longer theoretical.** Measured on the desktop
(RTX 3080 Ti, driver 591.86) on 2026-09-08:

```
docker compose run --rm deeptex python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
2.10.1 [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

So TF 2.10.1 / CUDA 11.2 really does see sm_86 on this card. Only the *lower*
end of the window (2.5–2.9) remains table-derived.

That run also emits three messages that look alarming and are not. Do not chase
them: `Unable to register cuBLAS factory: ... already been registered` is known
TF 2.10 duplicate-registration noise; `could not open file to read NUMA node`
is normal under Docker Desktop's WSL2 VM, which exposes no NUMA topology; the
oneDNN lines are informational.

2.10.1 was chosen as the newest version that runs the code unmodified. Moving
closer to upstream's 2.1.0 would be more faithful to the paper but means older
images and unpatched CVEs.

**To close:** only if fidelity to the original TF version becomes a stated
requirement. If so, verify the sm_86 lower bound with a real build before
committing to it.

---

## 6. DeepTextures PSNR pre-training stage

**Status:** deferred to the training round.

Upstream deleted `train_psnr.py` and `configs/psnr.yaml`, and set
`pretrain_name: './checkpoints/'` so the pretrain lookup silently fails and GAN
training starts from **random initialisation**. This departs from the original
ESRGAN recipe (PSNR pre-train → GAN fine-tune) and is the single biggest
unknown for reproduction quality.

`external/esrgan-tf2` is pinned as a submodule specifically as the donor for
those two deleted files.

**To close:** decide whether to train with the restored pre-train stage, without
it, or both (both is the only way to tell "faithful reproduction" apart from
"upstream defect"). Not needed until the training round.

---

## 7. DeepTextures LR schedule never fires

**Status:** observed in upstream config; no decision yet.

`configs/esrgan.yaml` sets `niter: 40000` while
`lr_steps: [50000, 100000, 200000, 300000]` with `lr_rate: 0.5`. The first decay
step lands 10k iterations *after* training ends, so both learning rates stay
constant at `1e-4` for the whole run — the multi-step decay that
`train_esrgan.py` logs at every step is dead code as configured.

**Why it matters:** same class as #6, and compounding with it. ESRGAN's
published recipe is PSNR pre-train → GAN fine-tune *with* LR decay; this config
has neither. So "DeepTextures as published" sits two deviations away from
"ESRGAN as published", and a weak reproduction cannot be attributed to the
paper's method without separating them.

**To close:** decide together with #6 — train as-configured (faithful to the
DeepTextures repo) or with `lr_steps` scaled into the 40k window (faithful to
ESRGAN). Either way state the choice explicitly in the writeup, since the paper
does not mention the discrepancy. Not needed until the training round.

---

## 8. `opencv-python` major-version drift

**Status:** RESOLVED 2026-09-08 — pinned `opencv-python<5` in both images.

Neither `requirements.txt` pinned `opencv-python`. Measured in the built images
on 2026-09-08, before the pin below was added:

| | Python | opencv-python | numpy |
|---|---|---|---|
| matup | 3.10 | **4.11.0.86** | 1.26.4 |
| deeptex | 3.8 | **5.0.0.93** | 1.23.4 |

Two separate problems, and the second is the worse one:

1. **deeptex resolved to OpenCV 5.x** against a 2020-era codebase. Auditing the
   call sites, every one is bedrock API that 5.x keeps — `cv2.imread`,
   `imwrite`, `resize` (`test.py`, `modules/utils.py:370,402,433-438`) and
   `getGaussianKernel` / `filter2D` (`modules/utils.py:302-312`, the SSIM
   implementation). So this probably *works*. It is undeclared drift rather than
   a known break.

2. **matup's 4.x is held there by accident.** Nothing pins opencv; it stayed on
   4.11 only because `numpy<2` forced pip to backtrack off the 5.x wheels. That
   coupling is invisible and load-bearing: relax or retarget the numpy pin later
   and matup silently jumps a major version of OpenCV too.

Note the Dockerfiles' build-time checks cannot catch either case — both only do
`import cv2`, which succeeds on any version.

**Why it matters:** this is precisely the time-rot failure that
`docker/matup/requirements.txt` was written to prevent, reappearing in the
dependency that file left open. SSIM is a reported comparison metric, so a
behavioural change in `filter2D` would move published numbers, not just crash.

**Decision:** `opencv-python<5` added to both requirements files, with the
reasoning in a comment beside each. matup's behaviour does not change (it was
already on 4.11.0.86) — the pin only converts an accident into an intention.
deeptex actually moves off 5.0.0.93, so its pip layer must be rebuilt:

```
docker compose build deeptex
```

**Verified after rebuild**, same day:

| | before the pin | after |
|---|---|---|
| matup | 4.11.0.86 | 4.11.0.86 (unchanged, as predicted) |
| deeptex | **5.0.0.93** | **4.14.0.94** |

The two images land on different 4.x releases because their Python and numpy
constraints differ; the pin only ever had to exclude the major bump, so that is
fine. To re-check later:

```
docker compose run --rm matup   pip list --format=freeze | grep opencv
docker compose run --rm deeptex pip list --format=freeze | grep opencv
```

(from PowerShell, replace `grep opencv` with `Select-String opencv`)

Unrelated but visible in that output: the deeptex image carries pip 20.2.4 and
says so on every invocation. That is what the TF 2.10.1 / Python 3.8 base ships.
Leave it — upgrading pip inside the image would change how every unpinned
requirement in this file resolves, which is the opposite of what #8 is for.

---

## 9. Deferred housekeeping

| Item | Note |
|---|---|
| ~~`.gitignore`~~ | **Done.** Covers `docker/.env`, `runs/`, `checkpoints/`, `logs/`, `*.pth`, `*.tfrecord`, and `docker/_das_placeholder/*` (with `.gitkeep` negated). |
| ~~`.gitattributes`~~ | **Done.** `* text=auto eol=lf` so the desktop checks out LF regardless of its own `core.autocrlf`, since it feeds `docker/` straight into `docker build`. |
| SwinIR checkpoint | Not downloaded yet. Must land at `external/matup_egsr/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`, **exactly 142,473,939 bytes**. `sr_swinir.py` hardcodes the path and `exit(0)`s if missing. |
| ~~Org repository registration~~ | **Done.** `https://github.com/GPU-TechLab-personal/TexSR-comp1` (private). |
| ~~SSH remote execution~~ | **Abandoned.** Verified that Docker Desktop is not headless: over SSH the daemon is unreachable (`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`) and the session dies on disconnect. Experiments now run directly on the desktop via screen streaming. |
| ⚠️ **Two Docker daemons on the desktop** | Docker Desktop **and** a native docker-ce 29.8.0 inside WSL `Ubuntu-24.04` (systemd `enabled`, so it starts with the distro; nvidia-container-toolkit 1.20.0 present, `nvidia` runtime registered). **They share no images, containers, or build cache.** Running `docker compose build` from inside the WSL shell builds into docker-ce, and those images will not appear in Docker Desktop — an easy way to re-download 6 GB and lose an afternoon. **Convention: run all docker commands from PowerShell on Windows.** Verify with `docker context ls` / `docker info --format "{{.Name}}"` if anything looks wrong. |
| `.gitmodules` section label | Reads `[submodule "external/matup_esgr"]` (typo) while `path`/`url` are correct. Harmless; left as-is. |
