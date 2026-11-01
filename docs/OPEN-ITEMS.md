# Open items

Decisions and unknowns that are deliberately unresolved. Each entry says what
blocks it, why it matters, and how to close it.

Last updated: 2026-09-16

Paths under `scripts/`, `experiments/` or `configs/` cited with a line number
point at the old code at tag `variants-verified` (`git show variants-verified:<path>`).
That code now lives under `variants/`; `variants/README.md` maps old files to new folders.

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

**RESOLVED 2026-09-16 — 복원하되, 현실적으로 "잘린" 복원만 가능하다.**
`variants/deeptex/paper_loop/main.py --experiment fixed --stage psnr`가 donor의 사전학습 레시피를 돌리고,
`variants/deeptex/fixed/esrgan_scaled.yaml`의 `pretrain_dir`이 그 출력을 가리킨다.
이번 조사에서 위 항목이 프레이밍하지 않았던 두 가지가 나왔다:

1. **donor의 단계를 복원하는 것만으로는 고쳐지지 않았을 것이다.** 탐색이 깨진 방식이,
   사전학습 체크포인트를 아무리 잘 만들어도 만족시킬 수 없는 형태다 — upstream은
   `pretrain_dir = cfg['pretrain_name']`(`:71`)인데 donor는 `'./checkpoints/' + name`이다.
   `pretrain_name: './checkpoints/'`면 `./checkpoints/` **바로 아래**의 체크포인트 상태 파일을
   찾으므로 `./checkpoints/psnr_pretrain/` 하위 디렉터리는 보이지 않는다. 두 단계를 다 돌리고,
   `generator.summary()` 300줄 밑으로 `print` 한 줄이 지나가는 걸 본 뒤, 결국 랜덤 초기화로
   학습했을 것이다. 우리 config 키는 전체 경로이고, 설정됐는데 못 찾으면 **치명(exit 9)**,
   복원되면 generator 가중치 전후 비교로 **증명**한다.

2. **donor의 `niter: 1000000`은 이 카드에서 70~180시간**이라 선택지가 아니다.
   `variants/deeptex/fixed/psnr_texsr.yaml`은 `niter: 40000`으로 자르고 `lr_steps`를 같은 1/5 간격
   (`[8000, 16000, 24000, 32000]`)으로 맞췄다. 이것 자체가 편차이고 그렇게 기술해야 한다 —
   `docs/DEEPTEX-DEVIATIONS.md` §4.

**2026-09-18 최종 결정: 사전학습을 쓰지 않는다.** 논문이 안 하기 때문이다. 비교는
`external/DeepTextures/train_esrgan.py` **원본**을 그대로 돌린 결과로 한다
(`-Stage paper`, `docs/DEEPTEX-DEVIATIONS.md` §0.4).

사전학습을 복원한 판(`variants/deeptex/paper_loop/main.py --experiment fixed --stage psnr` + `--stage esrgan`)은
지우지 않고 남겨 뒀다. 다만 그 arm은 **네 변수가 동시에 다르므로**(사전학습·shuffle·손실·LR
스케줄) 사전학습만의 기여를 분리하지 못한다. 그걸 알고 싶으면 세 번째 실행이 필요하다.

**2026-09-18 실측 — 예측이 확인됐다.** 40k 완주 로그에 `[*] cannot find pretrain model ./checkpoints/.`가 그대로 찍혔고, **40,000 스텝 전체가 랜덤 초기화에서 출발했다.** 그 결과 `loss_D`가 학습 시간의 **50%** 동안 `0.0000`으로 찍혔다 — 이 항목이 예고한 #8과의 결합이다 (`DEEPTEX-DEVIATIONS.md` §2.6).
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

**RESOLVED 2026-09-16 — config 2종으로 배포하고, 한쪽을 다른 쪽으로 편집하지 않는다.**

- `variants/deeptex/fixed/esrgan_scaled.yaml` — `lr_steps: [5000, 10000, 20000, 30000]`.
  upstream이 `niter`를 나눈 것과 같은 10으로 경계값도 나눈 것이라, ESRGAN이 발표한 스케줄의
  *형태*가 복원된다. **비교는 이걸로 돌린다.**
- `variants/deeptex/fixed/esrgan_asconfigured.yaml` — upstream의 `[50000, ...]` 그대로. 감쇠는 여전히 안 걸린다.

`variants/deeptex/fixed/hooks.py`가 **`lr_steps[0] < niter`를 단언**하고, `--allow-dead-lr-schedule`이
없으면 시작을 거부한다. 조용한 no-op을 기록된 선택으로 바꾸는 것이고, 실제 문제가 바로 그것이었다 —
아무도 감쇠가 안 걸리게 하기로 *결정*한 적이 없다.

한 파일을 편집하지 않고 2종으로 두는 이유: `batch_size`와 `lr_steps`는 프로토콜 가시적이라,
읽는 사람이 git 히스토리를 뒤지지 않고 두 상태를 다 볼 수 있어야 한다.

**2026-09-18: 비교는 `esrgan_paper.yaml`로 돈다** — 서브모듈 config에서 tfrecord 경로 한 줄만
바꾼 것이라, `lr_steps [50000, ...]`가 그대로 들어 있고 감쇠는 여전히 안 걸린다. 그게 논문이
배포한 상태이기 때문이다. `esrgan_scaled.yaml`은 남겨 두되 이번 비교에는 쓰지 않는다.

**2026-09-18 실측 — 예측이 확인됐다.** 40k 완주 로그의 `[metrics]` **411줄이 전부 `lr_G=1.0e-04`**다. 감쇠는 한 번도 걸리지 않았다 (`DEEPTEX-DEVIATIONS.md` §2.6).

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

## 9. MatUp output is less sharp than its own bilinear initialisation

**Status:** open. Cause narrowed to the optimisation stage; four hypotheses
measured and ruled out.

**Observation.** MatUp initialises from a bilinear x4 upsample
(`dataLoader.load_material`, `cv2.INTER_LINEAR`) and learns a residual on top,
so a ratio below 1.0 against that initialisation means the optimisation
*removed* sharpness. On `0559_brick_uneven_stones` (MatSynth, 128 -> 512):

| map | \|grad\| ratio | Laplacian-var ratio |
|---|---|---|
| baseColor | 0.883x | 0.850x |
| normal | 0.908x | 0.774x |
| roughness | 0.972x | 0.914x |

Output reaches 27% of the GT `|grad|`.

### Ruled out, with the measurement that ruled it out

1. **Our invocation.** The author's bundled `victorian_wood`, run with the
   author's own default command in the same container and checkpoint, gives
   **1.399x / 4.945x**. The setup works.

2. **`--ogl_normals`.** With the flag 0.883x/0.850x, without it 0.881x/0.845x —
   no effect on sharpness. (It *does* pick the output convention. Note
   `data_utils.py` hardcodes `opengl_normals=False` in the `write_normal` call,
   so output is always DirectX regardless. MatSynth is OpenGL — its dataset card
   states "We adopt the OpenGL standard for the Normal map" — so passing the
   flag is correct and the output needs a Y flip before comparison.)

3. **LR construction.** Suspected the 32x decimation (4096 -> 128) band-limited
   the input past what the upsampler can work with. Rebuilt the pair as a
   native-resolution 512 crop for GT with a x4 decimation for LR
   (`matsynth_extract.py --mode crop`): **0.851x / 0.835x**. No improvement.
   Hypothesis refuted.

4. **The SwinIR checkpoint.** Ran the render and upsample stages in isolation
   and measured SwinIR against a bilinear x4 of the same renders:

   | | \|grad\| | Laplacian-var |
   |---|---|---|
   | victorian_wood | 1.629x | 48.7x |
   | brick | 1.159x | **13.4x** |

   Weaker than on the author's example but plainly working. The optimisation is
   handed a target 13x sharper than bilinear and still returns 0.88x.

### Measured across all 12 materials of `test-00000`

The single-material conclusion did not survive. `matup/bilinear` on `|grad|`,
from `variants/compare/matup_postprocess.py --metrics`:

| map | above 1.0 | range |
|---|---|---|
| baseColor | **5 / 12** | 0.440 – 1.831 |
| roughness | 5 / 11 | 0.547 – 11.440 |
| **normal** | **1 / 12** | 0.577 – 1.175 |

So **baseColor and roughness are strongly material-dependent** — `acg_plastic_009`
gets 0.440x while `acg_roofing_tiles_006` gets 1.831x, and `th_brick_wall_005`
sharpens roughness by 11.4x. The brick this item was opened on (0.879x) sits
mid-range; generalising from it was the mistake.

**The normal map is the exception, and it is the real finding.** It is smoothed
on 11 of 12 materials with a narrow spread, while the other maps swing wildly.
That is systematic rather than material-dependent, and it is what hypothesis (b)
predicts: the optimisation can fit the upsampled radiance by moving baseColor and
roughness, which is easier than moving geometry, so the normal is left smooth.
The paper reports no per-map numbers, so this is not something it contradicts —
it is something it never measured.

### What is left

Why the normal specifically. Untested: inverse-problem ambiguity (the paper
itself concedes metalness diverges and adds a regulariser, §3.2);
`--alpha_reg 1e-1` possibly over-regularising; and whether the effect survives
outside this shard.

Note the reconstructed `height` (variants/matup/baseline/matup_height.py) exceeds 1.0 on all 12,
but that says nothing about MatUp — that map is built by integration plus a
high-pass, so adding high frequency is what it does by construction.

### Not defects — do not chase these

- **The gap to GT.** GT is a 512 view of a 4096 material; recovering 16x the
  pixels from a band-limited 128 input is not possible. A large gap is expected.
- **The 8px border** — the paper's stated design (README operational note 3).
- **The inverted normal Y** — convention, not error.

### Caveats on judging this at all

The paper reports **no quantitative metrics whatsoever** (zero occurrences of
PSNR, SSIM, RMSE or LPIPS), so there is no published figure for how sharp MatUp
"should" be. And the repo is the first author's own reimplementation, carrying
his caveat that it "may [have] slight differences which prevent me from
replicating all results perfectly".

All 12 materials come from one shard (`test-00000`) of one dataset, and
`|grad|` / Laplacian variance are proxies for sharpness, not for quality — a
result can be sharper and still wrong.

**To close:** decide whether the normal-map result is worth pursuing. It is the
only systematic effect found, and the comparison this project exists to make
runs on rendered appearance, where geometry matters. Widening to other shards
would establish whether it holds beyond these 12.

---

## 10. Deferred housekeeping

| Item | Note |
|---|---|
| ~~`.gitignore`~~ | **Done.** Covers `docker/.env`, `runs/`, `checkpoints/`, `logs/`, `*.pth`, `*.tfrecord`, and `docker/_das_placeholder/*` (with `.gitkeep` negated). |
| ~~`.gitattributes`~~ | **Done.** `* text=auto eol=lf` so the desktop checks out LF regardless of its own `core.autocrlf`, since it feeds `docker/` straight into `docker build`. |
| SwinIR checkpoint | Not downloaded yet. Must land at `external/matup_egsr/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth`, **exactly 142,473,939 bytes**. `sr_swinir.py` hardcodes the path and `exit(0)`s if missing. |
| ~~Org repository registration~~ | **Done.** `https://github.com/GPU-TechLab-personal/TexSR-comp1` (private). |
| ~~SSH remote execution~~ | **Abandoned.** Verified that Docker Desktop is not headless: over SSH the daemon is unreachable (`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`) and the session dies on disconnect. Experiments now run directly on the desktop via screen streaming. |
| ⚠️ **Two Docker daemons on the desktop** | Docker Desktop **and** a native docker-ce 29.8.0 inside WSL `Ubuntu-24.04` (systemd `enabled`, so it starts with the distro; nvidia-container-toolkit 1.20.0 present, `nvidia` runtime registered). **They share no images, containers, or build cache.** Running `docker compose build` from inside the WSL shell builds into docker-ce, and those images will not appear in Docker Desktop — an easy way to re-download 6 GB and lose an afternoon. **Convention: run all docker commands from PowerShell on Windows.** Verify with `docker context ls` / `docker info --format "{{.Name}}"` if anything looks wrong. |
| `.gitmodules` section label | Reads `[submodule "external/matup_esgr"]` (typo) while `path`/`url` are correct. Harmless; left as-is. |

---

## 11. DeepTextures upstream 결함 — 조사 완료

**Status:** RESOLVED 2026-09-16 — 전체 목록과 대응은 `docs/DEEPTEX-DEVIATIONS.md`.

#6·#7만 알려져 있었는데, 학습을 실제로 돌리려고 코드를 전부 읽은 결과 **열여덟 개**가 나왔다.
심각도 순으로 넷만 여기 옮긴다 — 나머지는 위 문서의 대조표에 있다:

1. **`test.py`로는 직접 학습한 모델을 평가할 수 없다.** `:36`이 체크포인트 경로를
   `'./checkpoints/original'`로 하드코딩하는데(+주석 처리된 대안 8개) 학습은
   `./checkpoints/<sub_name>`에 쓴다. CLI 플래그도 없다. donor는
   `'./checkpoints/' + cfg['sub_name']`이었다. → `variants/deeptex/infer/deeptex_sr.py`로 대체.
2. **`train_esrgan.py:23-25`가 엉뚱한 로거를 죽인다.** Python 로거를 끄는데, 정작 시끄러운
   cuBLAS/NUMA는 C++ 로거에서 나온다(#5). 결과적으로 체크포인트 불일치 경고가 사라져서
   **#6이 조용했던 것**이다.
3. **`modules/dataset.py:40-50`의 `tf.Assert` 3개가 반환값을 버린다.** HR≠LR×4가 에러가 아니라
   **어긋난 쌍**으로 학습된다. → `texsr_verify_tfrecord.py`가 학습 전 전수 검사.
4. **`modules/utils.py:401`이 평가 타일을 JPEG q95로 굽는다.** SR 경로만 그 손실을 먹고
   bicubic 기준선은 안 먹어서, 내장 비교가 SR에 불리하게 편향된다.

**2026-09-19 — 귀속을 나눠야 한다.** 이 저장소는 `github.com/peteryuX/esrgan-tf2`의 포크이고,
`modules/models.py`·`losses.py`·`dataset.py`·`lr_scheduler.py`가 donor와 **바이트 동일**하다.
`train_esrgan.py`는 **한 줄**만 다른데, 하필 그 줄이 사전학습 로딩을 깨뜨린 줄이다(#6).
따라서 위 네 개 중 2번(로거)은 **donor 결함이지 논문 저자의 코드가 아니다.** 3번(`tf.Assert`)도
마찬가지다. 논문에 귀속 가능한 것은 `train_esrgan.py`의 그 한 줄, config의 `niter` 1/10 축소(#7),
그리고 `test.py`+타일링(위 1번·4번)뿐이다. 전체 대조는 `DEEPTEX-DEVIATIONS.md` §0.7·§1.1·§1.2.

측정된 것 하나: **#8(`from_logits=False`)의 실제 증상은 NaN이 아니라 그래디언트 소멸**이고,
D 포화 시 upstream은 기울기가 **정확히 `0.000e+00`**, 수정판은 `1.095e-27`이다
(`DEEPTEX-DEVIATIONS.md` §2.5). 사전학습 없이 GAN을 시작하면 `loss_D`가 즉시 0으로 떨어지는
것이 관찰되며, 이것이 #6을 해소해야 하는 실증적 이유다.

**서브모듈은 한 바이트도 고치지 않았다.** `scripts/matup/matup_render.py:14`가 `matup_egsr`에 대해
세운 규약(바깥에서 import해 구동)을 그대로 따랐고, 그래서 `esrgan_asconfigured.yaml`로
"upstream 그대로"를 계속 돌려볼 수 있다 — #6·#7이 요구하는 분리가 그것 없이는 불가능하다.
