# 실험 운용 매뉴얼

`README.md`가 **한 번 하는 구축**이라면, 이 문서는 **매번 하는 실험**이다.
`docs/OPEN-ITEMS.md`는 아직 안 정해진 것들이고.

원칙 하나: **아래 어떤 주장도 그대로 믿지 말고 확인할 수 있게 써 뒀다.** 각 항목에
검증 명령을 같이 적은 이유다. 측정값은 전부 2026-09-08 데스크탑(RTX 3080 Ti,
드라이버 591.86) 기준이다.

모든 명령은 **Windows PowerShell**, `docker/` 디렉터리 기준이다.

---

## 1. 실험 전 환경 점검 (2분)

순서가 중요하다. 위가 깨지면 아래는 볼 필요가 없다.

```powershell
docker context show
```
→ `desktop-linux`. **다른 값이면 여기서 멈출 것.** 이 머신엔 Docker 데몬이 둘이고
(Docker Desktop / WSL `Ubuntu-24.04`의 docker-ce) 이미지·캐시를 공유하지 않는다.

```powershell
docker compose run --rm matup python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
→ `2.0.0 True`. `False`면 GPU 패스스루가 죽은 것이라 MatUp은 아예 못 돈다
(`torch.cuda.CUDAGraph()`를 쓰고 CPU 폴백 경로가 없다).

```powershell
docker compose run --rm matup ls /data/MatSynth/data | Measure-Object -Line
```
→ `303`. `0`이거나 `No such file`이면 `.env`의 `DAS_ROOT`나 DAS 연결 문제다.
**DAS는 베이가 조용히 사라진 전력이 있다** (OPEN-ITEMS #1). 긴 실험 전에는 호스트에서:

```powershell
Get-Volume | Where-Object DriveLetter | Format-Table DriveLetter, FileSystemLabel, HealthStatus
```
→ T·U·V 세 개가 다 보여야 한다. U:에 MatSynth 단일 사본이 있다.

```powershell
docker ps
```
→ **비어 있어야 한다.** 남아 있는 MatUp 컨테이너가 있으면 지금 실험을 파괴할 수 있다
(§5의 첫 항목).

---

## 2. 재질 하나 돌리기 (end-to-end)

### 2.1 샤드에서 뭐가 있는지 본다

```powershell
docker compose run --rm dataprep python variants/common/matsynth_extract.py --list
```

기본 샤드는 `test-00000-of-00008.parquet`(12개 재질). 다른 샤드는 `--shard`로.
**test 스플릿은 8/8 완비, train은 295/431로 결손**이 있다 (OPEN-ITEMS #4).
MatUp은 학습이 없으므로 test만으로 충분하다.

### 2.2 뽑는다

```powershell
docker compose run --rm dataprep python variants/common/matsynth_extract.py --index 8
```

`runs/matsynth/<이름>/` 아래에 만들어진다:

| | 크기 | 용도 |
|---|---|---|
| `SVBRDF/` | 128² | MatUp 입력 (LR) |
| `GT/` | 512² | 비교용 정답 |
| `meta.json` | | 출처·가정 기록 |

원본 4096²에서 LR·GT를 **각각 직접** area-average 한다. 배율이 정수(32×, 8×)라
두 결과는 정확히 ×4 쌍이 된다. **원본 비트뎁스를 보존**한다 — height는 uint16이다.

### 2.3 MatUp을 돌린다

스크립트가 마지막에 이 명령을 그대로 찍어준다. 복사해 쓰면 된다.

```powershell
$m = "<재질이름>"
docker compose run --rm matup python main.py --upscaler swinir --ogl_normals `
  --input_mat       "/workspace/runs/matsynth/$m/SVBRDF/" `
  --render_dir      "/workspace/runs/matsynth/$m/radiance_LR/" `
  --upsamplings_dir "/workspace/runs/matsynth/$m/radiance_Upscaled/" `
  --output_folder   "/workspace/runs/matsynth/$m/matup-output-$m/"
```

**경로 4개를 실행마다 분리하는 것이 핵심이다.** 기본값을 쓰면 동시 실행이 서로를
파괴한다 (§5). `--ogl_normals`는 MatSynth 법선이 OpenGL 규약이라 필수다.

3080 Ti에서 재질 1개당 **1~2분**. 정상 종료 로그:

```
loading model from ./pretrained_models/003_...pth
Upsampling took NN seconds
loss 1 ... loss 10
...done
```

결과는 `matup-output-<재질이름>/<타임스탬프>/`에 **512² 16-bit PNG 5장**. 타임스탬프
하위 폴더는 MatUp이 `--output_folder` 아래에 스스로 만드는 것이라 없앨 수 없다.

### 2.4 전부 자동으로 — 권장

위 명령을 재질마다 손으로 치는 대신:

```powershell
powershell -ExecutionPolicy Bypass -File variants\matup\baseline\run.ps1 -PostProcess
```

`runs/matsynth`의 모든 재질을 순차 실행하고 후처리 체인까지 돈다. 사전 점검(§1)을
자동으로 수행하고, **이미 결과가 있는 재질은 건너뛴다**(`-Force`로 무시). GPU 이탈
서명을 만나면 즉시 중단하고, 그 외 실패는 2회 연속이면 멈춘다. 재질별 로그는
`runs/_logs/<타임스탬프>/`에 남는다.

`-PostProcess`가 도는 네 단계 — 순서가 중요하다:

| 스크립트 | 컨테이너 | 하는 일 |
|---|---|---|
| `matup_height.py` | dataprep | 논문 §3.3 법선 적분으로 height 복원 (MatUp은 출력 안 함) |
| `matup_render.py` | **matup** | LR·bilinear·GT를 MatUp 렌더러·같은 광원으로 렌더 |
| `matup_postprocess.py` | dataprep | 테두리 크롭·법선 G 교정·크기 통일 → `compare/`, `--metrics` |
| `matup_montage.py` | dataprep | 4행×6열 몽타주 → `runs/montages/` |

앞의 둘이 MatUp이 만들지 않는 두 열을 채우고, 셋째가 전부를 `compare/`로 정규화하며,
넷째는 거기서만 읽는다. 따로 돌릴 때도 이 순서를 지켜야 한다.

`matup_render.py --verify`는 MatUp 자신의 출력 맵을 다시 렌더해 `render_opt.png`와
비교한다. 12개 전부 max ≤ 3.3e-4로 일치했다 — 맵 변환이 맞다는 확인이다.

---

## 2.5 DeepTextures 파이프라인

MatUp과 달리 **학습이 필요하다**. 사전학습 가중치가 배포된 적이 없다.

학습 소스는 서브모듈 안의 `external/DeepTextures/data/textures_db` 300장이고, 평가 대상은
이미 추출해 둔 `runs/matsynth/` 12개 재질이다. **`/data`(DAS)를 전혀 쓰지 않는다** (§2.6 MatSynth arm만 예외).

### 2.5.1 한 줄씩

```powershell
powershell -ExecutionPolicy Bypass -File variants\deeptex\paper\run.ps1 -Stage prep    # 정규화 -> 타일 -> tfrecord -> 검증 (10~30분)
powershell -ExecutionPolicy Bypass -File variants\deeptex\paper\run.ps1 -Stage smoke   # 20스텝 학습 + 평가 self-test (약 4분)
powershell -ExecutionPolicy Bypass -File variants\deeptex\paper\run.ps1 -Stage paper   # 논문 코드 40k 학습 (약 2.3시간)
powershell -ExecutionPolicy Bypass -File variants\deeptex\paper\run.ps1 -Stage eval    # SR -> compare/ -> 5행 몽타주
```

> **`-ExecutionPolicy Bypass`가 왜 붙어 있나:** 이 머신은 실행 정책이 전 범위 `Undefined`라
> Windows 기본값 `Restricted`가 걸린다. 그냥 `variants\...\run.ps1`로 부르면 `UnauthorizedAccess`로
> 거부된다. `Bypass`는 **그 실행 하나에만** 적용되고 시스템 설정을 바꾸지 않는다.
> 매번 붙이기 싫으면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 한 번 하면 되는데,
> 그건 계정 전역 설정이다.

`-Stage all`이 이 넷을 순서대로 돈다. 다른 컨테이너가 떠 있으면
`-AllowContainers <이름>`으로 허용해야 preflight를 통과한다.

**smoke를 건너뛰지 말 것** — 4분에 config 경로·CUDA·VGG19 캐시·쓰기 권한·`sys.path`,
그리고 `test.py`로는 애초에 불가능했던 **학습→평가 연결**까지 증명한다.

### 2.5.2 prep 4단계 — 각 단계가 막는 것

| 스크립트 | 컨테이너 | 하는 일 |
|---|---|---|
| `texsr_normalize.py` | dataprep | 8-bit·3채널·4의 배수로 정규화. **JPG 100장을 구제**한다(upstream 변환기는 `*.png`만 glob) |
| `texsr_make_pairs.py` | **deeptex** | 타일링 + ×4 LR 생성 |
| `texsr_build_tfrecord.py` | deeptex | 스키마는 upstream과 동일, 나머지는 교체 |
| `texsr_verify_tfrecord.py` | deeptex | **학습 전 게이트.** PASS 아니면 학습이 시작을 거부한다 |

`make_pairs`가 dataprep이 아니라 **deeptex**에서 도는 이유: `imresize_np`(MATLAB bicubic)를 써야 하는데,
함수는 numpy 전용이어도 **모듈**이 아니다 — `modules/utils.py:8,10`이 모듈 스코프에서 `import tensorflow`와
`from modules.dataset import ...`를 한다. 복제본을 만들면 학습·평가가 실제로 쓰는 커널과 드리프트할 수 있어
그렇게 하지 않았다.

`verify`의 핵심은 **`channels=0`으로 디코드**하는 것이다. `modules/dataset.py:12-13`처럼 `channels=3`으로
열면 알파·팔레트·그레이·16bit 문제가 바로 그 지점에서 숨는다.

### 2.5.3 학습 — 논문 코드를 그대로 돌린다

```powershell
powershell -ExecutionPolicy Bypass -File variants\deeptex\paper\run.ps1 -Stage paper
```

**실행되는 것은 `external/DeepTextures/train_esrgan.py` 원본이다.** 우리 트레이너가 아니다.
`variants/deeptex/upstream_runner/run_upstream_train.py`가 그것을 subprocess로 띄우고 stdout만 재포맷한다 —
진행바가 `\r`로 뭉쳐 나오는 걸 `[metrics]` 줄로 쪼개는 것뿐이고, 학습에는 손대지 않는다.

작업 디렉터리는 `runs/deeptex/paper/`다. upstream이 `./checkpoints`·`./logs`를 CWD 상대로 만들기
때문에, 이렇게 해야 **서브모듈이 더러워지지 않는다.**

정상 로그는 이렇게 생겼다 — 논문의 결함 두 개가 그대로 보인다:

```
[*] cannot find pretrain model ./checkpoints/.     <- 사전학습 탐색 실패, 랜덤 초기화 (OPEN-ITEMS #6)
[metrics] step=100 niter=40000 loss_G=... loss_D=... lr_G=1.0e-04 steps_per_sec=4.7
```

`lr_G`가 끝까지 `1.0e-04`에서 안 움직이는 것도 정상이다 — `lr_steps [50000,...]`가 niter 40000
밖이라 감쇠가 한 번도 안 걸린다 (OPEN-ITEMS #7). **이 셋 다 논문이 배포한 상태 그대로다.**

`loss_D`가 빠르게 0에 붙는 것도 예상된 거동이다. 사전학습 없이 시작하면 D가 G를 즉시 이긴다.

**성공 판정은 로그가 아니라 상태다.** `[metrics]`가 niter에 닿는 것과 **별개로**, 체크포인트의
`step` 텐서가 niter 이상인지 컨테이너에서 직접 확인한다. 둘 중 하나라도 안 맞으면 실패다.

> 결함 4개를 고친 판(`variants/deeptex/fixed/` + PSNR 사전학습)은 `variants\deeptex\fixed\run.ps1 -Stage pretrain` / `-Stage train`으로
> 여전히 돌릴 수 있지만 **이번 비교에서는 쓰지 않는다.** 이유는 `docs/DEEPTEX-DEVIATIONS.md` §0.4.

### 2.5.4 재개와 재시작

**재개가 정상이다.** 같은 명령을 다시 치면 체크포인트에서 이어간다. 이미 niter에 도달했으면
`already complete: resumed at step N`을 찍고 정상 종료한다 — 실패가 아니다.

`-Force`만으로는 체크포인트를 지우지 않는다. 처음부터 다시 하려면 `-Force -DeleteCheckpoints`가
필요하고, 그래도 기존 디렉터리는 `.bak-<타임스탬프>`로 **옮겨질 뿐 삭제되지 않는다.**

짧게 시험하려면 `-MaxSteps 20`. 이때는 `niter`와 `save_steps`를 낮춘 scratch config가
`runs/deeptex/paper/`에 만들어지고, **원본 config는 건드리지 않는다.**

### 2.5.5 비교에 붙이기

`-Stage eval`이 세 단계를 돈다:

| 스크립트 | 컨테이너 | 하는 일 |
|---|---|---|
| `deeptex_compare.py` | deeptex | 재질마다 `SVBRDF/baseColor.png`와 `render_lr.png`를 ×4 |
| `matup_postprocess.py --deeptex-root` | dataprep | 5번째 열 `deeptex`를 `compare/`에 추가 + PSNR/SSIM |
| `matup_montage.py` | dataprep | 5행 몽타주 |

**6열 중 2열만 채워진다.** DeepTextures는 3채널 RGB 전용이라 normal(단위 벡터장)·roughness·metallic(1채널)·
height(uint16)를 처리할 수 없다. 나머지 4칸은 placeholder로 남고, 그게 정직한 표현이다.

## 2.6 DeepTextures MatSynth arm

같은 upstream 트레이너를 **MatSynth 다섯 맵을 섞은 데이터**로 돌린다. 논문이 실제로 한 방식(모델 하나 ×
다섯 채널 종류)이다. 근거와 한계는 `docs/DEEPTEX-DEVIATIONS.md` §0.9.

**이 arm만 `/data`를 쓴다.** `docker/.env`에 `DAS_ROOT=D:/experimentdata`가 있어야 컨테이너 안에서
`/data/MatSynth/data/*.parquet`이 보인다.

```powershell
powershell -ExecutionPolicy Bypass -File variants\deeptex\matsynth\run.ps1 -Stage prep-matsynth   # 게이트 -> 쌍 -> tfrecord -> 검증 (처음 약 20분, 재실행 약 5분)
powershell -ExecutionPolicy Bypass -File variants\deeptex\paper\run.ps1 -Stage paper     -PaperConfig variants/deeptex/matsynth/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all   # 40k (약 2.6시간)
powershell -ExecutionPolicy Bypass -File variants\deeptex\matsynth\run.ps1 -Stage eval-arms -PaperConfig variants/deeptex/matsynth/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all   # SR 2회 + 리포트 (약 10분)
```

| 단계 | 스크립트 | 컨테이너 | 막는 것 |
|---|---|---|---|
| 게이트 | `matsynth_make_pairs.py --check-eval` | dataprep | 평가 12개를 다시 만들어 `runs/matsynth`와 **픽셀 비교**. 다르면 학습 데이터를 만들지 않는다 |
| 쌍 | `matsynth_make_pairs.py --jobs 4` | dataprep | train shard만, LFS 스텁 건너뜀, 평가 재질 블렌드 제외. **재실행하면 이어서 한다**(`_meta/<소재>.json`이 완료 표시) |
| tfrecord | `recipes.py --recipe <R>` (→ `texsr_build_tfrecord.py`, 다섯 맵) | deeptex | 다섯 맵을 **한 번에 섞는다** — upstream `shuffle=False` |
| 검증 | `texsr_verify_tfrecord.py --align-sample 16` | deeptex | `alignment` 경고는 정상이다. LR이 `imresize_np(HR)`이 아니라 원본에서 area로 만든 것이라서 |
| 평가 | `eval_arms.py` ×2 → `deeptex_arms_report.py` | deeptex → dataprep | textures_db·MatSynth 체크포인트를 같은 6개 입력에. 결과는 `runs\deeptex\report_matsynth_all\summary.md` |

리포트 폴더(`runs\deeptex\report_<tag>\`)에 생기는 것:

| 파일 | 내용 |
|---|---|
| `montages\<재질>.png` | 행 = 방법(LR / bilinear / MatUp / DeepTextures 두 arm / GT), 열 = 다섯 맵 + render |
| `overview_<맵>.png` | 행 = 12개 재질, 열 = 방법 |
| `sheets\<재질>_<맵>.png` | 맵 하나를 496² 원본 크기로 (확대해서 볼 때) |
| `summary.md` | 맵별 평균표와 arm 간 맞대결 횟수. 표 위에 열 설명이 있다 |
| `per_material.md` | 재질별 PSNR / SSIM (normal은 각도도) |
| `metrics_long.csv` | 재질 × 맵 × 열 한 줄씩. 엑셀이나 pandas 피벗용 |
| `report.json` | 모든 숫자 |

평균은 **모든 열이 유한 PSNR을 갖는 재질에서만** 낸다(`n`). `offset` / `PSNR -off`는 진단용이다
(DEEPTEX-DEVIATIONS §0.9의 색 이동).

⚠️ **`-Stage eval`을 이 arm에 쓰지 말 것.** 그 stage는 `runs/montages/`(커밋됨)를 다시 쓴다.
`eval-arms`는 `runs/deeptex/eval/`와 `runs/deeptex/report_<tag>/`에만 쓴다. tag는 `-PaperRoot`의 마지막 폴더명이다.

### 2.6.1 비율 ablation — `chratio`, `basecolor` (DEEPTEX-DEVIATIONS §0.10)

학습 쌍은 2.6과 같은 파일을 쓴다. TFRecord만 recipe별로 따로 만든다.

```powershell
# 데이터 (각 수 분). -Recipe all은 기존 matsynth_all_lrx4.tfrecord를 그대로 다시 만든다
powershell -ExecutionPolicy Bypass -File variants\deeptex\matsynth\run.ps1 -Stage prep-matsynth -Recipe chratio
powershell -ExecutionPolicy Bypass -File variants\deeptex\matsynth\run.ps1 -Stage prep-matsynth -Recipe basecolor
# 학습 (각 약 2.6시간). -KeepAllCheckpoints가 5k마다 저장되는 8개를 checkpoints_all\에 모두 남긴다
powershell -ExecutionPolicy Bypass -File variants\deeptex\paper\run.ps1 -Stage paper -PaperConfig variants/deeptex/matsynth/esrgan_matsynth_chratio.yaml   -PaperRoot runs/deeptex/matsynth_chratio   -KeepAllCheckpoints
powershell -ExecutionPolicy Bypass -File variants\deeptex\paper\run.ps1 -Stage paper -PaperConfig variants/deeptex/matsynth/esrgan_matsynth_basecolor.yaml -PaperRoot runs/deeptex/matsynth_basecolor -KeepAllCheckpoints
# 평가. 두 번째 명령이 네 arm을 한 표로 묶는다 (report_matsynth_basecolor)
powershell -ExecutionPolicy Bypass -File variants\deeptex\matsynth\run.ps1 -Stage eval-arms -PaperConfig variants/deeptex/matsynth/esrgan_matsynth_chratio.yaml   -PaperRoot runs/deeptex/matsynth_chratio
powershell -ExecutionPolicy Bypass -File variants\deeptex\matsynth\run.ps1 -Stage eval-arms -PaperConfig variants/deeptex/matsynth/esrgan_matsynth_basecolor.yaml -PaperRoot runs/deeptex/matsynth_basecolor -ExtraArms matsynth_all,matsynth_chratio
```

| recipe | 맵 반복 | 레코드 |
|---|---|---|
| `all` | 5맵 × 1 (논문 비율) | 14,520 |
| `chratio` | baseColor × 3, normal × 3, 회색 맵 × 1 (채널 수 비례 3:3:1:1:1) | 26,136 |
| `basecolor` | baseColor × 1 | 2,904 |

**색 이동 측정 (`deeptex_drift.py`, deeptex 컨테이너):** 체크포인트마다 자기 학습 쌍 60개와 평가 12개의 채널별
평균 이동과 초록 쏠림을 잰다.
- 보존한 8개를 보려면 `--ckpt-dir`를 `checkpoints_all\esrgan_40k`로 준다.
- `--sheet-material <재질>`을 주면 체크포인트마다 그 재질의 다섯 맵 + render 한 줄 시트도 만든다.

```powershell
docker compose -f docker/compose.yaml run --rm deeptex python /workspace/variants/deeptex/matsynth/deeptex_drift.py --cfg /workspace/variants/deeptex/matsynth/esrgan_matsynth_chratio.yaml --ckpt-dir /workspace/runs/deeptex/matsynth_chratio/checkpoints_all/esrgan_40k --pairs /workspace/runs/deeptex/datasets/matsynth/baseColor --out /workspace/runs/deeptex/drift/matsynth_chratio --sheet-material 0844_syntetic_foam
```

⚠️ **`--jobs`를 올리지 말 것.** shard 하나가 ~1 GB row group 하나다. 워커 6개로 시작했다가 16 GB VM에서
ENOMEM으로 두 번 죽었다. 지금은 열 하나씩 읽고 shard마다 새 프로세스(`maxtasksperchild=1`)를 쓴다.
4개에서 약 6 GB를 쓴다.

---

## 3. 결과를 비교하기 전에 반드시 할 것

이걸 안 하면 숫자가 조용히 틀린다. 셋 다 크래시가 아니라 **잘못된 수치**로 나타난다.

1. **8px 검은 테두리를 잘라낸다.** 512² 중 유효 영역은 496²다. 결과·GT·다른 방법
   전부에서 동일하게 잘라야 공정하다.
   확인: `a[:8].max()`가 0, `a[8].max()`가 0이 아니면 테두리가 8px이다.

2. **`normal_opt`의 녹색 채널을 뒤집는다.** 입력은 OpenGL로 넣었는데
   (`--ogl_normals`) MatUp **출력은 DirectX 고정**이다 —
   `file_utils.write_normal()`이 `array[..., 1] = -array[..., 1]`을 한다.
   GT는 OpenGL이므로 그대로 비교하면 Y축이 반대다.

3. **height는 비교 대상이 아니다.** MatUp은 height를 출력하지 않는다. 산출 맵은
   baseColor·metallic·normal·roughness 4장뿐이다.

그리고 **baseColor는 sRGB 가정 위에 있다** — `load_albedo()`가 `srgb2linrgb()`를
적용한다. MatSynth가 실제로 sRGB인지는 upstream 미문서다 (OPEN-ITEMS #3). 두 방법에
같은 가정이 걸리므로 비교 자체는 유효하지만, 절대 수치를 논할 때는 이 가정을 밝혀야
한다.

---

## 4. 코드 지도 — 뭘 바꾸려면 어디를 보나

> ⚠️ **아래 줄 번호는 고정된 서브모듈 커밋 기준이다** — `matup_egsr` @ `c74ec99`,
> `DeepTextures` @ `e4ee562`. 서브모듈 포인터를 올리면 **전부 조용히 틀어진다**
> (엉뚱한 줄을 가리키지, 에러가 나지 않는다). upstream을 올릴 일이 생기면 이 절을
> 먼저 다시 확인할 것. 현재 커밋 확인: `git submodule status --recursive`

### MatUp (`external/matup_egsr` @ c74ec99)

| 하려는 것 | 볼 곳 |
|---|---|
| 인자 전체 목록·기본값 | `main.py` (argparse 블록 하나에 다 있다) |
| 입력 맵 파일명·로딩 | `matup/dataLoader.py:36-67` (`load_roughness`/`normal`/`albedo`/`metallic`/`height`) |
| 비트뎁스·채널 순서 처리 | `matup/file_utils.py:54-62` `process_image_as_float` |
| 업샘플러 붙는 지점 | `sr_swinir.py:20-40` (체크포인트 경로가 `:27`) |
| 최적화 루프 | `main_optim.py` → `matup/train.py` |
| 출력 저장·법선 규약 | `matup/data_utils.py:123`, `file_utils.write_normal` |
| 중간 산출물 삭제 | `main_optim.py:103-104` |

파일명은 **대소문자 구분**이다. 컨테이너가 Linux라 `baseColor.png`의 대문자 C가 틀리면
못 찾는다.

### DeepTextures (`external/DeepTextures` @ e4ee562)

| 하려는 것 | 볼 곳 |
|---|---|
| 학습 하이퍼파라미터 | `configs/esrgan.yaml` |
| 학습 루프 | `train_esrgan.py` |
| 지각 손실(VGG19) | `modules/losses.py:25` |
| SSIM·이미지 I/O | `modules/utils.py:302-312`(SSIM), `:370,402,433-438` |
| 사전학습 체크포인트 로드 | `train_esrgan.py:70-79` |

> ⚠️ **위 파일들은 읽기만 한다. 우리는 서브모듈을 고치지 않는다.**
> 실제로 돌리는 코드는 상위 저장소에 있고 `modules/*`를 바깥에서 import한다:

| 하려는 것 | 볼 곳 |
|---|---|
| 학습 하이퍼파라미터 | `variants/deeptex/fixed/{psnr_texsr,esrgan_scaled,esrgan_asconfigured}.yaml` |
| 학습 루프 | `variants/deeptex/paper_loop/loop.py` + `variants/deeptex/fixed/hooks.py` (`train_esrgan.py`와 평행 구조) |
| 적대 손실 (수정판) | `variants/deeptex/fixed/losses_corrected.py` |
| 평가·추론 | `variants/deeptex/infer/deeptex_sr.py` (`test.py` 대체) |
| 비교 연결부 | `variants/deeptex/infer/deeptex_compare.py` |
| 데이터 준비 | `variants/deeptex/paper/texsr_{normalize,make_pairs}.py`, `variants/deeptex/upstream_runner/texsr_{build_tfrecord,verify_tfrecord}.py` |
| 오케스트레이션 | `variants/deeptex/{paper,matsynth,fixed,render_loss}/run.ps1` (공용 함수는 `variants/common/lib.ps1`) |

**upstream 결함은 둘이 아니라 열여덟이었다.** 전체 목록과 각각에 대한 우리 대응은
`docs/DEEPTEX-DEVIATIONS.md`에 있다. 원래 알려져 있던 둘은 이제 해소됐다:

- 사전학습 탐색이 **구조적으로 성공 불가**였다 (donor 단계를 복원해도 못 찾는다).
  이제 못 찾으면 **exit 9**이고, 찾으면 가중치 변화로 복원을 증명한다 (OPEN-ITEMS #6)
- `niter: 40000` vs `lr_steps [50000, ...]` → config 2종으로 분리하고 트레이너가
  `lr_steps[0] < niter`를 단언한다 (OPEN-ITEMS #7)

---

## 5. 증상 → 원인 (역추적표)

원인에서 멀리 떨어진 곳에서 터지는 것들만 모았다.

| 증상 | 진짜 원인 |
|---|---|
| `ZeroDivisionError` (`data_utils.py:11`) | **MatUp이 두 개 돌고 있다.** 먼저 끝난 쪽이 `main_optim.py:103-104`의 `rmtree`로 상대 입력을 지웠고, `load_renderings`의 glob이 빈 리스트를 반환해 `output_map_size`가 0으로 남았다. → `docker ps` 확인, 경로 분리 |
| `please download model at:` 후 조용히 종료 | 체크포인트가 없거나 CWD가 틀렸다. `sr_swinir.py:27`이 `./pretrained_models/...`를 **CWD 상대**로 찾고 없으면 `exit(0)` 한다. devcontainer 터미널은 `/workspace`에서 열리므로 `cd external/matup_egsr` 필요 |
| 체크포인트가 있는데도 로드 실패 | 부분 다운로드. 크기가 정확히 **142,473,939** 바이트여야 한다 |
| `/data`가 비어 있다 | `.env`의 `DAS_ROOT` 미설정, 또는 Docker Desktop 파일 공유 누락, 또는 **DAS 베이가 빠졌다**. 마운트 실패는 에러가 아니라 빈 디렉터리로 나타난다 |
| 빌드한 이미지가 `docker images`에 없다 | WSL 창에서 빌드해 docker-ce로 들어갔다. `docker context show` |
| 워커가 bus error로 죽는다 | `/dev/shm` 부족. compose의 `shm_size: 8gb`를 거치지 않고 `docker run`으로 돌린 경우 |
| Ctrl+C 했는데 계속 돈다 | **`docker compose run`은 클라이언트를 죽여도 컨테이너가 산다.** `docker ps` → `docker stop` |
| 렌더가 누락되거나 중복된다 | `--nb_lights`만 바꿨다. `--up_batch_size`와 짝이다 (`up_batch_size`가 `nb_renders`를 나눠떨어뜨려야 한다). 기본 100/10 |
| TF 실행 시 cuBLAS·NUMA 경고 | 무해하다. 쫓지 말 것 (OPEN-ITEMS #5) |
| 서브모듈 폴더가 비어 있다 | 초기화 안 된 것. `SRFlow`가 비어 있는 건 **의도**다 (재현 불가·라이선스 상이) |

DeepTextures 쪽은 실패가 다른 모양으로 나온다. `variants/common/lib.ps1`이 로그에서 아래를 찾아 분류한다:

| 증상 | 진짜 원인 |
|---|---|
| 학습이 "잘 도는데" 몇 주 걸릴 기세 | **TF가 조용히 CPU로 폴백했다.** MatUp은 CUDA 그래프라 GPU가 죽으면 크래시하지만 TF는 그냥 느려진다. `Skipping registering GPU devices`가 서명. 트레이너가 시작 시 GPU 가시성 + matmul 타이밍을 검사한다 |
| `training done`인데 `[metrics]`가 한 줄도 없다 | **0레코드 tfrecord.** `.repeat()`가 빈 데이터셋이면 아무것도 내놓지 않아 루프가 아예 안 돈다. upstream은 여기서 성공처럼 끝난다 |
| `[fatal] pretrain_dir is set ... but no checkpoint is there` (exit 9) | `-Stage pretrain`을 안 돌렸다. **upstream이라면 `print` 한 줄 뒤 랜덤 초기화로 진행했을 자리다** (OPEN-ITEMS #6) |
| `[fatal] lr_steps[0] >= niter` (exit 8) | 죽은 LR 스케줄. `esrgan_scaled.yaml`을 쓰거나, 의도라면 `--allow-dead-lr-schedule` (OPEN-ITEMS #7) |
| `[fatal] non-finite loss` (exit 4) | 디버그 체크포인트가 저장돼 있다. `--losses corrected`인지, 사전학습이 실제로 복원됐는지 확인 |
| `loss_D`가 즉시 `0.0000`으로 떨어진다 | **D가 G를 즉시 이겼다.** 사전학습 없이 GAN을 시작하면 나오는 정상적(그러나 나쁜) 거동이다. `docs/DEEPTEX-DEVIATIONS.md` §2.5 |
| `ResourceExhaustedError` / OOM | `batch_size`를 낮춘다. `train_esrgan.py:86`의 `persistent=True` 탓에 활성화가 전부 유지돼 footprint가 평범한 스텝의 약 3배다 |
| tfrecord를 열었는데 조용히 짧게 끝난다 | 잘렸다. 크기·sha256이 `.meta.json`과 다르다. `texsr_verify_tfrecord.py` |
| 체크포인트 복원 시 `Inconsistent references` 경고 | 무해하다. 우리가 로거를 다시 켜서 보이는 것뿐 (`docs/DEEPTEX-DEVIATIONS.md` §3) |

---

## 6. 파라미터 — 바꿔도 되는 것 / 짝으로 바꿔야 하는 것

| | |
|---|---|
| 단독으로 안전 | `--nb_epochs`, `--learning_rate`, `--alpha_reg`, `--tile_width`, `--batch_size` |
| **반드시 짝으로** | `--nb_lights` ↔ `--up_batch_size` |
| **반드시 짝으로** | 추출 `--lr-size` ↔ MatUp `--material_size` (둘 다 기본 128) |
| 실행마다 분리 | `--render_dir`, `--upsamplings_dir`, `--output_folder`, `--input_mat` |
| MatSynth엔 필수 | `--ogl_normals` |
| 건드리지 말 것 | `--upscaler srflow` (SRFlow는 체크포인트 배포 중단) |

OOM이면 `--batch_size`를 128에서 낮춘다. `--nb_lights`가 아니다.

---

## 7. 현재 환경 실측값

주장이 아니라 측정값이다. 재확인 명령을 같이 적는다.

| | matup | deeptex | dataprep |
|---|---|---|---|
| Python | 3.10 | 3.8 | 3.11 |
| 핵심 | torch 2.0.0 / CUDA 11.7 | TF 2.10.1 / CUDA 11.2 | pyarrow 17.0.0 |
| numpy | 1.26.4 | 1.23.4 | <2 |
| opencv | 4.11.0.86 | 4.14.0.94 | headless <5 |
| 기타 | torchvision **0.15.0**, timm 0.6.13 | keras 2.10.0 | |

```powershell
docker compose run --rm matup pip list --format=freeze | Select-String "^torch|^numpy|^opencv|^timm"
```

torchvision이 0.15.0인 것은 **오류가 아니라 확인된 편차**다 (OPEN-ITEMS #2).
베이스 이미지가 upstream 핀 0.15.1을 싣고 있지 않고, 이 프로젝트엔 호출 지점이 없다.

opencv가 두 이미지에서 다른 4.x인 것도 정상이다. 핀의 목적은 **메이저 5.x 회피**였고
(OPEN-ITEMS #8), 그 이하는 각 이미지의 Python·numpy 제약이 정한다.

### DeepTextures 학습 실측 (2026-09-16, RTX 3080 Ti, `llm-lab`이 VRAM 2 GB 점유 중)

| | PSNR 단계 | ESRGAN 단계 |
|---|---|---|
| batch_size | 16 | 8 |
| 처리량 | **3.62 steps/s** | **4.75 steps/s** |
| peak VRAM | 4427 MB | 3234 MB |
| 40k 예상 | **약 3.1시간** | **약 2.3시간** |
| 체크포인트 (3개 유지) | 192 MB × 3 | **375 MB** × 3 |

두 단계 합쳐 **약 5.4시간**. 200스텝 프로브로 잰 값이고, `[metrics]`의 `steps_per_sec`·`eta_h`가
실행 중 같은 값을 계속 갱신한다.

**ESRGAN 단계가 PSNR 단계보다 빠른 것은 오타가 아니다** — batch가 절반(8 vs 16)이라
D와 VGG19를 추가로 돌리고도 스텝당 비용이 더 싸다.

### 데이터셋 실측 (`textures_db`, 300장 → 2,965 타일)

`texsr_normalize.py`가 보고한 것 (`normalize_manifest.json`):

| | 장수 |
|---|---|
| 처리됨 | 299 / 300 (`221.png` 64²는 `gt_size` 미만이라 제외) |
| **JPG 구제** (upstream `*.png` glob이 버렸을 것) | **100** |
| 16-bit → 8-bit | 45 |
| 알파 삭제 | 119 (**그중 11장은 실제로 투명 영역이 있었다**) |
| 4의 배수로 크롭 | 142 |

`texsr_make_pairs.py` (crop 480 / step 240): **train 2,965 + val 264 타일** (홀드아웃은 원본 이미지 단위 10%).
`texsr_verify_tfrecord.py`: **PASS**, 정렬 `max |lr - imresize_np(hr)| = 0`.

---

## 8. 새 실험을 추가할 때

1. 데이터 가공이 필요하면 **`dataprep`에서** 한다. `matup`/`deeptex`에 패키지를
   추가하지 않는다 — 그 둘은 논문의 환경을 고정하는 것이 존재 이유다.
2. 의존성을 늘릴 땐 **핀을 박고 사유를 줄 주석으로** 남긴다. 이 저장소가 이미 두 번
   당했다 (OPEN-ITEMS #2, #8).
3. 검증할 수 없는 주장을 주석에 쓰지 않는다. 측정했으면 값과 날짜를, 안 했으면
   "미확인"이라고 쓴다.
4. 결과물은 `runs/` 아래로 보낸다 (gitignore 대상). **DAS(U:)에는 쓰지 않는다** —
   읽기 전용으로 마운트돼 있고, 유일한 MatSynth 사본이 있고, 신뢰할 수 없다.
5. 오래 걸리는 실험 전에 §1을 다시 돌린다.
