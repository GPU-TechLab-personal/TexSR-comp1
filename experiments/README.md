# experiments/ — 실험 리스트 1–5 구현

`실험리스트.md`(2026-09-29)의 다섯 실험과 공통 평가 프로토콜을 구현한 폴더다.
**구현과 짧은 게이트까지만 돌렸다. 시간 단위 실행은 아래 명령으로 직접 돌린다.**

기존 파이프라인(`docker/`, `scripts/`, `runs/matsynth`, `runs/montages`)은 **한 줄도 바꾸지 않았다.**
MatUp 서브모듈도 수정하지 않고 바깥에서 import해 구동한다 (`matup_render.py`와 같은 규약).

### 실행 순서 (리스트의 우선순위대로. 데이터 준비는 ESRGAN 학습셋만 남음)

모두 `experiments\docker`에서. 시간은 §8 측정값에서 환산한 것.

```powershell
# 1. 실험 1 파일럿 (5재질) — 연구 동기 확인                          ~30분
docker compose run --rm exp python experiments/exp1_inconsistency/run_C.py --pilot --s swinir bilinear
docker compose run --rm exp python experiments/exp1_inconsistency/run_residuals.py --pilot --s swinir --with-init
docker compose run --rm exp python experiments/exp1_inconsistency/analyze.py
# 2. 실험 4 파일럿 (orig는 위 결과 재사용)                           ~1시간
docker compose run --rm exp python experiments/exp4_grf_stats/run.py --pilot --s swinir
docker compose run --rm exp python experiments/exp4_grf_stats/analyze.py
# 3. 실험 2·3: 학습셋 완성(~2-3시간, 이어하기) 후 팔마다 ~5-7시간 (2-2는 ~20시간)
docker compose run --rm exp python experiments/data/build_train_set.py
docker compose run --rm exp python experiments/exp2_esrgan/train.py --config experiments/exp2_esrgan/configs/2-0_base_scratch.yaml
# 4. 실험 5: 이미 1회 측정됨 (§9). 재측정은 같은 명령
```

---

## 0. 환경

| | |
|---|---|
| 이미지 | `texsr/exp:torch2.0.0-cu117` — **`texsr/matup` 위에 쌓은 파생 이미지.** MatUp과 같은 torch 2.0.0 스택이라 "MatUp 원본" 팔이 원본 그대로다. matup 이미지 자체는 건드리지 않았다 (RUNBOOK §8) |
| 추가 패키지 | pyarrow, flip-evaluator(FLIP), diffusers/transformers(확산 업샘플러·DiT 타이밍). 핀과 사유는 `docker/exp/requirements.txt` |
| `/workspace` | 저장소 |
| `/data` | **읽기 전용** MatSynth 클론 (`D:/experimentdata` → `/data/MatSynth/data/*.parquet`). `MATSYNTH_ROOT`로 변경 |
| `/expdata` | 실험이 쓰는 모든 것 (`D:/experimentdata/texsr_exp`). 추출 재질, SR 캐시, 체크포인트, 결과, 가중치 캐시. `EXP_DATA`로 변경 |

```powershell
cd experiments\docker
docker compose build          # texsr/matup 이 먼저 있어야 한다 (cd docker; docker compose build matup)
```

모든 명령은 `experiments\docker`에서 `docker compose run --rm exp python experiments/...` 형태다.

Docker VM 메모리는 15.5 GB다. **MatSynth 샤드는 샤드당 row group 1개(~1 GB)**라 행 단위로 읽어도
샤드 전체가 메모리에 올라온다 — 8 워커 인덱싱이 ENOMEM으로 죽었다. 그래서 (`common/shards.py`)
- 샤드를 컬럼 묶음 단위로 스트리밍하고 (normal+roughness는 LEAN 때문에 같이),
- 샤드마다 새 프로세스를 쓰고 (`maxtasksperchild=1` — 오래 사는 워커는 Arrow 할당자 때문에 2.7–4 GB까지 컸고 커널이 20번 OOM-kill 했다),
- 워커가 죽으면 `Pool`이 **멈춰 있는 대신** 타임아웃으로 에러를 낸다. 모든 데이터 준비 스크립트는 재실행 시 끝난 샤드를 건너뛴다.

**데이터 준비와 GPU 실험을 동시에 돌리지 말 것.** 같은 VM 메모리를 나눠 쓰며, 실제로 학습 DataLoader가
`read(): Cannot allocate memory`로 죽었다.

**MatSynth 한 재질 안에서도 맵 해상도가 다를 수 있다** (eval12 중 3개: 2048² 맵 + 4096² metallic). 추출은 맵마다
자기 배율로 줄이고, 같은 phase(normal+roughness) 안에서 크기가 다르면 작은 쪽을 큰 쪽으로 올린다(LEAN이 법선
분산을 원해상도에서 재야 하므로). 추출 끝에 모든 맵이 512²/128²인지 단언한다.

---

## 1. 폴더 구조

```
experiments/
  common/            모든 실험이 공유
    protocol.py      정규 재질 표현 + 통일 LR 프로토콜 + 증강(법선 회전 포함)
    metrics.py       공통 평가 프로토콜 (맵별 지표 + 렌더 공간 FLIP/LPIPS + 경계 영역)
    render.py        MatUp 렌더러(Cook-Torrance+GGX, 피보나치 광원 ×1.4) 래퍼
    upscalers.py     S: swinir / srflow / sdx4(확산) / bilinear / bicubic
    sr_cache.py      LR 렌더 + S 업샘플 캐시 (MatUp 1·2단계의 파일 단위 복제)
    matup_driver.py  MatUp 최적화 (eager 루프 + 추가 손실 훅) + 조명별 residual
    matup_runs.py    (재질, S, 변형) 1회 실행 = 실험 1 정밀지표 · 실험 4가 공유
    shards.py        메모리 제한 샤드 리더
  data/              check_eval_materials · matsynth_index · select_materials · extract · build_train_set
  exp1_inconsistency/  run_C.py (간단 지표 C) · run_residuals.py (정밀 지표) · analyze.py
  exp2_esrgan/       model · data · losses · train · evaluate · configs/2-*.yaml
  exp3_render_loss/  configs/3-*.yaml (트레이너는 exp2와 공유)
  exp4_grf_stats/    grf_stats.py (통계항) · run.py · analyze.py
  exp5_benchmark/    bench_infer · bench_train · gpu_hours · texel_cost
  gates/             짧은 검증 스크립트 (아래 §8)
  docker/            compose.yaml + exp/Dockerfile
```

---

## 2. 공통 평가 프로토콜 — 어디에 어떻게 구현됐나

| 리스트 항목 | 구현 |
|---|---|
| basecolor PSNR/SSIM/LPIPS(sRGB), ΔE | `metrics.basecolor_metrics` — PSNR/SSIM은 **저장소 공유 코드 `scripts/common/texsr_metrics.py`**(8-bit 사다리)를 그대로 호출, LPIPS-alex, CIEDE2000 |
| normal 평균 각도 오차, 5°/10° 비율 | `metrics.normal_metrics` (atan2 기반, 0 근처에서 arccos보다 안정) |
| roughness·metalness MAE, metalness 0/1 정확도 | `roughness_metrics`, `metallic_metrics` |
| height MAE, gradient 오차 | `height_metrics` (+평균 정렬 MAE: 법선 적분 height는 상수항이 자유) |
| 렌더 공간 FLIP/LPIPS, 피보나치 조명 평균 | `render_metrics` — MatUp 렌더러, 기본 16광원 (MatUp이 학습하는 100광원과 **다른** 세트) |
| 주장하지 않는 맵도 공개 | `evaluate(maps=claimed)` — claimed와 무관하게 존재하는 모든 맵을 보고 |
| LR 생성 통일 | `protocol.downsample` — basecolor는 **linear 공간** 평균, normal은 평균 후 **재정규화**, roughness는 **LEAN**(기본) / Toksvig / area, 모든 방법·실험이 이 함수 하나 |
| 같은 테스트셋·배율·crop | 평가 재질은 `/expdata/materials/eval12` 하나 (512² GT / 128² LR, ×4) |
| tiling 경계 영역 오차 | `band_mask` + `masked_pixel_metrics`, `exp2_esrgan/evaluate.py --tile N` |
| baseline off-the-shelf + 재학습 | 실험 2의 `2-0`(scratch 재학습) / `2-1`(자연영상 사전학습 init) |
| 입력 정보량 맞추기 | 실험 2-4 / 실험 3이 "모든 맵 입력"을 ablation 팔로 분리 |

**LR 프로토콜이 바뀌었으므로 평가 재질을 새로 추출했다** (`/expdata/materials/eval12`).
`runs/matsynth/`의 12개는 구 커널(sRGB 평균, 비정규화 법선)로 만든 것이고 커밋된 몽타주의 근거라 그대로 뒀다.

---

## 3. 데이터 준비

```powershell
# (0) 평가 12재질이 받은 샤드에 있는지 + 픽셀 동일성          ~1분
docker compose run --rm exp python experiments/data/check_eval_materials.py   # 또는 docker/의 dataprep
# (1) 평가 12재질을 새 프로토콜로 추출                         ~1분  (완료됨)
docker compose run --rm exp python experiments/data/extract.py --set eval12 --eval12
# (2) 받은 220샤드 전체 인덱스 (재질별 기술자)                  ~20분  (완료됨: 2,950재질 = test 45 + train 2,905, 전부 5맵 완비)
docker compose run --rm exp python experiments/data/matsynth_index.py
# (3) 실험 1 재질 50개 + 파일럿 5개 선정 → experiments/data/selection_exp1.json   (완료됨)
docker compose run --rm exp python experiments/data/select_materials.py --n 50
# (4) 실험 1 재질 추출                                          ~5분  (완료됨)
docker compose run --rm exp python experiments/data/extract.py --set exp1 --selection experiments/data/selection_exp1.json
# (5) ESRGAN 학습셋 (train 스플릿, eval12·실험1 재질 62개 제외, 512²)  ~2-3시간  ← 아직 55/2,871만 됨
docker compose run --rm exp python experiments/data/build_train_set.py
```

**(3)을 (5)보다 먼저** 돌려야 실험 1 재질이 학습셋에서 빠진다.

(5)는 시간 단위라 게이트에서 멈췄다(0.3–0.4재질/s, 55개 완료). 같은 명령을 다시 치면 이어서 한다.
`materials.json`은 **완료 시에만** 써지고, 트레이너는 그것이 없으면 시작을 거부한다 — 멈춘 빌드의 일부로
조용히 학습하는 일은 없다.

선정 방법(`select_materials.py`): normal 강도(평균 기울기 각) × roughness 평균을 각각 순위 분위로 바꾼
단위 정사각형을 ⌈√n⌉² 격자로 나누고 칸마다 중심에 가장 가까운 재질을 고른다 (test 스플릿 우선).
파일럿 5개는 네 모서리 + 중심.

실제 선정 결과: 풀 2,938(eval12 제외) 중 50개 = test 16 + train 34, 13개 카테고리 전부,
기울기 0.3°–32.9°(중앙 7.7°), roughness 0.10–0.98(중앙 0.60).
파일럿: `tc_marble_013`(1.0°, 0.10) · `acg_solar_panel_004`(0.5°, 0.93) · `acg_leather_008`(27.9°, 0.29) ·
`0315_concrete_010`(26.8°, 0.90) · `1105_damaged_broken_wall_v2`(6.2°, 0.55).

---

## 4. 실험 1 — MatUp 조명 간 불일치 원인 분석

| | |
|---|---|
| 간단 지표 | `run_C.py`: D_ω = S(R_ω) − bilinear↑(R_ω), C = 픽셀 평균 std_ω(D_ω). R_ω는 **MatUp 자신의 LR 렌더**(100광원, AA 2×2), S 입력은 MatUp과 바이트 동일 (게이트로 확인) |
| 정밀 지표 | `run_residuals.py`: MatUp 최적화 후 텍셀×조명 residual. "버린 조명 비율"은 정의가 하나가 아니라 둘 다 보고: 텍셀 중앙값의 2×/3× 초과 비율, 절대 임계 0.02/0.05/0.1 |
| 분석 | `analyze.py`: S별로 C(와 C_rel)와 normal gradient 분산 / tilt / roughness 평균 / **base color 대비(대조군)**의 Pearson·Spearman |
| S | `swinir`(SwinIR-L ×4 GAN), `sdx4`(stable-diffusion-x4-upscaler), `srflow`(아래 주의), `bilinear`(대조) |

```powershell
# 파일럿 (5재질)
docker compose run --rm exp python experiments/exp1_inconsistency/run_C.py --pilot --s swinir bilinear
docker compose run --rm exp python experiments/exp1_inconsistency/run_residuals.py --pilot --s swinir
# 본 실험 (50재질)
docker compose run --rm exp python experiments/exp1_inconsistency/run_C.py --s swinir sdx4 bilinear
docker compose run --rm exp python experiments/exp1_inconsistency/run_residuals.py --s swinir sdx4
docker compose run --rm exp python experiments/exp1_inconsistency/analyze.py
```

- **bilinear의 C ≈ 0.0011은 0이 아니라 8-bit 양자화 바닥이다** (1/(255·√12) = 0.00113). S의 C는 이 값과 비교한다.
- **SRFlow는 돌지 않는다.** 체크포인트 배포가 중단됐고 서브모듈은 의도적으로 비어 있다(README). 어댑터는
  구현돼 있고 코드+체크포인트를 직접 넣으면 `--s srflow`로 돈다. 없으면 SKIP을 출력하고 넘어간다.
- `sdx4` 설정: `--s-opt sdx4.steps=20 sdx4.noise_level=20 sdx4.guidance=0 sdx4.seed_mode=per_light`(기본값).
  `seed_mode=shared`는 모든 조명에 같은 노이즈 — 그 자체로 불일치를 낮추는 손잡이라 태그에 기록된다.
  첫 실행 시 가중치(~3.5 GB)를 `/expdata/cache/huggingface`에 받는다.
- SR 렌더는 `/expdata/cache/sr/`에 캐시되어 실험 4가 재사용한다 (확산 S가 가장 비싸다).

## 5. 실험 2 — ESRGAN 비-basecolor 맵 저하 원인 분리

트레이너 하나(`exp2_esrgan/train.py`) + 팔마다 config 하나. **모든 팔은 `2-0`을 상속하고 요인 하나만 바꾼다.**

| config | 바꾸는 것 | 검증하는 가설 |
|---|---|---|
| `2-0_base_scratch` | — (공유 RRDBNet, 맵 독립 3→64 입력, 전 맵 ESRGAN 손실, scratch) | 기준 |
| `2-1_base_pretrained` | init: 자연영상 RRDBNet (`RealESRGAN_x4plus`) | 도메인 prior |
| `2-2_separate` | 맵별 별도 네트워크 | 용량/fitting |
| `2-3_mapaware_loss` | normal=각도 오차, roughness·metallic=L1만 | loss |
| `2-4_concat_input` | 15→64 concat 입력 | 맵 간 정보 |

공통: MatSynth train 스플릿, HR 128 패치 / LR 32(통일 프로토콜, 온더플라이), 배치 4 패치, 100k iter
(앞 20k는 ESRGAN식 PSNR 워밍업), bf16. 사전학습 가중치는 **공식 ESRGAN이 Google Drive 전용이라**
같은 아키텍처의 `RealESRGAN_x4plus`(GitHub release)를 기본으로 받는다. 공식 `.pth`를 받아 두면
`--set init.weights=/expdata/cache/pretrained/RRDB_ESRGAN_x4.pth`로 그대로 로드된다(구 명명 자동 변환).

```powershell
docker compose run --rm exp python experiments/exp2_esrgan/train.py --config experiments/exp2_esrgan/configs/2-0_base_scratch.yaml
# ... 2-1 ~ 2-4 동일
docker compose run --rm exp python experiments/exp2_esrgan/evaluate.py --run exp2/2-0_base_scratch
```

선행 확인 항목(Alonso & Patow의 fine-tune 여부)은 논문 확인 사항이라 코드로 해결되지 않는다.

## 6. 실험 3 — ESRGAN + 렌더링 loss

| config | |
|---|---|
| `3-0_maploss` | concat 15ch, 사전학습 body + **새 head**(conv_first/conv_last 재초기화 — `setup.json`에 기록), 맵 공간 손실만 |
| `3-1_render` | + 렌더링 loss: 매 iter 100광원 중 k=8 랜덤, SR·GT 맵을 **MatUp 렌더러**로 같은 광원에서 렌더 → L1 (`metric: logl1`이면 linear 복사휘도 log-L1) + 약한 맵별 L1 (metallic·height 0.1, 나머지 0.01) |
| `3-2_render_jointD` | + joint D (15ch 전체를 보는 판별자 하나) |

법선은 렌더 전에 정규화된다(`data.decode_t`). 확인할 점(metalness·height 개선)은 `evaluate.py`의 맵별 표에서 본다.

## 7. 실험 4 — GRF 통계항 (MatUp 기반)

`grf_stats.py` 머리 주석에 유도가 있다. 요약:

- MLP가 bilinear init 위에 더하는 디테일 d(x)를 정상 가우시안 랜덤 필드로 보고 **2차 모멘트**를 맞춘다. FFT 없음.
- 목표 모멘트는 재질마다 최적화 전에 한 번 추정: 렌더를 init 주변에서 선형화하면
  E[EEᵀ] = JΣJᵀ + sI (E = S 렌더 − init 렌더, J = 3×7 렌더 야코비안). **Σ에 대해 선형**이라 최소제곱으로 풀린다.
  같은 식을 유한차분에 적용해 Σ_g = Cov(∇d)(lag-1, 스펙트럼 기울기 담당)도 구한다.
- **변형 A**: Σ가 맵별 블록 대각 (albedo 3×3 | metallic | normal xy 2×2 | alpha). **변형 B**: 전체 7×7 (맵 간 cross-covariance).
- 손실: P×P 출력 패치 배치의 표본 모멘트를 Σ, Σ_g에 상관 정규화해 맞춤. 총 손실 = MatUp(Lfit+Lreg) + λ·L_stat.

```powershell
docker compose run --rm exp python experiments/exp4_grf_stats/run.py --pilot --s swinir
docker compose run --rm exp python experiments/exp4_grf_stats/run.py --s swinir sdx4 --lambdas 0.001 0.003 0.01
docker compose run --rm exp python experiments/exp4_grf_stats/analyze.py
```

`orig`(MatUp 원본) 팔은 실험 1 `run_residuals.py`와 **같은 실행 결과를 공유**한다 (`/expdata/runs/matup/<set>/<S>/<재질>/orig`).
`analyze.py`는 변형별 이득(부호 정규화, Wilcoxon)과 **이득 vs 실험 1의 C** Spearman을 낸다 — 리스트의 핵심 분석.

**결정이 필요한 것 두 가지** (게이트에서 측정, `grf_stats.py`에 기록):
1. acg_wood_floor_060(비금속)에서 추정 Σ의 최대 분산이 **metallic**(4e-3, 다른 채널 < 3e-4)에 몰린다. 렌더가
   metallic에 가파르기 때문인데, MatUp 자신의 metallic 정규화(Lreg)와 정면으로 부딪힌다. 기본은 리스트대로
   전 맵이고, `--stat-exclude metallic`이 비교 팔이다.
2. λ=0.01에서 통계항이 첫 epoch 평균 l2,1의 4.5배였다 → 기본 λ=0.003. 스윕 권장.

## 8. 게이트 — 무엇을 확인했나 (2026-09-29, RTX 3080 Ti)

| 게이트 | 결과 |
|---|---|
| `data/check_eval_materials.py` | 평가 12재질 **전부** 받은 `test-00000`의 기록된 row에 있고, 그 샤드에서 재유도한 GT·LR이 `runs/matsynth/`와 **픽셀 동일**(12/12) |
| `gates/gate_protocol.py` | linear 평균(흑백 체커 → sRGB 0.7354), 법선 재정규화, 평탄 법선에서 roughness 보존, LEAN≈Toksvig, rot/flip 증강의 법선 변환(8가지, 높이장에서 재계산한 법선과 최대 0.015°), 16-bit 왕복, 지표 항등성(FLIP·LPIPS 0) |
| `gates/gate_matup_driver.py` | LR 렌더 100장·SwinIR 업샘플 100장이 MatUp 원본과 **바이트 동일**. eager 루프 vs 실제 `main.py`(CUDA graph): normal 0.6099° vs 0.6092°, basecolor PSNR 31.568 vs 31.557 dB, FLIP 0.0745 vs 0.0740 |
| `gates/gate_exp4.py` | 합성 데이터에서 알려진 Σ 복원(B 1.2%, A 0.7%, 노이즈 3.5%), 실제 재질에서 PSD·유한, 변형 B로 1 epoch 최적화 |
| `gates/gate_configs.py` | 각 팔 config가 기준 팔과 **이름 붙은 요인만** 다름 (2-3의 맵별 손실은 병합이 아니라 교체) |
| `gates/gate_train_smoke.py` | 8개 팔 전부: PSNR→GAN 두 단계, 평가, 체크포인트, **재개**. 사전학습 init 702 텐서 로드, concat은 head 3텐서만 재초기화 |

게이트가 잡은 버그 (전부 수정됨): GPU에서 8-bit 입력을 다시 양자화하면 1 ulp가 움직여 SwinIR 출력 1%가
MatUp과 달라짐 · 혼합 해상도 재질의 metallic이 1024²로 추출됨 · 샤드 리더 OOM과 Pool 무한 대기 ·
torch 2.0에 bf16 nearest-upsample 커널 없음 · z=0 법선에서 MatUp 렌더러의 0/0(렌더 loss NaN) ·
config 상속이 맵별 손실을 교체 대신 병합.

### 소요 시간 (측정값에서 환산)

| 작업 | 단위 비용 | 전체 |
|---|---|---|
| 실험 1 C, SwinIR | 재질당 ~25 s | 50재질 ~20분 |
| 실험 1 C, sdx4 | 미측정 (가중치 미다운로드) | — |
| MatUp 1회 (eager, SR 캐시 있음) | 재질당 ~3.5분 | 실험 1 정밀: 50 × S당 ~3시간 |
| 실험 4 변형 1회 | 재질당 ~6분 (통계항 배치 포함, 1 epoch 36 s) | 50 × 2변형 × S × λ당 ~10시간 |
| 실험 2/3 팔 1개, 100k iter | 모델만 4.1–6.4 it/s → **~5–7시간**, `2-2_separate` 1.41 it/s → **~20시간** | 8팔 합계 ~65시간 |

실험 2/3의 `niter`(100k)와 배치(4)는 제가 정한 기본값이다 — 시간 예산에 맞춰 `--set train.niter=...`로 바꾼다.
모든 팔을 같은 값으로 돌려야 비교가 성립한다.

## 9. 실험 5 — 속도·메모리

```powershell
docker compose run --rm exp python experiments/exp5_benchmark/bench_infer.py     # RRDBNet/SwinIR/DiT 1·4·20·50/MatUp MLP
docker compose run --rm exp python experiments/exp5_benchmark/bench_train.py     # 팔별 iter/s, MatUp step/s
docker compose run --rm exp python experiments/exp5_benchmark/texel_cost.py      # 텍셀당 MAC·저장량
docker compose run --rm exp python experiments/exp5_benchmark/gpu_hours.py --target normal_angle<=3 --target basecolor_psnr>=28
```

조건: LR 128² → ×4, fp16, 워밍업 10회 후 100회 평균, 매 회 `torch.cuda.synchronize()`,
`max_memory_allocated()`. DiT는 VAE encode/decode 포함. **DiT는 공개 SVBRDF 체크포인트가 없어 DiT-XL/2 구조의
랜덤 가중치로 잰다** — 시간·메모리는 구조가 정하고 가중치 값과 무관하다(출력에 명시). SwinIR은 attention
mask를 fp32로 만들어 `.half()`가 불가능해 autocast fp16으로 잰다. GPU-hours는 트레이너의 `train_seconds`(평가·저장
시간 제외, 동기화)에서 목표 첫 도달 시점을 읽는다 — 외삽하지 않는다.

### 측정값 (2026-09-29, RTX 3080 Ti, 게이트 겸 본 측정; `/expdata/runs/exp5/`)

추론 (128² → 512², fp16, 배치 1, 워밍업 10 + 100회):

| 모델 | ms | peak MB | params |
|---|---|---|---|
| RRDBNet, 맵 1장 | 33.7 ± 6.2 | 168 | 16.7M |
| RRDBNet, 5맵(독립 입력) | 124.6 ± 0.6 | 713 | 16.7M |
| RRDBNet, concat 15ch | 33.3 ± 4.9 | 169 | 16.7M |
| SwinIR-L (autocast) — MatUp은 이것을 렌더 100장에 돌린다 | 143.0 ± 12.1 | 399 | 28.0M |
| DiT-XL/2 + VAE, 1 / 4 / 20 / 50 step (**랜덤 가중치**) | 131 / 222 / 730 / 1662 | 2225 | 750M + 84M |
| MatUp MLP, 재질 전체 1회 평가 | 12.1 ± 0.3 | 145 | 0.31M |

학습 (모델만, 합성 텐서, 50 step): 실험 2/3 팔 4.1–6.4 it/s(peak 1.7–3.5 GB), `2-2_separate` 1.41 it/s(5.4 GB).
MatUp 최적화 116 step/s → 재질당 165 s(10 epoch, eager). 스모크 학습(데이터로더 워커 2)은 2.3 it/s였으니
실제 실행에서 데이터로더가 병목인지 `log.jsonl`의 `it_per_s`로 확인할 것 (기본 워커 6).

텍셀당 비용: ESRGAN 독립 입력 5.6e6 MAC/재질 텍셀, concat 1.13e6, MatUp MLP 3.1e5 (+ 재질마다 선행 최적화 ~165 s).
512² 재질 저장량: 원본 16-bit 4.5 MB, ESRGAN(LR만) 288 KB, MatUp(LR + MLP fp16) 893 KB.
