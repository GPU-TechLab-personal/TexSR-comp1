# variants/experiments/

실험 리스트(2026-09-29, 저장소 밖 문서)의 실험 1–5와 공통 평가 프로토콜을 구현한 코드다. 실험마다 폴더가 하나이고, 공통 코드
(`common/`), MatUp 구동기(`matup_driver/`), 게이트(`gates/`)를 함께 쓴다. 실험 계획, 실행 명령, 측정 기록, 소요 시간은
[`EXPERIMENTS.md`](EXPERIMENTS.md)에 있다.

## 구성

| 폴더 | 실험 / 답하는 질문 | 진입점 |
|---|---|---|
| `common/` | 모든 실험이 쓰는 공통 프로토콜: 정규 재질 표현, LR 생성, 평가 지표, MatUp 렌더러, SR 캐시, 업샘플러 S | — |
| `matup_driver/` | MatUp 최적화 루프를 서브모듈 밖에서 구동한다. extra_loss 훅이 있다. 실험 1·4·5가 공유한다 | — |
| `light_inconsistency/` | **실험 1.** MatUp의 조명 간 불일치는 재질(normal 디테일, roughness) 때문인가, 업샘플러 S 때문인가 | `run_C.py`, `run_residuals.py`, `analyze.py` |
| `esrgan_map_degradation/` | **실험 2.** ESRGAN에서 basecolor 외 맵이 나빠지는 원인은 도메인 prior, 용량, loss, 맵 간 정보 중 무엇인가. 실험 2·3의 트레이너와 평가기도 여기 있다 | `core/train.py`, `core/evaluate.py` |
| `esrgan_render_loss/` | **실험 3.** ESRGAN에 렌더링 loss(와 joint D)를 더하면 metalness·height가 나아지는가 | (트레이너는 `esrgan_map_degradation/core/train.py`) |
| `matup_grf_stats/` | **실험 4.** MatUp에 GRF 2차 모멘트 통계항을 더하면 나아지는가, 실험 1의 불일치 C가 클수록 더 나아지는가 | `run.py`, `analyze.py` |
| `speed_memory/` | **실험 5.** 방법별 추론·학습 속도, 메모리, 목표 지표까지의 GPU-hours, 텍셀당 비용 | `bench_infer.py`, `bench_train.py`, `gpu_hours.py`, `texel_cost.py` |
| `gates/` | 구현이 문서대로 동작하는지 확인하는 짧은 검사(수 초~5분) | `gate_*.py` |

### `common/`

| 파일 | 하는 일 |
|---|---|
| `paths.py` | 컨테이너 경로 상수(`/workspace`, `/data`, `/expdata/{materials,cache/sr,cache/pretrained,runs}`). `add_matup()`·`add_swinir()`는 `variants/util/paths.py`의 `use_matup()`·`use_swinir()`를 부른다 |
| `protocol.py` | 정규 재질 표현(float32 dict, OpenGL normal)과 디스크 형식(MatUp 레이아웃, 16-bit PNG), 통일 LR 프로토콜 `downsample`(basecolor linear 평균, normal 평균 후 재정규화, roughness LEAN/Toksvig/area), 증강 `flip_rot` |
| `metrics.py` | 공통 평가 프로토콜: 맵별 지표(basecolor PSNR/SSIM/LPIPS/ΔE, normal 각도, roughness·metallic MAE, height MAE·gradient), 렌더 공간 FLIP/LPIPS, tiling 경계 `band_mask`·`masked_pixel_metrics`. PSNR/SSIM은 `variants/util/texsr_metrics.py`를 부른다 |
| `render.py` | MatUp 렌더러(Cook-Torrance + GGX, 1.4 반경 피보나치 광원)를 서브모듈 밖에서 구동한다. MatUp 실행, 실험 3 렌더 loss, 렌더 공간 평가가 모두 이것을 쓴다 |
| `upscalers.py` | 업샘플러 S: `swinir`, `srflow`, `sdx4`, `bilinear`, `bicubic`. 모두 `up(x)` 하나로 부르고 출력은 8-bit 사다리로 반올림한다 |
| `sr_cache.py` | 재질·S 설정·광원 수별 LR 렌더와 S 업샘플 캐시(`/expdata/cache/sr/`). MatUp의 1·2단계와 파일 단위로 같아서 MatUpDataset에 그대로 넘길 수 있다. manifest는 디렉터리 옆 `<dir>.json`에 둔다 |
| `matup_runs.py` | MatUp 실행 1회 = (세트, 재질, S, 변형). `run_one`이 캐시 → 구동기 → 평가 → `result.json`을 한다. 실험 1과 4가 `orig` 실행을 공유한다 |
| `cli.py` | 진입점 공통 인자: `--set`, `--materials`, `--pilot`, `--selection`, `--limit`, `--s`, `--s-opt`. 없는 S는 SKIP을 출력하고 넘어간다 |
| `shards.py` | MatSynth parquet 샤드를 컬럼 묶음 단위로 읽는다(샤드 하나가 row group 하나라 행 단위 읽기는 메모리를 다 쓴다). 데이터 준비 스크립트가 쓴다 |

### `matup_driver/`

| 파일 | 하는 일 |
|---|---|
| `upstream.py` | 그대로 재사용하는 MatUp 모듈의 import(FullyConnected, MatUpDataset, mlp_forward_pass, loss_fn, 렌더러, eval_model, output_maps_from_list) |
| `args.py` | `matup_args`: `main.py` argparse 기본값의 Namespace, `opengl_normals=True`만 다르다 (i) |
| `optimise.py` | `optimise`: MatUp 최적화를 eager 루프로 돌린다. extra_loss 훅 (f)이 있다 |
| `reference.py` | `init_maps`: 최적화하지 않은 bilinear init(`init` 변형) (j) |
| `outputs.py` | `write_outputs`(MatUp PNG) (e), `to_canonical`(내부 맵 → 정규 재질, height는 bilinear LR), `save_json` |
| `residuals.py` | `per_light_residuals`: 실험 1 정밀 지표(텍셀 × 조명 residual) |
| `__init__.py` | 위 이름을 모두 다시 내보낸다(`from variants.experiments import matup_driver as D; D.optimise(...)`) |

### `light_inconsistency/` (실험 1)

| 파일 | 하는 일 |
|---|---|
| `run_C.py` | 간단 지표 C: D_w = S(R_w) − bilinear↑(R_w), C = 픽셀 평균 std_w(D_w). D_rms, consistent, C_rel과 재질 요인도 기록한다 → `/expdata/runs/exp1/C/<S tag>/<재질>.json`, `<재질>_Cmap.npy` |
| `run_residuals.py` | 정밀 지표: MatUp을 돌린 뒤 텍셀 × 조명 residual 분포. `orig` 변형으로 `/expdata/runs/matup/`에 저장되어 실험 4가 재사용한다. `--with-init`은 `init` 변형도 돌린다 |
| `analyze.py` | S별로 C(와 C_rel)와 재질 요인(normal_grad_var, normal_tilt_deg, roughness_mean, 대조군 basecolor 대비)의 Pearson·Spearman. residual 실행이 있으면 discard 비율도 상관 → `/expdata/runs/exp1/analysis/{summary.md, correlations.csv, scatter_<S>.png}` |

### `esrgan_map_degradation/` (실험 2, 실험 2·3 공통 트레이너)

| 파일 | 하는 일 |
|---|---|
| `core/config.py` | `load_config`: `base:` 상속, `--set a.b=v` 덮어쓰기. `loss.per_map` 아래는 병합하지 않고 교체한다(`REPLACE_CHILDREN`) |
| `core/assemble.py` | config 키에서 팔을 판정해 G, init, D, Criterion을 만든다(`build`). 판정 표가 머리 주석에 있다 |
| `core/model.py` | RRDBNet, VGG식 D, VGG perceptual(BasicSR 이름 그대로), 맵 ↔ 3채널 인코딩 정의 |
| `core/dataset.py` | `TrainSet`(`/expdata/train/*/HR`에서 HR 패치, LR은 통일 프로토콜로 즉석 생성), encode/decode |
| `core/train.py` | 트레이너. PSNR 워밍업 → GAN 단계, 주기적 quick eval, 체크포인트, 자동 이어하기 |
| `core/evaluate.py` | 학습된 G의 전체 프로토콜 평가(eval12, 맵별 + 렌더 공간 16광원). `--tile N`이면 tiling 경계를 따로 보고한다 |
| `arms/base.py` | 2-0 기준 팔과 모든 팔이 꽂히는 뼈대: `SharedSR`, `Discriminators`(D 없음), `SharedD`, `Criterion`, `L1Term`, `VGGTerm`, `PerMapGanTerm`, RaGAN |
| `arms/pretrained.py` | 2-1: RealESRGAN_x4plus(또는 공식 ESRGAN .pth) init. 3-x도 쓴다 |
| `arms/separate.py` | 2-2: 맵별 G 5개(`SeparateSR`)와 맵별 D 5개(`PerMapD`) |
| `arms/mapaware_loss.py` | 2-3: `AngularTerm`(normal 각도 오차) |
| `arms/concat.py` | 2-4: 15→15 `ConcatSR`. 3-x의 G이기도 하다 |
| `configs/2-*.yaml` | 팔마다 config 하나. 모두 `2-0_base_scratch.yaml`을 상속한다 |

### `esrgan_render_loss/` (실험 3)

| 파일 | 하는 일 |
|---|---|
| `arms/render_loss.py` | 3-1: `RenderLoss`(매 iter MatUp 100광원 중 k개, SR·GT 맵을 MatUp 렌더러로 렌더, L1 또는 log-L1), `ExtraL1Term`, `RenderTerm` |
| `arms/joint_d.py` | 3-2: 15채널 joint D(`JointD`)와 `JointGanTerm` |
| `configs/3-*.yaml` | `3-0_maploss`(concat + pretrained, 맵 공간 loss만, 2-0 상속), `3-1_render`, `3-2_render_jointD` |

### `matup_grf_stats/` (실험 4)

| 파일 | 하는 일 |
|---|---|
| `grf_stats.py` | 통계항: 렌더를 init 주변에서 선형화해 목표 모멘트 Σ, Σ_g를 최소제곱으로 추정하고(FFT 없음), 패치 배치의 표본 모멘트를 맞추는 `StatLoss`. 변형 A(맵별 블록 대각), B(전체 7×7). `factory`가 구동기의 `extra_factory`다 |
| `run.py` | 재질 × S마다 `orig`(실험 1 결과가 있으면 재사용)와 (변형, λ)마다 1회 실행(`A_l0.01` 같은 태그). `--variants`, `--lambdas`(기본 0.003), `--stat-exclude` |
| `analyze.py` | 변형별 `orig` 대비 이득(부호 정규화), Wilcoxon, 이득 vs 실험 1 C의 Spearman → `/expdata/runs/exp4/analysis/{summary.md, gains.csv, gain_vs_C_<S>.png}` |

### `speed_memory/` (실험 5)

| 파일 | 하는 일 |
|---|---|
| `bench_infer.py` | 추론 속도·peak 메모리, 모든 모델 같은 프로토콜(128² → ×4, fp16, 배치 1, 워밍업 10 + 100회): RRDBNet 1맵/5맵/concat, SwinIR, DiT(랜덤 가중치) 1·4·20·50 step, MatUp MLP → `/expdata/runs/exp5/infer_<gpu>.{json,md}` |
| `bench_train.py` | 학습 처리량: 팔별 ESRGAN 1 step(합성 텐서, `bench_esrgan`), MatUp 최적화 step(`bench_matup`, eval12 SwinIR SR 캐시가 없으면 건너뛴다) → `/expdata/runs/exp5/train_<gpu>.json` |
| `gpu_hours.py` | `/expdata/runs/{exp2,exp3}/*/eval.jsonl`의 `train_seconds`에서 목표 지표 첫 도달 GPU-hours. 외삽하지 않는다 → `/expdata/runs/exp5/gpu_hours.json` |
| `texel_cost.py` | 출력 텍셀당 MAC(forward hook으로 센 값)와 재질당 저장량 → `/expdata/runs/exp5/texel_cost.json` |

### `gates/`

| 파일 | 확인하는 것 | 시간 |
|---|---|---|
| `gate_configs.py` | 각 팔 config가 기준 팔과 이름 붙은 요인만 다르다 | 수 초, GPU 없음 |
| `gate_protocol.py` | LR 프로토콜, 증강, 지표가 docstring대로 동작한다 | 수 초, GPU 선택 |
| `gate_matup_driver.py` | `matup_driver/`가 여전히 MatUp이다: LR·SR 렌더가 MatUp과 바이트 동일, eager 루프 vs 실제 `main.py`의 지표 차이 | ~5분 |
| `gate_exp4.py` | 모멘트 추정기가 알려진 공분산을 복원한다, 실제 재질에서 PSD·유한, 변형 B로 1 epoch | ~3분 |
| `gate_train_smoke.py` | 모든 팔 config가 몇 it 동안 두 단계, 평가, 체크포인트, 이어하기까지 돈다. 팔별 peak 메모리·it/s | GPU |

## 동작 방식

### 데이터

데이터 준비 스크립트(`check_eval_materials`, `matsynth_index`, `select_materials`, `extract`, `build_train_set`)는
저장소 루트의 `experiments/data/`에 있는 로컬 파일이다. `.gitignore:6`의 `data/` 규칙 때문에 저장소에 들어 있지 않다.
이 트리의 코드는 그 결과물만 읽는다.

| 경로 (컨테이너) | 내용 | 읽는 곳 |
|---|---|---|
| `/data/MatSynth/data/*.parquet` | MatSynth 샤드(읽기 전용) | 데이터 준비 |
| `/expdata/materials/eval12/<재질>/{LR,GT}/` | 평가 12재질(512² GT / 128² LR) | 실험 2·3 평가, 게이트, `bench_train` |
| `/expdata/materials/exp1/<재질>/` | 실험 1 재질 50개 | 실험 1·4 (`--set exp1`) |
| `/workspace/experiments/data/selection_exp1.json` | 선정 결과와 파일럿 5개 | `--pilot` (`cli.py`의 `--selection` 기본값) |
| `/expdata/train/<재질>/HR/`, `materials.json` | ESRGAN 학습셋. `materials.json`은 빌드가 끝나야 써지고, 없으면 트레이너가 시작을 거부한다 | 실험 2·3 학습 |

### 실험 1·4·5: MatUp 경로

```
재질 LR ─► sr_cache.ensure_sr_renders ─► /expdata/cache/sr/<set>/<재질>/radiance_LR_L100/, radiance_<S tag>_L100/
        ─► matup_runs.run_one(variant, extra_factory)
             ├─ init  : matup_driver.init_maps
             └─ orig / <tag> : matup_driver.optimise(args, extra_factory) ─► write_outputs
        ─► to_canonical ─► metrics.evaluate ─► (orig이고 요청하면) per_light_residuals
        ─► /expdata/runs/matup/<set>/<S tag>/<재질>/<variant>/{result.json, maps/, residual_mean.npy}
```

- 실험 1 `run_residuals.py`는 `orig`을 residual과 함께 돌린다. 실험 4 `run.py`는 같은 `orig`을 읽고,
  `grf_stats.factory(variant, λ)`를 `extra_factory`로 넘겨 변형을 돌린다. 실험 1 `run_C.py`는 MatUp을 돌리지 않고 같은 SR 캐시를 만들거나 읽기만 한다.
- 실험 5 `bench_train.bench_matup`은 `matup_driver`로 MatUp step을 잰다. eval12 재질 하나의 SwinIR SR 캐시
  (`/expdata/cache/sr/eval12/*/radiance_swinir_L100`)가 있어야 한다. `run_C.py --set eval12`나 `gate_matup_driver.py`,
  `gate_exp4.py`가 그것을 만든다. 없으면 건너뛰고 이유를 기록한다.
- MatUp은 height를 출력하지 않는다. 평가의 height 자리는 bilinear LR height이고 `result.json`에 그렇게 적힌다.

**extra_loss 훅 (f).** `extra_factory(dataset, args, device)`는 데이터셋(LR init + 업샘플 렌더)을 읽은 뒤 한 번 불린다.
돌려준 `extra_loss(model, dataset, step) → (tensor, dict)`의 tensor를 매 step MatUp loss에 더하고, dict는 epoch 기록에
남는다(마지막 step 값). `extra_factory=None`이면 MatUp과 같다.

**원본 MatUp 대비 변경 (a)–(k).** 원본은 `external/matup_egsr`(c74ec99)의 `main.py`, `main_optim.py`, `matup/train.py`다.
import한 모듈만 쓰고 서브모듈 파일은 고치지 않는다. 각 변경은 코드에 주석 `# (x)`로 표시되어 있다.

| | 변경 | 원본 위치 | 파일 | 결과에 미치는 영향 |
|---|---|---|---|---|
| a | CUDA graph 2개와 stream 2개, warm-up 2×4 it 대신 eager 루프를 쓴다. step마다 실행하는 연산 5개(zero_grad, mlp_forward_pass, render, loss_fn, backward+step)는 같다 | `matup/train.py:59-139` | `optimise.py` | 커널 순서와 RNG 소비가 달라진다. `gate_matup_driver.py`로 잰 차이는 normal 0.6099° vs 0.6092°, PSNR 31.568 vs 31.557 dB다 |
| b | `Adam(capturable=False)`. capturable은 graph capture에서만 의미가 있다 | `main_optim.py:73` | `optimise.py` | 없음 |
| c | DataLoader에 `drop_last=True`, `num_workers` 인자(기본 8), `persistent_workers=num_workers>0`을 준다 | `main_optim.py:81-84` | `optimise.py` | 512² 재질은 타일 246016 = 128×1922개라서 버려지는 배치가 없다 |
| d | 1·2단계(LR 렌더, SR 업샘플)는 `common/sr_cache.py`의 캐시를 쓰고, 끝난 뒤 rmtree를 하지 않는다 | `main.py:64-74`, `main_optim.py:103-104` | `optimise.py`, `common/sr_cache.py` | 캐시가 MatUp 출력과 바이트 동일함은 `gate_matup_driver.py`로 확인했다 |
| e | 출력 폴더에 타임스탬프 하위 폴더를 만들지 않는다 | `main_optim.py:51-52` | `outputs.py` | 경로만 다르다 (`<run>/maps/`) |
| f | extra_loss 훅(위). 실험 4가 쓴다 | — | `optimise.py` | `None`이면 원본과 같다 |
| g | 매 step `float(loss)`로 동기화하고, epoch마다 `history` 레코드를 남긴다. tqdm과 `loss N : v` 출력이 없다 | `train.py:86, 145-147` | `optimise.py` | 속도만 달라진다 |
| h | 업샘플 렌더가 없거나 렌더 수가 `nb_lights`와 다르면 예외를 낸다(원본은 그대로 진행한다) | — | `optimise.py` | 없음 |
| i | `matup_args`: `main.py` argparse 기본값에서 `opengl_normals=True`만 바꿨다(MatSynth는 OpenGL) | `main.py:12-53` | `args.py` | `--ogl_normals`와 같다 |
| j | `init_maps`: bilinear init을 별도 변형 `init`으로 돌려준다(renorm-clamp 적용, PNG를 쓰지 않음) | `main_optim.py:23-33` | `reference.py` | 추가 기능이다 |
| k | epoch마다 `model.train()`을 부르지 않고, save_interval 스냅샷이 없고, `torch.cuda.set_device`와 장치 출력이 없다 | `train.py:84, 149-151`, `main_optim.py:16-21, 40-44` | `optimise.py` | FullyConnected에는 dropout도 BN도 없고, save_interval=999는 epoch 수 10보다 크다. 따라서 영향이 없다 |

원본에 없는 추가 기능: `outputs.to_canonical`, `residuals.per_light_residuals`(실험 1 정밀 지표), `outputs.save_json`.

### 실험 2·3: ESRGAN 트레이너와 팔

트레이너(`esrgan_map_degradation/core/train.py`)는 하나이고 팔마다 config가 하나다. config에는 팔 이름이 없다.
`core/assemble.py`가 config 키에서 요인을 읽어 팔을 고른다.

```
--config X.yaml [--set k=v ...] ─► config.load_config (base: 상속, per_map 교체)
  ─► assemble.build: G → init → D → VGG → RenderLoss (이 순서로 만든다. 같은 seed면 초기 가중치가 같다)
  ─► 학습: 앞 train.psnr_iters는 PSNR 단계(pixel 항, D 학습 없음), 그 뒤 GAN 단계
  ─► /expdata/runs/<experiment>/<run-name>/
        config.yaml   해석된 config
        setup.json    init 보고(pretrained 로드 텐서 수, 재초기화된 head 등)
        log.jsonl     log.every마다 loss, it/s, peak 메모리, train_seconds
        eval.jsonl    eval.every마다 eval12 맵별 지표(빠른 부분집합)
        ckpt/iter_<n>.pt   G, D, 옵티마이저, 스케줄러 (save.keep개 유지)
evaluate.py --run <experiment>/<run-name> ─► <run>/eval/<ckpt>[_tile<N>]/{results.json, summary.md, <재질>/*.png}
```

`<experiment>`는 config의 `experiment:` 키다(2-x는 `exp2`, 3-x는 `exp3`). 가장 새 체크포인트가 있으면 자동으로 이어서 학습한다.
`evaluate.py`는 체크포인트 안의 config로 G를 다시 만든다. `train_seconds`는 학습 step만 센다(평가·저장 제외, CUDA 동기화).
실험 5 `gpu_hours.py`가 이것을 읽는다.

| config | 2-0 대비 바꾸는 것 | 팔 코드 |
|---|---|---|
| `2-0_base_scratch` | 기준: 공유 RRDBNet 3→3(5맵을 배치로), 공유 3ch D 1개, 맵별 L1+VGG+RaGAN | `esrgan_map_degradation/arms/base.py` — `SharedSR`, `SharedD`, `Criterion`, `L1Term`, `VGGTerm`, `PerMapGanTerm` |
| `2-1_base_pretrained` | `init.mode: pretrained` (RealESRGAN_x4plus) | `arms/pretrained.py` — `load` |
| `2-2_separate` | `model.arch: separate` → G 5개 + D 5개 (아래 알려진 동작 1) | `arms/separate.py` — `SeparateSR`, `owns_discriminators`, `PerMapD` |
| `2-3_mapaware_loss` | normal=각도, roughness·metallic=L1만 | `arms/mapaware_loss.py` — `AngularTerm`; 교체 의미는 `core/config.py`의 `REPLACE_CHILDREN` |
| `2-4_concat_input` | `model.input: concat` (15→15 G) | `arms/concat.py` — `ConcatSR` |
| `3-0_maploss` | concat + pretrained(새 head) | 별도 팔 없음: concat.py + pretrained.py. conv_first와 conv_last 3텐서는 `setup.json`의 `skipped`로 남는다 |
| `3-1_render` | + `loss.render`, + `loss.extra_l1` | `esrgan_render_loss/arms/render_loss.py` — `RenderLoss`, `ExtraL1Term`, `RenderTerm` |
| `3-2_render_jointD` | + `disc.mode: joint`, `loss.joint_gan` (15ch D 1개) | `esrgan_render_loss/arms/joint_d.py` — `JointD`, `JointGanTerm` |

Criterion 항 순서는 고정이다(float 합이 순서에 의존한다): 맵마다 `[L1, Angular]` pixel 그룹 → `ExtraL1 → VGG → PerMapGan`,
그 뒤에 `JointGan → Render`. `arms/base.py` 머리 주석에 이 순서와 각 항이 PSNR 단계에서 도는지(`in_psnr_stage`)가 적혀 있다.

### 바꾸면 기존 실행이 깨지는 것

- config 키와 8개 yaml의 값. `gate_configs.py`가 팔 간 차이를 검사한다.
- 출력 경로 `/expdata/runs/{exp1,exp2,exp3,exp4,exp5,matup}/...`. 폴더 이름이 바뀌어도 출력 디렉터리와 `experiment:` 키는 `exp1`–`exp5`다.
- 실험 2·3 `state_dict` 키(G `net.*` / `nets.<map>.*`, D `shared.*` / `ds.<map>.*` / `joint.*`)와 체크포인트 객체 이름
  (`G, opt_g, sch_g, D, opt_d, sch_d, iter, train_seconds, config`). 이어하기와 `evaluate.py`가 이것을 읽는다.
- `log.jsonl`·`eval.jsonl`의 키(`gpu_hours.py`가 `train_seconds`와 지표 키를 읽는다).

### import 규칙

이 트리 밖에서 가져오는 코드는 `external/*`(MatUp과 그 안의 SwinIR)와 `variants/util/`뿐이다. 진입 스크립트는 저장소 루트를 `sys.path`에 넣고 패키지로
import한다. 서브모듈을 `sys.path`에 넣는 곳은 `variants/util/paths.py` 하나다(`common/paths.py`의 `add_matup()`이 부른다).

## 실행

이미지는 `texsr/exp:torch2.0.0-cu117`이다(`texsr/matup` 위에 쌓은 파생 이미지). Dockerfile과 compose 파일은 `docker/exp/`에 있다.

```powershell
cd docker\exp
docker compose build          # texsr/matup 이 먼저 있어야 한다 (cd docker; docker compose build matup)
docker compose run --rm exp python variants/experiments/...
```

compose는 `/workspace`(저장소), `/data`(MatSynth, 읽기 전용, `MATSYNTH_ROOT`로 변경), `/expdata`(`D:/experimentdata/texsr_exp`,
`EXP_DATA`로 변경)를 마운트하고 `shm_size: 8gb`를 준다. compose 없이 마운트를 직접 주는 경우(Git Bash):

```bash
MSYS_NO_PATHCONV=1 docker run --rm --gpus all --shm-size 8g \
  -v "$(pwd -W):/workspace" -v D:/experimentdata:/data:ro -v D:/experimentdata/texsr_exp:/expdata \
  -w /workspace texsr/exp:torch2.0.0-cu117 \
  python variants/experiments/esrgan_map_degradation/core/train.py --config variants/experiments/esrgan_map_degradation/configs/2-0_base_scratch.yaml
```

컨테이너 안에서 환경변수 `EXPDATA`를 바꾸면 모든 쓰기가 그 아래로 간다(`gate_train_smoke.py`의 학습셋과 실행 포함).
`MATSYNTH_SHARDS`는 샤드 경로를 바꾼다.

실험별 명령, 실행 순서, 소요 시간과 측정값은 [`EXPERIMENTS.md`](EXPERIMENTS.md)에 있다(실험 1 §4, 2 §5, 3 §6, 4 §7, 게이트와
소요 시간 §8, 실험 5 §9). 시간 단위 실행(실험 1 본 실험, 실험 4, 실험 2·3의 100k it)은 직접 돌린다.

게이트(`docker/exp`에서 실행):

```bash
docker compose run --rm exp python variants/experiments/gates/gate_configs.py        # config가 기준과 지정한 요인만 다른지   수 초
docker compose run --rm exp python variants/experiments/gates/gate_protocol.py       # 공통 평가 프로토콜                    수 초
docker compose run --rm exp python variants/experiments/gates/gate_matup_driver.py   # 구동기 vs MatUp main.py              ~5분
docker compose run --rm exp python variants/experiments/gates/gate_exp4.py           # 실험 4 통계항                         ~3분
docker compose run --rm exp python variants/experiments/gates/gate_train_smoke.py    # 모든 팔 몇 it 학습 (GPU, /expdata/_gate_train 필요)
```

## 알려진 동작과 결함

고치지 않고 이름을 붙인 동작이다. 바꾸면 해당 실험의 결과가 달라지므로 실험 설계의 결정으로 다룬다.

| # | 동작 | 코드 |
|---|---|---|
| 1 | **2-2는 판별기도 5개다.** config는 `arch: separate`만 바꾸지만, `separate.owns_discriminators`가 이 값을 D 생성에 넘겨 맵마다 D를 만든다. 그래서 2-2와 2-0의 차이는 "G 분리"만이 아니라 "G 분리 + D 분리"이고, 속도 1.41 it/s도 둘을 합친 값이다 | `arms/separate.py` 머리 주석, `owns_discriminators`, `PerMapD` |
| 2 | **3-1·3-2의 PSNR 단계에서도 render 항과 extra_l1이 걸린다.** 첫 20k it의 목적함수는 L1(가중치 1) + extra_l1 + 1.0·L_render다. ESRGAN식 "pixel 항만"이 아니다 | `render_loss.py` 머리 주석, `ExtraL1Term.in_psnr_stage = True`, `RenderTerm.in_psnr_stage = True` |
| 3 | 3-2(joint)에서는 2-0에서 상속한 맵별 `gan: 0.005`가 아무 일도 하지 않는다. 이 값은 `gan_maps` 출력에만 남는다 | `joint_d.py` 머리 주석, `base.PerMapGanTerm` |
| 4 | `sch_d`는 D를 학습하지 않는 PSNR 단계에서도 step한다. 따라서 D의 milestone은 1 it부터 센다 | `core/train.py` 머리 주석, `sch_d.step()` |
| 5 | PSNR 단계의 pixel 가중치는 config와 무관하게 1.0이다. `l1` 키만 있으면 가중치 0이어도 돈다. 3-1에서 basecolor·normal·roughness의 L1은 GAN 단계에서 per_map 0.01 + extra 0.01 = 0.02가 된다 | `base.L1Term`, `mapaware_loss.AngularTerm`, `render_loss.py` 머리 주석 |
| 6 | `TrainSet`은 `default_rng()`를 seed 없이 쓴다. 그래서 `train.seed`가 같아도 패치와 증강이 매번 다르다(`self.seed`는 쓰이지 않는다) | `core/dataset.py` `TrainSet` |
| 7 | `disc.mode`가 none/per_map/joint 외의 값이면(오타 포함) D가 조용히 0개가 된다 | `assemble.discriminators`, `base.Discriminators` |
| 8 | `bench_esrgan`(실험 5)은 pretrained init을 하지 않고, D 폭을 기본값으로 두고, G에는 num_block만 넘긴다. 속도 측정용이라 품질과 무관하다 | `speed_memory/bench_train.py` `bench_esrgan` docstring |
| 9 | **데이터 준비 스크립트가 지금은 돌지 않는다.** `experiments/data/*.py`는 `experiments.common`(`paths`, `protocol`, `shards`)을 import하는데 그 패키지는 저장소에 더 이상 없다. ESRGAN 학습셋(`build_train_set.py`, EXPERIMENTS.md §3 (5))과 `gate_train_smoke.py`의 scratch 학습셋을 만들려면 import를 `variants.experiments.common`으로 바꿔야 한다 | `experiments/data/{build_train_set,extract,matsynth_index,select_materials}.py` |

- **bilinear의 C ≈ 0.0011은 0이 아니라 8-bit 양자화 바닥이다**(1/(255·√12)). S의 C는 이 값과 비교한다.
- **SRFlow는 돌지 않는다.** 체크포인트 배포가 중단됐다. 코드와 체크포인트를 직접 넣으면 `--s srflow`로 돌고, 없으면 SKIP을 출력한다.
- 실험 4의 결정 사항 두 가지(metallic에 몰리는 Σ, 기본 λ=0.003)는 `EXPERIMENTS.md` §7에 있다.
