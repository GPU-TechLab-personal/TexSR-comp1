# variants/experiments/ — 실험 리스트 1–5

`experiments/`(설명은 `experiments/README.md`)를 `variants/` 규칙(`variants/README.md`)에 맞춰 옮기고 분해한 트리다.
**동작은 바꾸지 않았다.** 옛 코드는 그대로 남아 있고, `variants/tests/`의 L0·L6가 두 코드가 같다는 것을 확인한다.

- **자기완결.** 이 트리는 `scripts/`와 `experiments/`를 import하지 않는다. 밖에서 가져오는 코드는 `external/*`(MatUp)뿐이다.
  `texsr_metrics`는 `variants/common/texsr_metrics.py`에서 가져온다.
  sys.path 조작은 `variants/common/paths.py`가 맡는다. `common/paths.py`의 `add_matup()`과 `add_swinir()`는 그 파일의 `use_matup()`과 `use_swinir()`를 부른다.
- **외부 계약은 그대로다.** config 키와 파일(8개 yaml, 바이트 동일), 출력 경로(`/expdata/runs/{exp1,exp2,exp3,exp4,exp5,matup}/...`),
  exp2 `state_dict` 키(G `net.*` / `nets.<map>.*`, D `shared.*` / `ds.<map>.*` / `joint.*`), 체크포인트 객체 이름
  (`G, opt_g, sch_g, D, opt_d, sch_d, iter, train_seconds, config`), 로그 줄과 `log.jsonl` 키 순서도 바뀌지 않았다.
  옛 트레이너가 쓴 체크포인트는 새 트레이너로 이어서 학습할 수 있고, 새 `evaluate.py`로도 읽힌다(L6 GPU `compat`).
- **범위 밖: `experiments/data/*.py`.** `.gitignore:6`이 `data/`를 무시해서 이 파일들은 저장소에 없다. 원본은 데스크탑
  메인 체크아웃 `experiments/data/`에 있다(check_eval_materials · matsynth_index · select_materials · extract ·
  build_train_set). 이 트리로 옮기지 않았다. 이 트리의 코드는 그 결과물만 읽는다(`/expdata/materials`, `/expdata/train`,
  `cli.py`의 기본값 `--selection /workspace/experiments/data/selection_exp1.json`). 오류 메시지가 `experiments/data/...`를
  가리키는 것은 그대로 두었다.
- `experiments/docker/`(이미지 `texsr/exp:torch2.0.0-cu117`)도 옮기지 않았다. 같은 이미지를 `docker run`으로 쓴다(아래 §5).

## 1. 실험 → 폴더

| 실험 | 폴더 | 진입점 | 원본 |
|---|---|---|---|
| 공통 프로토콜 | `common/` | — | `experiments/common/` (`matup_driver.py`만 `matup_driver/`로 분리) |
| MatUp 구동 (1·4·5가 공유) | `matup_driver/` | — | `experiments/common/matup_driver.py` |
| 1 불일치 원인 | `exp1_inconsistency/` | `run_C.py`, `run_residuals.py`, `analyze.py` | 같은 이름 |
| 2 ESRGAN 저하 원인 | `exp2_esrgan/core/`, `exp2_esrgan/arms/`, `exp2_esrgan/configs/` | `core/train.py`, `core/evaluate.py` | `experiments/exp2_esrgan/{train,evaluate,model,losses,data}.py` |
| 3 렌더링 loss | `exp3_render_loss/arms/`, `exp3_render_loss/configs/` | (트레이너는 exp2 `core/train.py`) | `losses.py`의 RenderLoss·extra_l1·joint D 분기 |
| 4 GRF 통계항 | `exp4_grf_stats/` | `run.py`, `analyze.py` | 같은 이름 |
| 5 속도·메모리 | `exp5_benchmark/` | `bench_infer.py`, `bench_train.py`, `gpu_hours.py`, `texel_cost.py` | 같은 이름 |
| 게이트 | `gates/` | `gate_*.py` | 같은 이름 |

### 옮긴 파일과 분해한 파일

기록은 `variants/tests/moves/experiments.json`에 있다. L0(`python variants/tests/l0_static.py`)가 이 기록을 검사한다.

- **moves (파일 통째, import와 경로 상수만 바꿈):** `common/{cli,protocol,shards,render,upscalers,sr_cache,metrics,matup_runs}.py`,
  `exp1_inconsistency/*`, `exp2_esrgan/data.py → core/dataset.py`, `exp4_grf_stats/*`, `exp5_benchmark/{bench_infer,gpu_hours,texel_cost}.py`,
  `gates/{gate_protocol,gate_matup_driver,gate_configs}.py`. `gate_configs`는 `C2`·`C3` 상수만 `__file__` 기준 경로로 바꿨다.
- **functions (함수 단위로 옮김, AST 동일):** `matup_driver`의 7개 함수 전부, exp2 `model.py`의 RRDBNet·VGG D·perceptual·pretrained 로더,
  `losses.py`의 `angular`·`ragan_g/d`·`srgb_to_linear_t`·`render_tensors`·`RenderLoss`·`cat`·`Discriminators.active`,
  `train.py`의 `load_config`·`deep_merge`·`infinite`·`quick_eval`, `evaluate.infer`, `bench_train`의 `bench_matup`·`run`·`main`,
  `gate_exp4`의 `check`·`synthetic`·`main`.
- **decomposed (분해, 옛 줄마다 verbatim/rewritten/dropped 표):** `MultiMapSR`, `Discriminators`, `Criterion`, `train.main`,
  `evaluate.main`, `bench_train.bench_esrgan`, `gate_exp4.real`, `gate_train_smoke.main`, `paths.add_matup/add_swinir`.
  `Discriminators`와 `MultiMapSR`의 `else:` 같은 분기 줄 3개는 dropped로 처리했다. 그 분기는 클래스 하나씩으로 나뉘었다. 나머지 줄은 전부 verbatim이거나 rewritten이다.

## 2. matup_driver — 원본 MatUp 대비 변경 (a)–(k)

원본 코드는 `external/matup_egsr`(c74ec99)의 `main.py`, `main_optim.py`, `matup/train.py`다. 원본에서 쓰는 것은 import한 모듈뿐이고
(`upstream.py`: FullyConnected, MatUpDataset, mlp_forward_pass, loss_fn, batch_render_tn_explicit, eval_model,
output_maps_from_list), 서브모듈 파일은 고치지 않았다. 각 변경은 코드 주석 `# (x)`로 표시되어 있다.

| | 변경 | 원본 위치 | 새 위치 | 결과에 미치는 영향 |
|---|---|---|---|---|
| a | CUDA graph 2개와 stream 2개, warm-up 2×4 it 대신 eager 루프를 쓴다. step마다 실행하는 연산 5개(zero_grad, mlp_forward_pass, render, loss_fn, backward+step)는 같다 | `matup/train.py:59-139` | `matup_driver/optimise.py:61-82` | 커널 순서와 RNG 소비가 달라진다. `gate_matup_driver.py`로 잰 차이는 normal 0.6099° vs 0.6092°, PSNR 31.568 vs 31.557 dB다 |
| b | `Adam(capturable=False)`. capturable은 graph capture에서만 의미가 있다 | `main_optim.py:73` | `optimise.py:36-37` | 없음 |
| c | DataLoader에 `drop_last=True`, `num_workers` 인자(기본 8), `persistent_workers=num_workers>0`을 준다 | `main_optim.py:81-84` | `optimise.py:48-52` | 512² 재질은 타일 246016 = 128×1922개라서 버려지는 배치가 없다 |
| d | 1·2단계(LR 렌더, SR 업샘플)는 `common/sr_cache.py`의 캐시를 쓰고, 끝난 뒤 rmtree를 하지 않는다 | `main.py:64-74`, `main_optim.py:103-104` | `optimise.py:40`, `common/sr_cache.py` | 캐시가 MatUp 출력과 바이트 동일함은 게이트로 확인했다 |
| e | 출력 폴더에 타임스탬프 하위 폴더를 만들지 않는다 | `main_optim.py:51-52` | `matup_driver/outputs.py:13-17` | 경로만 다르다 (`<run>/maps/`) |
| f | **extra_loss 훅.** `extra_factory(dataset,args,device)`가 돌려준 `extra_loss(model,dataset,step) → (tensor, dict)`를 매 step loss에 더한다. 실험 4가 이 훅을 쓴다 | — | `optimise.py:6-13, 55-56, 76-79` | `None`이면 원본과 같다 |
| g | 매 step `float(loss)`로 동기화하고, epoch마다 `history` 레코드를 남긴다. tqdm과 `loss N : v` 출력이 없다 | `train.py:86, 145-147` | `optimise.py:74-75, 83-92` | 속도만 달라진다 |
| h | 업샘플 렌더가 없거나 렌더 수가 `nb_lights`와 다르면 예외를 낸다(원본은 그대로 진행한다) | — | `optimise.py:42-47` | 없음 |
| i | `matup_args`: `main.py` argparse 기본값에서 `opengl_normals=True`만 바꿨다(MatSynth는 OpenGL) | `main.py:12-53` | `matup_driver/args.py:5-19` | `--ogl_normals`와 같다 |
| j | `init_maps`: bilinear init을 별도 변형 `init`으로 돌려준다(renorm-clamp 적용, PNG를 쓰지 않음) | `main_optim.py:23-33` (output_initial_maps) | `matup_driver/reference.py` | 추가 기능이다 |
| k | epoch마다 `model.train()`을 부르지 않고, save_interval 스냅샷이 없고, `torch.cuda.set_device`와 장치 출력이 없다 | `train.py:84, 149-151`, `main_optim.py:16-21, 40-44` | `optimise.py:30, 83-84` | FullyConnected에는 dropout도 BN도 없고, save_interval=999는 epoch 수 10보다 크다. 따라서 영향이 없다 |

추가된 기능(원본에 없는 것): `outputs.to_canonical`(내부 맵 → 정규 재질), `residuals.per_light_residuals`(실험 1 정밀 지표), `outputs.save_json`.

## 3. 실험 2·3 — arm 모듈

트레이너(`exp2_esrgan/core/train.py`)는 하나이고 팔마다 config가 하나다. config 키는 예전 것을 그대로 쓴다.
어느 arm이 G, D, Criterion의 어느 부분을 만드는지는 `core/assemble.py`가 config에서 읽는다. 이 판정은 예전 `train/model/losses`에 흩어져 있던 if/else를 한곳으로 옮긴 것이다.
**생성 순서**(G → init → D → VGG → RenderLoss)도 예전과 같다. 그래서 같은 seed로 초기 가중치가 바이트 동일하게 나온다(L6 `build` 검사).

| config | 2-0 대비 바꾸는 것 | arm 코드 (새 file:line) |
|---|---|---|
| `2-0_base_scratch` | 기준: 공유 RRDBNet 3→3(5맵을 배치로), 공유 3ch D 1개, 맵별 L1+VGG+RaGAN | `exp2_esrgan/arms/base.py` — `SharedSR` :54, `Discriminators`(D 없음) :74, `SharedD` :95, `Criterion` :109, `L1Term` :149, `VGGTerm` :163, `PerMapGanTerm` :179 |
| `2-1_base_pretrained` | init: RealESRGAN_x4plus | `arms/pretrained.py` — `load` :70 (`ensure_pretrained` :38, `load_pretrained` :50); 선택 `core/assemble.py:91-92` |
| `2-2_separate` | **G 5개 + D 5개** (config는 `model.arch`만 말한다) | `arms/separate.py` — `SeparateSR` :23 (5개 명시 :33-39), `owns_discriminators` :49, `PerMapD` :55 (`ds` :62); 선택 `assemble.py:52-58, 99` |
| `2-3_mapaware_loss` | normal=각도, roughness·metallic=L1만 | `arms/mapaware_loss.py` — `AngularTerm` :27 (pixel 그룹, L1 뒤); config 교체 의미는 `core/config.py:30-33` (REPLACE_CHILDREN) |
| `2-4_concat_input` | 15→15 concat G | `arms/concat.py` — `ConcatSR` :14 |
| `3-0_maploss` | concat + pretrained(새 head) | 별도 arm 없음: concat.py + pretrained.py. conv_first와 conv_last 3텐서는 `setup.json`의 `skipped`로 남는다 |
| `3-1_render` | + 렌더링 loss, + extra_l1 | `exp3_render_loss/arms/render_loss.py` — `RenderLoss` :47, `ExtraL1Term` :77, `RenderTerm` :92 |
| `3-2_render_jointD` | + joint D(15ch 1개) | `exp3_render_loss/arms/joint_d.py` — `JointD` :21, `JointGanTerm` :33 |

Criterion 항 순서(합산 순서가 같아야 비트 동일): 맵마다 `[L1, Angular]` pixel 그룹 → `ExtraL1 → VGG → PerMapGan`, 그 뒤에 `JointGan → Render`.
`base.py` 머리 주석에 이 순서와 각 항이 PSNR 단계에서 도는지(`in_psnr_stage`)가 적혀 있다.

## 4. 보존한 이상 동작 (고치지 않고 이름을 붙였다)

| # | 동작 | 옛 위치 | 새 위치 (명시한 곳) | L6 검사 |
|---|---|---|---|---|
| 1 | **2-2는 판별기도 5개다.** config는 `arch: separate`만 바꾸지만, 옛 트레이너가 `mc["arch"] == "separate"`를 D에 넘겨 맵마다 D를 만들었다. 그래서 2-2와 2-0의 차이는 "G 분리"만이 아니라 "G 분리 + D 분리"이고, 속도 1.41 it/s도 둘을 합친 값이다 | `train.py:144`, `losses.py:152-155` | `arms/separate.py:1-15, 49-62` | `named: 2-2 has 5 G and 5 D` |
| 2 | **3-1·3-2의 PSNR 단계에서도 render 항과 extra_l1이 걸린다.** 옛 generator는 extra_l1을 `if psnr_stage: continue` 앞에 두었고, render 항은 단계를 보지 않았다. 그래서 첫 20k it의 목적함수는 L1(가중치 1) + extra_l1 + 1.0·L_render다. ESRGAN식 "pixel 항만"이 아니다 | `losses.py:118-121, 134-136` | `render_loss.py:12-26`, `ExtraL1Term.in_psnr_stage = True` :80, `RenderTerm.in_psnr_stage = True` :95 | `named: 3-x PSNR stage runs render + extra_l1` |
| 3 | 3-2(joint)에서는 2-0에서 상속한 맵별 `gan: 0.005`가 아무 일도 하지 않는다. 이 값은 `gan_maps` 출력에만 남는다 | `losses.py:127` | `joint_d.py:5-10`, `base.py:179-194` | `named: 3-2 per-map gan weights inert` |
| 4 | `sch_d`는 D를 학습하지 않는 PSNR 단계에서도 step한다. 따라서 D의 milestone은 1 it부터 센다 | `train.py:231-232` | `core/train.py:29-30, 188-190` | (루프는 verbatim, L0) |
| 5 | PSNR 단계의 pixel 가중치는 config와 무관하게 1.0이다. `l1` 키만 있으면 가중치 0이어도 돈다. 3-1에서 basecolor·normal·roughness의 L1은 GAN 단계에서 per_map 0.01 + extra 0.01 = 0.02가 된다 | `losses.py:100-108` | `base.py:149-160`, `mapaware_loss.py:27-38`, `render_loss.py:8-10` | generator[psnr/gan] 비트 동일 |
| 6 | `TrainSet`은 `default_rng()`를 seed 없이 쓴다. 그래서 `train.seed`가 같아도 패치와 증강이 매번 다르다(`self.seed`는 쓰이지 않는다) | `data.py:87` | `core/dataset.py:89-92` | (move, L0) |
| 7 | `disc.mode`가 none/per_map/joint 외의 값이면(오타 포함) D가 조용히 0개가 된다 | `losses.py:151-157` | `base.py:74-92`, `assemble.py:52-61` | build: D mode/active |
| 8 | `bench_esrgan`(실험 5)은 pretrained init을 하지 않고, D 폭을 기본값으로 두고, G에는 num_block만 넘긴다. 속도 측정용이라 품질과 무관하다 | `bench_train.py:40-51` | `exp5_benchmark/bench_train.py` `bench_esrgan` docstring | (decomposed, L0) |

## 5. 실행

이미지는 `texsr/exp:torch2.0.0-cu117`이다(`experiments/docker`에서 빌드). `docker compose`를 쓰지 않으므로 마운트는 직접 준다.
Git Bash에서 실행하는 예:

```bash
MSYS_NO_PATHCONV=1 docker run --rm --gpus all --shm-size 8g \
  -v "$(pwd -W):/workspace" -v D:/experimentdata:/data:ro -v D:/experimentdata/texsr_exp:/expdata \
  -w /workspace texsr/exp:torch2.0.0-cu117 \
  python variants/experiments/exp2_esrgan/core/train.py --config variants/experiments/exp2_esrgan/configs/2-0_base_scratch.yaml
```

나머지 명령은 `experiments/README.md`의 명령에서 `experiments/`를 `variants/experiments/`로 바꾸면 된다.
트레이너와 평가기만 `exp2_esrgan/core/` 아래에 있다. 시간 단위 실행(실험 1 본 실험, 4, 2·3의 100k it)은 직접 돌려야 한다. 소요 시간은 `experiments/README.md` §8 표와 같다.

## 6. 테스트

```bash
variants/tests/l6_experiments_all.sh        # L0 + L6 전부, 리포트까지   ~15분 (GPU 포함)
variants/tests/l6_experiments_all.sh cpu    # GPU 층 제외                ~4분
```

결과는 `runs/variants_verify/experiments/<stamp>/report.md`에 쓴다(메인 체크아웃).
GPU 학습 출력은 `D:/experimentdata/texsr_exp/variants_verify/expdata/`(컨테이너의 `EXPDATA=/vvx`)에 쓴다.
`/expdata`는 읽기 전용으로 마운트하므로 기존 runs와 캐시는 건드리지 않는다. GPU 단계는 다른 `texsr*` 컨테이너가 돌고 있으면 시작하지 않는다.

| 층 | 파일 | 무엇을 | 판정 | 시간 |
|---|---|---|---|---|
| L0 | `l0_static.py` + `moves/experiments.json` | moves AST 동일, functions AST 동일, decomposed 줄 커버리지 | 정확히 | <5 s (호스트) |
| L6 imports | `l6_experiments_imports.py` | 모든 모듈이 새 인터프리터에서 import되고, `experiments.*`, `scripts.*`, `texsr_metrics`를 끌어오지 않는다 | 정확히 | ~1분 |
| L6 cpu | `l6_experiments_cpu.py` | 상수, 해석된 config(dict와 config.yaml 덤프), arm마다 build(seed 0 state_dict), 키 집합, 옛 ckpt strict 로드, forward, `Criterion.generator`(PSNR/GAN; total, 로그 순서, dG, D BN 상태), `Discriminators.loss`(dD), 이상 동작 단언, matup_driver CPU 경로(init_maps, to_canonical, per_light_residuals, metrics.evaluate), gate_configs·gate_protocol(옛것과 stdout 동일)·gate_exp4 합성(seed 0) | 정확히 (`torch.equal`/`==`) | ~3분 |
| L6 gpu matup | `l6_experiments_gpu.py` | `optimise` 옛/새, eval12 `acg_wood_floor_060`, seed 1996, 1 epoch(1922 step), 원본 루프와 exp4 훅(B, λ 0.003) | `allclose(rtol=0, atol=1e-3)`, 실측 차이와 old-vs-old 바닥값을 함께 보고 | ~4분 |
| L6 gpu smoke | 〃 | 새 경로의 `gate_train_smoke.py`(8팔, 20 it 후 30까지 재개) | 정확히 (exit 0, ALL PASS) | ~6분 |
| L6 gpu compat | 〃 | 옛 트레이너 ckpt(2-0, 2-2, 3-2)를 새 트레이너가 재개하고, 옛·새 `evaluate.py` 결과를 비교 | 정확히 | ~6분 |

DataLoader 워커와 CUDA atomic 때문에 matup_driver만은 옛 코드끼리 두 번 돌려도 결과가 정확히 같지 않을 수 있다. 그래서 이 항목만 allclose로 판정한다.
