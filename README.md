# TexSR-comp1 — MatUp vs DeepTextures

두 텍스처 초해상 논문을 **같은 조건에서 비교**하고 결과 텍스처를 뽑는 실험 워크스페이스.

| | |
|---|---|
| **MatUp** (EGSR 2024) | 재질 1개당 test-time 최적화. 학습 불필요. LR 재질을 100개 조명으로 렌더 → SwinIR로 렌더를 ×4 업샘플 → HR 맵을 역최적화 |
| **DeepTextures** (Sci. Rep. 2026) | TF2 ESRGAN (3채널 RGB, ×4). **사전학습 가중치가 배포된 적 없어 직접 학습해야 함.** 논문 코드를 무수정으로 두 데이터에 돌린다: 서브모듈에 동봉된 논문 원본 300장(`data/textures_db`, 사실상 컬러 전용) → `docs/RUNBOOK.md` §2.5, 그리고 MatSynth train split 다섯 맵을 섞은 것(논문의 멀티채널 학습 방식) → §2.6. 평가는 둘 다 이미 추출해 둔 MatSynth 12개 재질(test split)로 한다 |

## 머신 역할

| | 노트북 (authoring) | 데스크탑 (compute) |
|---|---|---|
| GPU | Intel Arc iGPU — CUDA 없음 | **RTX 3080 Ti 12 GB** |
| 역할 | 코드·설정 작성 → GitHub push | clone → 빌드 → 학습·실험 |
| Docker | — | **Docker Desktop** |

코드는 GitHub로, **데이터는 GitHub를 타지 않는다** (DAS에 있음).

예외는 `runs/montages/` 하나다. 연구실 공유용 산출물 — 재질별 4행×6열 몽타주, `metrics.json`, `summary.md` — 만 커밋한다(약 36 MB). 나머지 `runs/`는 600 MB가 넘고 **코드와 MatSynth로부터 재생성 가능**하므로(`scripts\matup\run_matup.ps1 -PostProcess`) git 히스토리에 넣지 않는다.

---

## 데스크탑 초기 구축

### 1. clone

```powershell
git clone https://github.com/GPU-TechLab-personal/TexSR-comp1.git
cd TexSR-comp1
```

### 2. 서브모듈

**`--recurse-submodules`를 쓰지 말 것.** SRFlow까지 받아버리는데, SRFlow는 체크포인트 배포가 중단돼 재현 불가능하고 라이선스도 다르다(CC BY-NC-SA).

```powershell
git submodule update --init external/matup_egsr external/DeepTextures external/esrgan-tf2
git -C external/matup_egsr submodule update --init sr_modules/SwinIR
```

`DeepTextures`는 논문 학습 이미지 300장을 포함해 약 560 MB를 받는다.

확인 — **다섯 줄**이 나오고 `sr_modules/SRFlow`만 `-` 상태여야 한다. 최상위 3개에 더해
`matup_egsr`가 SwinIR·SRFlow를 중첩 서브모듈로 들고 있고, `--recursive`는 받지 않은
SRFlow까지 `-`를 달아 함께 나열한다:

```powershell
git submodule status --recursive
```

### 3. SwinIR 체크포인트 (136 MiB)

`.pth`는 gitignore 대상이라 별도로 받아야 한다. **`sr_swinir.py`가 경로를 하드코딩**하므로 파일명·위치가 정확해야 하고, 없으면 조용히 `exit(0)` 된다.

```powershell
curl.exe -L -o external/matup_egsr/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth
```

**반드시 크기를 검증할 것** — `142473939`가 나와야 한다 (부분 다운로드 검출):

```powershell
(Get-Item external/matup_egsr/pretrained_models/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth).Length
```

### 4. `.env`

```powershell
cd docker
Copy-Item .env.example .env
```

`DAS_ROOT`는 아직 미정이라 주석 처리돼 있다. 정해지기 전까지 compose가 `./_das_placeholder`로 폴백하므로 **빌드와 아래 스모크 테스트는 그대로 진행 가능하다.**

설정할 때는 **슬래시를 `/`로** 쓰고(`DAS_ROOT=F:/texsr`), 해당 드라이브를 Docker Desktop → Settings → Resources → **File sharing**에 공유해야 한다. 누락하면 마운트가 조용히 빈 디렉터리가 된다.

> ⚠️ **이 아래 docker 명령은 전부 Windows PowerShell에서 실행할 것.**
> 이 데스크탑에는 Docker 데몬이 **둘** 있다 — Docker Desktop과, WSL `Ubuntu-24.04` 안의
> 네이티브 docker-ce(systemd `enabled`라 배포판이 켜지면 같이 뜬다).
> **둘은 이미지·컨테이너·빌드 캐시를 공유하지 않는다.** WSL 셸 안에서 빌드하면 이미지가
> docker-ce에 만들어져 Docker Desktop에서는 보이지 않는다 (6 GB를 두 번 받게 된다).
> 이상하면 `docker context ls`로 어느 데몬에 붙어 있는지 확인할 것.

### 5. GPU 패스스루 — 이것부터

실패하면 나머지가 전부 무의미하다.

```powershell
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu22.04 nvidia-smi
```

### 6. 빌드 (베이스 이미지 약 6 GB)

```powershell
docker compose build
```

두 Dockerfile 끝에 **빌드 타임 검증**이 들어 있어, 의존성 핀이 틀리면 실험 도중이 아니라 여기서 실패한다.

다만 그 검증이 찍는 버전 문자열을 **빌드 로그에서 읽으려 하지 말 것.** BuildKit이 `RUN` 출력을 접어버리고, 재빌드 때는 레이어가 캐시돼 아예 찍히지 않는다. 런타임에서 확인한다:

```powershell
docker compose run --rm matup python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
```

`2.0.0 0.15.0 True`가 정상이다. **torchvision이 0.15.0인 것은 오류가 아니라 확인된 편차다** — 베이스 이미지가 upstream 핀 0.15.1을 싣고 있지 않다. torchvision은 이 프로젝트에서 호출 지점이 없어(timm이 끌고 오는 전이 import뿐) 그대로 두기로 했다. 근거는 `docs/OPEN-ITEMS.md` #2.

### 7. MatUp 스모크 테스트

MatSynth를 붙이기 **전에** 전체 경로를 검증한다. 저장소에 동봉된 `victorian_wood` 128² 예제를 쓴다.

```powershell
docker compose run --rm matup python main.py --upscaler swinir
```

`output/<타임스탬프>/`에 16-bit PNG 4장(`baseColor_opt`, `metallic_opt`, `normal_opt`, `roughness_opt`)과 `render_opt.png`가 나오고 **8px 검은 테두리가 보이면** 의존성·경로·체크포인트·GPU가 전부 맞은 것이다.

여기까지가 구축이다. **실제 실험을 돌리는 것은 `docs/RUNBOOK.md`를 볼 것** — 실험 전 점검, MatSynth 재질 추출부터 결과까지의 전체 절차, 코드 지도, 증상별 역추적표가 거기 있다.

---

## VS Code devcontainer

CLI 대신 컨테이너 안에서 편집·실행하려면 `.devcontainer/`에 두 스택이 정의돼 있다. `docker/compose.yaml`을 그대로 재사용하므로 마운트·`shm_size`·GPU 예약이 이중으로 정의되지 않는다.

`F1` → **Dev Containers: Reopen in Container** → 스택 선택:

| 설정 이름 | 서비스 | 인터프리터 |
|---|---|---|
| `MatUp - torch 2.0.0 / CUDA 11.7` | matup | `/opt/conda/bin/python` |
| `DeepTextures - TF 2.10.1 / CUDA 11.2` | deeptex | `/usr/bin/python3` |

터미널은 **`/workspace`(저장소 루트)에서 열린다.** compose의 `working_dir`와 다르므로 MatUp을 돌릴 때는 `cd external/matup_egsr`가 필요하다 — `sr_swinir.py`가 체크포인트를 `./pretrained_models/...`로 찾기 때문에 CWD가 틀리면 조용히 `exit(0)` 된다 (위 4단계 참조).

> ⚠️ **이중 데몬 함정이 여기서 형태를 바꿔 재현된다.**
> Dev Containers 확장은 컨테이너를 직접 만들지 않고 **자기가 붙어 있는 호스트에서 `docker`를
> 실행하는 클라이언트**다. VS Code 창이 `WSL: Ubuntu-24.04`에 매여 있으면 그 선택이 컨테이너로
> 상속돼, 컨테이너가 Docker Desktop이 아니라 WSL 안 docker-ce에 만들어진다.
> **안에서는 멀쩡해 보이는 게 함정이다** — 이미지는 Docker Desktop에 없고 6 GB를 다시 받는다.
> **반드시 Windows 경로(`C:\...`)로 연 창에서 Reopen할 것.** 왼쪽 아래 원격 표시기가 `WSL:`로
> 시작하면 잘못된 창이다. 의심되면 Windows PowerShell에서 `docker images texsr/matup`.

`features`와 `postCreateCommand`는 의도적으로 비워뒀다. 둘 다 컨테이너 생성 시점에 패키지를 설치하는데, 그게 바로 `docker/*/requirements.txt`가 막으려는 드리프트다.

---

## 구조

```
docker/          컨테이너 정의. 두 스택은 CUDA 요구가 배타적이라 완전 분리
  matup/         torch 2.0.0 + CUDA 11.7
  deeptex/       TF 2.10.1 + CUDA 11.2
  compose.yaml   /workspace(repo) · /data(DAS) 고정 마운트, GPU 예약, shm 8GB
  .env.example   DAS_ROOT — 유일한 머신별 값
.devcontainer/   VS Code devcontainer 정의. compose.yaml을 재사용만 한다
configs/deeptex/ DeepTextures 학습 config 3종. upstream 것을 편집하지 않고 따로 둔다
                 psnr_texsr / esrgan_scaled(실사용) / esrgan_asconfigured(결함 보존 기준선)
scripts/         실험별로 나뉜다. 공유되는 것만 common/에 있다
  common/        preflight.ps1        두 오케스트레이터가 쓰는 사전점검
                 texsr_metrics.py     두 방법이 **같은 코드로** 재도록 공유
                 matsynth_extract.py  MatSynth → LR/GT 쌍 (두 실험의 평가 데이터)
  matup/         matup_height.py · matup_render.py · run_matup.ps1  (→ RUNBOOK §2.4)
  deeptex/       texsr_*(데이터 준비 4단계) · deeptex_*(학습·평가·비교)
                 run_deeptex.ps1                                    (→ RUNBOOK §2.5)
  compare/       matup_postprocess.py · matup_montage.py
                 두 방법의 결과를 한 격자에 놓는 곳. 양쪽 오케스트레이터가 호출한다
external/        upstream 서브모듈 (커밋 고정, **수정하지 않는다**)
docs/            RUNBOOK.md            — 실험 운용. 점검·코드 지도·증상별 역추적
                 OPEN-ITEMS.md         — 미해결 결정·불확실성
                 DEEPTEX-DEVIATIONS.md — upstream 대비 편차 전체 목록 + 근거
```

컨테이너 내부 경로가 `/workspace`·`/data`로 고정돼 있어 **호스트 경로 차이는 `.env` 한 줄에만 존재한다.**

## 운영상 주의 (실험 전 필독)

1. **`--nb_lights`와 `--up_batch_size`는 함께 바꿔야 한다.** `sr_swinir.py`가 기본값 100/10에서만 렌더를 온전히 커버한다. `nb_lights`만 바꾸면 렌더가 **조용히 누락/중복**된다.
2. **MatSynth 법선은 OpenGL 규약**이므로 MatUp에 `--ogl_normals`가 필요하다. 반면 MatUp **출력은 항상 DirectX**로 하드코딩돼 있어 비교 전 되돌려야 한다.
3. **MatUp은 `height`를 출력하지 않는다.** 산출 맵은 4장뿐이고, 512² 중 **유효 영역은 496²**(8px 검은 테두리)라 비교 시 전 방법·GT에서 잘라내야 공정하다.
4. **DeepTextures의 학습 데이터 준비·학습·평가 스크립트는 이제 존재한다.** upstream이 삭제한 것들을
   `scripts/deeptex/` 아래에 새로 썼다. **서브모듈은 한 바이트도 고치지 않았다** —
   `modules/*`를 바깥에서 import해 구동한다(`matup_render.py:14`와 같은 규약).
   **학습은 `external/DeepTextures/train_esrgan.py` 원본을 그대로 돌린다** — 우리 트레이너가
   아니다(`scripts\deeptex\run_deeptex.ps1 -Stage paper`).
   전체 절차는 `docs/RUNBOOK.md` §2.5, 편차 목록은 `docs/DEEPTEX-DEVIATIONS.md`.
   `external/esrgan-tf2`는 여전히 삭제된 `train_psnr.py`/`psnr.yaml`의 공여자다.
5. **`requirements.txt`를 upstream 것으로 되돌리지 말 것.** `docker/*/requirements.txt`에 변경 사유가 줄별로 주석돼 있다.
6. **MatUp을 같은 저장소에서 둘 이상 동시에 돌리지 말 것.** 중간 산출물이 `--render_dir`·`--upsamplings_dir`의 **고정 기본 경로**(`./data/<재질>/radiance_LR`, `radiance_Upscaled`)에 쌓이고, `main_optim.py`가 성공 시 그 둘을 `shutil.rmtree`로 지운다. 저장소가 컨테이너에 바인드 마운트돼 있으므로 **먼저 끝난 실행이 아직 돌고 있는 실행의 입력을 삭제한다.** 피해자는 원인과 무관해 보이는 `ZeroDivisionError: division by zero`(`data_utils.py:11`)로 죽는다 — `load_renderings`의 glob이 빈 리스트를 반환해 `output_map_size`가 0으로 남기 때문이다. 실제로 한 번 당했다.
   - 재질을 병렬로 돌리려면 실행마다 경로를 분리할 것: `--render_dir ./data/<재질>/radiance_LR_<태그> --upsamplings_dir ./data/<재질>/radiance_Upscaled_<태그>`
   - devcontainer 창을 두 개 열어두면 이 상황을 만들기 쉽다.
   - **`docker compose run`은 클라이언트를 죽여도 컨테이너가 계속 돈다.** Ctrl+C나 터미널 종료로 멈췄다고 믿지 말 것 — `docker ps`로 확인하고 `docker stop`으로 끊어야 한다.
