# TexSR-comp1 — MatUp vs DeepTextures

두 텍스처 초해상 논문을 **같은 조건에서 비교**하고 결과 텍스처를 뽑는 실험 워크스페이스.

| | |
|---|---|
| **MatUp** (EGSR 2024) | 재질 1개당 test-time 최적화. 학습 불필요. LR 재질을 100개 조명으로 렌더 → SwinIR로 렌더를 ×4 업샘플 → HR 맵을 역최적화 |
| **DeepTextures** (Sci. Rep. 2026) | TF2 ESRGAN (3채널 RGB, ×4). **사전학습 가중치가 배포된 적 없어 직접 학습해야 함.** 논문의 OpenGameArt 300장을 MatSynth로 대체 |

## 머신 역할

| | 노트북 (authoring) | 데스크탑 (compute) |
|---|---|---|
| GPU | Intel Arc iGPU — CUDA 없음 | **RTX 3080 Ti 12 GB** |
| 역할 | 코드·설정 작성 → GitHub push | clone → 빌드 → 학습·실험 |
| Docker | — | **Docker Desktop** |

코드는 GitHub로, **데이터는 GitHub를 타지 않는다** (DAS에 있음).

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

확인 — 네 줄이 나오고 SRFlow만 `-` 상태여야 한다:

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

두 Dockerfile 끝에 **빌드 타임 검증**이 들어 있어, 의존성 핀이 틀리면 실험 도중이 아니라 여기서 실패한다. matup 쪽 마지막 줄의 torchvision 버전을 확인할 것 (→ `docs/OPEN-ITEMS.md` #2).

### 7. MatUp 스모크 테스트

MatSynth를 붙이기 **전에** 전체 경로를 검증한다. 저장소에 동봉된 `victorian_wood` 128² 예제를 쓴다.

```powershell
docker compose run --rm matup python main.py --upscaler swinir
```

`output/<타임스탬프>/`에 16-bit PNG 4장(`baseColor_opt`, `metallic_opt`, `normal_opt`, `roughness_opt`)과 `render_opt.png`가 나오고 **8px 검은 테두리가 보이면** 의존성·경로·체크포인트·GPU가 전부 맞은 것이다.

---

## 구조

```
docker/          컨테이너 정의. 두 스택은 CUDA 요구가 배타적이라 완전 분리
  matup/         torch 2.0.0 + CUDA 11.7
  deeptex/       TF 2.10.1 + CUDA 11.2
  compose.yaml   /workspace(repo) · /data(DAS) 고정 마운트, GPU 예약, shm 8GB
  .env.example   DAS_ROOT — 유일한 머신별 값
external/        upstream 서브모듈 (커밋 고정)
docs/            OPEN-ITEMS.md — 미해결 결정·불확실성
```

컨테이너 내부 경로가 `/workspace`·`/data`로 고정돼 있어 **호스트 경로 차이는 `.env` 한 줄에만 존재한다.**

## 운영상 주의 (실험 전 필독)

1. **`--nb_lights`와 `--up_batch_size`는 함께 바꿔야 한다.** `sr_swinir.py`가 기본값 100/10에서만 렌더를 온전히 커버한다. `nb_lights`만 바꾸면 렌더가 **조용히 누락/중복**된다.
2. **MatSynth 법선은 OpenGL 규약**이므로 MatUp에 `--ogl_normals`가 필요하다. 반면 MatUp **출력은 항상 DirectX**로 하드코딩돼 있어 비교 전 되돌려야 한다.
3. **MatUp은 `height`를 출력하지 않는다.** 산출 맵은 4장뿐이고, 512² 중 **유효 영역은 496²**(8px 검은 테두리)라 비교 시 전 방법·GT에서 잘라내야 공정하다.
4. **DeepTextures는 학습 데이터 준비 스크립트가 없다.** upstream에서 삭제됐으므로 우리가 작성해야 한다. `external/esrgan-tf2`가 삭제된 `train_psnr.py`/`psnr.yaml`의 공여자다.
5. **`requirements.txt`를 upstream 것으로 되돌리지 말 것.** `docker/*/requirements.txt`에 변경 사유가 줄별로 주석돼 있다.
