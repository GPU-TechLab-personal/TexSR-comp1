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
docker compose run --rm dataprep python scripts/matsynth_extract.py --list
```

기본 샤드는 `test-00000-of-00008.parquet`(12개 재질). 다른 샤드는 `--shard`로.
**test 스플릿은 8/8 완비, train은 295/431로 결손**이 있다 (OPEN-ITEMS #4).
MatUp은 학습이 없으므로 test만으로 충분하다.

### 2.2 뽑는다

```powershell
docker compose run --rm dataprep python scripts/matsynth_extract.py --index 8
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
docker compose run --rm matup python main.py --upscaler swinir --ogl_normals `
  --input_mat       /workspace/runs/matsynth/<이름>/SVBRDF/ `
  --render_dir      /workspace/runs/matsynth/<이름>/radiance_LR/ `
  --upsamplings_dir /workspace/runs/matsynth/<이름>/radiance_Upscaled/ `
  --output_folder   /workspace/runs/matsynth/<이름>/output/
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

결과는 `output/<타임스탬프>/`에 **512² 16-bit PNG 5장**.

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

**이미 파악된 upstream 결함 둘** — 학습 시작 전에 결정해야 한다:
- `pretrain_name: './checkpoints/'`라 사전학습 탐색이 조용히 실패하고 랜덤 초기화로
  시작한다 (OPEN-ITEMS #6)
- `niter: 40000`인데 `lr_steps`가 `[50000, ...]`이라 **LR 감쇠가 한 번도 발동하지
  않는다** (OPEN-ITEMS #7)

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
