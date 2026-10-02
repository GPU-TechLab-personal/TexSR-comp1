# DeepTextures 체크포인트별 결과 — 시트와 색 이동 표

**2026-10-02.** 학습 중에 5,000 스텝마다 저장된 체크포인트 각각의 출력이다.
- 시트 하나 = **체크포인트 하나의 출력만**이다. 재질은 `0844_syntetic_foam`이고, 다섯 맵 + render가 한 줄로 들어 있다.
- LR / bilinear / MatUp / GT 같은 비교 열은 넣지 않았다. 비교는 `../deeptex_ablation/`과 `../deeptex_matsynth/`에 있다.
- 배경은 `docs/DEEPTEX-DEVIATIONS.md` §0.9·§0.10에 있다.

| 폴더 | 모델 | 시트 | 색 이동 표 |
|---|---|---|---|
| `matsynth_all/` | MatSynth 5맵 1:1 (논문 비율) | 30k·35k·40k (3개) | `drift.md` (baseColor), `drift_normal.md` |
| `matsynth_chratio/` | 채널 수 비례 3:3:1:1:1 (ablation) | 5k–40k (8개) | `drift.md`, `drift_normal.md` |
| `matsynth_basecolor/` | baseColor 전용 (대조군) | 5k–40k (8개) | `drift.md` |
| `textures_db/` | 논문 동봉 300장 | 없음 | `drift.md` (자기 학습 쌍 기준) |

**체크포인트 개수가 다른 이유.**
- upstream 학습 코드는 최근 3개만 남긴다(`train_esrgan.py:64`, `max_to_keep=3`). 그래서 `matsynth_all`과 `textures_db`는 30k–40k만 있다.
- 두 ablation 모델은 `run_deeptex.ps1 -KeepAllCheckpoints`로 8개를 모두 보존했다.

## 색 이동 표 (`drift*.md`) 읽는 법

- **값의 정의:** 체크포인트마다 mean(SR) − mean(HR)을 채널별로, 8-bit 레벨로 쟀다. `cast` = G − (R+B)/2(초록 쏠림)이다.
- **두 기준:**
  - `train`: 그 모델의 자기 학습 쌍 60개(seed 7)
  - `eval`: 평가 12개
- **보는 법:** 같은 모델이 체크포인트마다 얼마나 색이 흔들리는지 본다.
  - 예: `matsynth_chratio/drift.md`의 cast가 −8.6 → −7.6 → +11.2 → +4.5 → +0.7 → −0.5 → −5.3 → +1.7이다.
  - 이 흔들림 때문에 초록 쏠림은 데이터 구성이 아니라 레시피의 색 불안정으로 판정했다(§0.10).

## 시트 읽기 전에

- **1채널 맵의 출력:** roughness·metallic·height는 네트워크 출력 세 채널의 평균(`mean`)으로 합쳤다. 비교 몽타주와 같은 방식이다.
- **baseColor 전용 모델:** 다섯 맵 중 baseColor만 학습했다. 나머지 칸은 학습 때 보지 못한 맵에 적용한 결과다.
- **시트를 만든 스크립트:**
  - `matsynth_all`의 3장은 이 기능이 정식 스크립트로 들어가기 전에 일회성 스크립트로 그렸다.
  - 나머지는 `scripts/deeptex/deeptex_drift.py --sheet-material`로 그렸다.
  - 두 쪽 모두 `deeptex_compare.py`와 같은 입력 인코딩·출력 경로이고 칸 렌더링도 같다.
- **재현성:** GPU 추론이 비트 단위로 재현되지 않아, 다시 만들면 값이 1 LSB 안팎 달라질 수 있다.

## 다시 만들기

> **2026-10-02: 보존 체크포인트를 지웠다.** 두 ablation 모델의 `checkpoints_all/`을 디스크 정리로 삭제했다.
> - 그래서 **5k–25k(ckpt-1~5)의 시트와 표는 이 폴더가 유일한 기록이다.**
> - 30k·35k·40k(ckpt-6~8)는 각 모델의 `checkpoints/esrgan_40k/`에 남아 있어서 다시 잴 수 있다. 아래 명령의 `checkpoints_all`을 `checkpoints`로 바꾸면 된다.
> - 전 구간을 다시 보려면 `-KeepAllCheckpoints`로 재학습해야 한다.

```powershell
docker compose -f docker/compose.yaml run --rm deeptex python /workspace/scripts/deeptex/deeptex_drift.py --cfg /workspace/configs/deeptex/esrgan_matsynth_chratio.yaml --ckpt-dir /workspace/runs/deeptex/matsynth_chratio/checkpoints_all/esrgan_40k --pairs /workspace/runs/deeptex/datasets/matsynth/baseColor --out /workspace/runs/deeptex/drift/matsynth_chratio --sheet-material 0844_syntetic_foam
```

- 모델을 바꾸려면 `--cfg`와 `--ckpt-dir`를 바꾼다.
- normal 표는 `--pairs .../matsynth/normal --eval-map normal`로 만든다.
