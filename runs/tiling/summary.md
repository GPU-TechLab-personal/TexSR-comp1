# 타일링 실험 — 논문의 "scalable tiling"을 실제로 돌린 결과

**2026-09-19.** 비교 실험(128²→512²)에서는 타일링이 발동하지 않는다 —
`split_large_image`가 `H > 512 or W > 512`에서만 동작하기 때문이다
(`docs/DEEPTEX-DEVIATIONS.md` §0.8). 이 실험은 그 조건을 만족하는 이미지로
타일링을 **실제로 실행**한 별도 실험이다.

## 데이터

`runs/deeptex/datasets/textures_db/normalized/` 중 **홀드아웃(val) 분할**에서
타일 격자가 실제로 만들어지는 6장. **학습에 쓰이지 않았다.**

| 이미지 | 크기 | 격자 |
|---|---|---|
| `080`, `081` | 2048² | **4×4 = 16타일** |
| `217`, `231`, `275`, `293` | 1024² | 2×2 = 4타일 |

MatSynth 샤드는 쓰지 않았다(DAS 미마운트). 논문 자신의 데이터셋이므로 타일링
주장을 검증하기에는 오히려 적절하다.

## 비교한 세 경로

| | 무엇 |
|---|---|
| `bicubic` | `imresize_np` ×4 — 논문 자신의 기준선 |
| `upstream` | **`external/DeepTextures/test.py` 원본 그대로.** 512 타일 → JPEG q95 → 타일별 LR → SR → 맞대기 접합 |
| `ours` | `deeptex_sr.py` — LR 공간 타일 + 오버랩, 메모리 내, JPEG 없음 |

체크포인트는 양쪽 모두 `esrgan_40k` step 40,000(`ckpt-8`). upstream은
`test.py:36`이 `./checkpoints/original`을 하드코딩하므로 그 이름으로 배치만 했고
**서브모듈은 수정하지 않았다.**

## 결과 1 — 화질

지표는 `scripts/common/texsr_metrics`로 **한 구현에서** 재계산했다. `test.py`가
직접 찍는 PSNR/SSIM은 BGR을 먹인 `rgb2ycbcr`를 거쳐 R·B 가중치가 뒤바뀌어 있어
다른 무엇과도 비교할 수 없다.

| 이미지 | 격자 | bicubic | upstream | ours |
|---|---|---|---|---|
| 080 | 4×4 | 30.19 | **33.62** | 32.93 |
| 081 | 4×4 | 27.10 | 28.83 | **29.09** |
| 217 | 2×2 | **36.90** | 33.15 | 32.50 |
| 231 | 2×2 | **26.43** | 21.58 | 22.26 |
| 275 | 2×2 | **22.98** | 21.14 | 21.43 |
| 293 | 2×2 | **23.89** | 22.79 | 22.69 |

**2048² 두 장은 SR이 이기고, 1024² 네 장은 bicubic이 이긴다.** 6장은 결론을
내기에 적은 표본이므로 경향으로만 읽을 것.

## 결과 2 — 이음새 (논문 주장의 직접 검증)

논문은 타일이 *"processed and reassembled **seamlessly**"* 된다고 서술한다.
서브모듈에 `seam`·`blend`·`overlap`·`feather` 처리는 **0줄**이고, Fig. 6은
오히려 *"with no overlap"*이라고 적는다.

측정: 타일 경계를 가로지르는 인접 픽셀 절대차의 평균 ÷ 경계가 아닌 모든 위치의
같은 값. 이음새가 없으면 1.0. **정답 이미지도 같은 방식으로 재어** 내용물 영향을
나눈 값(`seam_vs_gt`)이 읽어야 할 숫자다.

| 이미지 | GT 원시비 | upstream 원시비 | **upstream ÷ GT** |
|---|---|---|---|
| 080 | 3.65× | 23.76× | **6.51×** |
| 081 | 2.70× | 15.55× | **5.76×** |
| 217 | 1.05× | 3.63× | **3.48×** |
| 231 | 1.00× | 1.31× | **1.30×** |
| 275 | 1.01× | 1.73× | **1.72×** |
| 293 | 1.11× | 1.84× | **1.65×** |

**6장 모두 1.0을 넘는다.** 평균 3.4배, 4×4 격자에서는 6배. GT의 원시비가
1024² 네 장에서 ≈1.00인 것이 측정이 올바르게 교정돼 있음을 보인다.

`seam_081.png`에서 육안으로도 확인된다 — upstream의 타일 경계에 **청색 프린지**가
있고 `ours`와 GT에는 없다.

⚠️ `ours`의 이음새 수치는 **같은 기준이 아니다.** 이 측정은 upstream의 고정 512
격자에 묶여 있는데 `deeptex_sr.py`는 LR 공간에서 오버랩을 두고 자르므로 경계가
다른 곳에 있다. 직접 비교하지 말 것.

## 산출물

```
src/                 입력 6장 (홀드아웃 원본)
upstream/<stem>/     test.py 원본 출력 — JPEG 타일, SR 타일, 모자이크, 비교 스트립
ours/                deeptex_sr.py 출력
bicubic/             기준선
sheet_<stem>.png     bicubic | upstream | ours | GT
seam_<stem>.png      x=512 경계 확대: upstream | ours | GT
tiling_report.json   전체 수치
```

## 재현

```
docker compose -f docker/compose.yaml run --rm \
  -w /workspace/runs/deeptex/tiling/upstream \
  -e PYTHONPATH=/workspace/external/DeepTextures \
  deeptex python /workspace/external/DeepTextures/test.py \
    --cfg_path /workspace/configs/deeptex/esrgan_paper.yaml \
    --img_path /workspace/runs/deeptex/tiling/src/081.png

docker compose -f docker/compose.yaml run --rm deeptex \
  python /workspace/scripts/deeptex/deeptex_sr.py \
    --cfg /workspace/configs/deeptex/esrgan_paper.yaml \
    --ckpt-dir /workspace/runs/deeptex/paper/checkpoints/esrgan_40k \
    --in-dir /workspace/runs/deeptex/tiling/src \
    --out-dir /workspace/runs/deeptex/tiling/ours \
    --make-lr bicubic --tile-size 128 --overlap 16 \
    --save-bicubic /workspace/runs/deeptex/tiling/bicubic

docker compose -f docker/compose.yaml run --rm dataprep \
  python scripts/deeptex/deeptex_tiling_report.py
```
