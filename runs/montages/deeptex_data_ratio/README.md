# DeepTextures 학습 데이터 구성 비교 — 몽타주와 지표 (네 모델)

**2026-10-02.** 논문의 DeepTextures 학습 코드를 **무수정**으로, 학습 데이터 구성만 바꿔 학습한 네 모델을
MatUp·GT와 나란히 놓은 결과다. 평가 재질은 상위 폴더의 12개(MatSynth test split)와 같다.

| 열 이름 | 학습 데이터 | 체크포인트 |
|---|---|---|
| DeepTextures (textures_db) | 논문 동봉 300장 (사실상 컬러 모델) | `runs/deeptex/paper/checkpoints/esrgan_40k/ckpt-8` |
| DeepTextures (matsynth_all) | MatSynth 5맵 1:1 — **논문과 같은 비율** | `runs/deeptex/matsynth_all/checkpoints/esrgan_40k/ckpt-8` |
| DeepTextures (matsynth_chratio) | MatSynth, 맵의 실제 채널 수에 비례 3:3:1:1:1 — **비율 비교 실험** | `runs/deeptex/matsynth_chratio/checkpoints/esrgan_40k/ckpt-8` |
| DeepTextures (matsynth_basecolor) | MatSynth baseColor만 — **대조군** | `runs/deeptex/matsynth_basecolor/checkpoints/esrgan_40k/ckpt-8` |

- 네 모델 모두 40k 스텝이고 config가 같다. 데이터 경로 한 줄만 다르다.
- 설계와 결과는 `docs/DEEPTEX-DEVIATIONS.md` §0.9·§0.10에 있다.
- 상위 폴더의 `deeptex_matsynth/`(앞의 두 모델만)는 그대로 두었다. 이 폴더는 그 옆에 더한 것이다.

## 파일

| 파일 | 내용 |
|---|---|
| `<재질>.png` | 행 = 방법(LR / bilinear / MatUp / DeepTextures 네 모델 / GT), 열 = 다섯 맵 + render |
| `overview_<맵>.png` | 행 = 12개 재질, 열 = 방법 |
| `summary.md` | 맵별 평균표, 모델 간 맞대결 횟수 |
| `per_material.md` | 재질별 PSNR / SSIM (normal은 각도 오차도) |
| `metrics_long.csv` | 재질 × 맵 × 열 한 줄씩 (피벗용) |
| `report.json` | 모든 숫자 |

## 결론 (요약, 자세히는 §0.10)

- **비율을 바꿔도 얻은 것이 없다.** 40k에서 논문 비율(1:1)이 가장 낫거나 비슷하다.
  - baseColor PSNR은 셋이 같고(26.85–26.92), 색 이동을 뺀 구조는 1:1이 가장 좋다.
  - chratio는 roughness·height에서 약 2 dB를 잃었다. 회색 맵 샘플이 20 → 11%로 줄어서다.
- **초록 쏠림은 데이터 구성 탓이 아니라 레시피의 색 불안정이다.**
  - 체크포인트 8개를 모두 보면 비율 비교 실험 두 모델의 쏠림이 −12 ~ +11을 오간다.
  - 40k 값은 어디서 멈췄느냐에 달린 값이다.

## 읽기 전에

- **각 열은 40k 한 시점의 결과다.** 이 레시피는 체크포인트마다 전체 색이 크게 흔들린다(위 결론).
  - 다른 체크포인트에서는 색이 꽤 다르게 나온다.
  - 체크포인트별 결과는 `runs/deeptex/drift/<모델>/`(`drift.md`와 체크포인트별 시트)에 따로 있다.
  - 이 몽타주와는 섞지 않았다.
- **`summary.md`의 `offset` / `PSNR -off` 열은 진단용이다.** 그 색 이동을 따로 보여 준다.
- **학습 때 보지 못한 맵에 적용한 칸이 있다.**
  - textures_db 모델과 basecolor 모델은 normal·roughness·metallic·height를 학습 때 (거의) 보지 못했다.
  - 그 칸은 "컬러 모델을 다른 맵에 적용한 결과"다.
- **PSNR은 흐린 결과를 선호한다.** 디테일 양은 `|grad| %GT`와 함께 읽어야 한다.
- **metallic.** GT 12개 중 9개가 전부 0이다. 평균은 금속 재질 3개 기준이고, 나머지는 `=GT` 열로 센다.
- **재현성.** GPU 추론이 비트 단위로 재현되지 않아, 다시 만들면 PSNR 소수점 둘째 자리가 바뀔 수 있다.

## 다시 만들기

```powershell
powershell -ExecutionPolicy Bypass -File scripts/deeptex/run_deeptex.ps1 -Stage eval-arms -PaperConfig configs/deeptex/esrgan_matsynth_chratio.yaml   -PaperRoot runs/deeptex/matsynth_chratio
powershell -ExecutionPolicy Bypass -File scripts/deeptex/run_deeptex.ps1 -Stage eval-arms -PaperConfig configs/deeptex/esrgan_matsynth_basecolor.yaml -PaperRoot runs/deeptex/matsynth_basecolor -ExtraArms matsynth_all,matsynth_chratio
```

결과는 `runs/deeptex/report_matsynth_basecolor/`에 생긴다. 이 폴더는 거기서 `sheets/`(맵별 원본 크기 시트, 176 MB)만
빼고 복사한 것이다.
