# DeepTextures MatSynth arm — 몽타주와 지표

**2026-10-01.** 논문의 DeepTextures 학습 코드를 **무수정**으로 두 데이터에 돌린 모델을
MatUp·GT와 나란히 놓은 결과다. 평가 재질은 이 폴더 상위의 12개(MatSynth test split)와 같다.

| 열 이름 | 무엇 | 체크포인트 |
|---|---|---|
| DeepTextures (textures_db) | 논문 동봉 300장으로 학습 (사실상 컬러 모델) | `runs/deeptex/paper/checkpoints/esrgan_40k/ckpt-8` |
| DeepTextures (matsynth_all) | MatSynth train split 2,904 재질 × 5맵을 섞어 모델 하나로 학습 | `runs/deeptex/matsynth_all/checkpoints/esrgan_40k/ckpt-8` |

두 모델 모두 40k 스텝, 같은 config다. 데이터 경로 한 줄만 다르다.
근거와 한계는 `docs/DEEPTEX-DEVIATIONS.md` §0.9에 있다.

**textures_db 열을 읽을 때.**
- 9월 18일 학습한 **기존 체크포인트 그대로**다. 재학습하지 않았다.
- 원래 비교(상위 폴더의 `<재질>_deeptex.png`)에서는 baseColor·render에만 썼다.
- 여기서는 기준선으로 다섯 맵 + render 전부에 돌렸다. 그래서 normal·roughness·metallic·height 칸은
  "**컬러 위주로 학습된 모델을 학습 때 거의 보지 못한 맵에 적용한 결과**"다. normal에 보이는 빗살무늬 잡음이 그 흔적이다.
- 1채널 맵은 출력 세 채널의 평균(`mean`)으로 합쳤다. MatSynth 열과 같은 방식이다.
- baseColor·render도 다시 추론해서, 상위 폴더 결과와 픽셀이 최대 1/255 다를 수 있다.
- 자세한 내용은 `docs/DEEPTEX-DEVIATIONS.md` §0.5의 2026-10-01 갱신에 있다.

## 파일

| 파일 | 내용 |
|---|---|
| `<재질>.png` | 행 = 방법(LR / bilinear / MatUp / DeepTextures 두 모델 / GT), 열 = 다섯 맵 + render |
| `overview_<맵>.png` | 행 = 12개 재질, 열 = 방법 |
| `summary.md` | 맵별 평균표, 모델 간 맞대결 횟수. 열 설명이 표 위에 있다 |
| `per_material.md` | 재질별 PSNR / SSIM (normal은 각도 오차도) |
| `metrics_long.csv` | 재질 × 맵 × 열 한 줄씩 (피벗용) |
| `report.json` | 모든 숫자 |

상위 폴더의 `<재질>.png`(MatUp 4×6), `<재질>_deeptex.png`(textures_db 모델, baseColor·render),
`overview_*.png`는 그대로 두었다. 이 폴더는 그것들을 대체하지 않고 옆에 더한다.

## 읽기 전에

- **색 이동.** 이 학습 레시피(사전학습 없음, 픽셀 손실 가중치 0.01, 판별자 포화)는 출력 색을 일정하게 민다.
  - 평탄한 normal에서 MatSynth 모델은 R이 약 −17 레벨 밀린다.
  - 자기 학습 쌍 40개에서도 같은 방향으로 밀려서(R −20) 평가 버그가 아니다.
  - `summary.md`의 `offset` / `PSNR -off` 열이 이 효과를 따로 보여 주는 **진단용** 수치다.
- **PSNR은 흐린 결과를 선호한다.** 그래서 bilinear가 PSNR로 자주 1등이다. 디테일 양은 `|grad| %GT`(GT 대비 선명도)와 함께 읽어야 한다.
- **metallic.** GT 12개 중 9개가 전부 0이다. 평균은 금속 재질 3개 기준이고, 나머지는 `=GT`(정확 재현) 열로 센다.
- **재현성.** GPU 추론이 비트 단위로 재현되지 않아, 다시 만들면 PSNR 소수점 둘째 자리가 바뀔 수 있다.

## 다시 만들기

```powershell
powershell -ExecutionPolicy Bypass -File scripts\deeptex\run_deeptex.ps1 -Stage eval-arms -PaperConfig configs/deeptex/esrgan_matsynth.yaml -PaperRoot runs/deeptex/matsynth_all
```

결과는 `runs\deeptex\report_matsynth_all\`에 생긴다. 이 폴더는 거기서 `sheets\`(맵별 원본 크기 시트, 124 MB)만 빼고
복사한 것이다.
