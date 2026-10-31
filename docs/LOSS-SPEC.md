# 손실식 명세: 원본 DeepTextures → R0a/R0b → R1

작성 2026-10-30. 이 문서는 두 가지만 다룹니다.

1. **차이표**: 원본 DeepTextures 학습과 render loss 실험의 세 arm(R0a, R0b, R1)이 어디서 다른지
2. **수식 ↔ 코드 대응표**: 이 arm들이 실제로 최적화한 손실식의 각 부분을 어느 코드가 계산하는지

모든 `file:line`은 2026-10-30 기준 작업 트리(커밋 `e0eb4c5`)를 가리킵니다. 기존 config 주석, `deeptex_train.py`·`grouped_dataset.py` docstring, `DEEPTEX-DEVIATIONS.md`는 고치지 않고 줄 번호로만 인용합니다.

## 0. 학습 코드는 두 개입니다

| | 원본 | 우리 트레이너 |
|---|---|---|
| 파일 | `external/DeepTextures/train_esrgan.py` (git submodule, 수정하지 않음) | `scripts/deeptex/deeptex_train.py` |
| 쓴 arm | 논문 arm: textures_db, `matsynth_all`, data_ratio (`run_deeptex.ps1:444`, "upstream train_esrgan.py VERBATIM") | **R0a, R0b, R1** (`run_deeptex.ps1:1306-1308`) |
| 손실·네트워크 | `modules/losses.py`, `modules/models.py` | **원본 모듈을 그대로 import합니다**(`deeptex_train.py:98-103`). R0/R1은 `--losses upstream`이라 RaGAN도 원본 함수입니다(`:444-445`, `:645-653`) |

R0/R1 실행 명령은 다음과 같습니다(`run_deeptex.ps1:1306-1308`).

```
deeptex_train.py --stage esrgan --losses upstream --shuffle off --allow-dead-lr-schedule
                 --pretrain-dir none --no-val --log-every 100 --render-monitor-every 100
```

`--shuffle off`, `--allow-dead-lr-schedule`, `--pretrain-dir none`은 원본이 실제로 한 동작(셔플 없음, LR 감쇠가 한 번도 걸리지 않음, scratch 시작)을 그대로 재현하려고 넣은 옵션입니다.

## 1. 차이표: 원본 → R0a/R0b → R1

"원본" 열은 `train_esrgan.py`의 동작입니다. 같은 레시피를 MatSynth에 적용한 arm이 `matsynth_all`입니다.

| 항목 | 원본 (`matsynth_all`) | R0a / R0b | R1 | 결과에 영향 | 근거 (원본 / 우리) |
|---|---|---|---|---|---|
| 학습 스크립트 | `train_esrgan.py` | `deeptex_train.py` grouped 분기 | 같음 | 아래 행들로 나뉨 | `run_deeptex.ps1:444` / `:1306-1308` |
| 데이터 레코드 | 이미지 1장 (MatSynth 2,904 재질 × 5맵 = 14,520 레코드) | **재질 1개의 5맵 + 위치 grid** (2,904 레코드) | 같음 | 있음 | `dataset.py:4-29` / `grouped_dataset.py:32-50`, `r0a.yaml:62-63` |
| 배치 | 이미지 $B=8$ | **재질 2 × 5맵 = 이미지 10장** | 같음 | 있음: 평균 분모 $\lvert I\rvert$, RaGAN 평균 $\bar C$, 판별기 BN 통계를 내는 묶음이 바뀝니다 | `esrgan.yaml:2` / `r0a.yaml:38`, `deeptex_train.py:723-724` |
| crop · flip · rot | 이미지마다 따로 뽑음, stateful 난수, 시드 없음 | **한 재질의 5맵이 같은 crop·flip/rot 공유**. 난수 키는 (seed, 스트림 순번) | 같음 | 있음 | `dataset.py:52-81` / `grouped_dataset.py:118-195`, `:330-335` |
| 셔플 | 없음 (`shuffle=False`) | 없음 (`--shuffle off`) | 같음 | 없음 (같은 동작) | `train_esrgan.py:37` / `deeptex_train.py:458`, `:832-835` |
| 배치 구성의 패스 간 고정 | $14520 \bmod 8 = 0$ → 고정 | $2904 \bmod 2 = 0$ → 고정 | 같음 | 없음 (둘 다 고정) | `dataset.py:95-102` (`repeat` 뒤 `batch`) |
| 초기화 시드 | 없음 | seed 1 (R0a) / 2 (R0b) | seed 1 (R0a와 init 비트 단위 동일) | 재현성. R1 − R0a를 render 항 하나로 좁힙니다 | — / `deeptex_train.py:563-577`, `r0a.yaml:48`, `r0b.yaml:49`, `r1.yaml:53` |
| 사전학습 | 경로 오류로 PSNR 모델을 못 찾고 scratch (`[*] cannot find pretrain model` 출력) | `--pretrain-dir none`, 명시적 scratch | 같음 | 없음 (결과 같음) | `train_esrgan.py:71`, `:77-79` / `run_deeptex.ps1:1308` |
| 네트워크 G/D | `RRDB_Model`, `DiscriminatorVGG128` | 원본 import | 같음 | 없음 | `models.py:79`, `:114` / `deeptex_train.py:98` |
| pixel / VGG / RaGAN 손실 | `modules/losses.py` | 원본 import, `--losses upstream` | 같음 | 없음 | `losses.py:5-97` / `deeptex_train.py:100-102`, `:645-653` |
| 손실 가중치, Adam, LR | 0.01 / 1.0 / 0.005, Adam 1e-4 (0.9, 0.99) | 같은 값 | 같음 | 없음 | `esrgan.yaml:25-42` / `r0a.yaml:12-23` (diff 목록에 없음) |
| LR 감쇠 | `lr_steps` [50k, …]가 40k에서 한 번도 걸리지 않음 | 같음 (`--allow-dead-lr-schedule`) | 같음 | 없음 | `esrgan.yaml:27` |
| 비유한 기울기 | 검사 없이 적용 | 적용 **전에** 검사하고, 비유한이면 **학습 중단**(건너뛰지 않음) | 같음 | 중단이 없으면 없음 | `train_esrgan.py:102-109` / `deeptex_train.py:752-756`, `:860-874` |
| 체크포인트 | 최근 3개만, 복원 결과 검사 없음 | 8개 모두 보관, 복원 검증 | 같음 | 없음 | `train_esrgan.py:62-66` / `deeptex_train.py:671-676`, `-KeepAllCheckpoints` |
| **render 손실** | 없음 | 없음 (`w_render: 0.0`) | **$+10\,\mathcal L_{render}$** | **실험 변수** | — / `r0a.yaml:94`, `r1.yaml:105`, `deeptex_train.py:741-743` |
| render monitor | 없음 | 100 step마다 render $\ell_{2,1}$를 **기록만** (`stop_gradient`, `training=False`) | 같음 | 없음 | — / `--render-monitor-every 100` |

**정리.**
- **원본 → R0:** 손실 함수는 같습니다. 다른 것은 데이터를 묶는 방식(재질 단위 배치, 5맵 공유 crop)과 학습 루프의 안전장치입니다.
- **R0 → R1:** `w_render` 한 값(0 → 10)만 다릅니다. 세 config의 나머지 값은 같습니다(`r0a.yaml` 대비 `r1.yaml`은 `w_render`와 출력 경로만, `r0b.yaml`은 seed와 출력 경로만 다릅니다).

## 2. 수식 ↔ 코드 대응표

### 2.1 기호

- 배치의 이미지 $i=1..B$. R0/R1은 $B=10$(재질 2 × 맵 5)입니다.
  - LR $x_i\in[0,1]^{32\times32\times3}$, HR $y_i\in[0,1]^{128\times128\times3}$, SR $\hat y_i=G(x_i)$
  - 값은 PNG를 /255 한 것입니다(`grouped_dataset.py:163-164`).
- $|I|=B\cdot128\cdot128\cdot3$은 픽셀 손실의 원소 수입니다.
- $C(\cdot)$는 판별기의 logit(스칼라 1개)입니다. $\bar C_r=\frac1B\sum_i C(y_i)$, $\bar C_f=\frac1B\sum_i C(\hat y_i)$입니다.
- 재질 $b$의 다섯 맵은 $S_{b,k}$로 씁니다. $k$는 baseColor, normal, roughness, metallic, height이고, 예측은 $\hat S_{b,k}$입니다. 평평한 순번으로는 이미지 $5b+k$가 재질 $b$의 맵 $k$입니다(`deeptex_train.py:723-724`, `:743`의 `reshape`).

### 2.2 전체 손실 (R1, 한 step)

$$
\mathcal L_G=\mathcal L^G_{reg}+0.01\,\mathcal L_{pix}+1\cdot\mathcal L_{feat}+0.005\,\mathcal L^G_{Ra}+w_r\,\mathcal L_{render},\qquad w_r=\begin{cases}0&\text{R0a, R0b}\\10&\text{R1}\end{cases}
$$
$$
\mathcal L_D=\mathcal L^D_{reg}+\mathcal L^D_{Ra}
$$

| 항 | 식 | 계산 코드 | 호출 코드 | 출처 | 의미 |
|---|---|---|---|---|---|
| $\mathcal L^G_{reg}$, $\mathcal L^D_{reg}$ | $\sum$ `model.losses` $=0$ | `models.py:7-8` (`l2(wd)`), `wd=0` 기본값 `:79`, `:114` | `deeptex_train.py:735-736` | 원본 | 항은 있지만 weight decay가 0이라 항상 0입니다. 논문 서술(L2 weight decay)과 다릅니다 |
| $\mathcal L_{pix}$ | $\frac{1}{\lvert I\rvert}\sum_{i,h,w,c}\lvert\hat y-y\rvert$ | `losses.py:5-8` (`MeanAbsoluteError`) | `deeptex_train.py:737` | 원본 | 원소 평균 L1입니다. 원소당 기울기는 $0.01\,\mathrm{sign}(\Delta)/\lvert I\rvert\approx2\times10^{-8}$로 사실상 꺼져 있습니다 |
| $\mathcal L_{feat}$ | $\frac{1}{B\cdot8\cdot8\cdot512}\sum\lvert\phi(P(\hat y))-\phi(P(y))\rvert$ | `losses.py:16-51` | `deeptex_train.py:738` | 원본 | $\phi$ = VGG19 `layers[20]`(conv5_4), 활성화 이전입니다(`:27-36`). 128² 입력이면 출력은 8×8×512입니다 |
| $P(x)$ | $\mathrm{caffe}(255x)/12.75$ | `losses.py:44-45` | — | 원본 | caffe = RGB→BGR 후 ImageNet 평균을 뺍니다. 주석(`:43`)은 "feature map rescale"이라 하지만, 실제로는 **VGG 입력**을 나눕니다. conv에 bias가 있으므로 $\phi(x/12.75)\ne\phi(x)/12.75$입니다 |
| $\mathcal L^G_{Ra}$ | $\tfrac12\big[\mathrm{BCE}(1,\sigma(C(\hat y)-\bar C_r))+\mathrm{BCE}(0,\sigma(C(y)-\bar C_f))\big]$ | `losses.py:83-86` | `deeptex_train.py:739` | 원본 | $\bar C$는 배치 전체 logit 평균(`tf.reduce_mean`)이고, BCE도 배치 평균입니다 |
| $\mathcal L^D_{Ra}$ | $\tfrac12\big[\mathrm{BCE}(1,\sigma(C(y)-\bar C_f))+\mathrm{BCE}(0,\sigma(C(\hat y)-\bar C_r))\big]$ | `losses.py:59-62` | `deeptex_train.py:740` | 원본 | $\bar C_f,\bar C_r$에 stop-gradient가 없어서 D 기울기가 상대 배치 평균을 통해서도 흐릅니다. BasicSR은 여기에 detach를 겁니다 |
| $w_r\,\mathcal L_{render}$ | §2.4 | `render_tf.py:566-578` (`RenderLoss.loss`) | `deeptex_train.py:741-743` | **우리 코드** | 10장을 (재질 2, 맵 5)로 다시 묶어 재질 단위로 렌더해 비교합니다 |

**BCE 수치 형태 (실측 2026-10-30).** 원본은 `sigmoid` 뒤에 Keras `BinaryCrossentropy(from_logits=False)`를 씁니다. 하지만 `train_step`이 `@tf.function`이라(`train_esrgan.py:84`, `deeptex_train.py:721`) Keras가 Sigmoid op의 logits를 되돌려 `sigmoid_cross_entropy_with_logits`로 계산합니다. 그래서 확률 clip($10^{-7}$)은 걸리지 않습니다.
- texsr/deeptex:tf2.10.1에서 logit −30, target 1로 확인했습니다.
  - `tf.function` 안: loss 10.158(정확한 값), 기울기 −1/3
  - eager: loss 5.30, 기울기 0
- clip은 eager 실행에서만 생기므로, 학습에서는 BCEWithLogits와 동치입니다.

### 2.3 갱신 방식

| 내용 | 코드 | 출처 |
|---|---|---|
| G와 D를 같은 forward, 같은 persistent tape에서 동시에 갱신 | `deeptex_train.py:729-756` (원본 `train_esrgan.py:86-109`) | 원본과 같은 구조 |
| D는 $y$와 $\hat y$를 따로 forward → BN 배치 통계도 real과 fake에서 따로 계산 | `deeptex_train.py:731-732` (원본 `:88-89`) | 원본과 같음 |
| $\mathcal L_D$는 D 변수로만 미분 → G로 흐르지 않음(detach와 동치) | `deeptex_train.py:748` | 원본과 같음 |
| 기울기 노름이 유한한지 확인한 뒤에 적용, 아니면 중단 | `deeptex_train.py:752-756`, `:860-874` | **우리 코드** |
| GT 렌더는 tape 밖에서 계산 | `deeptex_train.py:726-728` | **우리 코드** |

### 2.4 render 항 내부

$$
\mathcal L_{render}=\frac{1}{|\Omega|\,L}\sum_{p\in\Omega}\sum_{l=1}^{L}\sqrt{\max\Big(\sum_{c\in\{R,G,B\}}\big(\underbrace{T(R_l(\mathcal D(\hat S)))}_{\text{SR 렌더}}-\underbrace{\mathrm{sg}\,T(R_l(\mathcal D(S)))}_{\text{GT 렌더}}\big)_{p,c}^2,\;10^{-15}\Big)}
$$

$p$는 배치 안 두 재질의 모든 텍셀이고($N=2\cdot128^2$), $L=100$입니다. 계산은 모두 **우리 코드**(`scripts/deeptex/render_tf.py`)에서 합니다. 이 코드는 MatUp torch 렌더러를 TF로 한 줄씩 옮긴 것이며, 각 줄 주석에 MatUp 원래 위치가 적혀 있습니다. parity 게이트에서 torch와 float64 대비 약 1e-6 수준으로 일치함을 확인했습니다(2026-10-07).

| 기호 | 요소 | 식 | 코드 | 의미 |
|---|---|---|---|---|
| E1 | 디코딩 $\mathcal D$ | 아래 §2.5 | `decode_maps` `render_tf.py:346-394` | 네트워크 출력(5장의 RGB 이미지)을 렌더러 입력(반사율, 법선, 거칠기, 금속성, 위치)으로 바꿉니다 |
| E2 | BRDF 렌더 $R_l$ | $R_l(p)=I\,(n\cdot\omega_l)\big[\tfrac{(1-m)a}{\pi}+V F D\big]$, $I=8$ | `render_radiance` `:399-468` | 텍셀 $p$가 광원 $l$ 아래서 내는 선형 밝기입니다. **맵들이 서로 곱해지는 곳**입니다 |
| E3 | 광원 | $p_l=1.4\cdot\mathrm{Fib}_l$, $l=1..100$; $\omega_l(p)=\mathrm{nrm}(p_l-\mathrm{pos}(p))$, 감쇠 없음 | `matup_lights.py:131` (`training_lights`), `render_tf.py:411-414` | 광원은 절대 위치이므로 방향이 텍셀마다 다릅니다 |
| E4 | 톤맵 $T$ | $T(x)=\mathrm{lin2srgb}\big(\tfrac{x}{1+x}\big)$, 채널별 | `process_raw_render` `:242-244`, `render_display` `:471-474` | 화면에 보이는 값(display space)으로 바꾼 뒤 비교합니다 |
| E5 | 마스크 $\Omega$ | GT에서 $\lVert2s-1\rVert\ge0.1$이고 단위 $n_z\ge10^{-3}$ | `decode_maps` `:373`, `l21` `:485-496` | GT normal이 망가진 텍셀을 평균에서 뺍니다(실측 텍셀의 0.185%) |
| E6 | 노름 $\ell_{2,1}$ | (텍셀, 광원)마다 RGB 오차 벡터의 L2 길이, 그 뒤 마스크 평균 | `_texel_light_error` `:479-482`, `l21` `:485-496` | 안쪽은 L2, 바깥은 평균(L1)입니다. 텍셀당 기울기는 $\Delta/\lVert\Delta\rVert$(크기 1)입니다 |
| E7 | 목표 | GT 맵에 E1–E4를 똑같이 적용하고 stop-gradient | `RenderLoss.target` `:557-564`, `deeptex_train.py:728` | 학습 대상이 아닌 고정된 정답 렌더입니다 |
| E8 | 가중치 | $w_r=10$ | `r1.yaml:105`, `deeptex_train.py:742` | 아래 규칙으로 정했습니다 |

**$w_r$ 보정 규칙** (`render_calibrate.py:68-70`, `:448-461`, 결과 `runs/deeptex/render_gate/calibration.json`):
$$
w^*=\operatorname*{median}_{6\ \text{states}}\ \operatorname*{median}_{20\ \text{batches}}\ \frac{\max\big(\lVert\nabla_\theta\mathcal L_{pix}\rVert,\lVert\nabla_\theta\mathcal L_{feat}\rVert,\lVert\nabla_\theta\mathcal L_{gan}\rVert\big)}{\big\lVert\nabla_\theta\mathcal L_{render}\big(\hat S-\mathrm{sg}(\mu_{hw}(\hat S-S))\big)\big\rVert}
$$
- 6개 상태는 `matsynth_all`과 `matsynth_chratio`의 30k/35k/40k입니다. 분모는 색 오프셋을 뺀 출력에서 잰 render 기울기입니다.
- 결과는 9.54이고, 유효숫자 1자리로 반올림해 10입니다. 분자는 20개 배치 모두에서 VGG 항이었습니다.

### 2.5 디코딩 $\mathcal D$ (E1) 세부, `render_tf.py:346-394`

$\bar x$는 세 채널의 평균입니다.

| 맵 | 식 | 줄 | 비고 |
|---|---|---|---|
| baseColor → $a$ | $a=\mathrm{srgb2lin}(\mathrm{clip}(\hat S_{bc},0,1))$ | `:367` (`srgb_to_linear` `:219`) | [0,1] 밖의 값은 render 기울기가 0입니다. 렌더러가 `:416`에서 한 번 더 clip합니다 |
| normal → $n$ | $n=\mathrm{nrm}\big(n_x,\,-n_y,\,\max(n_z,10^{-3})\big)$, 여기서 $(n_x,n_y,n_z)=\mathrm{nrm}(2\hat S_{nrm}-1)$이고 $\lVert\cdot\rVert^2$ 하한은 $10^{-4}$ | `:370-378` (`_unit_normal` `:335`) | y 부호를 뒤집어 MatUp 좌표계로 맞춥니다. $n_z$ guard 때문에 $n_z<10^{-3}$인 SR 텍셀은 $n_z$ 쪽 기울기가 0입니다 |
| roughness → $\alpha$ | $\rho=\max(\mathrm{clip}(\bar{\hat S}_{rgh},0,1),0.045)$, $\alpha=\mathrm{clip}(\rho^2,2\cdot10^{-3},1)$ | `:381-383` | $\rho^2\ge0.002025$라서 $\alpha$의 clip은 항상 비활성입니다. 실효 하한은 $\rho\ge0.045$입니다 |
| metallic → $m$ | $m=\mathrm{clip}(\bar{\hat S}_{mtl},0,1)$ | `:386` | |
| height → 위치 | $\mathrm{pos}=\big(x,\,y,\,0.01(\mathrm{clip}(\bar{\hat S}_{hgt},0,1)-0.5)\big)$ | `:389-390` | $(x,y)$는 재질 전체 512² 기준 $[-1,1]^2$ grid를 같은 crop·flip/rot으로 자른 것입니다(`grouped_dataset.py:85`, `:118-165`) |

회색 맵(roughness, metallic, height)은 **채널 평균만** 렌더에 들어갑니다. 세 채널이 서로 달라도 render 항은 그 차이를 벌하지 않습니다.

### 2.6 BRDF (E2) 세부, `render_tf.py:399-468`

$$
R_l=I\,(n\cdot\omega_l)\Big[\frac{(1-m)\,a}{\pi}+V\,F\,D\Big],\qquad h=\mathrm{nrm}(\omega_l+v),\ v=(0,0,1)
$$

| 성분 | 식 | 줄 |
|---|---|---|
| 확산 | $(1-m)\,a/\pi$ | `:419`, `:450` |
| 프레넬 $F$ | Schlick $F_0+(1-F_0)(1-\omega_l\cdot h)^5$, $F_0=0.04(1-m)+m\,a$ | `:420-421`, `:463` (`f_schlick` `:319`) |
| 분포 $D$ | $A=\alpha^2$로 두면 $D=\dfrac{1}{\pi A\big(\frac{(t\cdot h)^2+(b\cdot h)^2}{A}+(n\cdot h)^2\big)^2}$ × heaviside$(n\cdot h)$ | `:459-461` (`d_ggx_anisotropic` `:276`) |
| 가시성 $V$ | $V=\dfrac{0.5}{\Lambda_v+\Lambda_l}$, $\Lambda_v=(n\cdot\omega_l)\sqrt{A(1-(n\cdot v)^2)+(n\cdot v)^2}$, $\Lambda_l$은 $v\leftrightarrow\omega_l$로 바꾼 것 × heaviside$(\omega_l\cdot h)$ | `:455-457` (`v_smith_ggx_correlated_aniso` `:293`) |
| 합치기 | $(n\cdot\omega_l)\,[\text{확산}+VFD]\cdot I$ | `:466-468` |
| 내적 clip | $n\cdot h$, $n\cdot v$, $n\cdot\omega_l$, $\omega_l\cdot h$ 모두 $[10^{-8},\,0.9999]$ | `:443-446` |

- baseColor $a$가 렌더 값에 들어가는 길은 둘입니다: 확산항 $(1-m)a$와 금속 반사율 $F_0$의 $m\,a$. 그리고 $a$에는 normal($n\cdot\omega_l$, $D$, $V$), roughness($D$, $V$), metallic($m$)이 곱해집니다.
- $D$와 $V$는 MatUp의 이방성 식에 $a^2=b^2=\alpha^2$, $c=0$을 넣은 것입니다. clip이 걸리지 않는 곳에서는 표준 isotropic GGX(roughness 매개변수 $\alpha_{GGX}=\rho^2$)와 같고, clip이 걸리는 경계에서는 다를 수 있습니다(`:279-281` 주석).
- 하한 상수: `VIS_MIN` 1e-8, `NDF_DENOM_MIN` 1e-8, `DET_MIN` 1e-15, `LAMBDA_INNER_MIN` 1e-15, `NORM_EPS` 1e-20 (`render_tf.py:110-125`)

### 2.7 R1 render 설정값 (`r1.yaml:106-116`)

| 키 | 값 | 출처 |
|---|---|---|
| `n_lights` | 100 | MatUp `main_optim.py:122` |
| `light_radius` | 1.4 | MatUp `main_optim.py:87` |
| `intensity` | 8.0 | MatUp `renderer.py:14` |
| `disp_val` | 0.01 | MatUp `dataLoader.py:154` |
| `min_roughness` | 0.045 | MatUp `renderer.py:15` |
| `min_alpha` | 2e-3 | MatUp `renderer.py:16` |
| `nz_min` | 1e-3 | 우리 추가 ($n_z$ guard) |
| `mask_min_norm` | 0.1 | 우리 추가 (GT 마스크) |
| `normal_norm_floor` | 1e-4 | 우리 추가 |
| `light_chunks` | 1 | 메모리 대비용, 값은 같음 |

MatUp 원래 손실(`external/matup_egsr/matup/train.py:13-29`)과의 차이:
- **목표:** MatUp은 SwinIR로 업샘플한 렌더를 쓰고, R1은 GT 맵의 렌더를 씁니다.
- **추가된 것:** 마스크와 $n_z$ guard는 MatUp에 없고 우리가 넣었습니다.
- **빠진 것:** MatUp의 metallic 정규화 항 $0.1\,\mathrm{MSE}(m,m_0)$는 옮기지 않았습니다.
