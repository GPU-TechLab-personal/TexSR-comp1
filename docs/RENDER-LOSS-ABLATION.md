# Render 손실 ablation 설계 노트: 렌더인가, ℓ2,1인가

작성 2026-10-31. **상태: 설계 보류.** 실험 2(SwinIR·ESRGAN의 MatSynth 파인튜닝)를 먼저 진행하고, 이 문서 §9의 미결정 사항과 §10의 RaGAN 검토를 마친 뒤 다시 시작합니다. 아직 코드는 없습니다.

이 문서는 다음을 정리합니다.
- R1(render·l21)이 최적화한 손실을 MatUp 논문 표기로 적은 식
- 그 식과 DeepTextures 원본 손실의 비교
- 요소를 하나씩 빼는 ablation 팔의 식과 판정 질문
- 설계 중에 직접 잰 값

원본 DeepTextures → R0a/R0b → R1의 차이표와 줄 단위 대응은 [LOSS-SPEC.md](LOSS-SPEC.md)에 있고, 여기서는 반복하지 않습니다. `file:line`은 커밋 `0dac128` 기준입니다.

## 1. 질문

R1은 R0a/R0b보다 baseColor 색 편차가 작습니다. 체크포인트 8개에 걸친 녹색 편향 span이 R0a는 약 29레벨, R1은 10k 이후 약 6레벨 이내입니다. 이 차이의 원인이 다음 중 무엇인지 가르려고 합니다.

1. **미분 가능 렌더링.** 맵 5장이 BRDF 안에서 서로 결합됩니다.
2. **ℓ2,1 노름의 억제 효과.** 오차가 작아져도 기울기 크기가 줄지 않습니다.
3. **충분히 큰 충실도 항 자체(앵커).** 원본의 pixel L1보다 유효 가중치가 수천 배 큽니다.

## 2. 기호와 MatUp의 손실

| 기호 | 뜻 |
|---|---|
| $\mathcal M_{LR}$, $\mathcal M_{HR}$, $\mathcal M_{SR}=G(\mathcal M_{LR})$ | 저해상도 입력, GT, 생성기 출력 재질 (맵 5장, 각각 RGB) |
| $k$ | 맵: baseColor, normal, roughness, metallic, height |
| $\mathcal P$ | 배치 안 두 재질의 텍셀 ($2\times128^2=32{,}768$개) |
| $\mathcal P_v\subset\mathcal P$ | GT normal이 정상인 텍셀 (§6 마스크) |
| $\Omega$ | 광원 집합 ($\lvert\Omega\rvert=100$), 원소 $\omega_i$ |
| $\mathcal R^{\omega_i}\{\mathcal M\}$ | 재질 $\mathcal M$을 광원 $\omega_i$로 렌더한 display 이미지 |
| $\langle x\rangle$ | $\mathrm{clip}(x,10^{-8},0.9999)$, 내적 clamp |
| $\chi^+(x)$ | $x>0$이면 1, 아니면 0 |
| $\bar x$ | 세 채널의 평균 |

혼합 노름(Kowalski 2009)은 다음과 같습니다. 행마다 L2 길이를 재고 그 길이를 더하는 것이 $(2,1)$입니다.

$$
\lVert\alpha\rVert_{p,q}=\Big(\sum_i\Big(\sum_j|\alpha_{i,j}|^p\Big)^{q/p}\Big)^{1/q},\qquad
\lVert\alpha\rVert_{2,1}=\sum_i\lVert\alpha_{i,\cdot}\rVert_2
$$

MatUp 식 (1)은 다음과 같습니다. $\mathcal S$는 이미지 업샘플러(SwinIR 또는 SRFlow)입니다.

$$
\mathcal L_{\text{fit}}^{\text{MatUp}}=\sum_{p\in\mathcal P}\sum_{\omega_i\in\Omega}\Big\lVert\,\mathcal S\big(\mathcal R^{\omega_i}\{\mathcal M_{LR}\}\big)(p)-\mathcal R^{\omega_i}\{\mathcal M_{SR}\}(p)\Big\rVert_2
$$

MatUp은 렌더링 손실을 "in-network renderer"로 구현한다고 쓰며 Aittala et al. 2016을 인용합니다(*Reflectance Modeling by Neural Texture Synthesis*, ACM TOG 35(4)).

## 3. R1의 render 손실

렌더 연산자는 $\mathcal R^{\omega_i}=T\circ R_{\omega_i}\circ\mathcal D$입니다(`scripts/deeptex/render_tf.py`). MatUp torch 렌더러를 한 줄씩 옮겼고, 2026-10-07 parity 게이트에서 약 1e-6 수준으로 일치했습니다.

### 3.1 디코딩 $\mathcal D$ (`render_tf.py:346-394`)

$$
\begin{aligned}
a_p&=\mathrm{srgb2lin}\big(\mathrm{clip}(\mathcal M_{bc}(p),0,1)\big)\\[1mm]
s_p&=\frac{2\mathcal M_{nrm}(p)-1}{\sqrt{\max\big(\lVert2\mathcal M_{nrm}(p)-1\rVert^2,\ 10^{-4}\big)}},\qquad
n_p=\mathrm{nrm}\big(s_{p,x},\ -s_{p,y},\ \max(s_{p,z},10^{-3})\big)\\[1mm]
\rho_p&=\max\big(\mathrm{clip}(\bar{\mathcal M}_{rgh}(p),0,1),\ 0.045\big),\qquad \alpha_p=\mathrm{clip}(\rho_p^2,\ 2\!\cdot\!10^{-3},\ 1)=\rho_p^2\\[1mm]
m_p&=\mathrm{clip}\big(\bar{\mathcal M}_{mtl}(p),0,1\big)\\[1mm]
\mathrm{pos}_p&=\big(x_p,\ y_p,\ 0.01\,(\mathrm{clip}(\bar{\mathcal M}_{hgt}(p),0,1)-0.5)\big)
\end{aligned}
$$

- **srgb2lin**은 구간별 식입니다(IEC 61966-2-1). $c\le0.04045$이면 $c/12.92$, 그 위는 $((c+0.055)/1.055)^{2.4}$입니다. `tf.where`의 선택하지 않은 가지에서 NaN 기울기가 생기지 않도록 pow 안에 clamp가 있습니다(`:219-224`).
- **α의 clip은 걸리지 않습니다.** $\rho^2\ge0.002025$이므로 항상 $\alpha_p=\rho_p^2$입니다.
- **회색 맵은 채널 평균만 들어갑니다.** 세 채널이 서로 달라도 render 항은 그 차이를 벌하지 않습니다.

### 3.2 광원 (`scripts/common/matup_lights.py:120-136`, `render_tf.py:411-414`)

$$
\ell_i=1.4\cdot\mathrm{Fib}^{+}_i,\qquad
\omega_i(p)=\mathrm{nrm}(\ell_i-\mathrm{pos}_p),\qquad v=(0,0,1),\qquad h=\mathrm{nrm}(\omega_i+v)
$$

- $\mathrm{Fib}^{+}$는 위쪽 반구의 Fibonacci 점입니다.
- 광원은 절대 위치이므로 방향이 텍셀마다 다르고, 거리 감쇠는 없습니다.
- $t_p$, $b_p$는 $n_p$에서 만든 branchless ONB입니다(Duff 2017, `:255`).

### 3.3 BRDF $R_{\omega_i}$ (Cook–Torrance, metallic–roughness 형태, `render_tf.py:399-468`)

$$
R_{\omega_i}(p)=8\,\langle n_p\cdot\omega_i\rangle\Big[\frac{(1-m_p)\,a_p}{\pi}+V\,F\,D\Big]
$$

$$
F_0=0.04\,(1-m_p)+m_p\,a_p,\qquad
F=F_0+(1-F_0)\big(1-\langle\omega_i\cdot h\rangle\big)^5
$$

$$
D=\frac{\chi^+(n_p\cdot h)}{\pi\,\max\Big(\alpha_p^2\Big(\dfrac{(t_p\cdot h)^2+(b_p\cdot h)^2}{\alpha_p^2}+\langle n_p\cdot h\rangle^2\Big)^2,\ 10^{-8}\Big)}
$$

$$
V=\frac{0.5\ \chi^+(\omega_i\cdot h)}{\max(\Lambda_v,10^{-8})+\max(\Lambda_l,10^{-8})}
$$

$$
\Lambda_v=\langle n_p\cdot\omega_i\rangle\sqrt{\alpha_p^2\big((t_p\cdot v)^2+(b_p\cdot v)^2\big)+\langle n_p\cdot v\rangle^2},\qquad
\Lambda_l=\langle n_p\cdot v\rangle\sqrt{\alpha_p^2\big((t_p\cdot\omega_i)^2+(b_p\cdot\omega_i)^2\big)+\langle n_p\cdot\omega_i\rangle^2}
$$

- 구성 요소는 표준입니다: GGX $D$(Trowbridge–Reitz, Walter 2007), height-correlated Smith $V$(Heitz 2014), Schlick $F$, Lambert 확산.
- 표준과 다른 점은 다섯 가지입니다.
  - 확산항에 $(1-F)$ 가중이 없습니다(UE4와 같음).
  - 내적 상한이 0.9999입니다.
  - clamp는 $n$과의 내적에만 걸립니다. 그래서 isotropic GGX와 같아지는 것은 clamp가 걸리지 않는 곳뿐입니다.
  - 시점이 직교 시점 $(0,0,1)$로 고정입니다.
  - 광원 세기 8, 점광원, 거리 감쇠 없음입니다.
- 렌더러는 $a_p$와 $m_p$를 한 번 더 $[0,1]$로 자릅니다(`:416-417`).

### 3.4 톤맵 $T$ (`render_tf.py:237-244`, MatUp `render_utils.py:134-137`)

$$
T(x)=\mathrm{lin2srgb}\Big(\frac{x}{1+x}\Big)
$$

순서는 Reinhard → lin2srgb(감마)이고, 채널마다 따로 적용합니다. 손실은 채널별이 아니라 RGB를 묶어 계산합니다(§3.6).

### 3.5 정답 렌더

$\mathrm{sg}\,\mathcal R^{\omega_i}\{\mathcal M_{HR}\}$는 GT 맵에 3.1–3.4를 똑같이 적용해, GradientTape를 열기 전에 한 번만 계산합니다(`RenderLoss.target :557-564`, `deeptex_train.py:726-728`). sg는 GT에 학습 변수가 없으므로 결과를 바꾸지 않고, 계산 그래프를 tape에 남기지 않는 역할만 합니다.

### 3.6 오차 행렬과 혼합 노름 (`render_tf.py:479-496`, `:566-578`)

$$
A_{(p,\omega_i),\,c}=\mathrm{sg}\,\mathcal R^{\omega_i}\{\mathcal M_{HR}\}(p)_c-\mathcal R^{\omega_i}\{\mathcal M_{SR}\}(p)_c,\qquad p\in\mathcal P_v,\ \omega_i\in\Omega,\ c\in\{R,G,B\}
$$

$$
\boxed{\ \mathcal L_{render}=\frac{\lVert A\rVert_{2,1}}{\max(|\mathcal P_v|,1)\,|\Omega|}=\frac{1}{\max(|\mathcal P_v|,1)\,|\Omega|}\sum_{p\in\mathcal P_v}\sum_{\omega_i\in\Omega}\sqrt{\max\Big(\textstyle\sum_c A_{(p,\omega_i),c}^2,\ 10^{-15}\Big)}\ }
$$

$$
\frac{\partial\mathcal L_{render}}{\partial\,\mathcal R^{\omega_i}\{\mathcal M_{SR}\}(p)}=-\frac{1}{|\mathcal P_v|\,|\Omega|}\,\frac{A_{(p,\omega_i),\cdot}}{\lVert A_{(p,\omega_i),\cdot}\rVert_2}
$$

- 기울기는 행마다 단위벡터라 크기가 오차와 상관없습니다.
- 가중치로 보면 제곱 L2에 $1/(2\lVert A_{(p,\omega_i)}\rVert)$를 곱한 것과 같습니다. 즉 작은 오차일수록 상대적으로 세게 밉니다.
- $\varepsilon=10^{-15}$는 $\lVert A\rVert<3.2\times10^{-8}$에서만 작동하는 수치 보호입니다.

### 3.7 MatUp과 다른 점

| 항목 | MatUp | R1 |
|---|---|---|
| 목표 | LR 렌더를 SwinIR 또는 SRFlow로 업샘플한 것 (`main.py:12`, `dataLoader.py:98-143`) | GT 맵을 렌더한 것 |
| 최적화 대상 | 재질마다 학습하는 MLP. 첫 층이 타일 크기 conv라 텍셀 주변을 보고, 중심 텍셀의 보정값을 bilinear 초기 맵에 더함 (`model.py:18-21`) | 여러 재질이 공유하는 생성기 G의 가중치 |
| 법선 표현 | 기울기 2개, $z=\sqrt{\max(1-x^2-y^2,0.001)}$ → $n_z>0$이 구조적으로 보장 (`render_utils.py:20-38`) | RGB 3채널 → guard $\max(s_z,10^{-3})$가 같은 역할 |
| 거칠기 변수 | α를 직접 최적화 (로드할 때 $\rho^2$로 변환) | 생성기가 ρ를 내고 디코딩에서 $\alpha=\rho^2$ |
| 렌더 전 clip | 선형 albedo, metallic, α를 자르고 법선 재정규화 (`data_utils.py:74-88`) | sRGB baseColor를 자른 뒤 srgb2lin. **결과는 같음**(srgb2lin은 [0,1]을 [0,1]로 보내는 증가함수) |
| 텍셀 집합 | $\mathcal P$ 전체 | $\mathcal P_v$ |
| 정규화 | 식에는 없음. 코드는 배치 × 광원 수 (`train.py:18-21`) | $\max(\lvert\mathcal P_v\rvert,1)\cdot\lvert\Omega\rvert$ |
| 추가 항 | metallic 정규화 $0.1\,\mathrm{MSE}(m,m_0)$ (`train.py:23-27`) | 없음 |

### 3.8 전체 손실과 가중치

$$
\mathcal L_G=0.01\,\mathcal L_{pix}+\mathcal L_{feat}+0.005\,\mathcal L^G_{Ra}+w_r\,\mathcal L_{render},\qquad
w_r=\begin{cases}0 & \text{R0a, R0b}\\ 10 & \text{R1}\end{cases}
$$

$$
w_r=\underset{6\ \text{상태}}{\mathrm{median}}\ \underset{20\ \text{배치}}{\mathrm{median}}\ \frac{\max\big(\lVert\nabla_\theta 0.01\mathcal L_{pix}\rVert,\ \lVert\nabla_\theta\mathcal L_{feat}\rVert,\ \lVert\nabla_\theta 0.005\mathcal L^G_{Ra}\rVert\big)}{\Big\lVert\nabla_\theta\mathcal L_{render}\Big(\mathcal M_{SR}-\mathrm{sg}\big(\mu_{hw}(\mathcal M_{SR}-\mathcal M_{HR})\big)\Big)\Big\rVert}=9.54\ \to\ 10
$$

- 6개 상태는 `matsynth_all`과 `matsynth_chratio`의 30k/35k/40k입니다.
- $\mu_{hw}$는 이미지·맵·채널마다 공간 평균을 낸 SR−GT 차이(전역 색 오프셋)입니다(`render_calibrate.py:363-365`).
- 원리는 Adam에서 각 항의 기울기 크기 비율이 갱신 방향을 정한다는 것입니다. 그래서 render 항을 기존 최대 항(VGG)과 같은 크기로 맞춥니다.

## 4. DeepTextures 원본 손실과의 비교

같은 혼합 노름 표기로 쓰면 다음과 같습니다.

$$
\mathcal L_{pix}=\frac{\lVert\Delta\rVert_{1,1}}{|I|},\qquad \Delta_{(b,k,p),c}=\mathcal M_{SR,k}(p)_c-\mathcal M_{HR,k}(p)_c,\qquad |I|=10\cdot128^2\cdot3=491{,}520
$$

$$
\mathcal L_{feat}=\frac{\lVert\Phi\rVert_{1,1}}{|F|},\qquad \Phi_{(b,k,q),f}=\phi(P(\hat y_{b,k}))_{q,f}-\phi(P(y_{b,k}))_{q,f},\qquad |F|=10\cdot8^2\cdot512=327{,}680
$$

| 항목 | pixel L1 | VGG feature | RaGAN | render ℓ2,1 |
|---|---|---|---|---|
| 비교 공간 | 인코딩된 맵 RGB | 맵 이미지의 VGG conv5_4 특징 | 판별기 logit | 렌더 display RGB × 광원 100 |
| 비교 단위 | 맵 1장 | 맵 1장 | 맵 1장 | 재질 1개 (맵 5장을 함께) |
| 맵 사이 결합 | 없음 | 없음 | 없음 | 있음 (BRDF) |
| 노름 | (1,1) | (1,1) | BCE | (2,1) |
| 회색 맵 3채널 | 셋 다 | 셋 다 | 셋 다 | 채널 평균만 |
| 맵별 가중 | 같음 | 같음 | 같음 | 보이는 정도에 따라 다름 (height 0.01배, roughness는 톤맵에 눌린 하이라이트로만) |
| 마스크 | 없음 | 없음 | 없음 | 0.185% 제외 |
| 가중치 | 0.01 | 1 | 0.005 | 10 |

**기울기 크기.**
- SR 출력 텍셀 하나가 받는 baseColor 기울기 추정치(자릿수 비교용)는 pixel이 $0.01\sqrt3/|I|\approx3.5\times10^{-8}$, render가 $10\times0.6/|\mathcal P_v|\approx2\times10^{-4}$입니다. render가 약 6,000배 큽니다.
- θ 기준 기울기 노름(`calibration.json`, 학습된 상태 6개, 가중치 곱한 값)은 다음과 같습니다.

| 항 | $\lVert\nabla_\theta\rVert$ |
|---|---|
| pixel | 0.022–0.034 |
| feature | 9.6–11.3 |
| RaGAN | 1.8–4.1 |
| render (색 오프셋 제거 후) | 9–15 |
| render (원래 출력) | 72–152 |

**정리.**
- 원본 손실은 맵 5장을 각각 별개의 사진으로 다룹니다. 전역 색 오프셋을 직접 벌하는 pixel L1은 사실상 꺼져 있습니다.
- R1은 원본 대비 세 가지를 한꺼번에 바꿨습니다: 맵 결합, (2,1) 노름, 수천 배 큰 유효 가중치. ablation은 이 셋을 떼어 보는 것입니다.

## 5. 색 변환과 톤맵의 감도

모두 `render_tf.py`와 같은 식으로 계산한 값입니다.

**srgb2lin 기울기.** 같은 8bit 오차가 선형 공간에서 얼마가 되는지입니다.

| sRGB c | 선형 값 | d lin / d c |
|---|---|---|
| 0.02 | 0.0015 | 0.077 |
| 0.10 | 0.0100 | 0.155 |
| 0.25 | 0.051 | 0.400 |
| 0.50 | 0.214 | 0.926 |
| 0.75 | 0.523 | 1.558 |
| 1.00 | 1.000 | 2.275 |

**$T$의 오차 증폭률.**

| 선형 radiance x | T(x) | dT/dx |
|---|---|---|
| 0.01 | 0.099 | 6.36 |
| 0.1 | 0.334 | 1.47 |
| 0.5 | 0.613 | 0.371 |
| 1 | 0.735 | 0.165 |
| 3 | 0.881 | 0.033 |
| 10 | 0.959 | 0.0038 |

**확산 반사 사슬** $\text{disp}=T(k\cdot\mathrm{srgb2lin}(bc))$의 $d\,\text{disp}/d\,bc$.

| k | bc=0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|---|---|---|---|---|---|
| 0.3 | 0.598 | 0.587 | 0.554 | 0.506 | 0.448 |
| 1 | 0.986 | 0.905 | 0.760 | 0.592 | 0.439 |
| 3 | 1.516 | 1.193 | 0.783 | 0.473 | 0.284 |

**해석.**
- 디코딩 단계에서 어두운 색과 밝은 색의 감도 차이는 약 30배입니다. 렌더 끝(display)에서는 1.3–5배로 줄어듭니다. 감마 쌍이 거의 상쇄되고, 남는 차이는 Reinhard가 밝은 쪽을 누르기 때문입니다.
- $T$는 어두운 텍셀 오차를 하이라이트 오차보다 약 1,700배 크게 셉니다. R1은 사실상 확산 반사와 그늘진 영역을 맞추는 손실입니다.
- 공간별 baseColor 가중은 이렇게 다릅니다.
  - render: 거의 지각적으로 균일
  - phys(선형): 어두운 색 오차를 약 30배 덜 봄
  - raw(sRGB): 균일

## 6. 마스크 $\mathcal P_v$

$$
\mathcal P_v=\Big\{p\in\mathcal P:\ \lVert2\mathcal M_{HR,nrm}(p)-1\rVert\ge0.1\ \ \text{그리고}\ \ s^{HR}_{p,z}\ge10^{-3}\Big\}
$$

GT에서만 계산하고 sg를 겁니다(`render_tf.py:370-373`).

**마스크가 필요한 이유.** 정답 렌더도 GT 맵을 렌더한 것이지만, 망가진 GT normal은 렌더러가 정답을 지어내게 만듭니다.
- 길이가 0에 가까운 법선은 방향이 8bit 양자화 잡음으로 정해집니다. 정규화 기울기도 $1/\lVert s\rVert$로 커집니다.
- 아래를 향한 법선은 guard가 거의 수평으로 만들어, 낮게 비추는 광원 아래서 가짜 하이라이트가 생깁니다. guard가 걸린 성분은 기울기도 0입니다.

**학습 데이터 실측** (`runs/deeptex/datasets/matsynth_grouped/verify_report.json`)

| 항목 | 값 |
|---|---|
| 무효 텍셀 | 1,408,888 / 761,266,176 (0.185%) |
| 그중 $n_z<10^{-3}$ | 1,356,792 (96%) |
| 무효 텍셀이 있는 재질 | 181 / 2,904 |
| 1% 초과 | 29 |
| 10% 초과 | 8 |
| 전체 무효 | 4 (`1652/1653/1654/1656_fur_cow_long`, 상수색 normal 맵) |

- `*_wood_planks_aged` 계열은 8–13%가 아래를 향한 법선이고, `js_fabric_carpet_001`은 9%가 길이 0.1 미만인 법선입니다.
- fur_cow는 마스크를 빼면 render ℓ2,1이 0.052가 나옵니다(`render_gate/mask.json`). 이 값은 지어낸 정답에서 나온 것입니다.

**결정 (사용자, 2026-10-31).** 새 팔도 모두 R1과 같은 마스크를 씁니다. 모든 팔의 유효 영역을 같게 해서, R1과의 차이를 요인 하나로 좁히기 위해서입니다.

## 7. ablation 팔

이름은 {비교 공간}·{노름}입니다.
- 공간: `render` = 렌더 display, `phys` = 디코딩된 물리량, `raw` = 네트워크가 낸 맵 RGB 그대로
- 노름: `l21` = (2,1), `l2sq` = (2,2)의 제곱(MSE), `l1` = (1,1)

모든 팔은 $\mathcal P_v$와 가중치 결정 규칙(§3.8, 해당 항으로 바꿔서)이 같습니다.

| 팔 | 기존 이름 | 바꾸는 곳 | 묻는 질문 |
|---|---|---|---|
| render·l21 seed 1 | R1 | 기준 (완료) | |
| render·l21 seed 2 | R1b | seed | 우연에 의한 차이 τ1 |
| render·l2sq | Q2 / RL2 | $\lVert A\rVert_{2,1}\to\lVert A\rVert_{2,2}^2$ | ℓ2,1의 억제 효과 |
| render·l1 | Q11 | $\lVert A\rVert_{2,1}\to\lVert A\rVert_{1,1}$ | RGB를 묶는 것의 효과 (선택) |
| phys·l21 | D21 | $\mathcal R^{\omega_i}\to\mathcal D$ | 미분 가능 렌더링의 효과 |
| raw·l21 | M21 | $\mathcal R^{\omega_i}\to$ 항등 | 디코딩의 효과 |
| raw·l2sq | M2 | raw에서 (2,2)² | 형태(크기 일정) 대 앵커 (§9 참고) |
| none seed 1/2 | R0a/R0b | render 항 없음 (완료) | τ0 |

$$
\mathcal L_{\text{render·l2sq}}=\frac{\lVert A\rVert_{2,2}^2}{|\mathcal P_v|\,|\Omega|},\qquad
\mathcal L_{\text{render·l1}}=\frac{\lVert A\rVert_{1,1}}{|\mathcal P_v|\,|\Omega|}
$$

$$
\mathcal L_{\text{phys·l21}}=\frac{1}{5|\mathcal P_v|}\sum_{p\in\mathcal P_v}\Big[\lVert\hat a_p-a_p\rVert_2+\lVert\hat n_p-n_p\rVert_2+|\hat\alpha_p-\alpha_p|+|\hat m_p-m_p|+|\hat z_p-z_p|\Big]
$$

$$
\mathcal L_{\text{raw·l21}}=\frac{1}{5|\mathcal P_v|}\sum_{p\in\mathcal P_v}\sum_k\lVert\mathcal M_{SR,k}(p)-\mathcal M_{HR,k}(p)\rVert_2,\qquad
\mathcal L_{\text{raw·l2sq}}=\frac{1}{5|\mathcal P_v|}\sum_{p\in\mathcal P_v}\sum_k\lVert\mathcal M_{SR,k}(p)-\mathcal M_{HR,k}(p)\rVert_2^2
$$

| 비교 | 바뀐 것 하나 | 질문 |
|---|---|---|
| render·l21 vs phys·l21 | 렌더 유무 | 미분 가능 렌더링이 편차를 줄였는가 |
| phys·l21 vs raw·l21 | 디코딩 유무 | sRGB→linear, clip 등의 영향 |
| render·l21 vs render·l2sq | 노름 | ℓ2,1의 억제 효과인가 |
| render·l21 vs render·l1 | RGB 묶음 | RGB를 한 벡터로 묶는 것이 중요한가 |
| raw·l21 vs raw·l2sq | 노름 (렌더 없이) | 크기 일정한 기울기 형태인가, 강한 충실도 항 자체인가 |

## 8. 설계 중 측정값 (텍셀 하나, `texsr/deeptex:tf2.10.1`)

**노름 비교.** 렌더 공간에서 baseColor 전 채널에 같은 오프셋을 주고 $\lVert\partial\mathcal L/\partial\,bc\rVert$를 쟀습니다(부록 A).

| 오프셋 (레벨) | l21 | l2sq | l1 |
|---|---|---|---|
| 유전체 +1 | 0.667 | 0.0061 | 1.155 |
| 유전체 +4 | 0.659 | 0.0241 | 1.141 |
| 유전체 +16 | 0.625 | 0.0891 | 1.083 |
| 유전체 +64 | 0.487 | 0.248 | 0.843 |
| 금속 +1 | 0.604 | 0.0052 | 1.047 |
| 금속 +64 | 0.462 | 0.219 | 0.800 |

- l21은 오프셋이 64배 커져도 기울기가 거의 같습니다. l2sq는 오프셋에 거의 비례합니다. 1레벨에서 그 비는 약 110배입니다.
- 회색 방향 오프셋에서 l1은 l21의 정확히 √3배입니다. 방향이 같아서, 둘은 색조 오차에서만 갈립니다.

**공간 비교.** ℓ2,1을 고정하고 같은 baseColor 오프셋에서 맵별 기울기를 쟀습니다(부록 B).

| 경우 | 공간 | \|g bc\| | g rough | \|g normal\| | g metal |
|---|---|---|---|---|---|
| 유전체 +1 | render | 0.667 | +0.039 | 0.037 | −0.136 |
| 유전체 +1 | phys | 0.187 | 0 | 0 | 0 |
| 유전체 +1 | raw | 0.200 | 0 | 0 | 0 |
| 금속 +1 | render | 0.604 | +0.798 | 0.033 | −0.341 |
| 유전체 +32 | render / phys / raw | 0.579 / 0.246 / 0.200 | | | |

- 렌더 공간에서는 baseColor만 틀려도 다른 맵에 기울기가 새어 나갑니다. 금속에서는 roughness가 baseColor보다 큰 기울기를 받습니다. 맵 공간에서는 0입니다.
- 이 값들은 텍셀 하나의 기울기입니다. 생성기를 거친 학습 결과를 예측하지는 않습니다.

## 9. 다시 시작하기 전에 정할 것

1. **팔 구성.** 현재 후보는 다음입니다.
   - 2×2: {render, raw} × {l21, l2sq}, 즉 R1, render·l2sq, raw·l21, raw·l2sq
   - 여기에 R1b, 그리고 phys·l21과 render·l1 중 무엇을 넣을지
   - **raw·l2sq를 빼면** "ℓ2,1의 형태 효과"와 "강한 충실도 항 자체의 효과"를 가를 수 없습니다. raw·l21은 사실상 "가중치를 키우고 RGB를 묶은 pixel L1"이기 때문입니다.
2. **render·l1은 측정 후 결정.** 회색 오프셋에서는 l21과 구분되지 않습니다. R0a/R1의 baseColor 편차가 회색(밝기) 방향인지 색조 방향인지 먼저 재고 정합니다.
3. **가중치 규칙의 부작용.** §3.8 규칙을 l2sq 항에 적용하면 오프셋을 뺀 잔차가 작아 가중치가 크게 나올 수 있습니다. 그러면 큰 오프셋에서는 l2sq의 복원력($\propto w\Delta$)이 l21보다 강해질 수 있습니다. 학습 전에 오프셋별 복원력 곡선(D0 probe)으로 미리 보여야 합니다.
4. **R1의 가중치 자체가 원인일 가능성.** $w_r=10$에서 오프셋이 있는 상태의 render 기울기는 VGG의 7–15배입니다. 다만 이 값은 기존 체크포인트에서 잰 것이고, R1은 scratch에서 시작했습니다. 팔마다 학습 중 "원본 / 오프셋 제거" 기울기 비율을 기록해야 합니다.
5. **phys·l21의 해석 주의.**
   - z 그룹은 0.01배라 height 감독이 거의 없습니다(R1도 같음).
   - 선형 공간이라 어두운 색 오차를 약 30배 덜 봅니다(§5).
6. **pixel L1(0.01)은 모든 팔에 남아 있습니다.** 어떤 팔도 "L1형 항이 전혀 없는" 팔이 아닙니다.
7. **판정 규칙.** 보관한 계획을 따릅니다.
   - 주 지표: D1(baseColor 평균 오프셋 최댓값), D2(밝기 span)
   - 잡음: τ0 = |R0a−R0b|, τ1 = |R1−R1b|
   - 기존 값만으로도 D1은 검정력 전제를 이미 만족하지 못합니다(8.75 < 2τ0 = 9.8). 그래서 재질 단위 쌍비교(12개 중 9개 이상)를 공동 주 판정으로 둡니다.
   - 결과를 본 뒤 기준을 바꾸지 않습니다.
8. **비용.** 팔 하나가 약 2.8–3.1 GPU-시간입니다. 2×2 + R1b면 약 12 GPU-시간에 평가가 추가됩니다.
9. **RaGAN 안의 기댓값 검토.** §10의 검토를 먼저 끝냅니다.

## 10. 검토 필요: RaGAN 안의 기댓값

R0a/R0b/R1을 설계할 때, DeepTextures의 기반 모델(ESRGAN)이 relativistic average GAN이라는 점, 즉 판별 기준에 상대 쪽 logit의 **기댓값**이 들어간다는 점을 render 항과의 상호작용 관점에서 검토하지 않았습니다. R1의 결과를 해석하기 전과, ablation을 설계하기 전에 아래를 확인해야 합니다.

**식.** 논문의 RaGAN은 기댓값을 씁니다.

$$
\tilde D(x_r)=\sigma\big(C(x_r)-\mathbb E_{x_f}[C(x_f)]\big),\qquad \tilde D(x_f)=\sigma\big(C(x_f)-\mathbb E_{x_r}[C(x_r)]\big)
$$

구현은 이 기댓값을 **한 배치의 평균**으로 대신합니다(`DeepTextures/modules/losses.py:59-62`, `:83-86`).

$$
\mathbb E_{x_f}[C(x_f)]\approx\bar C_f=\frac1B\sum_{i=1}^{B}C(\hat y_i),\qquad \mathbb E_{x_r}[C(x_r)]\approx\bar C_r=\frac1B\sum_{i=1}^{B}C(y_i)
$$

**검토할 점.**

1. **기댓값이 서로 다른 맵 종류를 섞습니다.**
   - R0/R1의 배치는 재질 2개 × 맵 5장 = 10장입니다. 그래서 $\bar C$는 baseColor, normal, roughness, metallic, height logit의 평균입니다.
   - 그 결과 가짜 normal 맵의 판별이 "진짜 맵 5종의 평균 logit"과 비교됩니다. 맵 종류별 logit 수준이 크게 다르면, 상대 판별 항이 이미지 품질보다 맵 종류를 구분하는 쪽으로 기울 수 있습니다.
   - 원본 DeepTextures(`matsynth_all`)도 무작위 8장이라 섞이는 건 같지만, 섞이는 구성이 다릅니다. grouped 배치는 매 step 다섯 종류가 정확히 2장씩 들어갑니다.
2. **기댓값 추정이 작은 배치에 의존합니다.** $B=10$의 평균이고, 그중 독립 재질은 2개뿐입니다.
3. **D 손실의 $\bar C$에 stop-gradient가 없습니다.** D 기울기가 상대 배치 평균을 통해서도 흐릅니다. BasicSR의 ESRGAN은 여기서 detach합니다(LOSS-SPEC §2.2).
4. **G 손실에서는 진짜 이미지 항으로도 기울기가 흐릅니다.** $\mathrm{BCE}(0,\sigma(C(y)-\bar C_f))$에서 $\bar C_f$가 모든 가짜 이미지에 의존하므로, G는 배치 전체 가짜 logit의 평균을 통해서도 기울기를 받습니다. 이 경로가 전역 색 오프셋을 키우거나 줄이는지는 측정하지 않았습니다.
5. **실험 2와의 비교에 영향을 줍니다.** BasicSR ESRGAN 파인튜닝은 detach가 있는 RaGAN과 무작위 패치 배치(기본 16장)를 씁니다. 그래서 `esrgan_ft`와 DeepTextures `matsynth_all`의 차이에는 이 GAN 구현 차이도 섞입니다.

**그대로 유효한 것.** R0a, R0b, R1과 이 문서의 모든 ablation 팔은 같은 RaGAN 구현과 같은 배치 구성을 씁니다. 그래서 **팔 사이의 차이는 여전히 통제된 비교**입니다. 다만 결론은 "이 RaGAN 설정 아래에서"라는 조건부입니다.

**제안하는 확인 (기존 체크포인트, 각 수 분).**
- **맵 종류별 logit 분포.** R0a/R1 체크포인트의 D로 진짜/가짜 맵을 종류별로 넣고 $C(\cdot)$의 평균과 분산을 잽니다. 종류 간 차이가 크면 1번이 실제 문제입니다.
- **GAN 항이 색 오프셋을 미는 정도.** w_r 보정과 같은 방식으로 GAN 항의 기울기를 원래 출력과 오프셋 제거 출력에서 각각 재서 비교합니다(`render_calibrate.py`의 `render_raw`/`render_offrm`과 같은 구조).
- 두 결과에 따라 ablation에 "GAN 항 없이" 또는 "맵 종류별 $\bar C$" 같은 대조 팔이 필요한지 정합니다.

## 부록 A. 노름 비교 스크립트

`texsr/deeptex:tf2.10.1`에서 저장소를 `/workspace`로 마운트해 실행합니다.

```python
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "/workspace/scripts/deeptex")
import numpy as np, tensorflow as tf, render_tf as rt

rl = rt.RenderLoss({})
pos = tf.zeros([1, 1, 1, 2], tf.float32)

def stack(bc=0.5, rough=0.5, metal=0.0):
    maps = [np.full(3, bc), np.array([0.5, 0.5, 1.0]), np.full(3, rough), np.full(3, metal), np.full(3, 0.5)]
    return tf.constant(np.stack(maps)[None, :, None, None, :], tf.float32)

def render(sr5):
    return rt.render_display(rt.decode_maps(sr5, pos), rl.lights)  # (N, 3, L)

NORMS = {
    "l21":  lambda d: tf.reduce_mean(tf.sqrt(tf.maximum(tf.reduce_sum(d * d, 1), rt.L21_EPS))),
    "l2sq": lambda d: tf.reduce_mean(tf.reduce_sum(d * d, 1)),
    "l1":   lambda d: tf.reduce_mean(tf.reduce_sum(tf.abs(d), 1)),
}
for kind, kw in (("dielectric", {}), ("metal", {"metal": 1.0})):
    gt = stack(**kw)
    gd = tf.stop_gradient(render(gt))
    for lv in (1, 4, 16, 64):
        off = tf.constant([lv / 255.0, 0, 0, 0, 0], tf.float32)[None, :, None, None, None]
        for name, fn in NORMS.items():
            sr = tf.Variable(gt + off)
            with tf.GradientTape() as tape:
                l = fn(render(sr) - gd)
            g = tape.gradient(l, sr).numpy()[0, 0, 0, 0, :]
            print(kind, lv, name, float(l), np.linalg.norm(g))
```

## 부록 B. 공간 비교 스크립트

부록 A와 같은 `stack`, `pos`, `rl`을 씁니다.

```python
EPS = 1e-15
def vnorm(d):
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(d), axis=-1), EPS))

def decoded_l21(sr5, gt5):
    a, b = rt.decode_maps(sr5, pos), rt.decode_maps(gt5, pos)
    groups = [vnorm(a["albedo"] - b["albedo"]), vnorm(a["normal"] - b["normal"]),
              vnorm(a["alpha"] - b["alpha"]), vnorm(a["metallic"] - b["metallic"]),
              vnorm(a["pos"][:, 2:3] - b["pos"][:, 2:3])]
    return tf.reduce_mean(tf.stack(groups))

def encoded_l21(sr5, gt5):
    return tf.reduce_mean(vnorm(sr5 - gt5))

for kind, kw in (("dielectric", {}), ("metal", {"metal": 1.0})):
    gt = stack(**kw)
    gd, va = rl.target(gt, pos)
    for lv in (1, 8, 32):
        off = tf.constant([lv / 255.0, 0, 0, 0, 0], tf.float32)[None, :, None, None, None]
        for name, fn in (("render", lambda s: rl.loss(s, pos, gd, va)),
                         ("phys", lambda s: decoded_l21(s, gt)),
                         ("raw", lambda s: encoded_l21(s, gt))):
            sr = tf.Variable(gt)
            with tf.GradientTape() as tape:
                l = fn(sr + off)
            g = tape.gradient(l, sr).numpy()[0, :, 0, 0, :]
            print(kind, lv, name, float(l), np.linalg.norm(g[0]), g[2].sum(), np.linalg.norm(g[1]), g[3].sum())
```

## 참고문헌

- Gauthier et al., *MatUp: Repurposing Image Upsamplers for SVBRDFs*, CGF 43(4), EGSR 2024. 식 (1).
- Aittala, Aila, Lehtinen, *Reflectance Modeling by Neural Texture Synthesis*, ACM TOG 35(4), 2016. in-network renderer.
- Deschaintre et al., *Single-Image SVBRDF Capture with a Rendering-Aware Deep Network*, ACM TOG 37(4), 2018. 렌더링 손실.
- Kowalski, *Sparse regression using mixed norms*, ACHA 27(3), 2009. 혼합 노름.
- Cook & Torrance 1982; Trowbridge & Reitz 1975. 미세면 BRDF와 GGX 분포. MatUp 참고문헌에 있습니다.
- Walter et al. 2007 (GGX), Heitz 2014 (height-correlated Smith), Schlick 1994, Reinhard et al. 2002, Duff et al. 2017 (ONB). 코드에서 쓰지만 OpenAlex의 MatUp 참고문헌 목록에서는 찾지 못했습니다.
