# DeepTextures — 우리가 upstream과 다르게 하는 것 전부

`external/DeepTextures` @ **`e4ee562`** 기준. **서브모듈은 한 바이트도 고치지 않았다.**
수정분은 전부 상위 저장소의 새 파일로 들고, `modules/*`를 바깥에서 import해 구동한다 —
`scripts/matup/matup_render.py:14`가 `matup_egsr`에 대해 이미 세운 규약과 같다.

> 경로 주의: `scripts/`·`experiments/`·`configs/` 아래 파일을 줄 번호(또는 줄 수)와 함께 인용한 곳은
> 태그 `variants-verified`의 옛 코드를 가리킨다(`git show variants-verified:<경로>`). 그 코드는 이제
> `variants/` 아래에 있고, 옛 파일 → 새 폴더 대응은 `variants/README.md`에 있다.

왜 in-place 패치가 아닌가:

- `docs/RUNBOOK.md` §4의 줄 번호가 이 커밋에 묶여 있다. 패치하면 **에러 없이 조용히** 틀어진다.
- `git submodule status`가 "이게 upstream인가?"에 한 명령으로 답하는 성질이 유지된다.
- **"논문 저장소 그대로"가 계속 실행 가능하다.** `OPEN-ITEMS.md` #6·#7이 요구하는
  "충실한 재현" vs "upstream 결함"의 분리가 이것 없이는 불가능하다.

대가는 중복이다. 실측(2026-09-17, 주석·공백 제외):

| | upstream 줄 | 우리 줄 | 글자 그대로 남은 줄 |
|---|---|---|---|
| `train_esrgan.py` → `deeptex_train.py` | 121 | 450 | **24** (upstream의 20%, 우리 파일의 5%) |
| `modules/losses.py` → `deeptex_losses.py` | 75 | 52 | **17** (upstream의 23%, 우리 파일의 33%) |

트레이너는 **복제가 아니라 같은 구조를 따라 다시 쓴 것**이다 — 늘어난 330줄은 시작 단언·검증
루프·`[metrics]` 로깅·NaN 검사·복원 증명처럼 원본에 대응물이 없다. 남은 24줄은
`optimizer_G = tf.keras.optimizers.Adam(...)` 처럼 달리 쓸 수 없는 줄들이다.
손실 쪽은 반대로 **복제에 가깝다** — RaGAN 두 함수의 구조를 그대로 두고 `from_logits`만 뒤집었다.

완화책 둘: (a) 트레이너를 `train_esrgan.py`와 줄 단위로 평행하게 써서 `diff`가 읽히게 했고,
(b) 시작 시 `git rev-parse HEAD`가 `e4ee562`가 아니면 크게 경고한다.

⚠️ 그래도 남는 위험: 서브모듈 포인터가 올라가면 이 중복본은 **조용히 낡는다.** 경고는 뜨지만
동작은 계속한다.

> ⚠️ 서브모듈 포인터를 올리면 **이 문서의 모든 줄 번호가 조용히 무효**가 된다. 먼저 여기를 갱신할 것.

---

## 0. 논문 대조 (2026-09-16 실측)

논문: Alonso, K. & Patow, G. *Deep learning for high-resolution material texture enhancement in
3D environments.* **Sci. Rep. 16, 12532 (2026).** doi:10.1038/s41598-026-42313-5

**이 서브모듈이 논문의 공식 코드가 맞다.** 논문 참고문헌 1번이
`https://github.com/NethamGH/DeepTextures` (Version 1.0)이고 서브모듈 태그가 `v1.0`이다.
추가 정황: `test.py:37-42`의 주석 처리된 체크포인트 목록
`esrgan_1k/5k/10k/20k/40k/80k`가 논문 Table 2의 **여섯 열과 정확히 일치한다.**
`:43-44`에는 논문에 없는 `esrgan_lstm`·`esrgan_transformer`도 있다 — 발표되지 않은 변형이다.

⚠️ **§0.7을 먼저 읽을 것.** 이 저장소는 `peteryuX/esrgan-tf2`의 포크이고, 핵심 모듈은
바이트 동일하다. 그 사실이 아래 전부의 전제다.

### 0.1 논문 서술이 자기 코드와 어긋나는 곳

우리 잘못이 아니지만, 재현 보고서에 반드시 적어야 한다.

| 논문 주장 | 실제 코드 (실측) |
|---|---|
| 생성기 "about 16.7 million" 파라미터 | **16,697,987** ✅ 일치 |
| "Residual scaling is also employed" | `res_beta=0.2` (`models.py:37,59,76`) ✅ 일치 |
| "two **PixelShuffle**-based upsampling stages" | `tf.image.resize(method='nearest')` + conv (`models.py:102-107`). PixelShuffle 아님 ❌ |
| 판별기 "around **2.6 million**, 8 conv + 2 dense" | **14,499,401**, conv 10개 ❌ (5.6배) |
| 전체 "**19.3 million** trainable parameters" | **31,197,388** ❌ |
| Fig 2: "Multi-step LR scheduling **is applied** to both optimizers" | `niter 40000` + `lr_steps [50000,...]` → **한 번도 안 걸린다**. 논문의 80k 실행에서만 5만에서 1회 발동 ❌ |
| 증강: 랜덤 크롭 + **스케일** + 회전 + **밝기** + **노이즈** | `dataset.py:58-83`는 flip + rot90뿐. 스케일·밝기·노이즈 없음 ❌ |
| 주 지표: **ΔE + StdDev + NVIDIA FLIP** | `test.py`는 PSNR/SSIM만. **ΔE·FLIP 코드가 저장소에 없다** ❌ |

### 0.2 논문은 PSNR 사전학습을 하지 않는다

논문 전체에서 "PSNR"은 **한 번만** 나오고, Fig 3의 평가 지표로서다. 2단계 학습·warm-up·
사전학습 서술이 전혀 없다. 즉 `pretrain_name: './checkpoints/'`가 조용히 실패해
**랜덤 초기화로 학습된 것이 논문의 실제 모델**이다.

→ OPEN-ITEMS #6이 "재현 품질 최대의 불확실성"이라 한 항목의 답: **논문은 안 했다.**
→ 따라서 우리의 사전학습 복원은 *논문 복원이 아니라 논문으로부터의 이탈*이다.
   그래서 arm을 둘로 나눈다 (§0.4).

### 0.3 논문의 학습 설정 vs 우리

| | 논문 | 저장소 config | 우리 |
|---|---|---|---|
| GPU | RTX 2060 6 GB | — | RTX 3080 Ti 12 GB |
| batch size | **4** (메모리 제약) | 8 | 8 |
| 스텝 | 20k 기본, 40k·80k 추가 | `niter 40000` | 40k |
| LR/HR 크기 | 128² / 512² | `input_size 32` / `gt_size 128` 크롭 | 동일 |
| **학습 이미지** | **1,050장** (210 텍스처 × 5채널) | — | **269장** (아래 참조) |
| 분할 | 텍스처 단위 210/45/45 | — | 원본 이미지 단위 90%/10% |

**데이터셋이 재현 불가능하다.** 논문은 300 텍스처 × 5채널 = **1,500장**을 쓰는데,
저장소의 `data/textures_db`에는 **300개 파일**뿐이다 — 사실상 color 채널 한 벌이다.
나머지 1,200장(displacement·metalness·normals·roughness)은 배포된 적이 없다.
그래서 우리 모델은 논문의 1,050장이 아니라 269장(홀드아웃 후)으로 학습된다.

→ **§0.6**에 2026-09-19 추가 확인(Zenodo 아카이브, PNG/JPG 구성, 실제 학습 입력의
부재, 공개 300장의 맵 종류 분포)이 있다.

#### 모델은 하나다 — 모든 맵을 같은 생성기에 섞어 넣는다 (2026-10-01 원문 확인)

"한 모델로 다섯 맵을 학습했다"는 문장이 논문에 그대로 있지는 않다. 아래가 모두 그 방향을 가리킨다.

| 근거 | 내용 |
|---|---|
| 학습 세트 | Dataset 절: 300 텍스처 × 5채널, 텍스처 단위 210/45/45. Training 절: 학습 **1,050장** = 210 × 5. 맵별로 나눴다면 210장씩 다섯 세트로 적었을 것이다 |
| 단수 표현 | 다섯 채널이 "our model"이 재질 표현을 배우게 했다는 서술, Method 절의 "the model to simultaneously process" |
| 체크포인트 | Table 2는 1k–80k 체크포인트 한 줄. `test.py:37-42`의 목록도 `esrgan_1k…80k`뿐이고 맵 이름이 없다 |
| **Fig. 6** | Albedo / Normal / Roughness / Metalness 패치가 각각 **같은 RRDB Generator 하나**로 들어가고, Batch Queue가 여러 종류 패치를 그 생성기로 돌려보낸다 → 한 batch에 맵이 섞인다 |

**쌓아 넣는 것(15채널)은 아니다.** Method 절이 생성기 첫 conv를 3→64로 명시하고, config는
`ch_size: 3`이다. 각 맵을 별도의 3채널 이미지로 같은 모델에 넣는다. §0.9의 MatSynth arm이 이 구조다.

본문과 그림이 어긋나는 곳:

| | 내용 |
|---|---|
| 맵 개수 | 본문은 5개(color, displacement, metalness, normals, roughness), **Fig. 6은 4개**(displacement 없음) |
| 스텝 산술 | 학습 1,050장 ÷ batch 4 ≈ 263 스텝/epoch인데 본문은 **400**이라고 쓴다(= 1,600 ÷ 4). "50 epoch = 20,000 스텝"도 400 기준이다 |
| Fig. 6의 성격 | 개념도다. 오타("Preceptual", "authenticicty", "Metainess")가 있고, 판별자 → 데이터셋 → 학습으로 이어지는 화살표처럼 실제 흐름과 다른 부분이 있다. 세부 구현은 본문과 코드를 우선한다 |

### 0.4 실험은 paper arm 하나로 간다 (결정, 2026-09-18)

**돌리는 것은 `external/DeepTextures/train_esrgan.py` 원본이다.** 우리 트레이너를
upstream처럼 설정한 것이 아니라, 논문이 배포한 코드가 그대로 실행된다.

| | |
|---|---|
| 실행 코드 | `external/DeepTextures/train_esrgan.py` (서브모듈, 무수정) |
| 래퍼 | `variants/deeptex/upstream_runner/run_upstream_train.py` — stdout만 재포맷 |
| config | `variants/deeptex/paper/esrgan_paper.yaml` — 서브모듈 config에서 **정확히 한 줄**(tfrecord 경로)만 다름 |
| 작업 디렉터리 | `runs/deeptex/paper/` — upstream의 `./checkpoints`·`./logs`가 서브모듈 밖으로 떨어지도록 |
| 소요 | 약 2.6시간 (40k 스텝, **실측 4.30 st/s** — §2.6. 200스텝 프로브의 4.75는 낙관적이었다) |

**래퍼가 학습을 바꾸지 않는 이유:** upstream 진행바(`modules/utils.py:113-118`)에 이미
step·loss_G·loss_D·lr·step/sec가 다 있고 `\r`로 뭉쳐 있을 뿐이다. 래퍼는 그 스트림을 쪼개
`[metrics]` 줄로 다시 낼 뿐, 학습 프로세스의 가중치·그래디언트·옵티마이저·RNG를 건드리지 않는다.
(upstream은 `tf.random.set_seed`를 부르지 않아 애초에 실행마다 비결정적이다.)

#### 왜 "고친 arm"을 돌리지 않는가

`scripts/deeptex/deeptex_train.py`와 `deeptex_losses.py`(합 502줄, 지금은 `variants/deeptex/fixed/`)는 결함 4개
(사전학습·shuffle·손실 수치안정성·LR 스케줄)를 고친 판이고, 동작하며 테스트도 돼 있다.
**그래도 이번 비교에서는 쓰지 않는다.**

- 이 프로젝트의 질문은 "MatUp vs DeepTextures"다. 상대는 **논문이 배포한 것**이어야 한다.
- 고친 arm은 **네 변수가 동시에 다르다.** 둘을 비교해도 어느 수정이 얼마나 기여했는지 못 나눈다.
  진짜 ablation을 하려면 세 번째 실행(고친 config + 사전학습 없음)이 더 필요하다.
- "이 논문 코드에 결함이 있고 고치면 이만큼 달라진다"를 **따로** 주장하고 싶을 때만 값이 생긴다.

두 파일은 지운 게 아니라 남겨 뒀다. `variants/deeptex/fixed/run.ps1 -Stage pretrain` / `-Stage train`으로 여전히 돌릴 수 있고,
`-Stage all`의 기본 흐름에서만 빠졌다.

#### 성공 판정

로그 grep이 아니라 **상태 단언**이다. `[metrics]`가 niter에 도달하고, **그와 별개로**
체크포인트의 `step` 텐서가 niter 이상이어야 한다. 네 경로를 실측 검증했다(2026-09-18):

| 경로 | 결과 |
|---|---|
| 신규 학습 20스텝 | `[ckpt] ... step=20` → 통과 |
| 재개했더니 이미 완료 | `already complete: resumed at step 20` → 통과 |
| 체크포인트 없음 | `[ckpt] MISSING` → exit 1 |
| step 부족 | `step=20 (need >= 40000)` → exit 2 |

⚠️ 여기서 실제 버그를 하나 잡았다. 처음 작성한 검증은 PowerShell 배열 리터럴 안에서 문자열
연결이 되지 않아 `python -c "import tensorflow as tf,sys; "`만 실행됐고, **import만 하고 exit 0**
이라 조용히 통과하고 있었다. 지금은 `-c` 페이로드를 변수 하나로 만들고, 출력에 `[ckpt]` 줄이
없으면 그것도 실패로 처리한다.

### 0.5 비교 범위 (결정 2026-09-18, 근거 개정 2026-09-19)

> **개정 이유.** 이 절은 원래 네 맵을 뺀 근거를 "논문이 정하지 않은 것 셋"으로 적었다.
> 논문 전문을 다시 읽고 **그중 둘이 틀렸음**을 확인했다. 아래가 정정판이고, 무엇을
> 어떻게 틀렸는지는 §0.6에 남긴다.

논문은 5채널 전부를 다룬다 — 초록, Method(L279 "multi-channel material upscaling,
allowing the model to simultaneously process and enhance various material maps"),
Results("not only applies to color images but also extends to displacement,
metalness, normals, and roughness maps"), 그리고 Fig. 7이 네 맵의 개선 결과다.
**맵별 비교는 방법의 의도 안에 있다.**

**맵마다 장애물이 다르다. 하나로 뭉뚱그리면 안 된다.**

| 맵 | GT | 3→1 축약 | 실제 장애물 |
|---|---|---|---|
| baseColor | 3ch | 불필요 | 없음 |
| render | 3ch | 불필요 | 없음 — 두 방법이 동일한 LR 바이트에서 출발 |
| **normal** | **3ch** | **불필요** | 학습 분포(§0.6). 단위 길이 미보장은 **논문이 신경 쓰지 않는** 문제다 — 논문은 normal map을 이미지로 취급하고 이미지 오차를 잰다 |
| roughness | 1ch | **필요** | 저장할 *이미지*를 1채널로 만드는 규정이 저장소에 없다 (아래) |
| metallic | 1ch | **필요** | 위와 동일 |
| height | 16-bit 1ch | 필요 | **채널과 무관한 이유** — MatUp이 height를 출력하지 않는다(우리 것은 `matup_height.py`가 복원). uint8 통과로 8비트도 잃는다 |

축약이 실제로 걸리는 열은 **둘(roughness·metallic)이지 셋이 아니다.** normal을 여기
넣은 것이 원래 서술의 오류였다.

#### 저장소에는 축약이 있다 — 단 지표 경로에만

`test.py:91-94`가 PSNR/SSIM 양쪽 인자에 `rgb2ycbcr()`를 씌우는데, `modules/utils.py:340`의
기본값이 `only_y=True`라 결과가 **(H,W) 1채널**이다. 게다가 `cv2.imread`가 준 BGR을 먹이고
`:351`이 채널을 한 번 더 뒤집으므로 **실제 적용되는 가중치는 R과 B가 뒤바뀐 BT.601**이다.

→ 그래서 roughness·metallic 열을 **저장소 충실(repo-faithful)**하게 만들 수 있다.
`test.py` 자신의 경로를 그대로 쓰면 그 가중치는 "우리가 발명한 것"이 아니라
**"이 저장소가 실제로 쓰는 것"**이 된다. R/B 뒤바뀜까지 포함해서 그대로 쓰고, 그렇게
기술한다.

**논문 충실(paper-faithful)은 불가능하다.** 논문이 보고한 ΔE·StdDev·FLIP은 저장소에
코드가 없다(§0.1, §0.6). 그러니 어떤 선택을 하든 "논문과 같은 방식"이라고 주장할 수 없다.

#### 이 결정의 가장 강한 근거는 위의 어느 것도 아니다

**§0.6을 볼 것.** 공개된 학습 데이터가 논문의 학습 데이터가 아니다.

#### 몽타주는 분리한다 (결정, 2026-09-19)

DeepTex는 6열 몽타주의 다섯째 **행**이 아니라 **별도 시트**로 나간다.

| 파일 | 격자 | 내용 |
|---|---|---|
| `montage.png` | 4×6 | LR / bilinear / MatUp / GT × 다섯 맵 + 렌더 — **커밋된 그대로** |
| `montage_deeptex.png` | 5×2 | LR / bilinear / MatUp / **DeepTex** / GT × baseColor·render |

행으로 두면 여섯 칸 중 네 칸이 placeholder였고, 그 placeholder가 **공개 산출물이
지원하지 못하는 비교를 광고**했다. 결정적 근거: 서브모듈 전체에 **맵 종류를 아는 코드가
0줄**이다(config 하나, `sub_name` 하나, `test.py`는 `--img_path`뿐, 체크포인트 목록에
맵 이름 없음). 게다가 공개된 300장이 컬러 위주라 체크포인트는 **컬러 모델**이다(§0.6).

분리로 얻은 것: `runs/montages/*.png` 12개가 커밋 `2582c86`과 **바이트 동일**로 되돌아왔고
(검증 완료), DeepTex는 **모든 칸이 찬** 시트를 갖는다. MatUp 측정값도 42개 전부 불변이다.

#### normal·roughness·metallic 확장은 보류 (2026-09-19) → §0.9에서 기준선으로 실행 (2026-10-01)

위 근거 때문에 확장해도 재현되는 것은 논문 Fig. 7이 아니라 "**컬러로 학습된 모델이
노멀맵에 하는 일**"이다. 할 가치는 있지만(공개물로 논문의 멀티채널 주장을 검증하는 것이
되므로) 지금은 미룬다.

> **2026-10-01 갱신.** MatSynth arm(§0.9)의 비교 기준선으로 **같은 체크포인트**
> (`runs/deeptex/paper/checkpoints/esrgan_40k/ckpt-8`, 재학습 없음)를 다섯 맵 + render 전부에
> 돌렸다. `variants/deeptex/matsynth/run.ps1 -Stage eval-arms` → `runs/deeptex/eval/paper_allmaps`.
>
> - 그 결과 열이 보여 주는 것은 위 문장 그대로 "컬러로 학습된 모델이 학습하지 않은 맵에 하는 일"이다.
>   논문 Fig. 7의 재현이 아니다.
> - 1채널 맵의 출력 합치기는 `paper`가 아니라 **`mean`**이다. MatSynth arm과 같은 기준으로 비교하려는
>   선택이다. 이 체크포인트의 세 출력 채널 차이는 평균 5~10/255라서, `paper` 가중치를 쓰면
>   roughness·metallic·height 수치가 조금 달라진다.
> - baseColor·render도 다시 추론한 값이라, 커밋된 `eval/paper`와 최대 1 LSB 다르다(§0.9의 GPU 재현성).
> - 기존 `runs/montages/*_deeptex.png`(baseColor·render만)는 그대로 둔다.
>   전체 맵은 `runs/montages/deeptex_matsynth/`에 있다.

#### 기계 장치

`deeptex_compare.py --maps all`로 6열을 켜고 `--scalar-collapse {mean,luma,paper}`로
가중치를 고를 수 있다(`paper`가 위의 R/B 뒤바뀐 판이다). 맵별 `channel_spread`
— 세 출력 채널이 실제로 얼마나 벌어졌는지 — 가 `sr_manifest.json`에 기록된다.

---

### 0.6 공개된 산출물은 논문의 산출물이 아니다 (2026-09-19 확인)

| | 논문 | 공개된 것 |
|---|---|---|
| 학습 데이터 | 300 텍스처 × 5채널 = **1,500장**. 그중 **210 텍스처 / 1,050장 학습**, 45/225 검증, 45/225 테스트 (L385-389) | `data/textures_db/` **300장** (200 PNG + **100 JPG**), `001`~`300`, **맵 종류 라벨 없음** |
| 실제 학습 입력 | — | **없다.** `configs/esrgan.yaml:17`이 가리키는 `textures_db_normalized/{HR,LRx4}`와 tfrecord가 저장소에 없다. `textures_db`는 원본 풀일 뿐이다 |
| 주 지표 코드 | ΔE + StdDev + NVIDIA FLIP | **없다.** 서브모듈 전체에서 `flip`은 `dataset.py:59`의 증강 플래그뿐 |
| Zenodo | "All data and source code used in this study are openly available on GitHub and archived on Zenodo" (`10.5281/zenodo.17495609`) | 열어 보니 **`NethamGH/DeepTextures: v1.0`, 587.3 MB zip, "Is supplement to github.com/NethamGH/DeepTextures/tree/v1.0"** — GitHub v1.0의 스냅샷 그 자체. 서브모듈이 바로 그 태그이고 562 MB다 |

**논문의 1,500장은 어느 쪽에도 공개되지 않았다.**

공개된 300장의 구성을 재어 보니(정규화본 299장, 채널 통계 휴리스틱)
**컬러 262 / 준회색조 26 / 노멀맵류 11**이다.

→ **`esrgan_40k` 체크포인트는 논문의 모델이 아니다.** 논문 모델은 학습 중에 노멀맵·러프니스·
displacement를 봤고, 우리 것은 거의 못 봤다. 다른 맵에서 결과가 나쁘면 그것은
**방법의 한계가 아니라 공개 데이터의 한계**이며 반드시 그렇게 기술해야 한다.

⚠️ 이건 고칠 항목이 아니라 **기술할 한계**다. 논문의 데이터셋이 존재하지 않으므로
재학습으로 해소할 수 없다.

→ 코드 쪽의 같은 성격 발견은 **§0.7**(저장소가 donor 포크라는 것)과
**§0.8**(논문 고유 기여인 타일링이 우리 해상도에서 미발동)에 있다.

#### 원래 서술이 어떻게 틀렸었나 (기록)

| 원래 근거 | 판정 |
|---|---|
| "회색조를 1채널로 되돌리는 방법이 논문에 없다" | **부분적으로 맞음.** 논문 본문은 실제로 논하지 않는다(`grayscale`·`luma`·`single channel` 전문 검색 0건). 그러나 **저장소에는 있다** — 지표 경로에 한해 `rgb2ycbcr(only_y=True)`. "규정이 아예 없다"는 과장이었다 |
| "super-resolve된 normal이 단위 길이라는 보장이 없다" | **근거로는 무효.** 논문은 normal map을 이미지로 취급한다. 물리적 타당성은 우리 관심사이지 재현 장애물이 아니다. normal은 3ch in / 3ch out으로 **가장 쉬운 케이스**다 |
| "height는 8비트를 잃고 MatUp이 출력하지 않는다" | **유효.** 유일하게 그대로 살아남은 근거다 |

---

### 0.7 이 저장소는 `peteryuX/esrgan-tf2` 포크다 (2026-09-19 확인)

우리가 donor로 핀해 둔 `external/esrgan-tf2` (`github.com/peteryuX/esrgan-tf2` @ `90e8eec`)와
파일 단위로 diff한 결과다. **이것이 §1 대조표를 읽는 전제여야 한다.**

| 파일 | 변경 | |
|---|---|---|
| `modules/models.py` | **0줄** | 생성기·판별기 전부 donor 것 |
| `modules/losses.py` | **0줄** | 손실 전부 donor 것 |
| `modules/dataset.py` | **0줄** | 데이터 파이프라인 전부 donor 것 |
| `modules/lr_scheduler.py` | **0줄** | |
| `train_esrgan.py` | **1줄** | 아래 |
| `data/convert_train_tfrecord.py` | 기본 경로만 | DIV2K → `textures_db_normalized` |
| `modules/utils.py` | **함수 2개 추가** | `split_large_image` / `join_large_image` (§1.2) |
| `test.py` | 110줄 | 그 타일링을 쓰는 추론 스크립트 |

트레이너에서 바뀐 **딱 한 줄**:

```diff
-            pretrain_dir = './checkpoints/' + cfg['pretrain_name']
+            pretrain_dir = cfg['pretrain_name']
```

donor는 맞게 돼 있었다. **DeepTextures가 트레이너에 가한 유일한 변경이 사전학습 로딩을
깨뜨린 그 줄이다**(결함 #2). config도 같은 성격이다:

| | donor | DeepTextures |
|---|---|---|
| `niter` | 400000 | **40000** |
| `lr_steps` | `[50000,100000,200000,300000]` | **그대로** → 감쇠 영구 미발동 (결함 #3) |
| `pretrain_name` | `'psnr_pretrain'` (위 `+` 와 짝) | `'./checkpoints/'` |
| `batch_size` | 16 | 8 |
| `test_dataset` (Set5/Set14) | 있음 | **삭제** ← 평가 루프가 없는 이유 |
| `network_D: nf` | 있음 | 삭제 → Python 기본값 64로 폴백 |
| `num_samples` | 32208 | 삭제 |

`niter`를 1/10로 줄이면서 `lr_steps`를 같이 줄이지 않은 것이 **결함 #3의 정확한 출처**다.

donor에 있고 포크에서 삭제된 것: `train_psnr.py`, `configs/psnr.yaml`,
`data/extract_subimages.py`, `dataset_checker.py`, `net_interp.py`, `environment.yml`,
notebooks·photo 일체. 포크의 흔적은 README 13줄 `conda activate esrgan-tf2` 하나뿐이고,
donor를 명시적으로 크레딧하는 문장은 없다.

**논문과의 대조.** 논문 Method의 "Enhanced model" 절은 아키텍처 개선을 서술하지만
`models.py`는 donor와 **바이트 동일**하다. 논문이 주장하는 "custom modifications" 중
공개 코드에서 확인되는 것은 **타일링뿐**이다.

---

### 0.8 우리 비교가 실제로 무엇인지 (2026-09-19)

`§1.2`가 보이듯 타일링은 `H > 512 or W > 512`일 때만 동작한다. 우리 평가 입력은 128²이므로
`split_large_image`는 `≤512` 분기로 빠져 **원본 경로를 그대로 돌려준다** — 타일링은
한 번도 실행되지 않는다. 우리는 `deeptex_sr.py`로 자체 추론하고 `--tile-size 0`이 기본이라
어차피 경유하지도 않는다.

→ 따라서 **우리가 MatUp과 비교하는 것은 엄밀히 말하면
`peteryuX/esrgan-tf2`의 ESRGAN을 `textures_db`로 학습시킨 모델**이다.
논문 고유 요소(타일링)는 이 해상도에서 작동 범위 밖이라 개입하지 않는다.

이것이 실험을 무효화하지는 않는다 — 논문이 배포한 코드를 배포된 config로 돌린 것이 맞다.
**다만 보고서에 이 문장 그대로 적어야 한다.**

---

### 0.9 MatSynth arm — 논문의 멀티채널 학습을 공개 데이터로 재현 (2026-10-01)

§0.6의 한계(논문의 1,500장이 공개되지 않아 체크포인트가 컬러 모델)는 **재학습으로 해소할 수
없다**고 적었다. 논문 데이터로는 그렇다. 그러나 MatSynth에는 다섯 맵이 전부 있으므로, **논문이
한 방식 — 3→3 네트워크 하나에 다섯 채널 종류를 섞어 학습 — 을 다른 공개 데이터로** 할 수 있다.
이 절은 2026-09-16의 "학습은 textures_db만" 결정을 뒤집는다. textures_db arm은 지우지 않고
기준선으로 남긴다.

| | textures_db arm (§0.4) | MatSynth arm |
|---|---|---|
| 실행 코드 | `train_esrgan.py` 무수정 | **동일** |
| config | `esrgan_paper.yaml` | `esrgan_matsynth.yaml` — upstream과 tfrecord 경로 **한 줄만** 다름 |
| 작업 디렉터리 | `runs/deeptex/paper/` | `runs/deeptex/matsynth_all/` (`-PaperRoot`). 둘 다 `checkpoints/esrgan_40k` |
| 학습 데이터 | 300장 → 2,965 타일 (컬러 262 / 준회색 26 / 노멀류 11) | MatSynth **train split** 2,904 소재 × 5맵 = 14,520쌍 (2,905개 중 1개 제외, 3.5 GB) |
| 쌍 | 480² 타일, LR = bicubic `imresize_np` | 소재 전체를 512²(HR)와 128²(LR)로, **원본에서 각각** `INTER_AREA` |
| 평가 입력과의 관계 | 열화 커널이 다름 (bicubic vs area) | **픽셀 동일** — 아래 게이트 |

**열화를 평가와 같게 만든 방법.** `variants/deeptex/matsynth/matsynth_make_pairs.py`는
`matsynth_extract.py`(평가 12개를 만든 스크립트)의 함수를 **import해서** 같은 순서로 부르고,
`variants/util/map_encode.to_rgb8`로 8-bit RGB를 만든다. `deeptex_compare.load_as_rgb01`도 같은
함수를 부르도록 바꿨다(132개 입력에서 리팩터 전후 배열 동일 확인). `--check-eval`이 test-00000에서
평가 12개를 다시 만들어 `runs/matsynth`와 **120개 파일 전부 픽셀 동일**임을 확인한다 — 학습 전 게이트.

**누수 차단.**
- test split은 읽지 않는다. 평가 12개는 전부 `test-00000`에 있다.
- MatSynth 블렌드 중 평가 재질로 만든 것 제외: `ms_concrete_026__moss_001` (`acg_concrete_026`). 로컬 train shard 216개에서 유일한 히트.
- 남긴 것(이름만 겹침, 출처가 다름): `tc_concrete_026`, `tc_plastic_009` (TextureCan). 평가 쪽은 ambientCG다.
- 내려받지 않은 train shard 215개(LFS 스텁)는 확인하지 않았다 — 쓰지도 않는다.

**하나의 레코드, 한 번의 셔플.** upstream은 `shuffle=False`(`train_esrgan.py:37`)라 레코드 순서가 곧
커리큘럼이다. 맵별 폴더 다섯 개를 `texsr_build_tfrecord.py --hr ... --hr ...`로 묶어 **seed 0으로
한 번** 섞는다. 이어 붙이면 한 번에 한 맵만 학습하는 블록이 생긴다.

**보고서에 적을 한계 (고치지 않는다 — paper arm의 원칙):**

| 한계 | 왜 |
|---|---|
| normal의 flip/rot90 증강이 벡터 성분을 보정하지 않는다 | upstream `dataset.py:59-81`. LR·HR이 같이 뒤집히므로 쌍의 대응은 유지되고, 입력 분포가 "뒤집힌 노멀"까지 넓어지는 것이다 |
| height 16 → 8 bit | upstream `decode_png`가 uint8만 읽는다. 평가 쪽도 같은 `rint(/257)` |
| 회색 맵은 3채널 복제 | 3채널 네트워크가 회색 맵을 보는 유일한 방법. 출력의 세 채널 차이는 `channel_spread`로 잰다 |
| 상수 맵을 거르지 않는다 (상수 맵: metallic 2,439 · roughness 122 · baseColor 47 · height 22 · normal 0, 2,904개 중. 합 2,630 = 전체 14,520쌍의 18%) | 논문은 "텍스처 × 5채널"을 큐레이션 없이 썼다 |
| 맵 라벨이 이상한 원본도 그대로 | 예: `0325_marble_wall_01`은 roughness가 상수 0이고 metallic이 텍스처다. MatSynth 인덱스 통계도 같다(roughness_mean 0, metallic_mean 0.48) — 우리 코드가 아니라 데이터다 |
| 소재 약 2,900개 vs 논문 210 | 로컬에 받은 train shard 절반(216/431)만 쓴다. 더 받을 이유가 없다 |
| 다섯 번째 맵: 논문 displacement, 우리 **height** | 평가 12개를 `matsynth_extract.py`가 height로 뽑았고, 학습과 평가가 같은 맵이어야 한다. MatSynth에는 둘 다 있다. Fig. 6은 이 맵 없이 4개다(§0.3) |
| 학습 데이터 구성 | baseColor 20%, normal 20%, 회색 맵(roughness·metallic·height) 60%. 컬러가 아닌 맵이 80%다 (Fig. 6의 4맵 구성이면 75%) |

**평가는 `-Stage eval-arms`로만 한다.** `-Stage eval`은 `matup_postprocess` + `matup_montage --collect`로
커밋된 `runs/montages/`를 다시 쓴다. `eval-arms`는 `runs/deeptex/eval/{paper_allmaps,matsynth_all}`와
`runs/deeptex/report_<tag>/`(본학습이면 `report_matsynth_all`)에만 쓴다. 리포트(`deeptex_arms_report.py`)는 `matup_postprocess`의 열 생성
코드를 import하므로 크롭·테두리·DirectX→OpenGL 뒤집기가 같다 — 기존 textures_db 출력으로 돌려
`runs/montages/*.json`의 PSNR/SSIM **240개 항목과 전부 일치**함을 확인했다.

**실측 (2026-10-01).**

| 항목 | 값 |
|---|---|
| 쌍 생성 | 216 shard, 966초, 워커 4개 |
| 재실행 | 이미 만든 재질은 건너뛰어 15초 |
| TFRecord | 14,520 레코드, 3.6 GB, verify PASS |
| verify의 alignment 경고 | `max \|lr - imresize_np(hr)\| = 11`. LR이 HR에서 유도된 게 아니라서 정상이다 |
| 학습 처리량 | 400스텝 프로브 4.2–4.5 st/s. textures_db arm의 4.30과 같아서, 512² PNG 디코딩은 병목이 아니다. 40k ≈ 2.6시간 |
| 판별자 포화 | `loss_D`는 100스텝 안에 0.0000이 된다 — §2.6의 결함 #8. 그대로 둔다 |

**본학습 (2026-10-01).** 40k 완료: `runs/deeptex/matsynth_all/checkpoints/esrgan_40k/ckpt-8`, step 40000,
4.5–4.6 st/s, 약 2.5시간, NaN 없음. 결과는 `runs/deeptex/report_matsynth_all/`(`summary.md`, `per_material.md`,
`metrics_long.csv`, `montages/`, `overview_<맵>.png`).

⚠️ **이 레시피는 색이 일정하게 밀린다 — 결과를 읽기 전에 알아야 한다.** 평탄한 normal 입력
(127.5, 127.5, 254)에 MatSynth 체크포인트는 약 (111, 121, 249)를, textures_db 체크포인트는 약
(122, 117, 225)를 낸다. 평가 파이프라인 문제가 아니다 — **자기 학습 쌍 40개**에서도 평균
(SR − HR)이 normal (−20, −7, −6), baseColor (−6, 0, −4) 레벨이다. 사전학습 없음(#2) + 픽셀 손실
가중치 0.01 + 포화된 판별자(#8) + 상수 이동에 둔감한 VGG 손실이 겹친 결과로 본다. 이 이동 하나로
평탄한 normal 맵에서 약 8° 각도 오차와 수십 dB의 PSNR 손실이 생긴다. 그래서 리포트에는
`offset`(채널별 평균 이동)과 `PSNR -off`(이동을 뺀 PSNR)를 **진단용**으로 함께 싣는다.
예: normal에서 MatSynth arm은 23.65 dB → 이동 제거 후 30.86 dB (bilinear 33.42).

**색이 튀는 것은 레시피, 초록 쏠림은 MatSynth arm에만 (2026-10-01 측정).**
두 arm의 체크포인트 30k / 35k / 40k를 각자 **자기 학습 쌍 60개**에 돌려 평균(SR − HR)을 쟀다
(MatSynth는 baseColor 쌍. 체크포인트마다 `deeptex_sr.super_resolve`로 추론하고, 0–1로 자른 뒤 채널별 평균을 비교).

| 체크포인트 | textures_db: 밝기 / G−(R+B)/2 | MatSynth baseColor: 밝기 / G−(R+B)/2 |
|---|---|---|
| ckpt-6 (30k) | +0.9 / +0.9 | −5.3 / +3.4 |
| ckpt-7 (35k) | **+5.4** / +0.6 (B +9.2) | +2.3 / +4.9 |
| ckpt-8 (40k) | +0.6 / +1.3 | −3.1 / **+6.1** |
| 밝기가 오르내린 폭 | 4.8 | 7.6 |

20k 이후 로그도 같은 그림이다.

| | textures_db | MatSynth |
|---|---|---|
| `loss_G` 평균 / 표준편차 | 0.75 / 0.13 | 0.72 / 0.18 |
| `loss_D`가 0.0000인 비율 | 37% | 48% |

- **체크포인트마다 색이 튀는 것은 원래 데이터에서도 일어난다** → 데이터가 아니라 레시피 탓이다.
- **MatSynth arm은 튀는 폭이 더 크고, 초록 쏠림이 점점 커진다.**
  - 평가 12개 기준 baseColor 이동은 R −7.0 / G +0.8 / B −3.5이고, 12개 중 10개에서 G가 가장 높다.
  - G가 밝아진 것이 아니라 R·B가 내려가서 초록빛으로 보인다.
  - 같은 모델을 자기 학습 쌍에 돌려도 같은 이동이 나오므로 파이프라인 문제가 아니다. 채널 순서·열화·인코딩도 별도로 확인했다.
- **쏠림의 원인은 미검증이다.**
  - 후보: 컬러가 아닌 맵 80%를 한 모델에 섞은 것, MatSynth 색 분포, 학습 운.
  - 학습이 arm마다 한 번뿐이고 upstream은 시드를 고정하지 않아서, 지금 자료로는 가를 수 없다.
  - 가르는 대조군은 **MatSynth baseColor만으로 학습한 모델**이다(실험리스트 2-2와 같은 질문). 섞은 탓으로 확인되면 재현 실패가 아니라 **논문 설계의 한계**로 보고한다. 논문이 바로 이 방식으로 학습했기 때문이다(§0.3).
- **남은 체크포인트는 30k·35k·40k뿐이다.** `train_esrgan.py:64`의 `max_to_keep=3` 때문에 5k–25k는 학습 중에 지워졌다. 전 구간을 보려면 저장될 때마다 복사해 두면서 다시 학습해야 한다.

> **2026-10-02 정정 (§0.10).** 위의 "초록 쏠림이 점점 커진다"는 체크포인트 3개만 보고 내린 판단이었다.
> 체크포인트 8개를 모두 남긴 두 ablation arm에서는 쏠림이 체크포인트마다 −12에서 +11까지 오간다.
> - +3.4 → +4.9 → +6.1은 그 흔들림의 한 구간이었을 가능성이 크다.
> - 쏠림은 데이터 구성의 고정된 성질이 아니라 레시피의 색 불안정이 끝 체크포인트에서 드러난 값으로 본다.

⚠️ **GPU 추론은 비트 단위로 재현되지 않는다.** 같은 체크포인트, 같은 코드로 `deeptex_compare.py`를 두 번
돌리면 출력이 **최대 1 LSB, 픽셀의 약 3.5%** 에서 다르다. cuDNN 알고리즘 선택 같은 부동소수 차이가 반올림
경계를 넘기 때문이다. 그래서 `eval/paper_allmaps`의 baseColor·render는 커밋된 `eval/paper`와 바이트가 다르고,
리포트의 `dt_textures_db` 수치도 `runs/montages/*.json`과 소수점 아래에서 어긋날 수 있다. 입력 인코딩은 배열
단위로 동일함을 따로 확인했다(§0.9 위).

⚠️ **TFRecord의 sha256은 재현성 검사가 아니다.** 같은 입력·같은 코드로 다시 만들어도 해시가 달라진다
(`tf.train.Example`의 feature가 protobuf map이라 직렬화 순서가 프로세스마다 다르다). `.meta.json`의
sha256은 **절단 검사**로만 쓴다. 빌더 수정 전후는 레코드 단위 내용·순서 비교로 확인했다(2,965개 동일).

---

### 0.10 채널 수 비례 ablation — 논문에서 일부러 벗어난다 (설계 2026-10-01)

§0.9의 초록 쏠림이 "맵을 섞는 것"에서 오는지 가르기 위한 두 arm이다. **논문 재현이 아니다.**
논문은 맵당 1:1로 섞었고(§0.3), `matsynth_all`이 그 비율이다.

| arm | 레코드 구성 | 레코드 | 맵 비율 |
|---|---|---|---|
| `matsynth_all` (§0.9, 논문 비율) | 5맵 × 1 | 14,520 | 각 20% |
| **`matsynth_chratio`** | baseColor × 3, normal × 3, roughness·metallic·height × 1 | 26,136 | baseColor 33.3%, normal 33.3%, 회색 맵 각 11.1% |
| **`matsynth_basecolor`** | baseColor × 1 | 2,904 | baseColor 100% |

- **chratio의 근거 (사용자 결정):** 맵의 비율을 **실제 채널 수에 비례**하게 맞춘다.
  - baseColor(RGB)와 normal(xyz)은 3채널이다.
  - 회색 맵은 1채널을 3개로 복제한 것이라 정보가 1채널뿐이다.
  - 1:1로 섞으면 학습 데이터의 60%가 1채널짜리가 된다.
- **basecolor는 대조군이다.** 맵을 섞지 않으면 쏠림이 사라지는지를 본다.
- **바뀌는 것은 데이터 구성 하나뿐이다.**
  - 학습 쌍은 같은 파일이고, 데이터를 빼지 않고 반복만 한다(`texsr_build_tfrecord.py --repeat`).
  - config는 upstream과 경로 한 줄만 다르고, 40k 스텝이다.
- **chratio를 읽을 때 주의:** 컬러(20 → 33%)와 normal(20 → 33%) 비중이 함께 오르고, 회색 맵이 60 → 33%로 준다.
- **체크포인트는 `-KeepAllCheckpoints`로 8개 모두 보존한다** (`<PaperRoot>/checkpoints_all/`).

**판단 기준 (학습 전에 정함).** 주 지표는 `deeptex_drift.py`의 baseColor 초록 쏠림 G−(R+B)/2와 채널별 이동이다.
자기 학습 쌍 60개(seed 7)와 평가 12개를 체크포인트마다 잰다.

| 결과 | 해석 |
|---|---|
| basecolor의 쏠림이 textures_db 수준(약 ±1.5 이내) | 맵을 섞는 것이 원인이다. 재현 실패가 아니라 **논문 설계의 한계**로 보고한다 |
| 위에 더해 chratio가 all과 basecolor 사이 | 회색 맵 비중이 쏠림을 키운다. 채널 비례 구성이 완화책이다 |
| basecolor도 +5 안팎 | 섞은 탓이 아니다. MatSynth 색 분포나 레시피 쪽이다 |

**보조 지표:**
- 체크포인트 사이 밝기 흔들림
- chratio에서 normal 이동(R −20)이 줄었는지
- 회색 맵 샘플이 20 → 11%로 줄어서 roughness·metallic·height가 얼마나 나빠지는지 (리포트의 PSNR / SSIM / |grad| %GT / offset)

arm마다 학습이 한 번이고 시드가 없으므로, 차이가 작으면 결론을 유보한다.

**출발점 (`deeptex_drift.py`, baseColor, 자기 학습 쌍 60개 / 평가 12개).**

| arm | ckpt-6 (30k) 쏠림 | ckpt-7 (35k) 쏠림 | ckpt-8 (40k) 쏠림 | 40k 이동 (학습 쌍, R/G/B) |
|---|---|---|---|---|
| textures_db (자기 쌍) | +0.91 / −1.22 | +0.62 / −0.94 | +1.29 / −0.08 | −0.09 / +1.44 / +0.39 |
| matsynth_all | +3.35 / +3.13 | +4.86 / +4.28 | +6.14 / +6.02 | −7.17 / +1.00 / −3.12 |

(칸은 학습 쌍 / 평가 12개 순서다.)

#### 결과 (2026-10-02)

**학습.** 두 arm 모두 40k를 정상 완료했다.

| arm | 로그 | 상태 |
|---|---|---|
| chratio | `runs/_logs/20261002-000417` | step 40000, `[ckpt]` 확인, NaN 없음 |
| basecolor | `runs/_logs/20261002-023233` | step 40000, `[ckpt]` 확인, NaN 없음 |

- `-KeepAllCheckpoints`로 **체크포인트 8개를 모두 보존**했다(`checkpoints_all/`). 16개 파일 모두 374,821,028 바이트로 온전하다.
  - **2026-10-02 삭제:** 측정이 끝난 뒤 디스크 정리로 `checkpoints_all/`과 MatSynth 학습 레코드 3개(12.5 GB)를 지웠다.
  - ckpt-6~8은 `checkpoints/`에 남아 있다.
  - 5k–25k의 결과는 `runs/montages/deeptex_checkpoints/`의 시트와 표로만 남는다.
  - 레코드는 학습 쌍 PNG가 남아 있어 `-Stage prep-matsynth -Recipe …`로 다시 만들 수 있다.
- 평가는 `-Stage eval-arms` 두 번이고, 네 arm 비교는 `runs/deeptex/report_matsynth_basecolor/`에 있다.
- 색 이동은 `runs/deeptex/drift/<arm>/drift.md`, 체크포인트별 시트(`0844_syntetic_foam`)는 같은 폴더에 있다.

**주 지표: baseColor 초록 쏠림 G−(R+B)/2, 자기 학습 쌍 60개.**

| 체크포인트 | 5k | 10k | 15k | 20k | 25k | 30k | 35k | 40k |
|---|---|---|---|---|---|---|---|---|
| chratio | −8.56 | −7.59 | +11.20 | +4.54 | +0.73 | −0.51 | −5.30 | +1.69 |
| basecolor | −6.85 | −11.84 | −1.52 | +7.58 | −3.58 | +0.58 | −3.14 | −0.04 |
| matsynth_all | | | | | | +3.35 | +4.86 | +6.14 |
| textures_db | | | | | | +0.91 | +0.62 | +1.29 |

- 40k의 평가 12개 기준 쏠림: all +6.02, chratio +0.86, basecolor +0.65, textures_db −0.08.
- 체크포인트마다 평균 이미지별 최대 이동은 4–24 레벨이다. 앞쪽 체크포인트일수록 크다.

**판정.**
- 기준 문구 그대로면 basecolor의 40k 쏠림(+0.65)이 ±1.5 안이라 "맵을 섞는 것이 원인"에 해당한다. **그렇게 결론짓지 않는다.** 이유는 셋이다.
  1. **맵을 섞은 chratio도 40k에서 +0.86이다.** 섞는 것 자체가 원인이라면 chratio에도 쏠림이 남았어야 한다.
  2. **8개 궤적에서 두 arm 모두 쏠림이 −12 ~ +11을 오간다.** 끝 체크포인트의 값은 어디서 멈췄느냐에 거의 달렸다.
  3. **matsynth_all의 +3.4 → +6.1은 3개 점이다.** 같은 크기의 흔들림 안에 들어간다(§0.9 정정).
- **기준의 결함:** 기준을 끝 체크포인트 하나로 정한 것이 부족했다. 궤적까지 보면, 쏠림은 데이터 구성의 성질이 아니라 **레시피의 색 불안정**이다.
- **보조 지표도 같다.** normal의 R 이동은 chratio 40k에서 −9.1로 all의 −20.3보다 작지만, chratio의 8개 체크포인트에서 −22.9 ~ +16.6을 오간다.
- **단서:** arm마다 학습이 한 번뿐이다. 데이터 구성이 흔들림의 폭이나 방향에 미치는 작은 영향은 이 자료로 가를 수 없다.

**40k 품질 (평가 12개, 8-bit PSNR, `report_matsynth_basecolor/summary.md`).**

| 맵 | matsynth_all (1:1) | chratio | basecolor |
|---|---|---|---|
| baseColor PSNR / 이동 뺀 PSNR | 26.88 / **30.47** | 26.85 / 30.22 | 26.92 / 29.62 |
| baseColor SSIM / \|grad\| %GT | **0.628** / 95% | 0.604 / 98% | 0.566 / 111% |
| normal PSNR / 이동 뺀 PSNR / 각도 | 23.65 / 30.86 / 13.3° | 26.58 / 30.60 / 9.6° | 27.56 / 31.33 / 8.1° |
| roughness PSNR | **31.31** | 29.25 | 28.71 |
| metallic (=GT, 0맵 9개) | 9 | 9 | 0 |
| height PSNR | **34.08** | 32.11 | 31.77 |
| render PSNR | **25.96** | 25.50 | 24.68 |

- **normal의 차이는 거의 전부 색 이동 차이다.** 이동이 20.5 → 9.2 / 5.2 레벨로 줄었을 뿐, 이동을 빼면 30.6–31.3으로 비슷하다.
- **chratio는 roughness·height에서 대가를 치렀다** (−2.1 / −2.0 dB). 회색 맵 샘플이 20 → 11%로 준 결과로, 예상했던 손해다.
- **baseColor만 학습해도 baseColor가 더 좋아지지 않는다.** 이동 뺀 PSNR과 SSIM은 오히려 1:1 혼합이 가장 높다. 맵 혼합이 baseColor를 해치지 않는다.
- **종합:** 논문 비율(1:1)이 가장 낫거나 비슷하다. 비율 조정으로 얻은 것은 없다.

**다음.** 색 불안정을 줄이려면 데이터가 아니라 레시피를 바꿔야 한다. 후보는 다음과 같다.
- PSNR 사전학습 (결함 #2, `variants/deeptex/paper_loop/main.py --experiment fixed --stage psnr`)
- 픽셀 손실 가중치 상향
- 학습률 감쇠가 실제로 걸리는 스케줄 (결함 #3)
- 체크포인트 선택이나 가중치 평균

모두 논문에서 벗어나므로, 하게 되면 "결함을 고친 판"으로 따로 다룬다(§0.4).

---

## 1. 결함 → 대응 대조표

| # | upstream 위치 | 증상 | 우리 대응 | 서브모듈 수정 |
|---|---|---|---|---|
| 1 | `configs/esrgan.yaml:17` | 학습 tfrecord 없음. 변환기 입력(`HR/`·`LRx4/`)도 없음. **×4 LR 생성기가 두 저장소 어디에도 없다** (upstream은 DIV2K LR을 다운로드했다) | `variants/deeptex/paper/texsr_normalize.py` → `texsr_make_pairs.py` → `variants/deeptex/upstream_runner/texsr_build_tfrecord.py` → `texsr_verify_tfrecord.py`. LR은 `modules/utils.py:125 imresize_np`를 **import해서** 쓴다(복제하지 않는다 — 복제본은 학습·평가가 쓰는 커널과 드리프트할 수 있다) | 없음 |
| 2 | `train_esrgan.py:71` + `configs/esrgan.yaml:8` | `pretrain_dir = cfg['pretrain_name']`, 값은 `'./checkpoints/'`. donor는 `'./checkpoints/' + name`. **즉 donor의 PSNR 단계를 복원해도 `./checkpoints/psnr_pretrain/`를 못 찾는다.** 실패는 `:78-79`의 `print` 한 줄, 그 뒤 랜덤 초기화로 학습 진행 (OPEN-ITEMS #6) | config 키를 **전체 경로** `pretrain_dir`로. 설정됐는데 못 찾으면 **exit 9**. 추가로 restore 전후 `generator.trainable_variables[0]`을 비교해 **복원을 증명**하고, 안 바뀌면 abort | 없음 |
| 3 | `configs/esrgan.yaml:23,27` | `niter 40000` < `lr_steps[0] 50000` → LR 감쇠 영구 미발동. `:128,:142-144`가 로깅하는 스케줄은 죽은 코드 (OPEN-ITEMS #7) | config 2종. `esrgan_scaled.yaml`은 `[5000,10000,20000,30000]`(upstream 값의 1/10 = ESRGAN 원형의 형태), `esrgan_asconfigured.yaml`은 그대로. 트레이너가 `lr_steps[0] < niter`를 **단언**하고, `--allow-dead-lr-schedule` 없이는 거부 | 없음 |
| 4 | `train_esrgan.py:23-25` | `logger.disabled = True; setLevel(FATAL)` — **Python 로거**를 죽인다. 그런데 정작 시끄러운 cuBLAS/NUMA는 **C++ 로거**에서 나온다(OPEN-ITEMS #5). 엉뚱한 쪽을 막았고, 그 결과 체크포인트 불일치·retracing 경고가 사라졌다 | C++는 `TF_CPP_MIN_LOG_LEVEL=2`로 INFO만 죽이고 **Python 로거는 WARNING 유지**. 환경변수는 `import tensorflow` **전에** 설정한다 (upstream은 `:20`에서 `:4`의 import 뒤에 설정해 효과가 없다) | 없음 |
| 5 | `train_esrgan.py:37` | ESRGAN 단계가 `shuffle=False`. 크롭·플립·회전은 매 패스 새로 뽑히지만 **레코드 순서, 따라서 배치 구성**이 고정 — batch *k*는 ~1600 패스 내내 같은 8장. donor의 PSNR 단계는 `shuffle=True` | `load_dataset(..., shuffle=True)`가 기본. 플래그는 `modules/utils.py:38`에 이미 있다. `--shuffle off`로 upstream 거동 복원 | 없음 |
| 6 | `modules/dataset.py:40-50` | `tf.Assert` 3개의 **반환값을 버린다**(`control_dependencies` 없음). HR≠LR×4가 에러가 아니라 `:56-57`에서 성공적으로 슬라이스돼 **어긋난 쌍**으로 학습된다 | 패치하지 않는다. 대신 (a) `texsr_make_pairs.py`가 HR만 타일링하고 LR을 **타일마다 유도**해 `hr.shape == lr.shape*4`를 *구성상* 참으로 만들고, (b) `texsr_verify_tfrecord.py`가 학습 전 전수 검사한다. 스텝마다 도는 그래프 단언보다 강하고 학습 중 비용은 0 | 없음 |
| 7 | `modules/dataset.py:12-13` | `decode_png(channels=3)` uint8 → 알파 삭제·팔레트 전개·그레이 3배·**16-bit 절단**이 전부 무경고 | `texsr_normalize.py`가 상류에서 전부 처리하고 **각각 카운트**한다. `texsr_verify_tfrecord.py`는 **`channels=0`(네이티브)** 으로 디코드해 검사한다 — `channels=3`으로 열면 바로 그 지점에서 숨는다 | 없음 |
| 8 | `modules/losses.py:56,80` | `from_logits=False` + 수동 sigmoid. Keras가 `[1e-7, 1-1e-7]`로 clip하므로 NaN은 아니고, **그래디언트가 정확히 0이 된다** — D가 이기기 시작하면 sigmoid가 float32에서 정확히 1.0/0.0에 닿아 손실이 0으로 떨어지고 기울기도 **0.000e+00**이 된다(§2.5 실측). 랜덤 초기화(#2)면 이게 **기본** 실패 모드 | `variants/deeptex/fixed/losses_corrected.py`에 `from_logits=True` 판본(수학적으로 동일, softplus로 안정). `--losses {corrected,upstream}`로 **차이를 단언이 아니라 측정 가능**하게 둔다. `PixelLoss`/`ContentLoss`는 서브모듈 것을 그대로 import | 없음 |
| 9 | `test.py:36` | 체크포인트 경로가 `'./checkpoints/original'` 하드코딩(+주석 처리된 8개). 학습은 `./checkpoints/<sub_name>`에 쓴다 → **직접 학습한 모델을 평가할 수단이 없다.** CLI 플래그도 없음 | `variants/deeptex/infer/deeptex_sr.py`가 `test.py`를 완전히 대체. 경로는 `sub_name`에서 유도, `--ckpt-dir`로 override. 없으면 **exit 3** | 없음 |
| 10 | `modules/utils.py:401` | `split_large_image`가 타일을 **JPEG q95**로 디스크에 굽고 `test.py:68`이 다시 읽어 SR 입력으로 쓴다. bicubic 기준선은 원본 메모리 배열 → 내장 비교가 SR에 불리하게 편향되고, **PNG 입력이 조용히 JPEG를 경유**한다 | 타일링을 전부 메모리 내에서. 디코드~인코드 사이에 디스크가 없다 | 없음 |
| 11 | `modules/utils.py:376-405` + `test.py:79` | SR 캔버스는 `(H//512)*512`, HR은 `(H//4)*4` → 512의 정확한 배수가 아니면 `ValueError` | 캔버스가 정확히 `(H*4, W*4)`. 마지막 행/열은 가장자리 정렬로 나머지를 덮는다 | 없음 |
| 12 | `test.py:57,68,70` | 학습은 RGB(`tf.image.decode_png`), 추론은 **BGR**(`cv2.imread`). 학습된 채널별 거동이 엉뚱한 채널에 적용된다 | 추론 경로에서 **cv2를 아예 쓰지 않는다** — `tf.io.decode_image(channels=3)` → `tf.io.encode_png`. 실수로 되돌아올 수 없는 구조 | 없음 |
| 13 | `test.py:75` | `if len(matrix_images[0]) > 1:` — N행×1열이면 join을 건너뛰고 **좌상단 타일만** 저장 | 2차원 그리드 루프, 특례 없음 | 없음 |
| 14 | `data/convert_train_tfrecord.py:93-97` | `exit(1)`(출력 이미 존재)이 `except SystemExit: pass`에 먹혀 **프로세스 종료 코드 0**. `test.py:108-111`도 동일 | argparse + 평범한 `sys.exit(n)`. absl 래퍼 없음 | 없음 |
| 15 | `modules/utils.py:329-333` | `calculate_ssim` 3채널 분기가 `for _ in range(3): ssims.append(_ssim(img1, img2))` — 같은 전체 SSIM을 3번 평균. RGB 입력에 대해 무의미한 값 | 쓰지 않는다. `variants/util/texsr_metrics.py`가 채널별로 제대로 계산 | 없음 |
| 16 | `modules/utils.py:340-365` | `rgb2ycbcr`이 `:350`에서 인자를 **in-place 변형**하고 `:351`에서 채널을 뒤집는다 — 이름과 달리 실제로는 BGR을 기대한다 | 쓰지 않는다 | 없음 |
| 17 | `modules/utils.py:64` | `tensor2img`가 `astype(uint8)`로 **절단**(반올림 아님) → 약 0.5 LSB, PSNR 약 −0.1 dB | `np.rint` | 없음 |
| 18 | `modules/utils.py:16` | `yaml.load(..., Loader=yaml.Loader)` — 안전하지 않은 로더 | 우리 config는 우리 코드가 읽는다. 서브모듈 `load_yaml`은 우리가 쓴 파일에만 쓰이므로 실질 위험은 없지만 기록해 둔다 | 없음 |
| — | `cfg['network_D']`, `train_dataset.num_samples` | **두 저장소 모두에서 읽는 곳이 없는 죽은 키.** D는 `train_esrgan.py:33`에서 `DiscriminatorVGG128(gt_size, ch_size)`로 만들어져 `nf`가 Python 기본값에서 온다 | 우리 config에서 생략. 실제 레코드 수는 `<tfrecord>.meta.json`에 기록하고 **거기서 읽어 검증**한다 | 없음 |

---

## 1.1 결함 귀속 — donor 상속 vs DeepTextures 도입

§0.7 때문에 위 표는 **두 종류**를 섞고 있다. 재현 보고서에서 둘을 같이 묶으면 안 된다.

| DeepTextures가 도입 | donor(`peteryuX/esrgan-tf2`)에서 상속 |
|---|---|
| **#2** 사전학습 경로 (`train_esrgan.py:71`, 1줄) | **#4** Python 로거 차단 (`:23-25`) |
| **#3** LR 감쇠 미발동 (config의 `niter` 1/10) | **#5** `shuffle=False` (`:37`) |
| **#9** 체크포인트 하드코딩 (`test.py:36`) | **#6** `tf.Assert` 반환값 폐기 |
| **#10** JPEG q95 타일 (`utils.py:401`) | **#7** `decode_png(channels=3)` |
| **#11** 512 그리드 vs 4배수 불일치 | **#8** `from_logits=False` + 수동 sigmoid |
| **#13** N행×1열 버그 (`test.py:75`) | **#15** `calculate_ssim` 3채널 분기 |
| **#14** 종료 코드 0 (`except SystemExit`) | **#16** `rgb2ycbcr` in-place·BGR 기대 |
| **#12** RGB/BGR 불일치 (`test.py`가 cv2) | **#17** `tensor2img` 절단 |
| | **#18** `yaml.Loader` |
| | 생성기 출력 무경계 (`conv_last` 무활성화) |
| | `_kernel_init`이 인자를 2배로 |
| | `ResDenseBlock_5C.conv5`에 LeakyReLU (원본 ESRGAN은 선형) |
| | 판별기 `conv0_0`→`conv0_1` 사이 활성화 없음 |

**따라서:** "#5·#6·#7·#8 때문에 논문 결과가 약하다"는 서술은 **틀렸다.** 그것들은
논문 저자가 쓴 코드가 아니다. 논문에 귀속시킬 수 있는 것은 왼쪽 열뿐이며, 그중
**#2와 #3이 학습 결과에 직접 영향을 준다**(랜덤 초기화 + 감쇠 없음, §2.6 실측).

---

## 1.2 타일링 정밀 분석 — DeepTextures의 유일한 실질 기여

`split_large_image` / `join_large_image`(`utils.py:368-440`)와 그것을 쓰는 `test.py:57-104`.
논문의 "scalable tiling ... processed and reassembled seamlessly" 주장에 대응하는 코드다.

### 실제 동작

```
test.py:57   raw_img = cv2.imread(path)              uint8 HxWx3 BGR  (알파 삭제, 16bit 절단)
test.py:58   split_large_image(path)
               H>512 or W>512 ?
                 예 → (H//512)x(W//512) 타일, 각 512², CWD 상대 폴더에
                      part_{r}_{c}.jpg 로 저장          ← JPEG q95
                 아니오 → [[원본 경로]] 반환 그대로      ← 타일 아님
test.py:68-70  타일마다: imread → create_lr_hr_pair(tile, 4) → LR 128²
                        → model(LR/255) → tensor2img → 512²
test.py:74-76  sr = matrix[0][0];  if len(matrix[0]) > 1: join_large_image(...)
test.py:79     final_hr = create_lr_hr_pair(raw_img, 4)[1]      (H//4*4, W//4*4)
test.py:81     final_bic = imresize_np(final_lr, 4)             ← 원본 배열에서
test.py:91-94  PSNR/SSIM( rgb2ycbcr(...) )                      ← 1채널 Y로 축약
```

### 여기서 나오는 다섯 가지

1. **타일은 SR이 아니라 "축소 후 복원"이다.** 512 타일 → bicubic 128 → 모델 ×4 → 512.
   `test.py`는 주어진 LR을 키우는 스크립트가 **아니다.** HR을 받아 스스로 LR을 만들고
   되살린다. → 우리 128² LR을 512²로 만들 수단이 없다. **`deeptex_sr.py`가 존재해야 하는
   진짜 이유가 이것이다** (체크포인트 하드코딩 #9는 부차적이다).
2. **SR과 bicubic이 같은 입력이 아니다.** bicubic은 원본 배열(`:79→:81`)에서, SR은
   **JPEG 재압축된 타일**(`:68`)에서 나온다. JPEG 손상이 전부 모델 몫으로 계상된다 —
   내장 비교가 SR에 불리하게 편향된다(결함 #10). 보고된 숫자에는 이 사실이 안 보인다.
3. **나머지가 조용히 버려진다.** `num_rows = H//512`, `num_cols = W//512`. 1300² 이미지는
   2×2 타일만 처리하고 양 축에서 276px를 버린다. 경고 없음.
4. **캔버스가 안 맞는다.** SR은 `(rows*512, cols*512)`, HR은 `(H//4*4, W//4*4)`.
   512의 정확한 배수가 아니면 `calculate_psnr`가 `ValueError`(결함 #11).
   퇴화 격자는 더 나쁘다 — `H>512, W≤512`면 `num_cols=0`이라 `test.py:74`가 `IndexError`,
   반대면 `num_rows=0`이라 `:59`가 `IndexError`.
5. **`if len(matrix_images[0]) > 1`은 열 수만 본다**(`:75`). N행×1열이면 join을 건너뛰고
   **좌상단 타일 하나**만 최종 SR이 된다(결함 #13).

### 출력 파일 이름이 서로 모순된다

| 파일 | 실제 내용 |
|---|---|
| `HR_{name}_{r}_{c}.png` | **SR 타일** |
| `HR_{name}.png` | **SR 모자이크** |
| `HR_{name}` (확장자 없음) | **정답 HR** |
| `LR_{name}` | **bicubic ×4 업샘플** — LR이 아니다 |
| `SR_{name}` | SR (유일하게 이름과 일치) |

셋이 충돌하지 않는 것은 `:76`이 확장자까지 포함한 basename 뒤에 `.png`를 덧붙이기
때문이다. 그리고 `cv2.imwrite`는 **확장자로 인코더를 고르므로**, `.jpg` 입력이면
디스크의 `SR_x.jpg`는 **PSNR을 잰 배열과 다른** 손실 압축본이다.

### `join_large_image`

캔버스는 `(rows*512, cols*512, 3)` uint8. 타일마다 `cv2.resize(img, (512,512))`를
**무조건** 건다(`:434`). `scale=4`에서는 no-op이지만, 512를 나누지 않는 scale이면
픽셀 격자를 조용히 바꾼다.

### 우리 파이프라인과의 관계

`deeptex_sr.py`가 이 경로 전체를 대체한다 — 메모리 내 타일링, 정확히 `(H*4, W*4)` 캔버스,
JPEG 없음, RGB 일관, `rint` 양자화. 다만 **우리 128² 입력에서는 타일링 자체가 불필요**하며
(`--tile-size 0` 기본), 그래서 upstream 타일링의 결함 #10·#11·#13은 우리 결과에
영향을 주지 않는다. 동시에 **논문의 고유 기여도 우리 결과에 반영되지 않는다**(§0.8).

---

## 2. 우리가 추가한 것 (upstream에 대응물이 없음)

| 추가 | 왜 |
|---|---|
| **데이터셋 단위 검증 루프** (`--val-every`) | **두 저장소 어디에도 없다.** donor의 Set5/Set14 분기는 `test.py`와 함께 삭제됐다. 3시간 낭비를 1시간에 알아채는 유일한 수단이고, bicubic 기준선을 같이 찍어 "학습이 되고는 있나"에 답한다 |
| **`texsr_verify_tfrecord.py`** | 조용한 손상 3종(#6·#7과 잘린 tfrecord)을 학습 전 **한 번**에 시끄럽게 만든다 |
| **시작 단언** | `gt_size == input_size*scale`(누구도 교차 검증하지 않는다), tfrecord sha256·레코드 수, verify PASS, **GPU 가시성**, 디스크 여유 |
| **`[metrics]` 개행 로그** | `ProgressBar`(`modules/utils.py:113-118`)는 `\r`이라 파이프하면 로그가 한 줄 괴물이 된다. 오케스트레이터는 `[metrics]`만 파싱한다 |
| **매 스텝 `isfinite`** | 프로그레스바가 이미 `.numpy()`로 sync하므로 공짜. 비정상이면 디버그 체크포인트를 남기고 exit 4 |
| **PSNR/SSIM (GT 대비)** | 기존 비교 체인에는 **전참조 메트릭이 하나도 없었다**(grad·lapvar·normal 상관뿐). 두 방법이 같은 GT를 공유하는 baseColor·render에서는 오차가 측정 가능하고, 따라서 측정해야 한다 |

---

## 2.5 결함 #8은 측정됐다

`deeptex_losses.py`의 수정판과 upstream 판을 같은 입력에 물려 재본 값 (2026-09-16, 컨테이너 안):

| 상황 | upstream | corrected |
|---|---|---|
| 초기화 (logit 0) | 0.6931 | 0.6931 |
| D 포화 (hr=+30, sr=−30) 손실 | **−0** | 8.75651e-27 |
| D 포화 시 **그래디언트** | **0.000e+00** | 1.095e-27 |

초기화 시점에는 둘이 정확히 같다 — 수학적으로 같은 함수이므로 당연하다. 차이는 **포화 이후**에만 나온다:
upstream은 그래디언트가 **정확히 0**이 되어 D가 회복할 경로 자체가 사라지고, 수정판은 극히 작지만
표현 가능한 값을 유지한다.

수정판이 포화한 D를 되살리는 건 아니다 — 그 지점에서 참 그래디언트가 실제로 작다. 차이는
**0과 0이 아닌 것**의 차이이고, 그게 회복 가능성의 유무를 가른다.

**그리고 이건 관찰된 현상이다.** 사전학습 없이(`--pretrain-dir none`) ESRGAN 200스텝을 돌리면
`loss_D`가 즉시 `0.0000`으로 떨어진다 — 랜덤 초기화된 G를 상대로 D가 바로 이긴다.
OPEN-ITEMS #6이 예측하는 실패 모드가 그대로 재현되며, 사전학습을 복원해야 하는 이유가 이것이다.
**40k 규모에서도 학습 시간의 절반에서 같은 현상이 유지된다 — §2.6.**

---

## 2.6 40k 본학습에서 결함 #2·#3·#8이 전부 관측됐다 (2026-09-18)

`-Stage paper`로 40,000 스텝을 완주한 기록. 로그가 둘로 나뉜다 —
`runs/_logs/20260918-161457/`(step 1–5,962, GPU 이탈로 중단)와
`runs/_logs/20260918-212530/`(step 5,000에서 재개, 40,000 완주).

| | |
|---|---|
| 최종 체크포인트 | `runs/deeptex/paper/checkpoints/esrgan_40k/ckpt-8`, **`step` 텐서 = 40000** |
| 처리량 | 두 실행 모두 **중앙값 4.30 step/s** |
| 소요 | 재개분 35,000 스텝에 **2시간 15분** (21:25:32 → 23:40:58). 40k 환산 약 2.6시간 |
| `loss_G` | 862.5 → 0.7316. 15,000 스텝 이후 평평 |
| non-finite loss | `[metrics]` 411줄 중 **0건** |
| 서브모듈 | 학습 후에도 `e4ee562` clean — 체크포인트·로그가 `runs/deeptex/paper/`로 떨어졌다 |

재개는 `[*] load ckpt from ./checkpoints/esrgan_40k/ckpt-1 at step 5000.`으로 확인됐다.
`run_upstream_train.py`가 이 줄을 따로 파싱하는 이유가 이것이다 — 재개 실패와 0레코드
tfrecord는 둘 다 "스텝 0"으로 보이고, 이 줄만이 둘을 구분한다.

⚠️ **재개는 데이터 스트림을 레코드 0부터 다시 시작한다.** 데이터셋 이터레이터는
체크포인트에 포함되지 않고(`train_esrgan.py:57-61`) `shuffle=False`(`:37`)이므로,
global step은 5,000에서 이어지지만 레코드 순서는 처음부터 반복됐다. 크롭·flip·rot만
다시 뽑힌다.

⚠️ step 5,000–5,962 구간은 중단된 실행과 재개된 실행 양쪽에 존재해, 411줄 중 10줄이 중복이다.
아래 비율은 그 중복을 포함한 값이다.

### 결함 #2 — 사전학습은 로드되지 않았다

중단된 실행의 로그 105줄:

```
[*] cannot find pretrain model ./checkpoints/.
```

`generator.summary()` 300줄 아래를 `print` 한 줄이 지나간다. **40,000 스텝 전체가 랜덤
초기화에서 출발했다.** 재개된 실행에는 이 줄이 없는데, 결함이 사라져서가 아니라
`train_esrgan.py:67-74`가 체크포인트를 찾으면 사전학습 탐색을 **건너뛰기** 때문이다.
가중치의 출처는 동일하다.

### 결함 #3 — LR 감쇠는 한 번도 걸리지 않았다

`[metrics]` **411줄 전부 `lr_G=1.0e-04`**. `lr_steps[0]=50000 > niter=40000`이므로
`:128,:142-144`가 매 스텝 로깅하는 스케줄은 끝까지 죽은 코드였다.
**이제 예측이 아니라 실측이다.**

### 결함 #8 — 판별자 포화가 학습 내내 지속됐다

`loss_D`가 `0.0000`으로 찍힌 비율 **206/411 (50%)**:

| 구간 | `loss_D` = 0.0000 | `loss_G` 평균 |
|---|---|---|
| 0–5k | **80%** | 19.86 |
| 5k–10k | 58% | 0.960 |
| 10k–15k | 66% | 0.843 |
| 15k–20k | 46% | 0.788 |
| 20k–25k | 34% | 0.773 |
| 25k–30k | 34% | 0.762 |
| 30k–35k | 42% | 0.725 |
| 35k–40k | 40% | 0.740 |

초반 80%에서 후반 40%로 내려오지만 **끝까지 절반**이다. §2.5의 마이크로벤치가 예측한 모드가
40k 규모에서 그대로 나타났고, #2(랜덤 초기화) + #8(포화)의 조합이 실제로 성립했다.

⚠️ **과대 해석 금지.** 진행바의 포맷은 `%.4f`이므로 `0.0000`은 **"손실 < 5e-5"**이지
"그래디언트가 정확히 0"이 아니다. 그래디언트가 `0.000e+00`에 닿는다는 것은 §2.5에서 따로
측정한 별개의 사실이고, 이 로그가 말해 주는 것은 **D가 학습 시간의 절반을 그 근방에서
보냈다**는 것뿐이다. 두 문장을 합쳐 쓰지 말 것.

`loss_D` 최댓값은 3.5545(재개분)로, D가 완전히 죽은 것은 아니고 포화와 회복을 오갔다.

### 부수 관측 — 전력제한은 처리량을 깎지 않았다

재개된 실행은 `nvidia-smi -pl 300`(이 카드 기본값 400 W) 상태로 돌았는데 중앙값이
4.30 step/s로 제한 없이 돈 첫 실행과 **같다.** GPU 이탈 재발 방지 조치의 비용이 측정 한계
이하라는 뜻이므로, 이 머신에서는 그냥 걸어 두는 편이 낫다. 단 Windows는 이 값을 재부팅 뒤
유지하지 않는다.

---

## 3. 무해하지만 눈에 보이는 것들 — 쫓지 말 것

- **`Inconsistent references when loading the checkpoint into this object graph`**
  `deeptex_sr.py`/`deeptex_compare.py`가 체크포인트를 복원할 때 8줄쯤 나온다. `RRDB_Model`의
  `fea = fea + trunck`가 `TFOpLambda` 레이어를 만들고, 저장 시점 그래프와 참조 형태가 달라서 나는 경고다.
  **복원 자체는 정상**이며 `status.assert_existing_objects_matched()`와 가중치 변화 검사가 그걸 증명한다.
  이 경고가 보이는 이유는 우리가 #4에서 **로거를 다시 켰기 때문**이다 — upstream에서는 숨겨졌을 뿐 존재했다.
- **cuBLAS 중복 등록 / NUMA node / oneDNN 안내** — OPEN-ITEMS #5에 이미 기록된 TF 2.10 잡음.
- **`VarianceScaling is unseeded and being called multiple times`** — Keras 초기화 경고. 같은 이유로 이제 보인다.
- **pip 20.2.4 경고** — TF 2.10.1 / Python 3.8 베이스 이미지가 싣고 있는 것. OPEN-ITEMS #8 참조.

---

## 4. 재현할 때 반드시 함께 적어야 하는 것

1. **PSNR 사전학습은 잘랐다.** donor `configs/psnr.yaml:27`의 `niter: 1000000`은 이 카드에서 70~180시간이라
   선택지가 아니었다. `niter: 40000` + `lr_steps [8000,16000,24000,32000]`(donor와 같은 1/5 간격)로 줄였다.
   → OPEN-ITEMS #6의 "사전학습 복원"은 실제로는 "**잘린** 사전학습 복원"이다.
2. **`shuffle=False`·미발동 LR 감쇠·미로딩 사전학습은 코드 결함이지 방법 결함이 아니다.**
   `esrgan_asconfigured.yaml` 기준선을 돌려 약하게 나오더라도 **논문에 귀속시키지 말 것.** 셋을 따로 이름 붙여 기술한다.
3. **학습 소스는 `textures_db` 300장, 평가는 MatSynth 12종.** 도메인 갭이 있고, 이건 고칠 항목이 아니라
   기술할 한계다. MatUp은 학습이 없어 이 갭을 겪지 않는다.
4. **정규화가 버린 것**(`normalize_manifest.json`의 실측):
   16-bit → 8-bit **45장**, 알파 삭제 **119장**(그중 **11장은 실제로 투명 영역이 있었다**),
   4의 배수로 크롭 **142장**, 크기 미달로 제외 **1장**(`221.png`, 64²),
   그리고 upstream의 `*.png` glob이 조용히 버렸을 **JPG 100장을 구제**했다.
