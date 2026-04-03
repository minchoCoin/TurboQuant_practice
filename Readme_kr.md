# TurboQuant_practice (한국어 정리)

TurboQuant_practice

TurboQuant 논문: [https://arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)

## Lemma 1 설명

`TurboQuant/main.tex`의 Lemma 1은 다음을 말합니다.

- $x \in S^{d-1}$가 $R^d$의 단위구면에서 균일분포를 따른다면
- 임의의 좌표 $x_j$는 $[-1,1]$ 위의 Beta 형태 분포를 따릅니다.

논문 식은 다음과 같습니다.

$$
f_X(x) =
\frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d-1)/2)}
\left(1-x^2\right)^{(d-3)/2}, \quad x \in [-1,1].
$$

직관적으로, 구면 위 랜덤 점에서 특정 좌표를 $x_j=x$로 고정하면 나머지 좌표는 반지름 $\sqrt{1-x^2}$인 저차원 구면 위에 있어야 합니다. 따라서 해당 좌표값의 확률은 그 구면 단면의 크기로 결정됩니다.

## 감마 함수 정의

감마 함수는 팩토리얼을 실수/복소수로 확장한 함수입니다.

$$
\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt, \quad z > 0.
$$

유용한 항등식:

- $\Gamma(z+1) = z \Gamma(z)$
- 정수 $n$에 대해 $\Gamma(n) = (n-1)!$
- $\Gamma(1/2) = \sqrt{\pi}$

고차원 구/구면 공식에 감마 함수가 자연스럽게 등장하는 이유입니다.

## 구의 부피/면적 공식

반지름 $r$인 $d$차원 공의 부피:

$$
V_d(r) = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)} r^d
$$

반지름 $r$인 구면 $S^{d-1}$의 표면적:

$$
A_{d-1}(r) = \frac{2\pi^{d/2}}{\Gamma(d/2)} r^{d-1}.
$$

특히 $r=1$일 때:

- 단위공 부피: $V_d(1)=\pi^{d/2}/\Gamma(d/2+1)$
- 단위구면 표면적: $A_{d-1}(1)=2\pi^{d/2}/\Gamma(d/2)$

Lemma 1 증명에서 단면 구면의 크기를 계산하므로 감마 함수가 등장합니다.

## Lemma 1 증명 스케치

논문은 단면적(cross-section) 관점으로 증명합니다.

1. $x_j=x$를 고정하면 나머지 좌표는
   $x_1^2+\cdots+x_{j-1}^2+x_{j+1}^2+\cdots+x_d^2=1-x^2$
   를 만족합니다.
2. 즉 가능한 집합은 $R^{d-1}$에서 반지름 $\sqrt{1-x^2}$, 차원 $d-2$인 구면입니다.
3. 그 표면적은
   $A_{d-2}(\sqrt{1-x^2})=\frac{2\pi^{(d-1)/2}}{\Gamma((d-1)/2)}(1-x^2)^{(d-2)/2}$.
4. 전체 표본공간은 단위구면 $S^{d-1}$이고 표면적은
   $A_{d-1}(1)=\frac{2\pi^{d/2}}{\Gamma(d/2)}$.
5. 밀도는 단면적/전체면적 비율에 비례하며, 좌표 $x$에 대한 밀도로 바꾸면 Jacobian
   $\frac{1}{\sqrt{1-x^2}}$
   가 추가됩니다.
6. 정리하면
   $f_X(x)=\frac{\Gamma(d/2)}{\sqrt{\pi}\Gamma((d-1)/2)}(1-x^2)^{(d-3)/2}$.

핵심은 “구면 위 랜덤점의 한 좌표 분포는 해당 좌표에서의 구면 슬라이스 크기로 결정된다”는 점입니다. 차원이 커질수록 분포는 0 근처로 집중하고, 고차원에서 $N(0,1/d)$ 근사로 이어집니다.

## QJL 설명

`TurboQuant/main.tex` Definition 1의 QJL(Quantized Johnson-Lindenstrauss)은 inner product 추정을 위한 1-bit 양자화 맵입니다.

정방향 맵:

$$
Q_{\tt qjl}(\mathbf{x}) := \mathrm{sign}(\mathbf{S}\mathbf{x}), \quad \mathbf{x} \in \mathbb{R}^d,
$$

여기서 $\mathbf{S}\in\mathbb{R}^{d\times d}$는 i.i.d. $N(0,1)$ 가우시안 행렬이고 sign은 좌표별로 적용됩니다.

역양자화:

$$
Q_{\tt qjl}^{-1}(\mathbf{z}) := \frac{\sqrt{\pi/2}}{d}\mathbf{S}^\top \mathbf{z}, \quad \mathbf{z}\in\{-1,+1\}^d.
$$

역할:

- 가우시안 투영 후
- 부호만 남겨 1-bit로 압축하고
- $\mathbf{S}^\top$와 스케일링으로 복원 벡터를 만듭니다.

QJL은 벡터 자체를 정확 복원하는 목적보다, inner product를 기대값 관점에서 보존하는 데 목적이 있습니다. TurboQuant에서는 MSE 양자화 residual에 QJL을 붙여 inner product estimator의 편향을 줄입니다.

### $\sqrt{\pi/2}/d$ 계수가 들어가는 이유

표준정규 $g\sim N(0,1)$에 대해

$$
\mathbb{E}[\mathrm{sign}(g)g]=\mathbb{E}[|g|]=\sqrt{2/\pi}.
$$

따라서 $\mathbf{S}^\top\mathrm{sign}(\mathbf{S}\mathbf{x})$의 기대값에는 $d\sqrt{2/\pi}$ 배가 붙습니다. 이를 상쇄해 unbiased를 맞추는 계수가

$$
\frac{\sqrt{\pi/2}}{d}
$$

입니다.

엄밀히는

$$
\mathbb{E}\!\left[\frac{\sqrt{\pi/2}}{d}\mathbf{S}^\top\mathrm{sign}(\mathbf{S}\mathbf{x})\right]=\mathbf{x}
$$

가 성립합니다(단위벡터 기준). 즉 Definition 1의 계수는 임의가 아니라 unbiased inner-product 추정을 위한 정규화입니다.

## Lemma 4 설명: QJL 성능 보장

Lemma 4는 단위벡터 $\mathbf{x}\in S^{d-1}$, 임의 질의 벡터 $\mathbf{y}\in\mathbb{R}^d$에 대해

$$
\left\langle \mathbf{y}, Q_{\tt qjl}^{-1}(Q_{\tt qjl}(\mathbf{x})) \right\rangle
$$

추정량의 통계적 보장을 제공합니다.

핵심:

- unbiased:
  $$
  \mathbb{E}\!\left[\left\langle \mathbf{y}, Q_{\tt qjl}^{-1}(Q_{\tt qjl}(\mathbf{x})) \right\rangle\right]
  =
  \langle \mathbf{y}, \mathbf{x} \rangle
  $$
- variance bound:
  $$
  \mathrm{Var}\!\left(\left\langle \mathbf{y}, Q_{\tt qjl}^{-1}(Q_{\tt qjl}(\mathbf{x})) \right\rangle\right)
  \le
  \frac{\pi}{2d}\|\mathbf{y}\|_2^2
  $$

즉 1-bit 양자화임에도 평균은 정확하고, 분산은 $1/d$로 줄어듭니다.

### Step 1: 평균 형태로 재작성

$\mathbf{S}$의 row를 $\mathbf{s}_1,\dots,\mathbf{s}_d$라 두면

$$
\left\langle \mathbf{y}, Q_{\tt qjl}^{-1}(Q_{\tt qjl}(\mathbf{x})) \right\rangle
=
\frac{1}{d}\sum_{i=1}^d \sqrt{\pi/2}\,\mathbf{s}_i^\top\mathbf{y}\,\mathrm{sign}(\mathbf{s}_i^\top\mathbf{x}).
$$

이는

$$
\langle \mathbf{a},\mathbf{S}^\top\mathbf{b}\rangle=\langle \mathbf{S}\mathbf{a},\mathbf{b}\rangle
$$

을 이용한 직접 대입 전개입니다.

### Step 2: 왜 unbiased인가

앞 절의 정규화 결과로

$$
\mathbb{E}[Q_{\tt qjl}^{-1}(Q_{\tt qjl}(\mathbf{x}))]=\mathbf{x}
$$

이므로

$$
\mathbb{E}\!\left[\left\langle \mathbf{y}, Q_{\tt qjl}^{-1}(Q_{\tt qjl}(\mathbf{x})) \right\rangle\right]
=
\langle \mathbf{y},\mathbf{x}\rangle.
$$

### Step 3: 왜 분산이 제한되는가

독립 평균 형태이므로

$$
\mathrm{Var}\!\left(\frac{1}{d}\sum_{i=1}^d z_i\right)
\le
\frac{1}{d}\mathrm{Var}(z_1).
$$

또한 variance $\le$ second moment와 sign의 절댓값 불변성을 이용하면

$$
\mathrm{Var}(z_1)\le\frac{\pi}{2}\mathbb{E}\!\left[(\mathbf{s}_1^\top\mathbf{y})^2\right]
=\frac{\pi}{2}\|\mathbf{y}\|_2^2
$$

이 되어 최종적으로

$$
\mathrm{Var}\!\left(\left\langle \mathbf{y}, Q_{\tt qjl}^{-1}(Q_{\tt qjl}(\mathbf{x})) \right\rangle\right)
\le
\frac{\pi}{2d}\|\mathbf{y}\|_2^2.
$$

### 해석

Lemma 4는 정확성(기대값)과 집중도(분산)를 동시에 보장합니다. TurboQuant에서 QJL을 residual에 붙이는 이유가 여기에 있습니다.

## Section 3.1 설명: MSE 최적 TurboQuant

Section 3.1은 재구성 MSE 최소화를 목표로 한 설계를 설명합니다.

1. 입력 $\mathbf{x}$를 직접 양자화하지 않고 랜덤 회전 $\mathbf{y}=\mathbf{\Pi}\mathbf{x}$를 먼저 수행합니다.
2. Lemma 1에 의해 회전 좌표 분포 $f_X$를 알 수 있게 됩니다.
3. 고차원 벡터 양자화를 좌표별 1차원 scalar quantization 문제로 근사합니다.
4. $2^b$개 centroid를 갖는 연속 1D k-means 문제
   $$
   \mathcal{C}(f_X,b)
   $$
   를 풀어 코드북을 설계합니다.

목적함수:

$$
\mathcal{C}(f_X, b)
:=
\min_{-1 \le c_1 \le \cdots \le c_{2^b} \le 1}
\sum_{i=1}^{2^b}
\int_{\frac{c_{i-1}+c_i}{2}}^{\frac{c_i+c_{i+1}}{2}}
|x-c_i|^2 f_X(x)\,dx.
$$

의미:

- 구간은 인접 centroid 중점으로 나뉘는 Voronoi 셀입니다.
- 각 적분항은 해당 셀의 기대 제곱오차입니다.
- 전체 합은 scalar quantizer의 총 기대 MSE입니다.

연속형 k-means의 최적조건도 동일하게

$$
c_i=
\frac{\int_{R_i}x f_X(x)\,dx}{\int_{R_i}f_X(x)\,dx}
$$

입니다.

파이프라인:

1. 랜덤 회전 $\mathbf{\Pi}$ 샘플링
2. $f_X$ 기반 codebook $\{c_k\}$ 계산
3. $\mathbf{y}=\mathbf{\Pi}\mathbf{x}$
4. 각 좌표 $y_j$를 최근접 centroid index로 양자화
5. centroid로 $\tilde{\mathbf{y}}$ 복원
6. $\tilde{\mathbf{x}}=\mathbf{\Pi}^\top\tilde{\mathbf{y}}$

## Algorithm 1: $\mathrm{TurboQuant}_{\tt mse}$ (MSE 최적화)

논문 알고리즘(의미 동일한 한국어 요약):

입력:

- 차원 $d$
- 비트폭 $b$

사전 준비:

- 랜덤 회전행렬 $\mathbf{\Pi}$ 생성
- $\mathcal{C}(f_X,b)$를 최소화하는 centroid $c_1,\dots,c_{2^b}$ 계산

양자화 $\mathrm{Quant}_{\tt mse}(\mathbf{x})$:

1. $\mathbf{y}\leftarrow \mathbf{\Pi}\mathbf{x}$
2. 각 $j\in[d]$에 대해 $\mathrm{idx}_j\leftarrow \arg\min_{k\in[2^b]}|y_j-c_k|$
3. 인덱스 벡터 $\mathrm{idx}$ 저장

복원 $\mathrm{DeQuant}_{\tt mse}(\mathrm{idx})$:

1. 각 $j$에 대해 $\tilde y_j\leftarrow c_{\mathrm{idx}_j}$
2. $\tilde{\mathbf{x}}\leftarrow \mathbf{\Pi}^\top\tilde{\mathbf{y}}$
3. $\tilde{\mathbf{x}}$ 출력

## $\mathrm{TurboQuant}_{\tt mse}$ 성능 보장

오차 정의:

$$
D_{\tt mse}:=\mathbb{E}\!\left[\|\mathbf{x}-\tilde{\mathbf{x}}\|_2^2\right]
$$

(기대값은 회전 랜덤성에 대해 취함)

상한:

$$
D_{\tt mse}\le \frac{\sqrt{3}\pi}{2}\cdot\frac{1}{4^b}.
$$

즉 비트 1개 증가 시 왜곡이 대략 4배 줄어듭니다.

소비트폭 구간의 보고값:

- $b=1$: $D_{\tt mse}\approx 0.36$
- $b=2$: $D_{\tt mse}\approx 0.117$
- $b=3$: $D_{\tt mse}\approx 0.03$
- $b=4$: $D_{\tt mse}\approx 0.009$

## Inner-Product 최적 TurboQuant

$\mathrm{TurboQuant}_{\tt mse}$는 복원 MSE에는 좋지만 inner product 추정의 무편향성을 자동 보장하지는 않습니다. 이를 보완한 것이 $\mathrm{TurboQuant}_{\tt prod}$입니다.

구성:

- $(b-1)$비트 $\mathrm{TurboQuant}_{\tt mse}$
- residual에 대한 1비트 QJL

먼저

$$
\tilde{\mathbf{x}}_{\tt mse}=Q_{\tt mse}^{-1}(Q_{\tt mse}(\mathbf{x}))
$$

를 구하고 residual

$$
\mathbf{r}:=\mathbf{x}-\tilde{\mathbf{x}}_{\tt mse}
$$

를 계산한 뒤, QJL로 residual을 스케치합니다.

추정식:

$$
\left< \mathbf{y}, Q_{\tt mse}^{-1}(Q_{\tt mse}(\mathbf{x})) \right>
+
\|\mathbf{r}\|_2\cdot
\left< \mathbf{y}, Q_{\tt qjl}^{-1}(Q_{\tt qjl}(\mathbf{r})) \right>.
$$

저장 항목:

- MSE 인덱스 $\mathrm{idx}$
- residual QJL 부호벡터 $\mathrm{qjl}$
- residual 노름 $\gamma=\|\mathbf{r}\|_2$

즉 coarse reconstruction은 MSE가 담당하고, 마지막 1비트로 inner product 편향을 보정하는 구조입니다.

### 알고리즘 관점

1. $\mathbf{x}$를 $(b-1)$비트 $\mathrm{TurboQuant}_{\tt mse}$로 양자화
2. residual 계산
3. residual에 QJL 적용
4. $(\mathrm{idx},\mathrm{qjl},\gamma)$ 저장
5. 복원 시 MSE 복원 + QJL residual 보정 합산

## $\mathrm{TurboQuant}_{\tt prod}$ 성능 보장

무편향성:

$$
\mathbb{E}\!\left[\langle \mathbf{y},\tilde{\mathbf{x}}\rangle\right]
=
\langle \mathbf{y},\mathbf{x}\rangle.
$$

왜곡 정의/상한:

$$
D_{\tt prod}
:=
\mathbb{E}\!\left[\left|\langle \mathbf{y},\mathbf{x}\rangle-\langle \mathbf{y},\tilde{\mathbf{x}}\rangle\right|^2\right]
\le
\frac{\sqrt{3}\pi^2\|\mathbf{y}\|_2^2}{d}\cdot\frac{1}{4^b}.
$$

소비트폭 보고값:

- $b=1$: $D_{\tt prod}\approx \frac{1.57}{d}$
- $b=2$: $D_{\tt prod}\approx \frac{0.56}{d}$
- $b=3$: $D_{\tt prod}\approx \frac{0.18}{d}$
- $b=4$: $D_{\tt prod}\approx \frac{0.047}{d}$

하한:

$$
D_{\tt prod}(Q)\ge \frac{1}{d}\cdot\frac{1}{4^b}.
$$

즉 상수배 차이를 제외하면 최적 왜곡률에 근접합니다.

## Entropy Encoding Codebook Pointer

$\mathrm{TurboQuant}_{\tt mse}$의 centroid index 분포는 일반적으로 균일하지 않습니다. 각 index $\ell\in[2^b]$의 확률은

$$
p_\ell := \int_{\frac{c_{\ell-1}+c_\ell}{2}}^{\frac{c_\ell+c_{\ell+1}}{2}} f_X(x)\,dx.
$$

이 분포에 대해 prefix code 등 entropy coding을 적용하면, 고정 $b$비트 대신 평균 비트를 엔트로피 수준까지 줄일 수 있습니다.

중요한 점:

- 재구성 왜곡(MSE)은 변하지 않음
- 저장 비트만 감소
- 이득은 비균일 인덱스 분포의 무손실 부호화에서 발생

논문 보고에 따르면 $b=4$에서 raw 4비트 대비 엔트로피가 약 3.8비트이고, 평균 약 5% 절감이 가능합니다. 다만 복잡도 대비 이득이 크지 않아 본 알고리즘에는 기본 포함하지 않습니다.

## My Experiments

아래는 논문 아이디어를 구현하면서 생성한 실험 그림입니다.

### 1. Codebook Quantization Demo

저색상 이미지, 코드북 크기별 양자화 결과, 코드북 팔레트를 시각화했습니다.

![Codebook comparison](results/1.codebook_comparison.png)

### 2. Lemma 1 Coordinate Distribution

차원별로 다음 3가지를 비교했습니다.

- 단위구면 좌표의 empirical 분포
- Lemma 1 정확 밀도
- 가우시안 근사 $N(0,1/d)$

![Lemma 1 distribution](results/2.lemma1_distribution.png)

### 3. QJL Simulation

여러 차원에서 다음을 관찰했습니다.

- inner-product error 분포
- inner-product squared error 분포
- $x$와 QJL 복원벡터의 코사인 유사도

QJL은 cosine 최대화보다 inner product 무편향 추정에 초점이 있음을 확인할 수 있습니다.

![QJL simulation](results/3-1.QJL_simulation.png)

### 4. Central Limit Theorem

지수분포 기반 시뮬레이션으로 다음을 확인했습니다.

- 표준화 표본평균의 정규분포 수렴
- 표본 크기 증가에 따른 표본평균 분산 감소

![Central limit theorem](results/6.central_limit_theorem.png)

### 5. Concentration of Measure

고차원 구면에서 차원이 커질수록 좌표 하나가 0 근처에 더 집중됨을 시각화했습니다.

![Concentration of measure](results/7.concentration_of_measure.png)

### 6. TurboQuant\_mse Simulation

차원별로 다음 분포를 비교했습니다.

- $D_{\tt mse}=\|x-\tilde{x}\|_2^2$
- per-coordinate MSE
- inner-product squared error
- 원본/복원 코사인 유사도

![TurboQuant mse simulation](results/8-1.TurboQuant_mse_simulation_fix.png)

### 7. TurboQuant\_prod Simulation

차원별로 다음을 비교했습니다.

- $D_{\tt mse}$ 분포
- per-coordinate MSE 분포
- $D_{\tt prod}$ 분포
- $D_{\tt prod}$ 하한/상한
- 원본/복원 코사인 유사도

![TurboQuant prod simulation](results/9-1.TurboQuant_prod_simulation_fix.png)

### 8. Final Comparison Across Bit Widths

비트폭 변화에 따라 $\mathrm{TurboQuant}_{\tt mse}$와 $\mathrm{TurboQuant}_{\tt prod}$를 비교했습니다.

- raw inner-product error 히스토그램
- squared inner-product error 히스토그램
- 평균 inner-product distortion + 하한/상한
- 평균 MSE + 하한/상한

![Final TurboQuant comparison](results/10.TurboQuant_final_simulation_fix.png)

## My Conclusion

구현/실험을 통해 본 핵심은 **랜덤 회전 관점**입니다. 회전 후 좌표 분포가 예측 가능해지면서, 고차원 벡터 문제를 원리 있는 scalar codebook 설계 문제로 바꿀 수 있습니다.

$\mathrm{TurboQuant}_{\tt mse}$는 재구성 관점에서 단순하고 효과적이었고, MSE 제어도 안정적으로 확인했습니다. 반면 inner product 측면은 MSE 최소화만으로 충분하지 않다는 점도 확인했습니다.

현재 실험에서는 $\mathrm{TurboQuant}_{\tt prod}$가 항상 $\mathrm{TurboQuant}_{\tt mse}$보다 좋아 보이지는 않았습니다. 구현 세부(랜덤성 처리/실험 셋업 등)에서 오차 요인이 있을 수 있어 추가 점검이 필요합니다.
