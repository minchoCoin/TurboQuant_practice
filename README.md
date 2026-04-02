# TurboQuant_practice

TurboQuant_practice

TurboQuant paper: [https://arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874)

## Codebook demo

Run:

```bash
python codebook.py
```

This creates a simple low-color image, quantizes it with several codebook sizes, and saves:

- `codebook_comparison.png`

The figure shows:

- the original image
- the quantized result for each codebook size
- the codebook palette colors used by each result

The script also prints the MSE and palette values for each codebook size so you can compare reconstruction quality.

## Lemma 1 Explanation

Lemma 1 in `TurboQuant/main.tex` states the following:

- if $x \in S^{d-1}$ is uniformly distributed on the unit sphere in $R^d$
- then any coordinate $x_j$ follows a Beta-type distribution on $[-1, 1]$

The formula in the paper is:

$$
f_X(x) =
\frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d-1)/2)}
\left(1-x^2\right)^{(d-3)/2}, \quad x \in [-1,1].
$$

The intuition is that if we fix one coordinate of a random point on the sphere to $x_j = x$, then the remaining coordinates must lie on a lower-dimensional sphere of radius $\sqrt{1-x^2}$. So the probability of observing a given coordinate value is determined by the size of that spherical cross-section.

## Definition of the Gamma Function

The Gamma function extends the factorial function to real and complex arguments:

$$
\Gamma(z) = \int_0^\infty t^{z-1} e^{-t} dt, \quad z > 0.
$$

Useful identities are:

- $\Gamma(z+1) = z \Gamma(z)$
- for any integer $n$, $\Gamma(n) = (n-1)!$
- $\Gamma(1/2) = \sqrt{\pi}$

This is why the Gamma function naturally appears in formulas for high-dimensional balls and spheres.

## Volume and Surface Area Formulas

The volume of a $d$-dimensional ball of radius $r$ is

$$
V_d(r) = \frac{\pi^{d/2}}{\Gamma(d/2 + 1)} r^d
$$

and the surface area of the sphere $S^{d-1}$ of radius $r$ is

$$
A_{d-1}(r) = \frac{2\pi^{d/2}}{\Gamma(d/2)} r^{d-1}.
$$

In particular, for radius $1$:

- unit ball volume: $V_d(1) = \pi^{d/2} / \Gamma(d/2 + 1)$
- unit sphere surface area: $A_{d-1}(1) = 2\pi^{d/2} / \Gamma(d/2)$

The Gamma function appears in Lemma 1 because the proof computes the size of spherical cross-sections, and those cross-sections are lower-dimensional spheres whose surface area is given by the formula above.

## Proof Sketch for Lemma 1

The proof in the paper uses a cross-sectional area argument.

1. Fix $x_j = x$. Then the remaining $d-1$ coordinates must satisfy
   $$
   x_1^2 + ... + x_{j-1}^2 + x_{j+1}^2 + ... + x_d^2 = 1 - x^2.
   $$

2. Therefore, the feasible set is a sphere in $R^{d-1}$ with radius $\sqrt{1-x^2}$ and dimension $d-2$.

3. Its surface area is
   $$
   A_{d-2}(\sqrt{1-x^2})
   =
   \frac{2\pi^{(d-1)/2}}{\Gamma((d-1)/2)}
   (1-x^2)^{(d-2)/2}.
   $$

4. The total sample space is the unit sphere $S^{d-1}$, whose surface area is
   $$
   A_{d-1}(1) = \frac{2\pi^{d/2}}{\Gamma(d/2)}.
   $$

5. The density is proportional to the ratio between the cross-sectional area and the total surface area. When expressing the density with respect to the coordinate $x$, an additional Jacobian factor
   $$
   \frac{1}{\sqrt{1-x^2}}
   $$
   appears. In the paper this is described as coming from the Pythagorean-theorem geometry.

6. Therefore,
   $$
   f_X(x)
   =
   \frac{
   \frac{2\pi^{(d-1)/2}}{\Gamma((d-1)/2)}
   (1-x^2)^{(d-2)/2}
   }{
   \frac{2\pi^{d/2}}{\Gamma(d/2)}
   }
   \cdot
   \frac{1}{\sqrt{1-x^2}}
   $$
   and simplifying gives
   $$
   f_X(x)
   =
   \frac{\Gamma(d/2)}{\sqrt{\pi}\,\Gamma((d-1)/2)}
   (1-x^2)^{(d-3)/2}.
   $$

So the key idea behind Lemma 1 is that the distribution of one coordinate of a random point on the sphere is determined by the size of the spherical slice at that coordinate value. As the dimension grows, this distribution becomes increasingly concentrated near zero, which is why it converges to $N(0, 1/d)$ in high dimensions.
