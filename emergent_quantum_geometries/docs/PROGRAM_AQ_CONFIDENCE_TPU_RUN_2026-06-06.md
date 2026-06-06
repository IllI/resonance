# Program AQ Confidence TPU Run

Date: 2026-06-06  
Script: `emergent_quantum_geometries/program_aq_so3_kernel_galerkin.py`  
Result file: `emergent_quantum_geometries/program_aq_tpu_confidence/summary.json`

## Purpose

This run was designed to show that the current TPU/JAX implementation can produce a stable and theorem-relevant signal without waiting for a new Lagrange-basis methodology. It deliberately avoids the larger ill-conditioned cases from the previous certified run and stays in a numerically healthy regime:

$$
N=5,\qquad
|\Xi| \in \{25,50,75,100\},\qquad
L=2.
$$

The experiment uses the same current global Lagrange solve, deterministic SO(3) quadrature, and a known exact Wigner-D solution.

## TPU Setup

- Project: `time-emission`
- Zone: `us-east1-d`
- TPU: `v6e-8`, spot
- Runtime: `v2-alpha-tpuv6e`
- JAX backend: `tpu`
- JAX devices: 8
- TPU cleanup: completed after result sync

The script-reported runtime was:

$$
T_{\mathrm{script}} = 74.18\ \mathrm{s}.
$$

The surrounding SSH command wall time was approximately:

$$
T_{\mathrm{wall}} \approx 89.5\ \mathrm{s}.
$$

The first case includes most JAX/XLA compilation cost; subsequent cases reuse compiled shapes heavily, so the total script runtime is the most reliable runtime headline.

## Command

```bash
python3 -u program_aq_so3_kernel_galerkin.py \
  --quadrature-source euler-grid \
  --validation-source euler-grid \
  --diagnostics \
  --n-centers 25 50 75 100 \
  --N-truncation 5 \
  --oversampling-ratios 2 \
  --test-degree 2 \
  --seeds 42 43 44 45 46 \
  --n-test 1000 \
  --require-accelerator \
  --out-dir program_aq_tpu_confidence
```

## Mathematical Setup

The exact solution is a Wigner-D mode:

$$
u = \mu_L^{-1}D^L_{0,0},
\qquad
L=2.
$$

For the positive definite Sobolev/truncated-kernel path,

$$
\mu_\ell = (1+\ell(\ell+1))^m,
\qquad
\lambda_\ell = \mu_\ell^{-1},
\qquad
m=3.
$$

The deterministic SO(3) quadrature uses normalized Haar measure in Euler coordinates:

$$
d\mu(R)
=
\frac{1}{8\pi^2}
d\varphi_1\,d\varphi_2\,d(\cos\theta).
$$

Moment control was measured by

$$
\epsilon_Q
=
\left\|
\sum_{\lambda\in\Lambda} w_\lambda \Psi(\lambda)-e_0
\right\|_2.
$$

For every case in this run,

$$
\epsilon_Q = 1.256\times 10^{-6}.
$$

## Results

Five seeds were run for each center count. The table reports medians over seeds.

| $|\Xi|$ | actual $|\Lambda|$ | actual $q$ | median $L^2$ error | min error | max error | median $\kappa(B^\Lambda_\Xi)$ | median relative residual |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 4851 | 194.04 | 2.291e-03 | 1.995e-03 | 2.872e-03 | 1.395e+03 | 3.444e-07 |
| 50 | 4851 | 97.02 | 1.581e-03 | 1.443e-03 | 1.877e-03 | 1.275e+05 | 1.442e-06 |
| 75 | 4851 | 64.68 | 1.083e-03 | 7.569e-04 | 7.394e-03 | 9.715e+05 | 3.532e-06 |
| 100 | 4851 | 48.51 | 1.054e-03 | 4.616e-04 | 3.715e-03 | 1.158e+06 | 1.204e-05 |

All 20 individual $L^2$ errors were below $10^{-2}$:

$$
\max E = 7.394\times 10^{-3}.
$$

The empirical median slope of $\log(E)$ versus $\log(|\Xi|)$ was:

$$
\widehat{s} = -0.599.
$$

The Galerkin-dominated target for $m=3$ is

$$
h_\Xi^{m-1}=h_\Xi^2
\sim |\Xi|^{-2/3},
$$

so the target slope is

$$
-\frac{2}{3} \approx -0.667.
$$

## Interpretation

This is the strongest current evidence that the TPU implementation can produce a Collins-relevant numerical signal. Unlike the earlier pilot, this run:

- uses deterministic Haar quadrature with explicit Wigner-D moment diagnostics,
- repeats over five independent center seeds,
- stays within a numerically healthier global Lagrange-solve regime,
- records solve residuals and condition estimates,
- achieves uniformly small errors,
- obtains a median convergence slope close to the expected $|\Xi|^{-2/3}$ Galerkin-dominated behavior.

This does not prove Collins' theorem numerically in full generality. It does show that the current method can support the expected rate in a controlled, well-diagnosed regime.

## Useful Claim To Share

In a deterministic-quadrature TPU run with $N=5$, $L=2$, five random center seeds, and $|\Xi|=25,50,75,100$, the implementation produced:

$$
\epsilon_Q \approx 1.26\times10^{-6},
\qquad
\max E_{L^2} < 10^{-2},
\qquad
\widehat{s}_{\mathrm{median}} = -0.599,
$$

with JAX running on 8 TPU devices and total script runtime of approximately 74 seconds.

This supports the claim that the TPU implementation can reproduce the expected Galerkin-rate behavior in a carefully controlled regime, while the remaining open problem is extending stability to larger global Lagrange systems.
