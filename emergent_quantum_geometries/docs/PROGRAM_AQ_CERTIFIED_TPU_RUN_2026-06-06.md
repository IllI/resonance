# Program AQ Certified TPU Run

Date: 2026-06-06  
Script: `emergent_quantum_geometries/program_aq_so3_kernel_galerkin.py`  
Result file: `emergent_quantum_geometries/program_aq_tpu_certified/summary.json`

## What Changed

This run replaced the earlier learned/random quadrature path with deterministic SO(3) Euler-product quadrature and added theorem-facing diagnostics:

$$
d\mu(R)
=
\frac{1}{8\pi^2}
d\varphi_1\,d\varphi_2\,d(\cos\theta).
$$

The quadrature uses uniform nodes in $\varphi_1,\varphi_2$ and Gauss-Legendre nodes in $x=\cos\theta$. It records Wigner-D moment error:

$$
\epsilon_Q(B)
=
\left\|
\sum_{\lambda\in\Lambda} w_\lambda \Psi_B(\lambda)
- e_0
\right\|_2.
$$

The sweep also records:

$$
\kappa(K_\Xi),\qquad
\kappa(B_\Xi^\Lambda),\qquad
\frac{\|B_\Xi^\Lambda\gamma-\omega^\Lambda\|_2}{\|\omega^\Lambda\|_2}.
$$

## TPU Run

Configuration:

- TRC zone: `us-east1-d`
- TPU: `v6e-8`, spot
- Runtime: `v2-alpha-tpuv6e`
- JAX backend: `tpu`
- Devices: 8 TPU devices
- Seeds: 42, 43, 44
- Centers: Haar-random SO(3)
- Quadrature: deterministic Euler grid
- Validation: deterministic Euler grid
- Operator: Sobolev/truncated positive definite path
- Test degree: $L=2$

Command:

```bash
python3 -u program_aq_so3_kernel_galerkin.py \
  --quadrature-source euler-grid \
  --validation-source euler-grid \
  --diagnostics \
  --n-centers 50 100 200 \
  --N-truncation 3 5 \
  --oversampling-ratios 2 \
  --test-degree 2 \
  --seeds 42 43 44 \
  --n-test 1000 \
  --require-accelerator \
  --out-dir program_aq_tpu_certified
```

The TPU VM and queued resource were deleted after result sync.

## Aggregated Results

Median errors over three seeds:

| $N$ | $|\Xi|$ | actual $|\Lambda|$ | median $L^2$ error | median $\kappa(B)$ | median $\epsilon_Q$ |
|---:|---:|---:|---:|---:|---:|
| 3 | 50 | 1183 | 1.345e-03 | 2.760e+04 | 5.582e-07 |
| 3 | 100 | 1183 | 9.076e-03 | $\infty$ | 5.582e-07 |
| 3 | 200 | 1183 | 1.975e-02 | $\infty$ | 5.582e-07 |
| 5 | 50 | 4851 | 1.501e-03 | 2.009e+05 | 1.256e-06 |
| 5 | 100 | 4851 | 1.054e-03 | 1.158e+06 | 1.256e-06 |
| 5 | 200 | 4851 | 2.432e-03 | $\infty$ | 1.256e-06 |

Median empirical slopes of $\log(E)$ versus $\log(|\Xi|)$:

| $N$ | observed slope | Galerkin-dominated target |
|---:|---:|---:|
| 3 | +1.938 | $-2/3$ |
| 5 | +0.348 | $-2/3$ |

## Interpretation

The deterministic quadrature is working: moment errors are at approximately $10^{-6}$ and weights remain positive and normalized. This is a major improvement over the previous random/learned quadrature pilot.

The remaining obstacle is no longer quadrature. The diagnostics point to conditioning and dense linear solves. For $|\Xi|\ge 100$ in the $N=3$ path and $|\Xi|=200$ in the $N=5$ path, the stiffness condition number is reported as infinite in float32. The solve residuals also grow in the problematic cases, especially for $N=3,|\Xi|=200$.

## Verdict

This run furthers Collins' research more directly than the first TPU pilot because it measures the theorem's numerical assumptions. It supports:

$$
\text{TPU execution: yes}
$$

$$
\text{Deterministic SO(3) quadrature moment control: yes}
$$

It still does not verify the asymptotic Collins error rate:

$$
\text{Current convergence-rate verification: no}
$$

The next useful research step is to replace the dense ill-conditioned Lagrange solve with a stabilized solver path: float64 where supported, spectral projection baseline, QR/SVD diagnostics, or regularized/iterative solves with explicit residual control.

## Postulates For The Lagrange Solve

Based on the certified run and nearby kernel/RBF literature, the next research moves are probably these:

1. **Localized Lagrange bases should be the first serious replacement for global dense Lagrange solves.** Fuselier, Hangelbroek, Narcowich, Ward, and Wright construct localized, small-footprint kernel bases on $S^2$ and prove stability/decay properties for surface-spline-type kernels. Collins' SO(3) setting is not identical, but the failure mode here is exactly what localized bases are meant to relieve: global bases create dense, increasingly ill-conditioned linear systems.

2. **A spectral basis baseline should be run before another kernel-basis scale-up.** The present Chapter 6 Sobolev/truncated-kernel path is already represented spectrally by Wigner-D functions. A direct truncated spectral Galerkin projection can give a reference error floor and isolate whether the pathology is in the PDE approximation or specifically in reconstructing through the kernel Lagrange basis.

3. **Stable-basis ideas from RBF-QR are relevant by analogy, even though SO(3) surface splines are not Gaussian RBFs.** The RBF-QR literature shows a recurring theme: the mathematical interpolation problem may remain well posed while the direct kernel basis becomes numerically unusable. Our condition diagnostics fit that pattern. We should look for an SO(3) analogue of a basis change from kernel translates to a better-conditioned harmonic/Wigner-D representation.

4. **QR/SVD diagnostics should become part of every theorem-facing run.** Reporting only $E_{L^2}$ hides the real obstruction. For Collins, the useful artifact is likely a table of

   $$
   \sigma_{\min}(K_\Xi),\quad
   \sigma_{\max}(K_\Xi),\quad
   \kappa(K_\Xi),\quad
   \sigma_{\min}(B^\Lambda_\Xi),\quad
   \kappa(B^\Lambda_\Xi),\quad
   \frac{\|B^\Lambda_\Xi\gamma-\omega^\Lambda\|_2}{\|\omega^\Lambda\|_2}.
   $$

5. **If random centers remain in the loop, quasi-uniformity must be measured.** Collins' estimates assume geometric control through fill distance/separation. Haar-random centers are convenient, but not automatically quasi-uniform. A farthest-point or low-discrepancy SO(3) center construction may reduce the conditioning spikes now appearing at $|\Xi|=100,200$.

References worth asking Collins about:

- Fuselier, Hangelbroek, Narcowich, Ward, Wright, "Localized Bases for Kernel Spaces on the Unit Sphere" (SIAM J. Numer. Anal., 2013). The paper constructs localized, robust kernel bases for surface-spline spaces on $S^2$ and is the closest obvious analogue to the Lagrange-basis failure here. See [publisher/Boise State record](https://scholarworks.boisestate.edu/math_facpubs/123/) and [arXiv version](https://arxiv.org/abs/1205.3255).
- Narcowich, Rowe, Ward, "A Novel Galerkin Method for Solving PDEs on the Sphere Using Highly Localized Kernel Bases" (Math. Comp., 2017). This is especially relevant because it couples localized bases to a Galerkin PDE solve. See [arXiv version](https://arxiv.org/abs/1404.5263).
- Hangelbroek, Narcowich, Rieger, Ward, "Direct and Inverse Results on Bounded Domains for Meshless Methods via Localized Bases on Manifolds" (2014/2018). This extends the localized-basis story toward manifold approximation spaces. See [arXiv version](https://arxiv.org/abs/1406.1435).
- Hangelbroek, Narcowich, Ward, "Kernel Approximation on Manifolds I: Bounding the Lebesgue Constant" (SIAM J. Math. Anal., 2010). Relevant to stability of interpolation on compact manifolds. See [arXiv version](https://arxiv.org/abs/0909.0033).
- Fornberg and Piret, "A Stable Algorithm for Flat Radial Basis Functions on a Sphere" (SIAM J. Sci. Comput., 2007), and Fornberg, Larsson, Flyer, "Stable Computations with Gaussian Radial Basis Functions" (SIAM J. Sci. Comput., 2011). These are not SO(3) surface-spline papers, but they are useful cautionary examples: direct kernel bases can be the numerical problem even when the approximation problem is stable. See [SIAM page](https://epubs.siam.org/doi/10.1137/060671991) and [SIAM page](https://epubs.siam.org/doi/10.1137/09076756X).
