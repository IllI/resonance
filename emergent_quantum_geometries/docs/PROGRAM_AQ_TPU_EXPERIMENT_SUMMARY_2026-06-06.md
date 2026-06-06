# Program AQ TPU Experiment Summary

Date: 2026-06-06  
Experiment: SO(3) kernel-Galerkin pilot for Collins' truncated kernel theorem  
Script: `emergent_quantum_geometries/program_aq_so3_kernel_galerkin.py`  
Results:
- `emergent_quantum_geometries/program_aq_tpu_smoke/summary.json`
- `emergent_quantum_geometries/program_aq_tpu_pilot/summary.json`
- `emergent_quantum_geometries/program_aq_tpu_pilot/prog_aq_pilot.log`

## Hypotheses

**H1. TPU/JAX feasibility.** Program AQ can run on TRC Cloud TPUs using JAX on TPU devices, with the SO(3) experiment remaining on the accelerator backend.

**H2. Dataset choice.** Haar-uniform synthetic SO(3) nodes are the appropriate primary dataset for theorem-facing validation, because Collins' estimates are formulated in terms of fill distance, separation, and Haar-measure quadrature on SO(3).

**H3. Numerical theorem support.** The pilot sweep should show error behavior consistent with Collins' quadratized truncated Galerkin estimate:

$$
\left\|u - \tilde{u}^{\Lambda}_{\Xi}\right\|_{L^2}
\leq C
\left(
h_{\Xi}^{m-1}
+ h_{\Xi}^{-15/2-4m}N^{2-2m}
+ h_{\Xi}^{-7-2m}h_{\Lambda}^{m-1}
\right)
\|f\|_{H^s}.
$$

For the present pilot, $m=3$, so the leading Galerkin term is

$$
h_{\Xi}^{m-1} = h_{\Xi}^2.
$$

Since SO(3) has dimension $d=3$, a rough random-node scaling is

$$
h_{\Xi} \sim |\Xi|^{-1/3},
\qquad
h_{\Xi}^2 \sim |\Xi|^{-2/3}.
$$

If the pilot were in a clean Galerkin-dominated regime, the empirical slope of $\log(E)$ versus $\log(|\Xi|)$ would be near $-2/3$.

## Dataset And Operator

The experiment used synthetic Haar-uniform SO(3) samples:

$$
\varphi_1 \sim U[0,2\pi],
\qquad
\theta = \arccos(1-2u), \quad u \sim U[0,1],
\qquad
\varphi_2 \sim U[0,2\pi].
$$

The pilot used the Chapter 6 positive definite Sobolev/truncated-kernel path:

$$
\mu_{\ell} = (1+\ell(\ell+1))^m,
\qquad
\lambda_{\ell} = \mu_{\ell}^{-1}.
$$

The right-hand side was a Wigner-D basis element with test degree $L=2$, so the exact solution was

$$
u = \mu_L^{-1}D^L_{0,0}.
$$

This is not yet the closed-form conditionally positive definite rotational surface spline experiment. It is the positive definite truncated/Sobolev-kernel pilot needed to exercise the Chapter 6 pipeline.

## TPU Setup

The run followed the TRC email guidance:

- Project: `time-emission`
- Zone: `us-east1-d`
- TPU type: `v6e-8`
- Provisioning: spot
- Runtime: `v2-alpha-tpuv6e`
- Allocation method: Queued Resource API
- Cleanup: TPU VM and queued resource deleted after result sync

JAX confirmed the accelerator backend:

```text
jax 0.6.2
backend tpu
devices: 8 TPU devices
```

Both summary files record:

```json
"jax_backend": "tpu"
```

and list eight TPU devices.

## Commands

Smoke test:

```bash
python3 -u program_aq_so3_kernel_galerkin.py \
  --smoke-test \
  --require-accelerator \
  --out-dir program_aq_tpu_smoke
```

Pilot sweep:

```bash
python3 -u program_aq_so3_kernel_galerkin.py \
  --dataset-source haar \
  --operator sobolev \
  --aux-dim 0 \
  --n-centers 100 200 500 \
  --N-truncation 3 5 \
  --oversampling-ratios 5 10 \
  --test-degree 2 \
  --n-test 500 \
  --require-accelerator \
  --out-dir program_aq_tpu_pilot
```

## Results

Smoke result:

| $|\Xi|$ | $N$ | $q=|\Lambda|/|\Xi|$ | $|\Lambda|$ | $L^2$ error |
|---:|---:|---:|---:|---:|
| 50 | 3 | 5 | 250 | 1.310949e-01 |

Pilot sweep:

| $|\Xi|$ | $N$ | $q$ | $|\Lambda|$ | $L^2$ error |
|---:|---:|---:|---:|---:|
| 100 | 3 | 5 | 500 | 2.574280e-03 |
| 100 | 3 | 10 | 1000 | 3.199105e-03 |
| 100 | 5 | 5 | 500 | 1.211589e+00 |
| 100 | 5 | 10 | 1000 | 1.300661e+00 |
| 200 | 3 | 5 | 1000 | 3.131289e-02 |
| 200 | 3 | 10 | 2000 | 1.451272e-02 |
| 200 | 5 | 5 | 1000 | 1.001018e+00 |
| 200 | 5 | 10 | 2000 | 2.030323e-02 |
| 500 | 3 | 5 | 2500 | 9.145744e-03 |
| 500 | 3 | 10 | 5000 | 2.837914e-02 |
| 500 | 5 | 5 | 2500 | 1.263060e-01 |
| 500 | 5 | 10 | 5000 | 4.246209e-02 |

Empirical slopes of $\log(E)$ versus $\log(|\Xi|)$:

| $N$ | $q$ | observed slope | pure $h_{\Xi}^2$ expectation |
|---:|---:|---:|---:|
| 3 | 5 | +0.676 | -0.667 |
| 3 | 10 | +1.324 | -0.667 |
| 5 | 5 | -1.449 | -0.667 |
| 5 | 10 | -1.973 | -0.667 |

Oversampling comparison:

| $N$ | $|\Xi|$ | $E(q=10)/E(q=5)$ | Interpretation |
|---:|---:|---:|---|
| 3 | 100 | 1.243 | More quadrature nodes increased error |
| 3 | 200 | 0.463 | More quadrature nodes reduced error |
| 3 | 500 | 3.103 | More quadrature nodes increased error |
| 5 | 100 | 1.074 | More quadrature nodes increased error |
| 5 | 200 | 0.020 | More quadrature nodes strongly reduced error |
| 5 | 500 | 0.336 | More quadrature nodes reduced error |

## Interpretation

H1 is supported. The experiment ran on a TRC TPU VM with JAX reporting `backend=tpu` and eight TPU devices. The `--require-accelerator` guard passed, so the run did not silently fall back to CPU.

H2 is supported for theorem validation. Haar-uniform SO(3) nodes are the best primary dataset for this experiment. Real pose datasets may become useful later, but they introduce non-uniform sampling and application-specific pose noise. ModelNet40-C is not an SO(3) quadrature dataset; it is a point-cloud corruption benchmark.

H3 is not supported by this pilot. The result does not show stable convergence consistent with the expected $|\Xi|^{-2/3}$ Galerkin-dominated slope. The observed slopes are inconsistent in sign and magnitude, and the oversampling response is not monotone.

This does not falsify Collins' theorem. It rejects the narrower implementation-level hypothesis that the current pilot configuration already numerically verifies the theorem.

## Likely Causes

The pilot is not in the theorem-scale asymptotic regime. Fixed oversampling ratios $q=5$ and $q=10$ are engineering pilots, not the condition from Remark 6.6.9. For Chapter 6, the theorem-scale requirement is

$$
h_{\Lambda} \leq h_{\Xi}^{p},
\qquad
p = 3 + \frac{9}{m-1}.
$$

For $m=3$,

$$
p = 7.5.
$$

With fixed $q$, we have

$$
h_{\Lambda} \sim q^{-1/3}h_{\Xi},
$$

which does not approach $h_{\Xi}^{7.5}$ as $|\Xi|$ grows.

The dense collocation and stiffness solves are also likely ill-conditioned for some $(N,|\Xi|,q)$ settings, especially the small-center $N=5$ runs where errors exceeded $1$. The current quadrature weights are computed with projected gradient descent and a regularized objective, not a verified positive quadrature rule satisfying the lower-bound assumptions used in Collins' proof.

## Verdict

The experiment supports TPU feasibility and the use of Haar SO(3) nodes as the correct primary dataset. It does not yet support the numerical theorem-verification hypothesis.

Current verdict:

$$
\text{TPU/JAX feasibility: supported}
$$

$$
\text{Dataset choice: supported}
$$

$$
\text{Current numerical verification of Corollary 6.6.8: not supported}
$$

The next run should target numerical stability before scale: repeated seeds, condition-number diagnostics, residual norms, quadrature moment errors, and a comparison against an exact spectral projection baseline. Only after those diagnostics stabilize should we attempt larger TPU sweeps or theorem-scale oversampling.
