# Program AQ: Numerical Completion of Collins' SO(3) Kernel-Galerkin Theorem

> **Status:** Implementation Plan — For Review by Patrick Collins  
> **Date:** June 2026  
> **Repository:** `emergent_quantum_geometries/kernel/`  
> **TPU Access:** Google TRC — 64× v6e chips (`us-east1-d`, spot)

---

## 1. The Unfinished Theorem

Patrick Collins' 2021 dissertation (*Kernel-Based Galerkin Methods on Compact Manifolds Without Boundary, with an Emphasis on SO(3)*, University of Hawai'i at Mānoa) establishes a complete theoretical framework for mesh-free, coordinate-free PDE solvers on $\mathrm{SO}(3)$. The theory culminates in **Corollary 6.6.8** and **Remark 6.6.9** — results that were proved analytically but never numerically validated due to computational constraints.

The dissertation ends with the final chapter labeled `??` and the remark:

> *"We also must keep in mind the upper bound for $h_\Lambda$ from Lemma 4.1.25... If indeed the quadrature weights satisfy the lower bound discussed in Remark 4.1.26, then that upper bound is not needed, and the error estimates in Corollary 6.6.8 alone tell us how to choose $h_\Lambda$."*

The goal of **Program AQ** is to provide the first numerical verification of Corollary 6.6.8 and Remark 6.6.9 using Google Cloud TPUs, completing what could not be done on 2021 hardware.

---

## 2. Theoretical Setup

### 2.1 The PDE and Weak Formulation

We consider the elliptic PDE on $\mathrm{SO}(3)$:

$$\mathcal{L} u = f, \qquad \mathcal{L} = (-\Delta)^m + \text{lower order terms}$$

where $\Delta$ denotes the Laplace–Beltrami operator on $\mathrm{SO}(3)$ (dimension $d = 3$), and $f \in H^s(\mathrm{SO}(3))$ is a given right-hand side. The weak form seeks $u \in H^m(\mathrm{SO}(3))$ such that:

$$a(u, v) = \lambda_f(v) \quad \forall\, v \in H^m(\mathrm{SO}(3))$$

where the bilinear form $a(\cdot, \cdot)$ incorporates both the principal part (via the Riemannian gradient $\nabla$) and lower-order terms with bounded coefficients $a^\sharp$ and $b$.

### 2.2 Harmonic Analysis on SO(3)

The Laplace–Beltrami operator on $\mathrm{SO}(3)$ admits the **Wigner-D functions** $D^\ell_{j,k}$ as eigenfunctions:

$$-\Delta D^\ell_{j,k} = \lambda_\ell \, D^\ell_{j,k}, \qquad \lambda_\ell = \ell(\ell+1), \quad \ell \in \mathbb{N},\quad j,k \in \{-\ell, \ldots, \ell\}$$

In Euler angles $(\varphi_1, \theta, \varphi_2)$ via the ZXZ decomposition $x = R_z(\varphi_1) R_x(\theta) R_z(\varphi_2)$:

$$D^\ell_{j,k}(\varphi_1, \theta, \varphi_2) = e^{-ij\varphi_1}\, d^\ell_{j,k}(\theta)\, e^{-ik\varphi_2}$$

where $d^\ell_{j,k}$ are the **Wigner small-d matrix elements** (given by Jacobi polynomials). The Haar measure on $\mathrm{SO}(3)$ in these coordinates is:

$$\int_{\mathrm{SO}(3)} f \, d\mu = \frac{1}{8\pi^2} \int_0^{2\pi} \int_0^{\pi} \int_0^{2\pi} f(\varphi_1, \theta, \varphi_2) \sin\theta \, d\varphi_1 \, d\theta \, d\varphi_2$$

The normalized Wigner-D functions $\phi^\ell_{j,k} = \sqrt{2\ell+1}\, D^\ell_{j,k}$ form a complete orthonormal basis for $L^2(\mathrm{SO}(3))$, and by the Peter–Weyl theorem any $f \in L^2(\mathrm{SO}(3))$ expands as:

$$f = \sum_{\ell=0}^{\infty} \sum_{j,k=-\ell}^{\ell} \hat{f}^\ell_{j,k} \, \phi^\ell_{j,k}, \qquad \hat{f}^\ell_{j,k} = \langle f, \phi^\ell_{j,k} \rangle_{L^2}$$

### 2.3 The Truncated Kernel

The **exponential-rate kernel** $\kappa_m$ on $\mathrm{SO}(3)$ is given by its Hilbert–Schmidt series:

$$\kappa_m(x, y) = \sum_{\ell=0}^{\infty} \hat{\kappa}(\ell) \sum_{j,k=-\ell}^{\ell} \phi^\ell_{j,k}(x)\, \phi^\ell_{j,k}(y)$$

with Fourier coefficients $\hat{\kappa}(\ell) = (1 + \lambda_\ell)^{-m}= (1 + \ell(\ell+1))^{-m}$, so that the native space $\mathcal{N}_{\kappa_m}(\mathrm{SO}(3)) \cong H^m(\mathrm{SO}(3))$.

The **truncated kernel** $\tilde{\kappa}_m$ retains only degrees $\ell \leq N$:

$$\tilde{\kappa}_m(x, y) = \sum_{\ell=0}^{N} \hat{\kappa}(\ell) \sum_{j,k=-\ell}^{\ell} \phi^\ell_{j,k}(x)\, \phi^\ell_{j,k}(y)$$

This is a **finite-rank kernel** — the sum has $(N+1)^2$ distinct eigenvalues, each of multiplicity $2\ell+1$, giving a total of $\sum_{\ell=0}^{N}(2\ell+1)^2 = \frac{(N+1)(2N+1)(2N+3)}{3}$ basis functions.

**Key connection to Wesel & Batselier (2024):** Because the Wigner-D functions factorize over Euler angles as $D^\ell_{j,k}(\varphi_1,\theta,\varphi_2) = e^{-ij\varphi_1} d^\ell_{j,k}(\theta) e^{-ik\varphi_2}$, the truncated kernel is a **product kernel** (Definition 2.1 of Wesel–Batselier) over the three-dimensional Euler angle domain. Its associated kernel machine weights admit a natural **Tensor Train (TT) decomposition**, enabling TT-compressed stiffness matrix construction and TT-CG solves on TPU.

### 2.4 Kernel-Based Approximation Spaces

Given a finite center set $\Xi \subset \mathrm{SO}(3)$ of size $n = |\Xi|$, the approximation space is:

$$\tilde{V}_{\kappa_m, \Xi} = \mathrm{span}\{ \tilde{\kappa}_m(\cdot, \xi) : \xi \in \Xi \}$$

with associated **truncated Lagrange basis** $\{ \tilde{\chi}_\xi \}_{\xi \in \Xi}$ satisfying $\tilde{\chi}_\xi(\eta) = \delta_{\xi\eta}$ for $\xi, \eta \in \Xi$.

The **fill distance** of $\Xi$ measures the worst-case coverage of $\mathrm{SO}(3)$:

$$h_\Xi = \sup_{x \in \mathrm{SO}(3)} \min_{\xi \in \Xi} d(x, \xi)$$

where $d(\cdot,\cdot)$ is the geodesic distance on $\mathrm{SO}(3)$ (equivalently, $d(x,y) = \omega(y^{-1}x)$, the rotation angle of $y^{-1}x$). As $h_\Xi \to 0$, the approximation power of $\tilde{V}_{\kappa_m,\Xi}$ improves.

---

## 3. The Full Method and Its Error Estimate

### 3.1 The Quadratized Truncated Galerkin Approximation

The full discretized method uses **two** sets of points:

- $\Xi$ — Galerkin centers (size $n$, fill distance $h_\Xi$)
- $\Lambda$ — quadrature nodes (size $M \gg n$, fill distance $h_\Lambda \ll h_\Xi$)

**Step 1 — Truncated stiffness matrix:**

$$\tilde{B}_{\xi\eta} = a(\tilde{\chi}_\xi, \tilde{\chi}_\eta) = \int_{\mathrm{SO}(3)} \left[ a^\sharp(\nabla \tilde{\chi}_\xi, \nabla \tilde{\chi}_\eta) + b\, \tilde{\chi}_\xi \tilde{\chi}_\eta \right] d\mu \qquad \xi, \eta \in \Xi$$

**Step 2 — Quadratized truncated stiffness matrix** (replace integrals with quadrature rule $Q_\Lambda$ with weights $w_\lambda \geq 0$):

$$\tilde{B}^{\Lambda}_{\xi\eta} = Q_\Lambda\!\left( a^\sharp(\nabla \tilde{\chi}_\xi, \nabla \tilde{\chi}_\eta) + b\, \tilde{\chi}_\xi \tilde{\chi}_\eta \right) = \sum_{\lambda \in \Lambda} w_\lambda \left[ a^\sharp\!\left(\nabla \tilde{\chi}_\xi(\lambda), \nabla \tilde{\chi}_\eta(\lambda)\right) + b(\lambda)\, \tilde{\chi}_\xi(\lambda)\, \tilde{\chi}_\eta(\lambda) \right]$$

**Step 3 — Quadratized truncated load vector:**

$$\tilde{\omega}^\Lambda_\xi = Q_\Lambda(\tilde{\chi}_\xi \, f) = \sum_{\lambda \in \Lambda} w_\lambda \, \tilde{\chi}_\xi(\lambda)\, f(\lambda), \qquad \xi \in \Xi$$

**Step 4 — Solve the linear system** (size $n \times n$):

$$\tilde{B}^\Lambda_\Xi \, \tilde{\gamma}^\Lambda = \tilde{\omega}^\Lambda$$

**Step 5 — Quadratized truncated Galerkin approximation:**

$$\tilde{u}^\Lambda_\Xi = \sum_{\xi \in \Xi} \tilde{\gamma}^\Lambda_\xi \, \tilde{\chi}_\xi$$

### 3.2 The Master Error Estimate: Corollary 6.6.8

This is the theorem Collins proved but never numerically verified. Let $m > 5/2$ and $s = m-1$ if $5/2 < m \leq 7/2$, or $s = m-2$ if $m > 7/2$. For $f \in H^s(\mathrm{SO}(3))$, there exists a constant $C > 0$ such that for $h_\Xi$ sufficiently small:

$$\boxed{ \| u - \tilde{u}^\Lambda_\Xi \|_{L^2(\mathrm{SO}(3))} \leq C \left( \underbrace{h_\Xi^{m-1}}_{\text{Galerkin}} + \underbrace{h_\Xi^{-15/2 - 4m} \cdot N^{2-2m}}_{\text{Truncation}} + \underbrace{h_\Xi^{-7-2m} \cdot h_\Lambda^{m-1}}_{\text{Quadratization}} \right) \| f \|_{H^s} }$$

This bound has three independently controlled contributions:

| Error Term | Governing Parameter | Shrinks When |
|---|---|---|
| $h_\Xi^{m-1}$ | Galerkin center density | $h_\Xi \to 0$ (more centers) |
| $h_\Xi^{-15/2-4m} \cdot N^{2-2m}$ | Wigner-D truncation degree $N$ | $N \to \infty$ faster than $h_\Xi^{-\frac{15/2+4m}{2m-2}}$ |
| $h_\Xi^{-7-2m} \cdot h_\Lambda^{m-1}$ | Quadrature oversampling | $h_\Lambda \leq h_\Xi^p$ (see §3.3) |

### 3.3 The Oversampling Condition: Remark 6.6.9

Collins' final mathematical result states that to balance the quadratization error against the Galerkin error, the quadrature fill distance must satisfy:

$$h_\Lambda \leq h_\Xi^p, \qquad p = 3 + \frac{9}{m-1}$$

In this regime, $h_\Xi^{-7-2m} \cdot h_\Lambda^{m-1} \leq h_\Xi^{-7-2m} \cdot h_\Xi^{(m-1)p} = h_\Xi^{m-1}$, so the quadratization error matches the Galerkin error and the full approximation achieves optimal rate $O(h_\Xi^{m-1})$.

**Example for $m = 3$:** $p = 3 + 9/2 = 7.5$, so quadrature nodes must satisfy $h_\Lambda \leq h_\Xi^{7.5}$. If $|\Xi| \sim h_\Xi^{-3}$ (since $\dim(\mathrm{SO}(3)) = 3$), then $|\Lambda| \sim h_\Lambda^{-3} \sim h_\Xi^{-22.5}$ — roughly $|\Xi|^{7.5}$ points. This is the computational explosion that made the experiment infeasible in 2021.

**Example for $m = 4$:** $p = 3 + 3 = 6$, so $|\Lambda| \sim |\Xi|^6$.

For practical computation we fix $h_\Xi$ at a moderate value and verify the three-term convergence separately, sweeping each parameter independently.

### 3.4 Truncation Parameter Lower Bound: Remark 6.4.15

To ensure the truncation error does not dominate the Galerkin error, the truncation degree $N$ must satisfy:

$$N \geq h_\Xi^{-\frac{5}{2} - \frac{23}{4m-4}}$$

For $m = 3$ this gives $N \geq h_\Xi^{-5/2 - 23/8} = h_\Xi^{-43/8} \approx h_\Xi^{-5.375}$. If $|\Xi| = 10{,}000$ (so $h_\Xi \approx |\Xi|^{-1/3} \approx 0.046$), this requires $N \geq 0.046^{-5.375} \approx 7$. At $|\Xi| = 100{,}000$, $N \geq 15$.

---

## 4. Connection to Wesel & Batselier (2024)

Wesel & Batselier prove that a TT-constrained kernel machine

$$f^{\mathrm{TT}}(x) = \langle R_1(\mathrm{ten}(\phi(x))), \mathrm{TT}(\mathrm{ten}(w)) \rangle_F$$

converges in distribution to a Gaussian Process with product kernel $k(x,x') = \prod_{d=1}^{D} \phi^{(d)}(x_d)^T \Lambda^{(d)} \phi^{(d)}(x'_d)$ as all TT ranks $R_1, \ldots, R_{D-1} \to \infty$ (Theorem 3.2).

**The bridge to Collins:** The Euler-angle factorization

$$D^\ell_{j,k}(\varphi_1, \theta, \varphi_2) = e^{-ij\varphi_1} \cdot d^\ell_{j,k}(\theta) \cdot e^{-ik\varphi_2}$$

expresses every Wigner-D basis function as a **rank-1 TT** (or CPD) in the three-dimensional Euler-angle feature space $(\varphi_1, \theta, \varphi_2) \in [0,2\pi] \times [0,\pi] \times [0,2\pi]$. The truncated kernel $\tilde{\kappa}_m$ is therefore a product kernel in the sense of Wesel–Batselier Definition 2.1:

$$\tilde{\kappa}_m(x, y) = \sum_{\ell=0}^{N} (1+\ell(\ell+1))^{-m} \cdot U_{2\ell}\!\left(\cos\!\tfrac{\omega(y^{-1}x)}{2}\right)$$

where the addition formula $\sum_{j,k} D^\ell_{j,k}(x) D^\ell_{j,k}(y) = U_{2\ell}(\cos(\omega(y^{-1}x)/2))$ collapses the sum. In the feature-space representation, the quadratized stiffness matrix $\tilde{B}^\Lambda_\Xi$ has natural **TT structure** with bond dimensions equal to the number of Wigner-D functions per degree $(2\ell+1)$.

This means:
1. The $n \times n$ stiffness matrix can be stored in **TT-compressed form** with $O(N \cdot n)$ parameters instead of $O(n^2)$.
2. The Galerkin linear system $\tilde{B}^\Lambda_\Xi \tilde{\gamma}^\Lambda = \tilde{\omega}^\Lambda$ can be solved via **TT-CG** (conjugate gradient in TT format).
3. The connection to GPs established in Wesel–Batselier gives a probabilistic interpretation: as $N \to \infty$, the truncated Galerkin approximation approaches the posterior mean of a GP with kernel $\kappa_m$.

**Convergence rate comparison (Corollaries 3.3–3.4 of Wesel–Batselier):** For the same number of model parameters $P$, TT converges to the GP at rate $O\!\left(\tfrac{MD}{P}\right)^{\!\frac{D-1}{4}}$ versus CPD's $O\!\left(\tfrac{MD}{P}\right)^{\!1/2}$. For $D = 3$ (Euler angles), TT converges at rate $P^{-1/2}$ compared to CPD's $P^{-1/2}$ — they are equal for $D=3$ but TT is preferred for its hierarchical structure capturing all rank-$(R^2)$ tensor interactions.

---

## 5. Implementation Plan: `program_aq_so3_kernel_galerkin.py`

### 5.1 Data Generation (No External Data Required)

All data is synthetically generated:

**SO(3) sampling** (uniform Haar measure via Euler angles):
$$\varphi_1 \sim \mathrm{Uniform}[0, 2\pi], \quad \theta \sim \arccos(1 - 2U),\; U \sim \mathrm{Uniform}[0,1], \quad \varphi_2 \sim \mathrm{Uniform}[0, 2\pi]$$

**Test PDE** (analytic ground truth): We use $f = D^L_{0,0}$ (a single Wigner-D function of degree $L$) as right-hand side. Since $\mathcal{L} D^\ell_{j,k} = (1 + \lambda_\ell)^m D^\ell_{j,k}$, the exact weak solution is:

$$u = \frac{1}{(1 + L(L+1))^m} D^L_{0,0}$$

This gives an analytic ground truth for $L^2$ error computation.

### 5.2 Wigner-D Computation

The Wigner small-d matrix elements satisfy the three-term recurrence (stable for large $\ell$):

$$d^\ell_{j,k}(\theta) = \sqrt{\frac{(\ell+k)!(\ell-k)!}{(\ell+j)!(\ell-j)!}} \left(\sin\tfrac{\theta}{2}\right)^{k-j} \left(\cos\tfrac{\theta}{2}\right)^{j+k} P^{(k-j,j+k)}_{\ell-k}(\cos\theta)$$

where $P^{(\alpha,\beta)}_n$ are Jacobi polynomials. In JAX, this is implemented via `jax.scipy.special` Jacobi polynomial evaluation, fully JIT-compiled and `vmap`-able over all quadrature points.

The **truncated kernel matrix** at $n$ centers has entries:

$$\tilde{K}_{\xi\eta} = \tilde{\kappa}_m(\xi, \eta) = \sum_{\ell=0}^{N} (1+\ell(\ell+1))^{-m} \sum_{j,k=-\ell}^{\ell} \phi^\ell_{j,k}(\xi)\, \overline{\phi^\ell_{j,k}(\eta)}$$

Computed as: for each $\ell$, form the $n \times (2\ell+1)^2$ matrix of basis values, multiply by eigenvalue weight, add to $\tilde{K}$. Total FLOPs: $O(N^3 \cdot n^2)$.

### 5.3 Quadrature Weight Optimization

Collins §5.6: quadrature weights $\{w_\lambda\}_{\lambda \in \Lambda}$ are found by minimizing $\|w\|_{\ell^2}$ subject to exact integration of all Wigner-D functions of degree $\leq N_q$:

$$\min_{w \geq 0} \|w\|_2^2 \quad \text{s.t.} \quad \sum_{\lambda \in \Lambda} w_\lambda \phi^\ell_{j,k}(\lambda) = \int_{\mathrm{SO}(3)} \phi^\ell_{j,k} \, d\mu = \delta_{\ell,0}\delta_{j,0}\delta_{k,0}$$

This is a **non-negative least-squares** problem of size $|\Lambda| \times (N_q+1)^2$ solved via JAX LSQR.

### 5.4 Convergence Experiment

Sweep the following parameter grid:

| Parameter | Values | Role |
|---|---|---|
| $\|\Xi\|$ (Galerkin centers) | 500, 1000, 2000, 5000, 10000 | Controls $h_\Xi$ |
| $N$ (Wigner-D degree cutoff) | 5, 10, 15, 20, 30 | Truncation error |
| $q = h_\Xi / h_\Lambda$ (oversampling ratio) | 2, 4, 8, 16 | Quadratization error |
| $m$ (kernel smoothness order) | 3, 4, 5 | Convergence rate exponent |

For each parameter combination, compute:
1. $\|\tilde{u}^\Lambda_\Xi - u\|_{L^2}$ via quadrature (using a held-out evaluation set)
2. Log-log slope of error vs $h_\Xi$ to verify the predicted rate $m-1$
3. Log-log slope of error vs $N$ to verify the predicted rate $2m-2$
4. Log-log slope of error vs $h_\Lambda$ to verify the predicted rate $m-1$

### 5.5 Predicted Convergence Rates to Verify

From Corollary 6.6.8, fixing two parameters and varying the third:

$$\log \|u - \tilde{u}^\Lambda_\Xi\|_{L^2} \approx \begin{cases} (m-1) \log h_\Xi + C_1 & \text{Galerkin rate (fix } N, h_\Lambda\text{)} \\ (2-2m) \log N + C_2 & \text{Truncation rate (fix } h_\Xi, h_\Lambda\text{)} \\ (m-1) \log h_\Lambda + C_3 & \text{Quadratization rate (fix } h_\Xi, N\text{)} \end{cases}$$

For $m = 3$: predicted slopes are $2$, $-4$, and $2$ respectively.

### 5.6 TPU Configuration

Per TRC operational rules (`TRC_TPU_RULES.md`):

```
Zone:     us-east1-d
Type:     v6e-8  (8 chips, spot)
Runtime:  v2-alpha-tpuv6e
Flag:     --spot
Teardown: immediate after run
Output:   summary.json only (< 5 KB)
```

JAX parallelization strategy:
- `jax.pmap` over the 8 chips for parallel Wigner-D evaluation across quadrature points
- `jax.vmap` over center pairs for stiffness matrix assembly
- CG solver via `jax.scipy.sparse.linalg.cg`

### 5.7 Output Format

The experiment writes a compact `summary.json`:

```json
{
  "corollary_688_verification": {
    "m": 3,
    "galerkin_rate": {"predicted": 2.0, "observed": null},
    "truncation_rate": {"predicted": -4.0, "observed": null},
    "quadratization_rate": {"predicted": 2.0, "observed": null}
  },
  "remark_669_verification": {
    "oversampling_exponent_p": {"predicted": 7.5, "threshold_verified": null}
  },
  "timing_seconds": null,
  "n_galerkin_max": 10000,
  "N_truncation_max": 30
}
```

---

## 6. Open Questions for Patrick Collins

> **Q1 — Smoothness order:** The dissertation proves the bound for general $m > 5/2$. Should we start with $m = 3$ (simplest case with integer smoothness, so $\hat{\kappa}(\ell) = (1 + \ell(\ell+1))^{-3}$) or is there a value of $m$ that Collins thinks is most physically/numerically illustrative?

> **Q2 — Quadrature weight positivity:** Remark 4.1.26 discusses whether the optimal quadrature weights satisfy $w_\lambda \geq 0$. Collins notes that positivity is *expected* at large oversampling but not proven. Should the experiment specifically test the positivity threshold, i.e., find the minimum oversampling ratio $q^* = |\Lambda|/|\Xi|$ at which $\min_\lambda w_\lambda \geq 0$?

> **Q3 — The rotational surface splines:** Chapter 5 also develops the **rotational surface spline** kernel $\Phi(x,y) = \sin^3(\omega(y^{-1}x)/2)$ which has a closed-form expression. Should we run a parallel experiment with this kernel (algebraic energy estimate, polynomial convergence rate) versus the truncated exponential-rate kernel (exponential energy estimate)? This would demonstrate the improvement from using the harder-to-compute kernel.

> **Q4 — Missing final chapter:** The dissertation references a final chapter `??` discussing future directions. We can implement whatever experiments were intended for that chapter if Collins can describe what they were.

---

## 7. References

1. **Collins, P.** (2021). *Kernel-Based Galerkin Methods on Compact Manifolds Without Boundary, with an Emphasis on SO(3)*. Ph.D. dissertation, University of Hawai'i at Mānoa. [paper.md in this directory]

2. **Wesel, F. & Batselier, K.** (2024). *Tensor Network-Constrained Kernel Machines as Gaussian Processes*. arXiv:2403.19500v1 [cs.LG]. [2403.19500v1[1].md in this directory]

3. **Narcowich, F.J., Rowe, S.T., & Ward, J.D.** (2017). A novel Galerkin method for solving PDEs on the sphere using highly localized kernel bases. *Math. Comp.*, 86(303):197–231. [Collins' primary inspiration; ref [22] in paper.md]

4. **Hangelbroek, T. & Schmid, D.** (2011). Surface spline approximation on SO(3). *Appl. Comput. Harmon. Anal.*, 31(2):169–184. [ref [17] in paper.md]

5. **Solin, A. & Särkka, S.** (2020). Hilbert space methods for reduced-rank Gaussian process regression. *Statistics and Computing*, 30(2):419–446. [Used by Wesel–Batselier for basis function expansion]

6. **Schmid, D.** (2009). *Scattered Data Approximation on the Rotation Group and Generalizations*. Berichte aus der Mathematik, Shaker. [ref [26] in paper.md — primary reference for SO(3) harmonic analysis]

---

*Document prepared for review by Patrick Collins, June 2026.*  
*Implementation target: Google Cloud TPU v6e, project `time-emission`, zone `us-east1-d`.*
