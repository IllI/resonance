# Program AQ: Numerical Completion of Collins' SO(3) Kernel-Galerkin Theorem

> **Status:** Implementation Plan — For Review by Patrick Collins
> **Date:** June 2026 | **TPU:** Google TRC, 64× v6e (`us-east1-d`, spot)

---

## 1. The Unfinished Theorem

Collins (2021) proves **Corollary 6.6.8** and **Remark 6.6.9** analytically — the master $L^2$ error estimate for the quadratized truncated Galerkin approximation on $\mathrm{SO}(3)$ — but never runs the numerical validation. The final chapter is labeled `??`. Program AQ provides the first numerical verification using Cloud TPUs.

---

## 2. Kernel Selection: Rotational Surface Splines

### 2.1 Conditionally Positive Definite Kernels and the Auxiliary Space

The kernel shown in the screencap is the **rotational surface spline** (Collins §5.5, Example 5.5.1). For integer $m \geq 3$:

$$\Phi_m(x, y) := \left(\sin\frac{\omega(y^{-1}x)}{2}\right)^{2m-3}$$

where $\omega(g)$ denotes the rotation angle of $g \in \mathrm{SO}(3)$.

This kernel is **conditionally positive definite (CPD)** with respect to the auxiliary space

$$\Pi_{m-2} = \mathrm{span}\left\{ D^\ell_{j,k} : \ell \leq m-2,\; j,k \in \{-\ell,\ldots,\ell\} \right\}$$

i.e., the span of all Wigner-D functions of degree $\leq m-2$. The dimension of this auxiliary space is $\dim(\Pi_{m-2}) = (m-1)^2$.

**Why CPD requires the polynomial term:** $\Phi_m$ has Fourier coefficients (Collins §5.5, ref [17]):

$$\hat{\phi}_m(\ell) = \frac{(2m-2)!}{\pi(-4)^{m-1}(2\ell+1)^{m-1}} \prod_{j=0}^{m-1} \frac{1}{\ell(\ell+1) - \left(j^2 - \tfrac{1}{4}\right)}$$

These coefficients are **negative** for small $\ell \leq m-2$ (due to sign of $\ell(\ell+1) - (j^2 - 1/4)$ for small $\ell$), so $\Phi_m$ is not positive definite outright. Restricting to vectors $\alpha$ that annihilate $\Pi_{m-2}|_\Xi$ restores positivity, making $\Phi_m$ CPD w.r.t. $\Pi_{m-2}$.

The coefficients satisfy (2.8.1) with $\tau = m - \tfrac{1}{2}$, giving native space $\mathcal{N}_{\Phi_m} \cong H^{m-1/2}(\mathrm{SO}(3))$.

### 2.2 The Augmented Collocation System

For a $\Pi_{m-2}$-unisolvent center set $\Xi$ of size $n$, the interpolant is:

$$s = \sum_{\xi \in \Xi} a_\xi \Phi_m(\cdot, \xi) + \sum_{\ell=0}^{m-2} \sum_{j,k=-\ell}^{\ell} b^\ell_{jk}\, D^\ell_{j,k}$$

with coefficients determined by the augmented linear system:

$$K_{\Xi,m-2} \begin{pmatrix} a \\ b \end{pmatrix} = \begin{pmatrix} c \\ 0 \end{pmatrix}, \qquad K_{\Xi,m-2} = \begin{pmatrix} K_\Xi & P \\ P^T & 0 \end{pmatrix}$$

where $K_\Xi = [\Phi_m(\xi,\eta)]_{\xi,\eta \in \Xi}$ and $P = [D^\ell_{j,k}(\xi)]_{\xi \in \Xi,\, \ell \leq m-2}$ is the Vandermonde-type matrix of low-degree Wigner-D evaluations. Its column count is $\dim(\Pi_{m-2}) = \sum_{\ell=0}^{m-2}(2\ell+1)^2$.

**We choose $m = 3$** for our primary experiment:
- Auxiliary space $\Pi_1$: constant + 9 degree-1 Wigner-D functions, $\dim(\Pi_1) = 4$ (since $(m-1)^2 = 4$; concretely the $\ell=0$ constant and $\ell=1$ functions $D^1_{j,k}$, $(2\cdot1+1)^2 = 9$, but $\dim(\Pi_1) = 1 + 9 = 10$... correction: $\dim(\Pi_L) = \sum_{\ell=0}^{L}(2\ell+1)^2$, so $\dim(\Pi_1) = 1 + 9 = 10$)
- Kernel: $\Phi_3(x,y) = \sin^3(\omega(y^{-1}x)/2)$, fully closed-form
- Sobolev smoothness: $\tau = m - 1/2 = 5/2$, so $\mathcal{N}_{\Phi_3} \cong H^{5/2}(\mathrm{SO}(3))$

### 2.3 The Differential Operator

The kernel $\Phi_m$ is the fundamental solution to the operator (Collins §5.5):

$$L_m = \frac{(2m-2)!}{\pi(-4)^{m-1}} \prod_{j=0}^{m-1} \left(\Delta - \left(j^2 - \tfrac{1}{4}\right)\right)$$

For $m = 3$:

$$L_3 = \frac{4!}{\pi \cdot 16} \bigl(\Delta + \tfrac{1}{4}\bigr)\bigl(\Delta - \tfrac{3}{4}\bigr)\bigl(\Delta - \tfrac{35}{4}\bigr)$$

This is the **PDE we solve**: $L_3 u = f$ on $\mathrm{SO}(3)$.

---

## 3. The Verification Setup: Solving a PDE with Known Exact Solution

### 3.1 Strategy

We choose the right-hand side $f$ to be a specific Wigner-D function so that the **exact solution $u$ is known analytically**. We then run the full quadratized Galerkin method and measure whether $\|u - \tilde{u}^\Lambda_\Xi\|_{L^2}$ decays at the rate predicted by Corollary 6.6.8.

### 3.2 The Exact Solution for Wigner-D Right-Hand Sides

Since each $D^\ell_{j,k}$ is an eigenfunction of $\Delta$ with eigenvalue $-\lambda_\ell = -\ell(\ell+1)$:

$$L_m D^\ell_{j,k} = \frac{(2m-2)!}{\pi(-4)^{m-1}} \prod_{j'=0}^{m-1} \bigl(-\ell(\ell+1) - j'^2 + \tfrac{1}{4}\bigr) \cdot D^\ell_{j,k} = \hat{\phi}_m(\ell)^{-1} D^\ell_{j,k}$$

Therefore, if $f = D^L_{0,0}$ (the class function of degree $L$), the **exact weak solution** is:

$$\boxed{u = \hat{\phi}_m(L) \cdot D^L_{0,0}}$$

where $\hat{\phi}_m(L)$ is the closed-form Fourier coefficient from §2.1. This gives an exact $u$ computable to machine precision, with $\|u\|_{L^2} = \hat{\phi}_m(L)$ and $\|u\|_{H^{m-1/2}} = \hat{\phi}_m(L) \cdot (1+L(L+1))^{(m-1/2)/2}$.

### 3.3 Exactness on the Auxiliary Space (Wigner-D Functions)

**Proposition (Exactness).** Let $\Phi_m$ be the rotational surface spline with auxiliary space $\Pi_{m-2}$. If the right-hand side $f = D^\ell_{j,k}$ with $\ell \leq m-2$, then the Galerkin approximation satisfies $\tilde{u}_\Xi = u$ exactly for any $\Pi_{m-2}$-unisolvent center set $\Xi$.

**Proof sketch.** Since $D^\ell_{j,k} \in \Pi_{m-2}$ and the interpolant always includes a $\Pi_{m-2}$ component, $u = \hat{\phi}_m(\ell) D^\ell_{j,k}$ lies in the approximation space $V_{\Phi_m, \Xi, m-2}$. The Galerkin solution is the unique best approximation, so it recovers $u$ exactly. $\square$

This means: for test degrees $L \leq m-2 = 1$, the error is zero by construction — a built-in sanity check. We use test degrees $L \geq m-1$ where non-trivial approximation error arises.

### 3.4 The Norm We Measure

The quantity of interest is:

$$E(\Xi, \Lambda, N) = \|u - \tilde{u}^\Lambda_\Xi\|_{L^2(\mathrm{SO}(3))}$$

where:
- $u = \hat{\phi}_m(L) \cdot D^L_{0,0}$ — the **known exact solution**
- $\tilde{u}^\Lambda_\Xi = \sum_{\xi \in \Xi} \tilde{\gamma}^\Lambda_\xi \, \tilde{\chi}_\xi$ — the **quadratized truncated Galerkin approximation**

Computed via quadrature on a held-out validation set $V$ (distinct from $\Xi$ and $\Lambda$):

$$E \approx \left( \sum_{v \in V} w_v \left| u(v) - \tilde{u}^\Lambda_\Xi(v) \right|^2 \right)^{1/2}$$

---

## 4. The Master Error Estimate: Corollary 6.6.8

With the rotational surface spline $\Phi_m$ (algebraic energy estimate), the analogous quadratized Galerkin error from Corollary 4.2.7 / Example 5.5.1 is:

$$\left\| u - u^\Lambda_\Xi \right\|_{L^2} \leq C \left( \underbrace{h_\Xi^{m-1}}_{\text{Galerkin}} + \underbrace{\left(\frac{h_\Lambda}{h_\Xi}\right)^m h_\Xi^{-11/2} h_\Lambda^{-5/2}}_{\text{Quadratization}} \right) \|f\|_{H^s}$$

with oversampling exponent (Remark 4.2.8):

$$p = 2 + \frac{19}{2m-5}$$

For $m=3$: $p = 2 + 19 = 21$. For $m=4$: $p = 2 + 19/3 \approx 8.3$.

For the **truncated** kernel $\tilde{\kappa}_m$ (Chapter 6 version), the full three-term estimate (Corollary 6.6.8) is:

$$\left\| u - \tilde{u}^\Lambda_\Xi \right\|_{L^2} \leq C \left( h_\Xi^{m-1} + h_\Xi^{-15/2-4m} N^{2-2m} + h_\Xi^{-7-2m} h_\Lambda^{m-1} \right) \|f\|_{H^s}$$

| Term | Meaning | Make small by |
|---|---|---|
| $h_\Xi^{m-1}$ | Galerkin error | Increase $|\Xi|$ |
| $h_\Xi^{-15/2-4m} N^{2-2m}$ | Truncation error | Increase $N$ with $|\Xi|$ |
| $h_\Xi^{-7-2m} h_\Lambda^{m-1}$ | Quadratization error | Oversample: $|\Lambda| \gg |\Xi|$ |

**Our target:** if this bound is small, the approximation was close enough. We verify by comparing to the known $u$.

For the Chapter 6 truncated experiment, Remark 6.6.9 gives the oversampling exponent

$$p = 3 + \frac{9}{m-1}$$

so for $m=3$, $p=7.5$. This is the exponent needed for the quadratization term to match the Galerkin term in the theorem.

---

## 5. Quadrature Setup: TPU Pilot and Theorem-Scale Oversampling

For smoke and pilot TPU runs we can use fixed ratios such as $|\Lambda| = 5|\Xi|, 10|\Xi|, 20|\Xi|$ to verify the implementation and gather empirical scaling. Since $\mathrm{SO}(3)$ has dimension $d=3$:

$$h_\Xi \sim |\Xi|^{-1/3}, \qquad h_\Lambda \sim (10\,|\Xi|)^{-1/3} = 10^{-1/3} h_\Xi \approx 0.464\, h_\Xi$$

For a fixed oversampling ratio $q=|\Lambda|/|\Xi|$, the ratio $h_\Lambda / h_\Xi \sim q^{-1/3}$ is fixed regardless of $|\Xi|$, so the Chapter 6 quadratization error term scales as:

$$h_\Xi^{-7-2m} \cdot (q^{-1/3} h_\Xi)^{m-1} = q^{-(m-1)/3} \cdot h_\Xi^{-8-m}$$

For $m=3$ this is $q^{-2/3} h_\Xi^{-11}$, while the Galerkin term is $h_\Xi^2$. Therefore fixed 10x oversampling is useful as an engineering pilot, but it is not the theorem-scale oversampling regime. To probe Remark 6.6.9 directly, drive $h_\Lambda$ according to $h_\Lambda \le h_\Xi^{7.5}$, subject to TPU memory.

---

## 6. Dataset Choice

The primary validation dataset should be generated synthetically from the Haar measure on $\mathrm{SO}(3)$. This is the best dataset for Collins' theorem because the theorem is stated in terms of SO(3) fill distance, separation, and Haar-measure quadrature, and because the exact Wigner-D solution is known.

External datasets are useful only after the theorem-facing run works. They can stress non-uniform coverage, symmetry, and real-world pose noise, but they should not replace Haar-uniform nodes for the mathematical validation.

| Dataset | Points | SO(3) Values | Use case |
|---|---|---|---|
| **Haar synthetic SO(3)** | arbitrary | exact generated rotations | Primary theorem validation |
| **SYMSOL / symmetric solids pose data** | large synthetic pose sets | random rotations with object symmetries | Best downstream ambiguity/symmetry stress test |
| **Pascal3D+ / BOP-style pose datasets** | thousands to millions | object pose rotations | Applied non-uniform $\Xi$ after pose extraction |
| **Cryo-EM particle orientations** (EMPIAR) | $10^4$–$10^6$ | SO(3) viewing angles | High-density quadrature test |
| **Hitchhiker's Guide** ([github](https://github.com/martius-lab/hitchhiking-rotations)) | Synthetic | Various SO(3) representations | Representation benchmarks |
| **ModelNet40-C** ([github](https://github.com/jiachens/ModelNet40-C)) | point-cloud corruption benchmark | not native SO(3) labels | Not a primary SO(3) node dataset |

**Recommendation:** Use Haar-measure uniform sampling for the mathematical validation. After that, use SYMSOL or pose-estimation datasets as realistic non-uniform stress tests. Treat ModelNet40-C as a point-cloud robustness benchmark, not as an SO(3) quadrature dataset.

---

## 7. Implementation: `program_aq_so3_kernel_galerkin.py`

### 7.1 Data Generation

Haar-uniform SO(3) samples via Euler angles:

$$\varphi_1 \sim U[0, 2\pi],\quad \theta = \arccos(1 - 2u),\; u \sim U[0,1],\quad \varphi_2 \sim U[0, 2\pi]$$

**Test PDE:** $L_3 u = f$ with $f = D^L_{0,0}$ for $L = 5, 10, 20$.

**Exact solution:** $u = \hat{\phi}_3(L) \cdot D^L_{0,0}$, computed analytically.

The current TPU-first script defaults to the Chapter 6 positive definite truncated/Sobolev kernel path:

```
--dataset-source haar --operator sobolev --aux-dim 0
```

This keeps the kernel coefficients and PDE operator paired as $\lambda_\ell = \mu_\ell^{-1}$, with $\mu_\ell = (1+\ell(\ell+1))^m$, so the known exact solution remains $u=\mu_L^{-1}D^L_{0,0}$.

### 7.2 Kernel and Stiffness Matrix

**Kernel evaluation** (closed form, no series needed):
$$\Phi_3(x, y) = \sin^3\!\left(\frac{\omega(y^{-1}x)}{2}\right) = \sin^3\!\left(\frac{1}{2}\arccos\frac{\mathrm{tr}(y^T x) - 1}{2}\right)$$

**Stiffness matrix entries** (via quadrature, §7.3):
$$B^\Lambda_{\xi\eta} = \sum_{\lambda \in \Lambda} w_\lambda \left[ a^\sharp\!\left(\nabla\chi_\xi(\lambda), \nabla\chi_\eta(\lambda)\right) + b(\lambda)\,\chi_\xi(\lambda)\chi_\eta(\lambda) \right]$$

The **Lagrange basis functions** $\chi_\xi$ for the CPD kernel satisfy the augmented system $K_{\Xi,1} [\alpha_\xi; \beta_\xi]^T = e_\xi$ (standard unit vector). Each $\chi_\xi$ evaluation at $\lambda$ costs $O(n + \dim\Pi_1)$ via the precomputed decomposition.

### 7.3 Quadrature Weights

Solve non-negative least squares on $|\Lambda| = 10n$ nodes to exactly integrate $\Pi_{N_q}$:

$$\min_{w \geq 0} \|w\|_2 \quad \text{s.t.} \quad \sum_{\lambda \in \Lambda} w_\lambda D^\ell_{j,k}(\lambda) = \delta_{\ell 0}\delta_{j0}\delta_{k0} \quad \forall\, \ell \leq N_q,\; |j|,|k| \leq \ell$$

System size: $10n \times (N_q+1)^2$. Solved via JAX LSQR.

### 7.4 Convergence Sweep

| Parameter | Values | Expected slope |
|---|---|---|
| $\|\Xi\|$ | 500, 1000, 2000, 5000, 10000 | $\log E \sim (m-1)\log h_\Xi$ |
| $N$ (truncation degree) | 5, 10, 15, 20 | $\log E \sim (2-2m)\log N$ |
| $\|\Lambda\|/\|\Xi\|$ | 5, 10, 20, 50 | $\log E \sim (m-1)\log h_\Lambda$ |

For $m=3$: predicted slopes $+2$, $-4$, $+2$ in log-log space.

**Sanity check:** For $L \leq 1$ (degree in auxiliary space $\Pi_1$), verify $E = 0$ to machine precision.

### 7.5 TPU Configuration

```
Zone:     us-east1-d  (TRC-covered)
Type:     v6e-8  (8 chips, spot, --spot flag)
Runtime:  v2-alpha-tpuv6e
Output:   summary.json only (<5KB, no large downloads)
Teardown: immediate after run
```

JAX strategy: `vmap` over center pairs for kernel matrix, `pmap` over 8 chips for parallel quadrature evaluation, `jax.scipy.sparse.linalg.cg` for the Galerkin solve.

---

## 8. Open Questions for Patrick Collins

**Q1 — Auxiliary space degree:** The screencap shows $k_m(x,\alpha) = \sin^{2m-3}(\omega(\alpha^{-1}x)/2)$. Collins §5.5 states $\Phi_m$ is CPD w.r.t. $\Pi_{m-2}$. For $m=3$ this is $\Pi_1$ ($\dim = 10$). Is this the auxiliary space Collins intended for the numerical experiments?

**Q2 — The differential operator $L_3$:** The operator $L_3 = \frac{4!}{16\pi}(\Delta + \tfrac{1}{4})(\Delta - \tfrac{3}{4})(\Delta - \tfrac{35}{4})$ is not the intuitive $(I-\Delta)^m$ but the natural one for $\Phi_m$. Should we also run the ideal kernel $\kappa_m$ experiment with $L = (I-\Delta)^m$ (positive definite, no auxiliary space) in parallel, as Collins compares the two in §5.5?

**Q3 — Non-uniform quadrature:** The EMPIAR cryo-EM orientation datasets have millions of non-uniform SO(3) samples. Would it be meaningful to use these as the quadrature set $\Lambda$ to test the method on a real-world coverage pattern?

**Q4 — Missing `??` chapter:** The dissertation references a final chapter `??` on future directions. The natural candidates are: (a) extending to other compact Lie groups, (b) adaptive refinement of $\Xi$, (c) time-dependent PDEs on $\mathrm{SO}(3)$. Which was intended?

---

## 9. References

1. **Collins, P.** (2021). *Kernel-Based Galerkin Methods on Compact Manifolds Without Boundary, with an Emphasis on SO(3)*. Ph.D. dissertation, UH Mānoa. [paper.md / paper.pdf]
2. **Wesel, F. & Batselier, K.** (2024). *TN-Constrained Kernel Machines as Gaussian Processes*. arXiv:2403.19500. [2403.19500v1[1].md]
3. **Hangelbroek, T. & Schmid, D.** (2011). Surface spline approximation on SO(3). *Appl. Comput. Harmon. Anal.*, 31(2):169–184. [ref [17] — Fourier coefficients of $\Phi_m$]
4. **Narcowich, Rowe & Ward** (2017). A novel Galerkin method on the sphere. *Math. Comp.*, 86:197–231. [ref [22] — sphere analog]
5. **Schmid, D.** (2009). *Scattered Data Approximation on the Rotation Group*. Shaker. [ref [26] — SO(3) harmonic analysis]
