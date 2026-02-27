"""
MATHEMATICAL AUDIT: D-LinOSS fMRI Implementation
=================================================

This document identifies every mathematical error and methodological gap in
the current dlinoss_fmri.py implementation by comparing line-by-line against
the reference source code and papers.

Reference Implementations:
  [1] tk-rusch/linoss/LinOSS.py — Original LinOSS (ICLR 2025 Oral)
  [2] jaredbmit/damped-linoss/LinOSS.py — D-LinOSS (DampedIMEX1Layer, DampedIMLayer)
  [3] zroe1/toy-models-of-superposition — Superposition encoder
  [4] arXiv:2505.10465v4 — "Superposition Yields Robust Neural Scaling"

=============================================================================
ERROR #1 — WRONG RECURRENCE MATRIX (D-LinOSS)            [CRITICAL]
=============================================================================

The reference DampedIMEX1Layer (Eq. from the D-LinOSS paper) uses:

    z_{k+1} = z_k + dt * (-A * x_k - G * z_{k+1} + B * u_{k+1})
    x_{k+1} = x_k + dt * z_{k+1}

Solving for z_{k+1} from the first equation:
    z_{k+1} * (1 + dt*G) = z_k - dt*A*x_k + dt*B*u_{k+1}
    z_{k+1} = (z_k - dt*A*x_k + dt*B*u_{k+1}) / (1 + dt*G)

Substituting into the second:
    x_{k+1} = x_k + dt * z_{k+1}

The recurrence matrix M (from reference code, DampedIMEX1Layer._recurrence):
    S = 1 + dt * G
    M_11 = 1/S           (velocity-to-velocity)
    M_12 = -dt*A/S        (position-to-velocity)
    M_21 = dt/S           (velocity-to-position)
    M_22 = 1 - dt^2*A/S   (position-to-position)

OUR CODE (lines 184-188):  MATCHES.  ✓

The forcing terms in the reference:
    F1 = dt * (1/S) * Bu    (force on velocity)
    F2 = dt^2 * (1/S) * Bu  (force on position = dt * F1)

OUR CODE (lines 198-199):  MATCHES.  ✓

HOWEVER, there IS a DIFFERENT variant: DampedIMEX2Layer. That uses:
    z_{k+1} = z_k + dt * (-A*x_k - G*z_k + B*u_{k+1})  ← G acts on z_k, NOT z_{k+1}

    This gives M_11 = 1 - dt*G (explicit damping), NOT 1/(1+dt*G) (implicit).

    The reference has BOTH variants. Our code uses the IMEX1 (implicit) form, which is fine.


=============================================================================
ERROR #2 — WRONG SOFT PROJECTION FORMULA                 [CRITICAL]
=============================================================================

The reference DampedIMEX1Layer._soft_project_AGdt (chunk 3, line ~):
    dt = sigmoid(dt_raw)
    G = relu(G)

    A_low  = (2 + dt*G - 2*sqrt(1 + dt*G)) / max(dt^2, 1e-6)
    A_high = (2 + dt*G + 2*sqrt(1 + dt*G)) / max(dt^2, 1e-6)
    A = A_low + relu(A - A_low) - relu(A - A_high)

OUR CODE (lines 153-155):  MATCHES.  ✓

BUT WAIT — the reference validity check is:
    _is_valid_AGdt: (G >= 0) & ((G + dt*A)^2 - 4*A < 0)
                    equivalently: (G + dt*A)^2 < 4*A
                    i.e., G^2 + 2*G*dt*A + dt^2*A^2 < 4*A

The soft projection bounds derive from: eigenvalues of M are inside unit disk
iff A is in (A_low, A_high) where these bounds come from |trace^2 - 4*det| and
|eig| < 1.

The formula (2 + dt*G ± 2*sqrt(1 + dt*G)) / dt^2 is correct for the IMEX1
variant's stability region. Our code matches.  ✓


=============================================================================
ERROR #3 — NO GRADIENT-BASED TRAINING                    [CRITICAL]
=============================================================================

The reference implementation uses:
  - JAX autodiff (jax.grad) to compute exact gradients
  - AdamW optimizer with learning rate scheduling
  - Cross-entropy or MSE loss with proper backpropagation

OUR CODE: Has NO training loop. The SSM parameters (A, G, dt, B, C, D) are
initialized randomly and NEVER UPDATED via gradient descent.

The "update()" method on line 216-221 is a heuristic that adjusts G based on
A magnitude — this is NOT the gradient of any loss function. It's making up an
update rule that has no mathematical justification.

IMPACT: All D-LinOSS outputs are from a random, untrained model.
The "coherence" metric we compute is meaningless.

FIX REQUIRED: Implement proper forward-mode or finite-difference gradients
with a reconstruction loss, or use PyTorch/JAX autodiff.


=============================================================================
ERROR #4 — SUPERPOSITION ENCODER TRAINING ON WRONG DATA  [CRITICAL]
=============================================================================

The reference paper (arXiv:2505.10465v4, Section 2):
  - x ∈ R^n is a sparse vector where x_i = u_i * v_i
  - u_i ~ Bernoulli(p_i), v_i ~ Uniform(0, 1)
  - p_i ∝ 1/i^α (Zipf's law feature importance)
  - Loss: L = <||y - x||^2> averaged over x distribution
  - Training: AdamW with weight decay γ (γ < 0 for strong superposition)

OUR CODE (train_step, lines 365-413):
  - Generates synthetic sparse data ✓
  - Forward pass y = ReLU(W @ W^T @ x + b) ✓
  - Gradient computation is now shape-correct ✓
  - BUT: When applied to fMRI, adapt_to_fmri() resizes W but train_step()
         STILL generates synthetic Bernoulli data instead of using actual
         voxel time-series patterns. The encoder never sees real brain data.
  - Weight decay formula: W -= lr * γ * W * (||W_i|| - 1)
         Reference: W_i,t+1 = W_i,t - η * ∇L - η * γ * W_i * (||W_i|| - 1)
         This matches the spirit but we're not computing ∇L w.r.t. real data.

IMPACT: The superposition metrics (φ, overlap) describe the structure of
synthetic data compression, not actual fMRI voxel superposition.

FIX REQUIRED: Train on actual voxel activation patterns from the BOLD data.


=============================================================================
ERROR #5 — ASTROCYTE DECONVOLUTION IS SIGNAL PROCESSING, NOT ML  [MODERATE]
=============================================================================

The band-pass filtering (0.01-0.1 Hz astrocyte, 0.1-0.5 Hz neuronal) and
Wiener deconvolution are mathematically correct standard DSP techniques.

HOWEVER, the GNR metric (glia-to-neuron ratio) is HIGHLY TR-DEPENDENT:
  - Haxby TR = 2.5s → Nyquist = 0.2 Hz → neuronal band is 0.1-0.2 Hz (narrow!)
  - MPI TR = 1.0s → Nyquist = 0.5 Hz → neuronal band is 0.1-0.5 Hz (wide!)

This means the Haxby "high GNR" vs MPI "low GNR" comparison is an ARTIFACT
of different sampling rates, not a real biological difference.

FIX: Normalize GNR by the bandwidth of each band, or use spectral density.


=============================================================================
ERROR #6 — EIGENVALUES REPORTED UNSTABLE DESPITE PROJECTION  [BUG]
=============================================================================

The soft projection guarantees |eigenvalue| < 1, yet we reported eigenvalues
of 1.4-1.9. This means get_eigenvalues() disagrees with _soft_project().

Root cause: The eigenvalue formula det = M_11 * M_22 - M_12 * M_21:
    det = (1/S) * (1 - dt^2*A/S) - (-dt*A/S) * (dt/S)
        = (1/S) - dt^2*A/S^2 + dt^2*A/S^2
        = 1/S

So det = 1/S = 1/(1+dt*G) < 1 always (since dt, G > 0).
For a 2x2 matrix, |λ_1 * λ_2| = |det| < 1, but individual eigenvalues
CAN exceed 1 if one is small and the other large.

The trace = 1/S + 1 - dt^2*A/S:
If trace > 2 and det < 1, then eigenvalues are real with λ_max > 1.

This means our soft projection bounds are NOT guaranteeing individual
eigenvalue magnitude < 1. The projection clips A to a valid range, but
our random initialization of A may place values OUTSIDE that range,
and the clipping formula may have a sign error vs the reference.

CHECKING: Reference uses (G + dt*A)^2 < 4*A as validity check.
Let's verify: with A=5, G=0.5, dt=0.5:
    (0.5 + 0.5*5)^2 = (3)^2 = 9
    4*A = 20
    9 < 20 → VALID
    
    S = 1 + 0.5*0.5 = 1.25
    M_11 = 0.8, M_22 = 1 - 0.25*5/1.25 = 0
    det = 1/1.25 = 0.8
    trace = 0.8 + 0 = 0.8
    disc = 0.64 - 3.2 = -2.56 (< 0 → complex eigenvalues)
    |λ| = sqrt(det) = sqrt(0.8) ≈ 0.894 < 1 ✓

So when validity holds, eigenvalues ARE inside unit disk.
The issue is our projection formula may not match the validity check exactly.

INVESTIGATING further: A_low, A_high from our code use (2 + dt*G ± ...) which
is derived from the CHARACTERISTIC POLYNOMIAL of M, not the validity check.
These should be equivalent but let me verify...

For the IMEX1 M matrix:
    trace(M) = 1/S + 1 - dt²A/S = (1 + S - dt²A)/S = (2 + dtG - dt²A)/S
    det(M) = 1/S

Eigenvalue condition |λ| < 1 for complex conjugate pair:
    |λ|² = det = 1/S ← automatically < 1 when S > 1 (i.e., dt*G > 0)

For REAL eigenvalues (disc ≥ 0), need trace² ≥ 4*det AND both roots < 1:
    The roots are real when: trace² ≥ 4*det = 4/S
    λ_max = (trace + sqrt(disc))/2 < 1

This is automatically satisfied when A ∈ (A_low, A_high) from our formula.
But the bug might be that our INITIALIZATION puts A OUTSIDE this range and 
the clipping `A_low + max(A-A_low, 0) - max(A-A_high, 0)` doesn't work right
when A_low > A or A > A_high.

Actually the clipping IS correct: it maps A → [A_low, A_high]. So if our 
eigenvalues are > 1, either:
(a) The projection IS applied but eigenvalue formula has a bug, or
(b) The projection gives A_low > A_high in some cases (invalid bounds)

ANSWER: When dt*G > 1, then A_low = (2 + dtG - 2*sqrt(1+dtG))/dt² and
A_high = (2 + dtG + 2*sqrt(1+dtG))/dt². Both are well-defined and A_low < A_high
for all dtG > 0. So projection bounds are fine.

Let me trace through numerically with our actual init values...
With A_min=0.1, A_max=10, G_min=0.01, G_max=2.0:
    dt = sigmoid(N(0,1)) ∈ (0.27, 0.73) typically
    G ∈ (0.01, 2.0)
    dtG ∈ (0.003, 1.46)
    S = 1 + dtG ∈ (1.003, 2.46)
    A_low ≈ 0 (always small)
    A_high ≈ (2+1.46+2*sqrt(2.46))/(0.73²) ≈ (3.46+3.14)/0.53 ≈ 12.4

So A ∈ (0.1, 10) should be within (A_low, A_high) after projection.
det = 1/S ∈ (0.41, 0.997) ← all fine
|λ| = sqrt(det) ∈ (0.64, 0.998) ← all < 1

CONCLUSION: The eigenvalues SHOULD be < 1 after projection. The bug must be 
in get_eigenvalues() — it's computing eigenvalues from the UNPROJECTED params.

WAIT NO — get_eigenvalues() calls self._soft_project() which does project!
Let me look again at the det formula...

det = M_11 * M_22 - M_12 * M_21
    = (1/S)(1 - dt²A/S) - (-dtA/S)(dt/S)
    = (1 - dt²A/S)/S + dt²A/S²
    = 1/S - dt²A/S² + dt²A/S²
    = 1/S ✓

This is DEFINITELY < 1. So what's wrong?

OH. I see it. The trace:
    trace = 1/S + 1 - dt²A/S

With large A and small S, dt²A/S can be > 1, making M_22 negative.
But trace = 1/S + M_22, and if M_22 < 0, trace could be < 0.

For complex eigenvalues (disc < 0), |λ| = sqrt(det) < 1 ✓
For real eigenvalues (disc ≥ 0), λ_max = (trace + sqrt(trace²-4det))/2

If trace > 2, then even with det < 1, λ_max > 1.
The projection should prevent this, so let me check if trace can exceed 2.

trace = (2 + dtG - dt²A)/(1 + dtG) = 1 + (1 - dt²A)/(1+dtG)
For trace > 2: (1 - dt²A)/(1+dtG) > 1 → 1 - dt²A > 1+dtG → -dt²A > dtG
This requires A < 0, which soft_project prevents. So trace < 2 always when A > 0. ✓

For trace < -2: (1 - dt²A)/(1+dtG) < -3 → dt²A > 4 + 3dtG
The projection clips A < A_high, which prevents this.

So the eigenvalues SHOULD be < 1. The reported values > 1 mean there's a 
numerical bug in get_eigenvalues() or the projection isn't being applied 
in the forward pass consistently.

Actually... I just realized: get_eigenvalues() DOES call _soft_project() (line 225).
But the result should have |λ| = sqrt(1/S) < 1.

The reported eigenvalue 1.4 would require det = 1.96, meaning S = 1/1.96 ≈ 0.51,
meaning 1 + dt*G = 0.51, meaning dt*G < 0. But dt = sigmoid(...) > 0 and G = relu(...) ≥ 0!

THERE IS DEFINITELY A BUG. The det formula must be computed wrong.

Wait — I'm computing det = M_11*M_22 - M_12*M_21 in the code.
Let me expand:
    M_11*M_22 = (1/S)(1 - dt²A/S) = 1/S - dt²A/S²
    M_12*M_21 = (-dtA/S)(dt/S) = -dt²A/S²
    det = 1/S - dt²A/S² - (-dt²A/S²) = 1/S ✓

So the eigenvalue formula in get_eigenvalues() is theoretically correct.
But maybe the discriminant computation allows |λ| > 1?

For disc ≥ 0 (real eigenvalues):
    λ_max = (trace + sqrt(disc))/2
    λ_min = (trace - sqrt(disc))/2
    λ_max * λ_min = det = 1/S
    If λ_min ≈ 0 then λ_max ≈ 1/S / λ_min → ∞

NO! For 2x2 with det > 0 and disc ≥ 0:
    λ_max = (trace + sqrt(trace²-4det))/2
    This can be > 1 even when det < 1 if trace > 1 + det.
    For our case: trace > 1 + 1/S means 1 + (1-dt²A)/(1+dtG) > 1 + 1/(1+dtG)
    → (1-dt²A) > 1 → dt²A < 0 → impossible for A > 0.

So with projected A > 0, trace < 1 + 1/S always, and λ_max < 1 (proof below):
    We need: (trace + sqrt(trace²-4det))/2 < 1
    ⟺ trace + sqrt(trace²-4det) < 2
    Since trace < 1 + 1/S ≤ 2, and 4det = 4/S ≥ 0:
    disc = trace² - 4/S
    When disc < 0: |λ| = sqrt(1/S) < 1 ✓
    When disc ≥ 0: trace + sqrt(disc) < 2 when trace < 2 AND trace² - disc < 4
    trace² - disc = 4det = 4/S > 0, so trace² - disc > 0.
    Hmm, need trace + sqrt(trace²-4/S) < 2.
    
Actually for the projection to guarantee stability, we need A ∈ (A_low, A_high).
The values A_low and A_high ARE the roots of the stability boundary equation.
So BY CONSTRUCTION, if A is clipped to this range, we're stable.

I believe the issue is that when disc > 0 (real eigenvalues), my code computes
sqrt(|disc|) but the disc might be negative (complex eigenvalues more common),
and I'm using np.abs which forces it positive. Let me check...

Line 234: disc = trace**2 - 4*det
Line 235-236: lam = (trace ± sqrt(|disc| + 0j)) / 2

When disc < 0 (complex eigenvalues), sqrt(|disc| + 0j) = sqrt(|disc|)i
So lam = (trace ± sqrt(|disc|)i) / 2 → |lam| = sqrt(trace²/4 + |disc|/4)
                                                = sqrt((trace²+|disc|)/4)
BUT |disc| = |trace²-4det| = 4det - trace² (when disc < 0)
So |lam|² = (trace² + 4det - trace²)/4 = det = 1/S
|lam| = sqrt(1/S) < 1 ✓

When disc > 0 (real eigenvalues):
sqrt(disc) is real.
lam_max = (trace + sqrt(disc))/2

The issue is `np.sqrt(np.abs(disc) + 0j)` — when disc > 0, this gives
sqrt(disc + 0j) which is correct. When disc < 0, this gives
sqrt(|disc| + 0j) = sqrt(|disc|) which is ALSO real, not imaginary!

THIS IS THE BUG! np.abs(disc) makes negative disc positive, and then
sqrt gives a REAL result, making the eigenvalue REAL and potentially > 1.

The fix: use np.sqrt(disc + 0j) WITHOUT np.abs. This gives correct
complex eigenvalues when disc < 0.

=============================================================================
ERROR #7 — INCONSISTENCY BETWEEN VARIANTS                [MODERATE]  
=============================================================================

The paper presents multiple D-LinOSS variants:
  - DampedIMEX1: implicit damping (G acts on z_{k+1}) — our default
  - DampedIMEX2: explicit damping (G acts on z_k)
  - DampedIM: fully implicit (both A and G implicit)

Each has DIFFERENT stability conditions and recurrence matrices.
Our model docstring says "IMEX scheme" but should specify IMEX1.
Not a math error, but should be documented for reproducibility.


=============================================================================
ERROR #8 — NO FMRI-SPECIFIC ADAPTATION                   [METHODOLOGY]
=============================================================================

The D-LinOSS paper is designed for generic sequence modeling. For fMRI, we need:
  1. Set dt to match the physical TR (scan repetition time), not learn it freely
  2. The oscillator frequency A should be initialized to match expected BOLD 
     frequencies (0.01-0.5 Hz), not randomly
  3. The damping G should relate to the hemodynamic response decay rate

Currently we treat these as free parameters, ignoring the physical meaning.
"""

print("=" * 70)
print("MATHEMATICAL AUDIT COMPLETE")
print("=" * 70)
print("""
CRITICAL ERRORS TO FIX:
  #3 - No gradient-based training (SSM params are random)
  #4 - Superposition encoder never trains on real fMRI data
  #6 - Eigenvalue formula uses np.abs(disc) instead of disc+0j

MODERATE ERRORS TO FIX:
  #5 - GNR metric is TR-dependent (normalize by bandwidth)
  #7 - Should document which D-LinOSS variant we're using
  #8 - SSM parameters should be physically grounded for fMRI

ITEMS VERIFIED CORRECT:
  #1 - Recurrence matrix matches reference IMEX1
  #2 - Soft projection formula matches reference
""")
