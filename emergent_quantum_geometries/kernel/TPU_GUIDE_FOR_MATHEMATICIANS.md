# Writing and Running Your Own SO(3) Kernel Experiments on Google Cloud TPUs

> **Audience:** Patrick Collins (or any mathematician who knows MATLAB, tinkers with Python, and wants to run large numerical experiments on tensor hardware without becoming a DevOps engineer).
>
> **What this guide covers:**
> 1. What the proof-of-concept TPU run proved
> 2. How JAX works (the 60-second version for MATLAB people)
> 3. The libraries we use and where their docs live
> 4. How to write your own kernel / modify the existing script
> 5. How to test locally before burning cloud time
> 6. How to deploy to a Cloud TPU from your laptop, step by step

---

## 1. What the Proof of Concept Proved

We ran [program_aq_so3_kernel_galerkin.py](./program_aq_so3_kernel_galerkin.py) on Google Cloud TPUs (v6e-8, 8 chips) under the TRC (TPU Research Cloud) free-tier program. The experiment:

- Sampled SO(3) under Haar measure
- Built real-valued Wigner-D basis matrices
- Assembled kernel collocation systems and Galerkin stiffness matrices
- Solved dense linear systems
- Measured L² error against known exact solutions

**The takeaway:** Your SO(3) kernel-Galerkin mathematics can run on tensor processing hardware. JAX compiled and executed the entire pipeline — Jacobi polynomial recurrences, Wigner-D evaluations, quadrature weight solves, dense linear algebra — on 8 TPU chips with no code changes versus CPU. The script ran in ~74 seconds on TPU versus what would have been many minutes on a laptop CPU for the same parameter grid.

This means you can write your math in Python, test it on your laptop's CPU, and then deploy the *exact same script* to a TPU with 8 or 64 chips. No rewriting required.

---

## 2. JAX for MATLAB People (The 60-Second Version)

### 2.1 What Is JAX?

JAX is "NumPy that runs on GPUs and TPUs." If you know MATLAB, you already think in terms of vectorized array operations. JAX is the same idea, but in Python, and it can run your arrays on tensor hardware.

| MATLAB | NumPy (CPU only) | JAX (CPU / GPU / TPU) |
|---|---|---|
| `A = rand(100,100)` | `A = np.random.rand(100,100)` | `A = jax.random.normal(key, (100,100))` |
| `B = A * A'` | `B = A @ A.T` | `B = A @ A.T` |
| `x = A \ b` | `x = np.linalg.solve(A, b)` | `x = jnp.linalg.solve(A, b)` |
| `sin(A)` | `np.sin(A)` | `jnp.sin(A)` |

The key difference: **JAX arrays live on whatever device JAX picks** (CPU on your laptop, TPU in the cloud). You write the math once. JAX handles the hardware.

### 2.2 The Three Things JAX Does That MATLAB Doesn't

#### `jax.jit` — Just-In-Time Compilation (≈ MATLAB's codegen, but automatic)

When you decorate a function with `@jax.jit`, JAX traces through your code once, builds an optimized computation graph, and compiles it for the hardware. The first call is slow (compilation). Every subsequent call with the same array shapes is fast.

```python
@jax.jit
def my_kernel_matrix(angles1, angles2):
    # ... math here ...
    return K

# First call: compiles (~seconds)
K = my_kernel_matrix(angles_a, angles_b)

# Second call with same shapes: instant
K = my_kernel_matrix(angles_c, angles_d)
```

**MATLAB analogy:** It's like if `mex` compilation happened automatically the first time you called a function.

**Gotcha for MATLAB people:** Inside a `@jax.jit` function, you **cannot** use Python `if` statements that depend on array *values* (only on array *shapes* or compile-time constants). Use `jax.lax.cond` instead. Similarly, Python `for` loops over data should become `jax.lax.fori_loop`. The existing script has examples of both.

#### `jax.vmap` — Vectorized Map (≈ MATLAB's implicit vectorization)

Instead of writing a for-loop over rows, you tell JAX "apply this function across this axis":

```python
# MATLAB: arrayfun(@my_func, A)  or just my_func(A) if already vectorized
# JAX:
batched_func = jax.vmap(my_func)
results = batched_func(A)
```

#### `jax.grad` — Automatic Differentiation

If you ever need gradients (for optimization, sensitivity analysis, adjoint methods):

```python
def loss(params):
    return jnp.sum((f(params) - target)**2)

gradient_of_loss = jax.grad(loss)
g = gradient_of_loss(params)  # exact gradient, not finite differences
```

### 2.3 Random Numbers Are Different

MATLAB: `rand(3,3)` just works, uses global state.

JAX: Random numbers require an explicit **key** (a seed token). This is because TPUs need deterministic, reproducible randomness across distributed chips.

```python
key = jax.random.PRNGKey(42)          # Create initial key from seed
key, subkey = jax.random.split(key)   # Split to get a fresh key
samples = jax.random.normal(subkey, (1000, 3))  # Use subkey for sampling
```

**Rule:** Never reuse a key. Always split before sampling.

### 2.4 Array Indexing Is (Slightly) Different

MATLAB-style mutation (`A(3,4) = 7`) doesn't work in JAX because arrays are immutable (this is required for compilation). Instead:

```python
# Instead of: A[3, 4] = 7
A = A.at[3, 4].set(7)
```

### 2.5 float32 vs float64

**TPUs default to float32.** This is 7 decimal digits of precision. For most kernel experiments with condition numbers below ~10⁶, this is fine. If you need float64 (15 digits, like MATLAB's default `double`), set:

```python
jax.config.update("jax_enable_x64", True)
```

or set the environment variable `PROGRAM_AQ_ENABLE_X64=1` before running. float64 on TPU is emulated and roughly 2-4× slower.

---

## 3. Libraries We Use (and Their Documentation)

### 3.1 Core Stack

| Library | What It Does | Docs | MATLAB Equivalent |
|---|---|---|---|
| **JAX** | Array computation on CPU/GPU/TPU | [jax.readthedocs.io](https://jax.readthedocs.io) | The engine itself |
| **jax.numpy** (`jnp`) | NumPy API on JAX arrays | [jax.readthedocs.io/en/latest/jax.numpy.html](https://jax.readthedocs.io/en/latest/jax.numpy.html) | Core MATLAB functions |
| **jax.lax** | Low-level ops (loops, conditionals) | [jax.readthedocs.io/en/latest/jax.lax.html](https://jax.readthedocs.io/en/latest/jax.lax.html) | `for`, `if`, `while` inside compiled code |
| **jax.scipy** | SciPy subset on JAX | [jax.readthedocs.io/en/latest/jax.scipy.html](https://jax.readthedocs.io/en/latest/jax.scipy.html) | Optimization, sparse solvers, special funcs |
| **NumPy** (`np`) | CPU-only arrays (used for setup) | [numpy.org/doc](https://numpy.org/doc/stable/) | Basic MATLAB arrays |
| **SciPy** | Scientific computing (CPU) | [docs.scipy.org](https://docs.scipy.org/doc/scipy/) | Toolboxes |

### 3.2 What Already Exists (Don't Rebuild These)

Before writing your own version of something, check if it already exists:

| You Need | It Already Exists In | Function / Module |
|---|---|---|
| Gauss-Legendre quadrature nodes/weights | `np.polynomial.legendre.leggauss(n)` | Returns nodes and weights on [-1,1] |
| Jacobi polynomials | Our script: `jacobi_p(n, alpha, beta, x)` | Uses `jax.lax.fori_loop` recurrence |
| Legendre polynomials | `jax.scipy.special.lpmn_values(...)` or our Jacobi with α=β=0 | |
| Gamma / log-gamma | `jax.lax.lgamma(x)` | Compiled, works on TPU |
| Bessel functions | `jax.scipy.special.bessel_jn(...)` | |
| Linear solve (Ax=b) | `jnp.linalg.solve(A, b)` | Dense direct solve |
| Eigenvalues | `jnp.linalg.eigh(A)` | For symmetric/Hermitian matrices |
| SVD | `jnp.linalg.svd(A)` | |
| Condition number | `jnp.linalg.cond(A)` | |
| Conjugate gradient (sparse) | `jax.scipy.sparse.linalg.cg(A, b)` | For large sparse or matrix-free systems |
| GMRES | `jax.scipy.sparse.linalg.gmres(A, b)` | |
| Cholesky decomposition | `jnp.linalg.cholesky(A)` | For SPD matrices |
| FFT | `jnp.fft.fft(x)`, `jnp.fft.fftn(x)` | N-dimensional FFT on TPU |
| Optimization (L-BFGS etc.) | `jax.scipy.optimize.minimize(...)` | |
| Automatic differentiation | `jax.grad(f)`, `jax.jacobian(f)` | Exact, not finite differences |

### 3.3 Key Documentation Pages to Bookmark

- **JAX Quickstart (start here):** https://jax.readthedocs.io/en/latest/quickstart.html
- **NumPy → JAX translation guide:** https://jax.readthedocs.io/en/latest/jax-101/01-jax-basics.html
- **"Thinking in JAX" (how jit works):** https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html
- **Random numbers explained:** https://jax.readthedocs.io/en/latest/jax-101/05-random-numbers.html
- **Sharp bits (gotchas):** https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html
- **Available math functions:** https://jax.readthedocs.io/en/latest/jax.numpy.html
- **Sparse / iterative solvers:** https://jax.readthedocs.io/en/latest/jax.scipy.html

---

## 4. How to Write Your Own Kernel Experiment

### 4.1 The Template Structure

Every experiment script follows this pattern. Copy [program_aq_so3_kernel_galerkin.py](./program_aq_so3_kernel_galerkin.py) and modify it:

```
┌──────────────────────────────────────────────┐
│  1. Imports + JAX setup                      │  ← Keep as-is
│  2. Watchdog (preemption listener)           │  ← Keep as-is
│  3. YOUR GEOMETRY (sampling, basis funcs)    │  ← Modify / replace
│  4. YOUR KERNEL (eigenvalues, operators)     │  ← Modify / replace
│  5. Solver (collocation, Galerkin, etc.)     │  ← Modify / keep
│  6. Sweep loop + JSON output                 │  ← Modify parameters
│  7. argparse CLI                             │  ← Add your flags
└──────────────────────────────────────────────┘
```

### 4.2 Example: Adding a New Kernel

Say you want to test a **Matérn-type kernel on SO(3)** with Fourier coefficients:

$$\hat{\phi}_\nu(\ell) = \frac{1}{(1 + \ell(\ell+1))^\nu}$$

Here is exactly what you'd change in the script:

**Step 1.** In `get_eigenvalues`, add a new `operator` branch:

```python
def get_eigenvalues(N, m, operator):
    eigenvals_mu = []
    eigenvals_lambda = []

    for l in range(N + 1):
        lam_l = l * (l + 1)

        if operator == "sobolev":
            mu_l = (1.0 + lam_l)**m
        elif operator == "l3":
            mu_l = (24.0 / (16.0 * jnp.pi)) * (-lam_l + 0.25) * (-lam_l - 0.75) * (-lam_l - 8.75)

        # ---- YOUR NEW KERNEL ----
        elif operator == "matern":
            nu = float(m)  # reuse the m parameter as nu
            mu_l = (1.0 + lam_l)**nu
        # --------------------------

        else:
            raise ValueError(f"Unsupported operator: {operator}")

        lambda_l = 1.0 / mu_l
        mult = (2 * l + 1)**2
        eigenvals_mu.extend([mu_l] * mult)
        eigenvals_lambda.extend([lambda_l] * mult)

    return jnp.array(eigenvals_mu, dtype=DTYPE), jnp.array(eigenvals_lambda, dtype=DTYPE)
```

**Step 2.** Add `"matern"` to the `--operator` choices in the argparse section at the bottom:

```python
p.add_argument("--operator", choices=["sobolev", "l3", "matern"], default="sobolev")
```

**Step 3.** Run it:

```bash
python3 kernel/program_aq_so3_kernel_galerkin.py \
  --operator matern --m 3 \
  --n-centers 50 100 200 \
  --N-truncation 5 \
  --smoke-test
```

That's it. Everything else — the Wigner-D basis, the Galerkin assembly, the solver, the error measurement, the JSON output — stays the same.

### 4.3 Example: Completely Custom Kernel Function

If your kernel isn't defined by Fourier coefficients but by a closed-form expression (like the rotational surface spline), you can bypass the spectral path entirely:

```python
def my_kernel(R1, R2):
    """
    Evaluate Phi(R1, R2) for rotation matrices R1, R2.
    Example: rotational surface spline Phi_m(x,y) = sin^{2m-3}(omega(y^-1 x)/2)
    """
    # R1, R2 are (..., 3, 3) rotation matrices
    tr = jnp.sum(R1 * R2, axis=(-2, -1))         # tr(R2^T R1)
    cos_half_omega = 0.5 * jnp.sqrt(jnp.maximum(0.0, tr + 1.0))
    sin_half_omega = jnp.sqrt(jnp.maximum(0.0, 1.0 - cos_half_omega**2))
    m = 3
    return sin_half_omega ** (2*m - 3)

# Build the kernel matrix directly:
def build_kernel_matrix(angles):
    R = angles_to_matrices(angles)  # (n, 3, 3)
    n = R.shape[0]
    # Use vmap to vectorize over pairs
    def row_fn(r1):
        return jax.vmap(lambda r2: my_kernel(r1, r2))(R)
    K = jax.vmap(row_fn)(R)  # (n, n)
    return K
```

### 4.4 Tips for Writing JAX-Compatible Math

1. **Use `jnp` instead of `np` for anything that touches the TPU.** Use `np` only for one-time setup (like generating quadrature nodes on the CPU before transferring to TPU).

2. **Avoid Python loops over data.** Instead of:
   ```python
   # SLOW (Python loop, not compiled)
   result = []
   for i in range(1000):
       result.append(f(x[i]))
   ```
   Use:
   ```python
   # FAST (compiled, vectorized)
   result = jax.vmap(f)(x)
   ```

3. **For recurrences (Jacobi, Chebyshev, etc.), use `jax.lax.fori_loop`:**
   ```python
   def body(i, carry):
       p_prev, p_curr = carry
       p_next = ((2*i+1) * x * p_curr - i * p_prev) / (i + 1)
       return p_curr, p_next

   _, result = jax.lax.fori_loop(1, n, body, (p0, p1))
   ```

4. **Mark compile-time constants with `static_argnames`:**
   ```python
   @jax.jit
   def f(x, n):  # n changes the computation graph
       ...

   # Better:
   f_jit = jax.jit(f, static_argnames=("n",))
   ```
   This tells JAX to recompile when `n` changes (since the graph shape depends on it).

5. **Use `jnp.maximum(0.0, x)` instead of `max(0, x)`** — the Python builtin doesn't work inside JIT.

---

## 5. Testing Locally Before Deploying

### 5.1 Install JAX (CPU) on Your Laptop

```bash
pip install jax jaxlib numpy scipy
```

This gives you JAX on CPU. Your code will be slower but functionally identical.

### 5.2 Run a Smoke Test

Every script should have a `--smoke-test` flag that uses small parameters:

```bash
cd emergent_quantum_geometries
python kernel/program_aq_so3_kernel_galerkin.py --smoke-test
```

This runs with `|Xi|=50, N=3, q=5` — small enough to finish in seconds on a laptop CPU. If this works, the same code will work on a TPU.

### 5.3 Check That JAX Sees Your Device

```python
import jax
print(jax.default_backend())  # "cpu" on laptop, "tpu" on cloud
print(jax.devices())          # lists available devices
```

### 5.4 The `--require-accelerator` Safety Net

Pass `--require-accelerator` when deploying to a TPU. This makes the script crash immediately if JAX only sees CPU — so you know right away if something went wrong with the TPU setup, instead of waiting hours for a CPU-speed run to finish.

---

## 6. Deploying to a Cloud TPU (Step by Step)

### 6.1 Prerequisites (One-Time Setup)

1. **Install Google Cloud SDK** on your laptop: https://cloud.google.com/sdk/docs/install
2. **Authenticate:** `gcloud auth login`
3. **Set project:** `gcloud config set project time-emission`
4. **Verify TRC access:** You should be able to list TPUs without errors:
   ```
   gcloud compute tpus tpu-vm list --project=time-emission
   ```

### 6.2 The Deployment Recipe

There are only 5 steps. Here they are in plain English and then in exact commands.

#### Step 1: Create a TPU VM

```powershell
# Variables — adjust these
$Project  = "time-emission"
$Zone     = "us-east1-d"            # TRC-covered zone
$NodeName = "my-experiment-node"    # Pick any name
$Type     = "v6e-8"                 # 8 TPU chips
$Runtime  = "v2-alpha-tpuv6e"       # Software image

# Create it (--spot = free under TRC, but Google can preempt it)
gcloud compute tpus tpu-vm create $NodeName `
    --project=$Project `
    --zone=$Zone `
    --accelerator-type=$Type `
    --version=$Runtime `
    --spot
```

Wait ~2-5 minutes for it to become READY.

#### Step 2: Install JAX on the TPU

```powershell
gcloud compute tpus tpu-vm ssh $NodeName `
    --project=$Project --zone=$Zone `
    --command="pip install -q -U 'jax[tpu]' scipy numpy -f https://storage.googleapis.com/jax-releases/libtpu_releases.html"
```

#### Step 3: Upload Your Script

```powershell
# Upload a single file:
gcloud compute tpus tpu-vm scp `
    "emergent_quantum_geometries/kernel/my_experiment.py" `
    "${NodeName}:/home/$env:USERNAME/" `
    --project=$Project --zone=$Zone

# Or upload a folder:
gcloud compute tpus tpu-vm scp --recurse `
    "emergent_quantum_geometries/kernel/" `
    "${NodeName}:/home/$env:USERNAME/kernel/" `
    --project=$Project --zone=$Zone
```

#### Step 4: Run Your Experiment

```powershell
gcloud compute tpus tpu-vm ssh $NodeName `
    --project=$Project --zone=$Zone `
    --command="cd /home/$env:USERNAME && python3 -u my_experiment.py --require-accelerator --out-dir results"
```

The `-u` flag gives you unbuffered output so you can see progress in real time.

For long runs, use `nohup` so it survives if your SSH drops:

```powershell
gcloud compute tpus tpu-vm ssh $NodeName `
    --project=$Project --zone=$Zone `
    --command="cd /home/$env:USERNAME && nohup python3 -u my_experiment.py --require-accelerator --out-dir results > run.log 2>&1 &"
```

Then check progress later:

```powershell
gcloud compute tpus tpu-vm ssh $NodeName `
    --project=$Project --zone=$Zone `
    --command="tail -n 30 /home/$env:USERNAME/run.log"
```

#### Step 5: Download Results and Delete the TPU

```powershell
# Download just the summary JSON (small, no egress charges)
gcloud compute tpus tpu-vm scp `
    "${NodeName}:/home/$env:USERNAME/results/summary.json" `
    "./my_results/" `
    --project=$Project --zone=$Zone

# DELETE THE TPU (important — don't leave it running!)
gcloud compute tpus tpu-vm delete $NodeName `
    --project=$Project --zone=$Zone --quiet
```

### 6.3 TRC Zones and Types (What's Free)

| Zone | TPU Type | Chips | Mode |
|---|---|---|---|
| `us-east1-d` | `v6e` | up to 64 | spot only |
| `europe-west4-a` | `v6e` | up to 64 | spot only |
| `us-central1-a` | `v5e` | up to 64 | spot only |
| `europe-west4-b` | `v5e` | up to 64 | spot only |
| `us-central2-b` | `v4` | up to 32 | on-demand or spot |

**Always pass `--spot`** (except v4 on-demand). Spot VMs are free under TRC but Google can preempt (kill) them at any time. That's why the watchdog/heartbeat code exists in the script — it detects preemption and saves partial results.

### 6.4 Don't Get Charged (The Short Version)

1. **Only use the zones/types listed above.**
2. **Always delete the TPU when you're done.** Check with:
   ```
   gcloud compute tpus tpu-vm list --project=time-emission
   ```
   The output should say `Listed 0 items.`
3. **Don't download big files.** A `summary.json` (a few KB) is fine. Don't `scp --recurse` a folder of `.npz` checkpoint files — that's network egress and it costs money.

---

## 7. Writing a Complete Script from Scratch (Minimal Template)

Here's the smallest possible experiment script that will run on a TPU:

```python
"""
my_experiment.py — Minimal TPU experiment template
"""
import argparse
import json
import os
import time

# --- JAX setup (keep this block as-is) ---
import jax
jax.config.update("jax_enable_x64", False)  # True if you need float64
import jax.numpy as jnp
import numpy as np

def main(args):
    print(f"Backend: {jax.default_backend()} | devices: {len(jax.devices())}")
    if args.require_accelerator and jax.default_backend() == "cpu":
        raise SystemExit("No accelerator found!")

    key = jax.random.PRNGKey(args.seed)
    t0 = time.time()

    # ============================================================
    # YOUR MATH GOES HERE
    # ============================================================
    #
    # Example: build a random kernel matrix and compute its spectrum
    #
    n = args.matrix_size
    key, subkey = jax.random.split(key)
    A = jax.random.normal(subkey, (n, n), dtype=jnp.float32)
    K = A @ A.T / n  # symmetric positive semidefinite

    eigenvalues = jnp.linalg.eigvalsh(K)

    results = {
        "n": n,
        "top_5_eigenvalues": [float(e) for e in eigenvalues[-5:]],
        "trace": float(jnp.trace(K)),
        "condition_number": float(eigenvalues[-1] / jnp.maximum(eigenvalues[0], 1e-12)),
    }
    # ============================================================

    runtime = time.time() - t0
    output = {
        "experiment": "my_experiment",
        "seed": args.seed,
        "runtime_seconds": runtime,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "results": results,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--matrix-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--require-accelerator", action="store_true")
    p.add_argument("--out-dir", default="my_results")
    main(p.parse_args())
```

Test locally: `python my_experiment.py --matrix-size 100`

Deploy to TPU: follow the 5-step recipe in §6.2, adding `--require-accelerator`.

---

## 8. MATLAB → JAX Cheat Sheet

| MATLAB | JAX |
|---|---|
| `A = zeros(m, n)` | `A = jnp.zeros((m, n))` |
| `A = ones(m, n)` | `A = jnp.ones((m, n))` |
| `A = eye(n)` | `A = jnp.eye(n)` |
| `A = rand(m, n)` | `A = jax.random.uniform(key, (m, n))` |
| `A = randn(m, n)` | `A = jax.random.normal(key, (m, n))` |
| `A(i, j)` | `A[i, j]` (0-indexed!) |
| `A(i, j) = v` | `A = A.at[i, j].set(v)` |
| `A(:, 3)` | `A[:, 3]` |
| `A(2:5, :)` | `A[1:5, :]` (0-indexed, exclusive end) |
| `A * B` (element-wise) | `A * B` |
| `A * B` (matrix multiply) | `A @ B` |
| `A'` (transpose) | `A.T` |
| `A \ b` | `jnp.linalg.solve(A, b)` |
| `eig(A)` | `jnp.linalg.eig(A)` |
| `eigs(A)` (symmetric) | `jnp.linalg.eigh(A)` |
| `svd(A)` | `jnp.linalg.svd(A)` |
| `norm(x)` | `jnp.linalg.norm(x)` |
| `cond(A)` | `jnp.linalg.cond(A)` |
| `chol(A)` | `jnp.linalg.cholesky(A)` |
| `fft(x)` | `jnp.fft.fft(x)` |
| `linspace(a, b, n)` | `jnp.linspace(a, b, n)` |
| `sum(A, 1)` | `jnp.sum(A, axis=0)` (axis numbering differs!) |
| `max(A(:))` | `jnp.max(A)` |
| `arrayfun(@f, A)` | `jax.vmap(f)(A)` |
| `integral(@f, a, b)` | Use quadrature: `np.polynomial.legendre.leggauss(n)` |
| `ode45(@f, tspan, y0)` | `jax.experimental.ode.odeint(f, y0, t)` |

> **Important indexing difference:** MATLAB is 1-indexed (`A(1,1)` is the first element). Python/JAX is 0-indexed (`A[0,0]` is the first element). MATLAB's `A(2:5,:)` gives rows 2,3,4,5. Python's `A[1:5,:]` gives rows 1,2,3,4 (the end index is exclusive).

---

## 9. Anatomy of the Existing Script

Here's a roadmap of [program_aq_so3_kernel_galerkin.py](./program_aq_so3_kernel_galerkin.py) so you know what each section does and what you'd modify:

| Lines | Section | What It Does | Modify? |
|---|---|---|---|
| 1–22 | JAX setup | Imports, float32/64 config | No |
| 24–49 | Watchdog | Heartbeat file + preemption detector for spot TPUs | No |
| 55–106 | SO(3) sampling | Haar-uniform random nodes + deterministic Euler quadrature | Only if changing the manifold |
| 108–142 | Rotation matrices | Euler angles → 3×3 matrices, geodesic distance | Only if changing the manifold |
| 148–221 | Wigner-D basis | Jacobi polynomial recurrence → real Wigner-D functions | Only if using different basis functions |
| 227–259 | Quadrature weights | NNLS solver for positive quadrature weights | Rarely |
| 265–346 | Solvers | Collocation system, Lagrange basis, operator evaluation | Only if changing the discretization method |
| 348–440 | `run_experiment` | Single experiment: assemble, solve, measure error | Modify to change what you measure |
| 446–585 | `run_sweeps` | Parameter sweep loop, JSON output | Modify to add/change sweep parameters |
| 587–618 | CLI | argparse flags | Add your own flags here |

### Key function to understand: `get_eigenvalues` (line 301)

This is where the **kernel is defined spectrally.** The kernel's Fourier coefficients λ_ℓ and the operator's eigenvalues μ_ℓ are computed here. If you want to test a different kernel, this is the function to modify (see §4.2).

### Key function to understand: `run_experiment` (line 348)

This is the **core math pipeline**: build basis matrices → solve collocation → assemble stiffness → solve Galerkin → measure error. If you want to change the PDE, the right-hand side, or the error metric, modify this function.

---

## 10. Quick Reference: Running the Existing Script

### Smoke test (laptop, ~5 seconds):
```bash
python kernel/program_aq_so3_kernel_galerkin.py --smoke-test
```

### Small sweep (laptop, ~1 minute):
```bash
python kernel/program_aq_so3_kernel_galerkin.py \
  --n-centers 50 100 \
  --N-truncation 3 5 \
  --oversampling-ratios 5 \
  --diagnostics
```

### Full sweep on TPU:
```bash
python3 -u kernel/program_aq_so3_kernel_galerkin.py \
  --n-centers 500 1000 2000 5000 10000 \
  --N-truncation 5 10 15 20 \
  --oversampling-ratios 10 \
  --require-accelerator \
  --out-dir program_aq_results
```

### With deterministic quadrature (recommended for theorem-facing runs):
```bash
python3 -u kernel/program_aq_so3_kernel_galerkin.py \
  --quadrature-source euler-grid \
  --validation-source euler-grid \
  --diagnostics \
  --n-centers 25 50 75 100 \
  --N-truncation 5 \
  --oversampling-ratios 2 \
  --test-degree 2 \
  --seeds 42 43 44 \
  --n-test 1000 \
  --require-accelerator \
  --out-dir program_aq_tpu_certified
```

---

## 11. Troubleshooting

| Problem | Cause | Fix |
|---|---|---|
| `JAX not found` | JAX isn't installed on the TPU VM | SSH in and run: `pip install 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html` |
| `--require-accelerator` crashes | JAX sees only CPU | Check `jax.devices()`. If empty, the TPU runtime wasn't installed correctly. Reinstall JAX. |
| Very slow first run, then fast | Normal! First call compiles the computation graph. | Don't worry. Subsequent calls with same array shapes are fast. |
| `ConcretizationTypeError` | You used a Python `if` on an array value inside `@jax.jit` | Use `jax.lax.cond(pred, true_fn, false_fn, operand)` instead |
| `TracerArrayConversionError` | You passed a JAX array to NumPy inside `@jax.jit` | Use `jnp` functions instead of `np` functions inside JIT |
| Infinite condition number | Matrix is ill-conditioned in float32 | Try `jax.config.update("jax_enable_x64", True)` or reduce problem size |
| `PREEMPTED` (run killed mid-execution) | Google reclaimed the spot VM | This is normal for spot instances. Re-run. The watchdog saves a heartbeat so you know when it died. |
| `Listed 0 items` after create | TPU is still provisioning | Wait 2-5 minutes, then check again with `gcloud compute tpus tpu-vm list` |

---

## 12. File Layout

```
emergent_quantum_geometries/
├── kernel/
│   ├── program_aq_so3_kernel_galerkin.py   ← The working experiment script
│   ├── PROGRAM_AQ_PLAN.md                  ← Mathematical plan for Collins' theorem
│   ├── TPU_GUIDE_FOR_MATHEMATICIANS.md     ← This file
│   └── paper.md / paper.pdf                ← Collins' dissertation
├── run_program_aq_shotgun.ps1              ← Automated TPU launcher (advanced)
├── docs/
│   ├── PROGRAM_AQ_TPU_EXPERIMENT_SUMMARY_2026-06-06.md
│   ├── PROGRAM_AQ_CERTIFIED_TPU_RUN_2026-06-06.md
│   └── PROGRAM_AQ_CONFIDENCE_TPU_RUN_2026-06-06.md
└── ...
```
