"""
Program AQ: Numerical Completion of Collins' SO(3) Kernel-Galerkin Theorem
=======================================================================
Verifies Corollary 6.6.8 and Remark 6.6.9 on Cloud TPUs.
"""

import argparse
import json
import os
import sys
import time
import numpy as np

# TPU-first default: keep arrays in float32 unless explicitly requested.
try:
    import jax
    jax.config.update("jax_enable_x64", os.environ.get("PROGRAM_AQ_ENABLE_X64", "0") == "1")
    import jax.numpy as jnp
except ImportError:
    raise SystemExit("JAX not found")

DTYPE = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

# Watchdog/preemption listener
import urllib.request
import threading

SHUTDOWN_FLAG = False

def watchdog_and_heartbeat():
    global SHUTDOWN_FLAG
    while not SHUTDOWN_FLAG:
        try:
            with open("heartbeat.txt", "w") as f:
                f.write(str(time.time()))
        except:
            pass
        try:
            req = urllib.request.Request("http://metadata.google.internal/computeMetadata/v1/instance/preempted", headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.read().decode('utf-8') == "TRUE":
                    print("[WATCHDOG] Preemption imminent!")
                    SHUTDOWN_FLAG = True
        except:
            pass
        time.sleep(5)

watchdog_thread = threading.Thread(target=watchdog_and_heartbeat, daemon=True)
watchdog_thread.start()

# ---------------------------------------------------------------------------
# 1. SO(3) Geometry and Sampling (Haar Measure)
# ---------------------------------------------------------------------------

def sample_so3_haar(key, num_samples):
    """Sample SO(3) uniformly under Haar measure using Euler angles (ZXZ)."""
    k1, k2, k3 = jax.random.split(key, 3)
    phi1 = jax.random.uniform(k1, (num_samples,), minval=0.0, maxval=2*jnp.pi, dtype=DTYPE)
    u2 = jax.random.uniform(k2, (num_samples,), minval=0.0, maxval=1.0, dtype=DTYPE)
    theta = jnp.arccos(1.0 - 2.0 * u2)
    phi2 = jax.random.uniform(k3, (num_samples,), minval=0.0, maxval=2*jnp.pi, dtype=DTYPE)
    return jnp.stack([phi1, theta, phi2], axis=-1)

def so3_euler_quadrature(min_samples, degree):
    """Deterministic product quadrature for normalized Haar measure on SO(3)."""
    angle_count = max(2 * degree + 1, int(np.ceil(min_samples ** (1.0 / 3.0))))
    theta_count = max(degree + 1, int(np.ceil(min_samples / (angle_count * angle_count))))

    x_nodes, x_weights = np.polynomial.legendre.leggauss(theta_count)
    phi1 = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    phi2 = np.linspace(0.0, 2.0 * np.pi, angle_count, endpoint=False)
    theta = np.arccos(x_nodes)

    p1, th, p2 = np.meshgrid(phi1, theta, phi2, indexing="ij")
    angles = np.stack([p1.ravel(), th.ravel(), p2.ravel()], axis=-1)

    # dmu = (1 / 8pi^2) dphi1 dphi2 d(cos theta). Uniform phi weights
    # and Gauss-Legendre x weights therefore normalize to sum exactly 1.
    weights = np.tile(x_weights[None, :, None], (angle_count, 1, angle_count)).ravel()
    weights = weights / (2.0 * angle_count * angle_count)

    meta = {
        "angle_count": int(angle_count),
        "theta_count": int(theta_count),
        "actual_samples": int(angles.shape[0]),
        "degree": int(degree),
    }
    return jnp.asarray(angles, dtype=DTYPE), jnp.asarray(weights, dtype=DTYPE), meta

def make_so3_dataset(key, num_samples, source):
    """Return SO(3) nodes for the selected experiment dataset."""
    if source != "haar":
        raise ValueError(f"Unsupported dataset source: {source}")
    return sample_so3_haar(key, num_samples)

def make_weighted_so3_nodes(key, num_samples, source, degree):
    """Return SO(3) nodes, optional quadrature weights, and generation metadata."""
    if source == "haar":
        angles = sample_so3_haar(key, num_samples)
        weights = jnp.ones((num_samples,), dtype=DTYPE) / jnp.asarray(num_samples, dtype=DTYPE)
        return angles, weights, {"source": "haar", "actual_samples": int(num_samples)}
    if source == "euler-grid":
        angles, weights, meta = so3_euler_quadrature(num_samples, degree)
        meta["source"] = "euler-grid"
        return angles, weights, meta
    raise ValueError(f"Unsupported weighted node source: {source}")

def angles_to_matrices(angles):
    """Convert ZXZ Euler angles to 3x3 rotation matrices."""
    phi1, theta, phi2 = angles[:, 0], angles[:, 1], angles[:, 2]
    c1, s1 = jnp.cos(phi1), jnp.sin(phi1)
    ct, st = jnp.cos(theta), jnp.sin(theta)
    c2, s2 = jnp.cos(phi2), jnp.sin(phi2)
    
    # Rz(phi1) * Rx(theta) * Rz(phi2)
    # Rz(a) = [[cos a, -sin a, 0], [sin a, cos a, 0], [0, 0, 1]]
    # Rx(b) = [[1, 0, 0], [0, cos b, -sin b], [0, sin b, cos b]]
    r00 = c1 * c2 - s1 * ct * s2
    r01 = -c1 * s2 - s1 * ct * c2
    r02 = s1 * st
    
    r10 = s1 * c2 + c1 * ct * s2
    r11 = -s1 * s2 + c1 * ct * c2
    r12 = -c1 * st
    
    r20 = st * s2
    r21 = st * c2
    r22 = ct
    
    matrices = jnp.stack([
        jnp.stack([r00, r01, r02], axis=-1),
        jnp.stack([r10, r11, r12], axis=-1),
        jnp.stack([r20, r21, r22], axis=-1),
    ], axis=-2)
    return matrices

def compute_omega_cos_half(r1, r2):
    """Compute cos(omega(r2^{-1} r1)/2) for two rotation matrices."""
    # tr(r2^T r1) = sum_ij (r2_ij * r1_ij)
    tr = jnp.sum(r1 * r2, axis=(-2, -1))
    cos_half = 0.5 * jnp.sqrt(jnp.maximum(0.0, tr + 1.0))
    return cos_half

# ---------------------------------------------------------------------------
# 2. Wigner-D Basis Functions (Real-Valued)
# ---------------------------------------------------------------------------

def jacobi_p(n, alpha, beta, x):
    """Evaluate Jacobi polynomial P_n^(alpha, beta)(x) using recurrence."""
    if n == 0:
        return jnp.ones_like(x)
    p0 = jnp.ones_like(x)
    p1 = 0.5 * (alpha - beta + (alpha + beta + 2.0) * x)
    if n == 1:
        return p1
    
    def loop_body(i, carry):
        p_prev, p_curr = carry
        d = 2 * i + alpha + beta
        den = 2.0 * (i + 1) * (i + alpha + beta + 1.0) * d
        coeff_curr = (d + 1.0) * (d * (d + 2.0) * x + alpha**2 - beta**2) / den
        coeff_prev = 2.0 * (i + alpha) * (i + beta) * (d + 2.0) / den
        p_next = coeff_curr * p_curr - coeff_prev * p_prev
        return p_curr, p_next

    _, pn = jax.lax.fori_loop(1, n, loop_body, (p0, p1))
    return pn

def wigner_d_element(l, j, k, theta):
    """Compute little Wigner-d element d^l_{j,k}(theta) via Jacobi polynomials."""
    # j and k are representation indices, so keep these branches static.
    if j < k:
        j1, k1 = k, j
        sign1 = (-1.0) ** (j - k)
    else:
        j1, k1 = j, k
        sign1 = 1.0

    if (j1 + k1) < 0:
        j2, k2 = -k1, -j1
    else:
        j2, k2 = j1, k1
    
    n = l - j2
    alpha = j2 - k2
    beta = j2 + k2
    
    log_fac = 0.5 * (
        jax.lax.lgamma(l - j2 + 1.0) +
        jax.lax.lgamma(l + j2 + 1.0) -
        jax.lax.lgamma(l - k2 + 1.0) -
        jax.lax.lgamma(l + k2 + 1.0)
    )
    fac = jnp.exp(log_fac)
    
    sin_half = jnp.sin(theta / 2.0)
    cos_half = jnp.cos(theta / 2.0)
    
    term = sign1 * fac * (sin_half**alpha) * (cos_half**beta) * jacobi_p(n, alpha, beta, jnp.cos(theta))
    return term

def wigner_d_basis(angles, N):
    """Compute all real-valued Wigner-D basis functions up to degree N."""
    phi1, theta, phi2 = angles[:, 0], angles[:, 1], angles[:, 2]
    basis_list = []
    
    for l in range(N + 1):
        for j in range(-l, l + 1):
            for k in range(-l, l + 1):
                if j > 0 or (j == 0 and k > 0):
                    d_val = wigner_d_element(l, j, k, theta)
                    arg = j * phi1 + k * phi2
                    basis_list.append(jnp.sqrt(2.0 * (2.0 * l + 1.0)) * d_val * jnp.cos(arg))
                    basis_list.append(jnp.sqrt(2.0 * (2.0 * l + 1.0)) * d_val * jnp.sin(arg))
                elif j == 0 and k == 0:
                    d_val = wigner_d_element(l, 0, 0, theta)
                    basis_list.append(jnp.sqrt(2.0 * l + 1.0) * d_val)
                    
    return jnp.stack(basis_list, axis=-1)

wigner_d_basis_jit = jax.jit(wigner_d_basis, static_argnames=("N",))

# ---------------------------------------------------------------------------
# 3. Solver: NNLS / PGD Quadrature Weights
# ---------------------------------------------------------------------------

@jax.jit
def solve_quadrature_weights(Psi, gamma=1e-3, max_iters=2000):
    """Solve min_{w >= 0} 0.5 * ||Psi^T w - b||^2 + 0.5 * gamma * ||w||^2."""
    M, d = Psi.shape
    b = jnp.zeros(d)
    b = b.at[0].set(1.0) # Integrate constant function to 1.0, others to 0.0
    
    step_size = 1.0 / (M + gamma)
    
    def body_fn(i, w):
        error = Psi.T @ w - b
        grad = Psi @ error + gamma * w
        w_new = jnp.maximum(0.0, w - step_size * grad)
        return w_new
        
    w0 = jnp.ones(M) / M
    w_opt = jax.lax.fori_loop(0, max_iters, body_fn, w0)
    return w_opt

def quadrature_moment_diagnostics(angles, weights, degree):
    """Measure how well weights integrate Wigner-D basis functions through degree."""
    Psi = wigner_d_basis_jit(angles, degree)
    target = jnp.zeros((Psi.shape[1],), dtype=Psi.dtype)
    target = target.at[0].set(1.0)
    residual = Psi.T @ weights - target
    return {
        "moment_degree": int(degree),
        "moment_l2_error": float(jnp.linalg.norm(residual)),
        "moment_linf_error": float(jnp.max(jnp.abs(residual))),
        "weight_sum": float(jnp.sum(weights)),
        "min_weight": float(jnp.min(weights)),
        "max_weight": float(jnp.max(weights)),
    }

# ---------------------------------------------------------------------------
# 4. Solvers: Collocation and Galerkin
# ---------------------------------------------------------------------------

def tpu_solve(A, b, jitter=1e-6):
    """Solve on the active JAX backend, with a tiny diagonal jitter for stability."""
    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        A = A + jnp.eye(A.shape[0], dtype=A.dtype) * jnp.asarray(jitter, dtype=A.dtype)
    return jnp.linalg.solve(A, b)

def condition_number(A):
    """Return a finite condition number when possible."""
    return float(jnp.linalg.cond(A))

def solve_collocation_system(Psi_Xi, D_lambda, aux_dim=0):
    """Solve the positive definite or CPD augmented collocation system."""
    n, d = Psi_Xi.shape
    K_Xi = (Psi_Xi * D_lambda) @ Psi_Xi.T

    if aux_dim == 0:
        alpha = tpu_solve(K_Xi, jnp.eye(n, dtype=Psi_Xi.dtype))
        beta = jnp.zeros((0, n), dtype=Psi_Xi.dtype)
        return alpha, beta

    P_Xi = Psi_Xi[:, :aux_dim]
    
    # Construct A matrix
    # [K_Xi, P_Xi]
    # [P_Xi^T, 0 ]
    A_top = jnp.concatenate([K_Xi, P_Xi], axis=1)
    A_bot = jnp.concatenate([P_Xi.T, jnp.zeros((aux_dim, aux_dim), dtype=Psi_Xi.dtype)], axis=1)
    A = jnp.concatenate([A_top, A_bot], axis=0)
    
    rhs = jnp.concatenate([jnp.eye(n, dtype=Psi_Xi.dtype), jnp.zeros((aux_dim, n), dtype=Psi_Xi.dtype)], axis=0)
    C = tpu_solve(A, rhs)
    
    alpha = C[:n, :] # (n, n)
    beta = C[n:, :]  # (aux_dim, n)
    return alpha, beta

def get_eigenvalues(N, m, operator):
    """Get operator eigenvalues and matching fundamental-kernel coefficients."""
    eigenvals_mu = []
    eigenvals_lambda = []
    
    for l in range(N + 1):
        lam_l = l * (l + 1)
        if operator == "sobolev":
            mu_l = (1.0 + lam_l)**m
        elif operator == "l3":
            # L_3 = 4! / (16 pi) * (Delta + 1/4)(Delta - 3/4)(Delta - 35/4)
            mu_l = (24.0 / (16.0 * jnp.pi)) * (-lam_l + 0.25) * (-lam_l - 0.75) * (-lam_l - 8.75)
        else:
            raise ValueError(f"Unsupported operator: {operator}")
        lambda_l = 1.0 / mu_l
        
        mult = (2 * l + 1)**2
        eigenvals_mu.extend([mu_l] * mult)
        eigenvals_lambda.extend([lambda_l] * mult)
        
    return jnp.array(eigenvals_mu, dtype=DTYPE), jnp.array(eigenvals_lambda, dtype=DTYPE)

def target_basis_index(L):
    """Index of the real Wigner-D basis element with degree L and j=k=0."""
    idx = sum((2 * l + 1) ** 2 for l in range(L))
    for j in range(-L, L + 1):
        for k in range(-L, L + 1):
            if j == 0 and k == 0:
                return idx
            if j > 0 or (j == 0 and k > 0):
                idx += 2
    raise ValueError(f"Could not find target basis index for L={L}")

def eval_lagrange(Psi_eval, Psi_Xi, kernel_lambda, alpha, beta):
    """Evaluate Lagrange basis functions at nodes represented by Psi_eval."""
    chi = (Psi_eval * kernel_lambda) @ (Psi_Xi.T @ alpha.T)
    if beta.shape[0]:
        chi = chi + Psi_eval[:, :beta.shape[0]] @ beta
    return chi

def eval_operator_lagrange(Psi_eval, Psi_Xi, kernel_lambda, operator_mu, alpha, beta):
    """Evaluate L chi. For a fundamental kernel, mu_l * lambda_l = 1."""
    l_chi = (Psi_eval * (operator_mu * kernel_lambda)) @ (Psi_Xi.T @ alpha.T)
    if beta.shape[0]:
        l_chi = l_chi + (Psi_eval[:, :beta.shape[0]] * operator_mu[:beta.shape[0]]) @ beta
    return l_chi

def run_experiment(
    Xi_angles,
    Lambda_angles,
    Lambda_weights,
    Test_angles,
    Test_weights,
    N,
    m,
    test_L,
    operator,
    aux_dim,
    quadrature_source,
    diagnostics=False,
    moment_degree=None,
):
    """Run a single Galerkin solver experiment and compute L2 error."""
    n = Xi_angles.shape[0]
    if test_L > N:
        raise ValueError(f"test_degree={test_L} must be <= N={N} for the truncated basis")
    
    # Compute basis functions
    Psi_Xi = wigner_d_basis_jit(Xi_angles, N)
    Psi_Lambda = wigner_d_basis_jit(Lambda_angles, N)
    Psi_Test = wigner_d_basis_jit(Test_angles, N)
    
    mu, kernel_lambda = get_eigenvalues(N, m, operator)
    
    # Collocation
    alpha, beta = solve_collocation_system(Psi_Xi, kernel_lambda, aux_dim=aux_dim)
    
    # Quadrature weights
    if quadrature_source == "learned":
        w = solve_quadrature_weights(Psi_Lambda)
    else:
        w = Lambda_weights
    
    Chi_Lambda = eval_lagrange(Psi_Lambda, Psi_Xi, kernel_lambda, alpha, beta)
    L_Chi_Lambda = eval_operator_lagrange(Psi_Lambda, Psi_Xi, kernel_lambda, mu, alpha, beta)
    
    # Stiffness matrix: B^Lambda = L_Chi_Lambda^T * diag(w) * Chi_Lambda
    B_Lambda = L_Chi_Lambda.T @ (w[:, None] * Chi_Lambda)
    
    # Build right hand side load vector: f = D^test_L_{0,0}
    # Note: Test RHS is the character of degree test_L.
    # The character of degree L is Chi^L(R) = U_2L(cos(omega/2)).
    # We choose one specific Wigner-D function of degree test_L: Psi_test_L
    # Since all Wigner-D functions are orthogonal, we choose the one corresponding to j=0, k=0.
    # Its index in the basis list can be found.
    # Let's write down the exact solution: u_exact = (1/mu_test_L) * Psi_test_L
    # Index of l=test_L, j=0, k=0 in the basis list:
    idx_target = target_basis_index(test_L)
                
    # RHS function f is the basis function itself: Psi_test_L
    f_Lambda = Psi_Lambda[:, idx_target]
    
    # Load vector: omega^Lambda = Chi_Lambda^T * diag(w) * f_Lambda
    omega_Lambda = Chi_Lambda.T @ (w * f_Lambda)
    
    # Solve Galerkin system: B_Lambda * gamma = omega_Lambda
    gamma = tpu_solve(B_Lambda, omega_Lambda)
    
    # Galerkin solution evaluated at Test points: u_approx = Chi_Test * gamma
    Chi_Test = eval_lagrange(Psi_Test, Psi_Xi, kernel_lambda, alpha, beta)
    u_approx = Chi_Test @ gamma
    
    # Exact solution evaluated at Test points
    u_exact = Psi_Test[:, idx_target] / mu[idx_target]
    
    # L2 Error
    if Test_weights is None:
        error_l2 = jnp.sqrt(jnp.mean((u_approx - u_exact)**2))
    else:
        error_l2 = jnp.sqrt(jnp.sum(Test_weights * (u_approx - u_exact)**2))

    out = {
        "l2_error": float(error_l2),
        "n_Lambda_actual": int(Lambda_angles.shape[0]),
    }

    if diagnostics:
        K_Xi = (Psi_Xi * kernel_lambda) @ Psi_Xi.T
        solve_residual = B_Lambda @ gamma - omega_Lambda
        out.update({
            "collocation_cond": condition_number(K_Xi),
            "stiffness_cond": condition_number(B_Lambda),
            "relative_solve_residual": float(
                jnp.linalg.norm(solve_residual) / (jnp.linalg.norm(omega_Lambda) + jnp.asarray(1e-12, dtype=DTYPE))
            ),
        })
        if moment_degree is not None:
            out["quadrature_diagnostics"] = quadrature_moment_diagnostics(Lambda_angles, w, moment_degree)

    return out

# ---------------------------------------------------------------------------
# 5. Master Sweep Script
# ---------------------------------------------------------------------------

def run_sweeps(args):
    t0 = time.time()
    print("=" * 80)
    print("Program AQ — SO(3) Kernel-Galerkin Solver Sweep")
    print("=" * 80)
    print(f"Backend: {jax.default_backend()} | devices={len(jax.devices())} | dtype={DTYPE}")
    print(
        f"Centers: {args.dataset_source} | quadrature={args.quadrature_source} | "
        f"validation={args.validation_source} | operator={args.operator} | aux_dim={args.aux_dim}"
    )
    if args.require_accelerator and jax.default_backend() == "cpu":
        raise SystemExit("--require-accelerator was set, but JAX only sees CPU")
    
    m = args.m
    seeds = args.seeds if args.seeds else [args.seed]
    validation_degree = args.validation_degree if args.validation_degree >= 0 else 2 * max(args.N_truncation)
    test_meta = None
    results = []

    for seed in seeds:
        print(f"[SEED] {seed}")
        key = jax.random.PRNGKey(seed)

        # Generate validation points
        key, subkey = jax.random.split(key)
        Test_angles, Test_weights, test_meta = make_weighted_so3_nodes(
            subkey,
            args.n_test,
            args.validation_source,
            validation_degree,
        )

        for n_Xi in args.n_centers:
            for N in args.N_truncation:
                key, k1 = jax.random.split(key)
                Xi_angles = make_so3_dataset(k1, n_Xi, args.dataset_source)
                for q in args.oversampling_ratios:
                    n_Lambda = q * n_Xi

                    print(f"Running: seed={seed}, |Xi|={n_Xi}, N={N}, q={q} (|Lambda|={n_Lambda})...")

                    key, k2 = jax.random.split(key)
                    moment_degree = args.moment_degree if args.moment_degree >= 0 else 2 * N
                    if args.quadrature_source == "learned":
                        Lambda_angles = make_so3_dataset(k2, n_Lambda, args.dataset_source)
                        Lambda_weights = None
                        quadrature_meta = {
                            "source": "learned",
                            "node_source": args.dataset_source,
                            "requested_samples": int(n_Lambda),
                            "actual_samples": int(n_Lambda),
                        }
                    else:
                        Lambda_angles, Lambda_weights, quadrature_meta = make_weighted_so3_nodes(
                            k2,
                            n_Lambda,
                            args.quadrature_source,
                            moment_degree,
                        )

                    try:
                        run_t0 = time.time()
                        exp_result = run_experiment(
                            Xi_angles,
                            Lambda_angles,
                            Lambda_weights,
                            Test_angles,
                            Test_weights,
                            N,
                            m,
                            args.test_degree,
                            args.operator,
                            args.aux_dim,
                            args.quadrature_source,
                            diagnostics=args.diagnostics,
                            moment_degree=moment_degree,
                        )
                        case_runtime = time.time() - run_t0
                        err = exp_result["l2_error"]
                        actual_lambda = exp_result["n_Lambda_actual"]
                        print(
                            f"  -> L2 Error: {err:.6e} "
                            f"(actual |Lambda|={actual_lambda}, runtime={case_runtime:.2f}s)"
                        )
                        row = {
                            "seed": int(seed),
                            "n_Xi": int(n_Xi),
                            "N": int(N),
                            "q": int(q),
                            "q_actual": float(actual_lambda / n_Xi),
                            "n_Lambda": int(n_Lambda),
                            "n_Lambda_actual": int(actual_lambda),
                            "case_runtime_seconds": case_runtime,
                            "quadrature_meta": quadrature_meta,
                        }
                        row.update(exp_result)
                        results.append(row)
                    except Exception as e:
                        print(f"  -> Failed: {e}")

                    if SHUTDOWN_FLAG:
                        print("[WATCHDOG] Exiting due to preemption!")
                        sys.exit(0)
                    
    # Write summary
    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.json")
    
    out = {
        "experiment": "SO3_kernel_galerkin_sweep",
        "m": m,
        "test_degree": args.test_degree,
        "seed": args.seed,
        "seeds": [int(s) for s in seeds],
        "dataset_source": args.dataset_source,
        "quadrature_source": args.quadrature_source,
        "validation_source": args.validation_source,
        "validation_meta": test_meta,
        "dataset_rationale": (
            "Haar-uniform synthetic SO(3) nodes are the primary validation set because "
            "Collins' estimates are stated against SO(3) fill distance and Haar measure. "
            "Real pose datasets such as ModelNet40-C are downstream non-uniform stress tests, "
            "not the calibration dataset for Corollary 6.6.8."
        ),
        "operator": args.operator,
        "aux_dim": args.aux_dim,
        "diagnostics": bool(args.diagnostics),
        "moment_degree_arg": int(args.moment_degree),
        "validation_degree": int(validation_degree),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(d) for d in jax.devices()],
        "dtype": str(DTYPE),
        "runtime_seconds": time.time() - t0,
        "results": results
    }
    
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2)
        
    print(f"Results written to {summary_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-centers", nargs="+", type=int, default=[100, 200, 500])
    p.add_argument("--N-truncation", nargs="+", type=int, default=[3, 5, 7])
    p.add_argument("--oversampling-ratios", nargs="+", type=int, default=[10])
    p.add_argument("--test-degree", type=int, default=2)
    p.add_argument("--m", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--dataset-source", choices=["haar"], default="haar")
    p.add_argument("--quadrature-source", choices=["learned", "haar", "euler-grid"], default="learned")
    p.add_argument("--validation-source", choices=["haar", "euler-grid"], default="haar")
    p.add_argument("--operator", choices=["sobolev", "l3"], default="sobolev")
    p.add_argument("--aux-dim", type=int, default=0)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--moment-degree", type=int, default=-1)
    p.add_argument("--validation-degree", type=int, default=-1)
    p.add_argument("--diagnostics", action="store_true")
    p.add_argument("--require-accelerator", action="store_true")
    p.add_argument("--out-dir", default="program_aq_results")
    p.add_argument("--smoke-test", action="store_true")
    args = p.parse_args()
    
    if args.smoke_test:
        args.n_centers = [50]
        args.N_truncation = [3]
        args.oversampling_ratios = [5]
        args.n_test = min(args.n_test, 200)
        print("[SMOKE TEST MODE]")
        
    run_sweeps(args)
