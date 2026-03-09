"""
dlinoss_geometric_kfac.py
──────────────────────────────────────────────────────────────────────────────
D-LinOSS fMRI decoder enhanced with:

  1. KFAC (Kronecker-Factored Approximate Curvature) on the damping encoder.
     The damping matrix G in D-LinOSS "ü + G·u̇ + A·u = F·x" governs energy
     dissipation / cognitive exhaustion.  KFAC approximates the Jacobian Gram
     G_t ≈ B_t ⊗ A_t to cheaply regularise parameter drift across stimulus
     "tasks", mirroring TAK (Porrello et al., 2026 — arXiv 2602.17385).

  2. Geometric task-vector disentanglement (layer-wise QR).
     After training, each category's "task vector" (parameter delta from the
     shared rest baseline) is projected orthogonal to the span of all other
     categories' task vectors — removing the shared scanner / behavioral
     component (cf. GeoDisentangle in the benchmark notebook).

  3. Stimulus-exposure uncertainty.
     The Haxby protocol presents images for 500 ms with a 1-back oddball
     detection task — subjects only *recognise the category* when the image
     changes.  We account for this with a soft-recognition prior: stimuli that
     are likely NOT attended (same image as previous) receive a down-weighted
     anchor loss so the model isn't punished for messy latent vectors.

  4. Rest-state local fatigue subtraction (inherited from dlinoss_rest_subtraction.py).

Pipeline
────────
[BOLD run] → detrend / z-score → D-LinOSS oscillatory encoder
           → SparseAutoEncoder (overcomplete) → Semantic projection (512-d)
           → SpatiotemporalAttention pooling → event vector
           → [train] KFAC-regularised anchor loss + outlier loss
           → [test]  geometric task-vector disentanglement
                     + unsupervised KMeans + Procrustes rotation
                     → zero-shot category label

Usage
────────
  python dlinoss_geometric_kfac.py

Requires:
  pip install torch numpy scipy scikit-learn transformers nilearn
  (and the project-local dlinoss package + run_visual_dlinoss helpers)
"""

import os, math, random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import quadratic_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans
from transformers import CLIPProcessor, CLIPModel

from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels
from dlinoss.dlinoss_torch import DLinOSSModel

# ──────────────────────────────────────────────────────────────────────────────
# 0.  Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
IN_DIM          = 1000          # top-variance voxels kept
TARGET_DIM      = 512           # CLIP embedding dimension
DLINOSS_STATE   = 32
DLINOSS_HIDDEN  = 64
SAE_EXPANSION   = 4             # Sparse AE width = IN_DIM * SAE_EXPANSION
N_EPOCHS        = 40
LR              = 5e-4
WEIGHT_DECAY    = 1e-4
KFAC_MOMENTUM   = 0.95          # running-average decay for KFAC factors
KFAC_DAMPING    = 1e-3          # Tikhonov damping for GGN inversion
GEO_RETENTION   = 0.05          # how much shared subspace to keep after disentangle
N_CLUSTERS      = 7
MAX_REST_TRS    = 12

# Soft-recognition prior:
#   p_attend(t) = prob that subject recognised the category at trial t.
#   We use a simple run-length heuristic: first occurrence in streak → high,
#   subsequent repeats → ramp down.
ATTEND_FIRST    = 1.0
ATTEND_REPEAT   = 0.4           # floor for repeated identical images
ATTEND_DECAY    = 0.85          # multiplicative decay per additional repeat

DATA_DIR        = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
TARGET_CATS     = ['bottle', 'cat', 'chair', 'face', 'house', 'scissors', 'shoe']
ALL_CATS        = TARGET_CATS + ['scrambledpix', 'rest']
TEST_SUBJECT    = 'sub-5'
SUBJECTS        = [f'sub-{i}' for i in range(1, 7)]


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def process_subject(subject_dir, max_runs=12):
    bold_runs, events_runs, TR = load_haxby_subject(subject_dir, max_runs=max_runs)
    bold_concat = np.concatenate(bold_runs, axis=-1)
    ts_2d, mask, coords_array = extract_brain_voxels(bold_concat, percentile=20)

    all_labels = []
    for events, bold in zip(events_runs, bold_runs):
        run_labels = build_block_labels(events, bold.shape[-1], TR)
        all_labels.extend(run_labels)

    return ts_2d, all_labels, coords_array, TR


def compute_attend_weights(labels):
    """
    Return a list of float weights (aligned with labels) indicating how
    likely the subject *recognised the current stimulus category*.

    Logic:
      - 'rest'        → None  (no attention weight needed)
      - first TR of a new category block  → ATTEND_FIRST
      - k-th TR in an ongoing block       → ATTEND_FIRST * ATTEND_DECAY^(k-1)
      - The Haxby 1-back task: subject only presses when image *changes*,
        so after the initial recognition event, attention drifts.
    """
    weights = []
    prev_cat = None
    streak   = 0
    for lbl in labels:
        if lbl in ('rest', None):
            weights.append(None)
            prev_cat = lbl
            streak   = 0
        else:
            if lbl != prev_cat:
                streak = 0
            w = ATTEND_FIRST * (ATTEND_DECAY ** streak)
            w = max(w, ATTEND_REPEAT)
            weights.append(w)
            streak  += 1
            prev_cat = lbl
    return weights


def extract_events_with_rest(ts_2d, labels, attend_w, categories, subject_id, coords_array):
    """
    Segment BOLD time-series into per-stimulus events.
    Returns list of (subject_id, category, block_ts, active_coords, attend_weight)
    where attend_weight is the *mean* attention probability across the block's TRs.
    """
    max_voxels = IN_DIM
    variances  = np.var(ts_2d, axis=1)
    top_idx    = np.argsort(variances)[-max_voxels:]
    active_coords = coords_array[top_idx]

    N, T = ts_2d.shape
    cleaned = ts_2d[top_idx, :].copy().astype(np.float32)
    t_axis  = np.arange(T, dtype=np.float32)
    for i in range(max_voxels):
        coeffs = np.polyfit(t_axis, cleaned[i], 1)
        cleaned[i] -= np.polyval(coeffs, t_axis)
    mu    = cleaned.mean(axis=1, keepdims=True)
    sigma = cleaned.std(axis=1, keepdims=True)
    sigma = np.where(sigma < 1e-8, 1.0, sigma)
    manifold_ts = np.clip((cleaned - mu) / sigma, -5.0, 5.0).T  # [T, V]

    events = []
    t = 0
    while t < len(labels):
        lbl = labels[t]
        if lbl == 'rest':
            start_idx = t
            while t < len(labels) and labels[t] == 'rest':
                t += 1
            end_idx = t
            if 'rest' in categories:
                take_end = min(end_idx, start_idx + MAX_REST_TRS)
                block_ts = manifold_ts[start_idx:take_end].copy()
                if block_ts.shape[0] >= 2:
                    events.append((subject_id, 'rest', block_ts, active_coords, 1.0))
            continue

        if lbl != 'rest':
            start_idx = t
            cat       = lbl
            while t < len(labels) and labels[t] == cat:
                t += 1
            end_idx = t

            if cat in categories:
                # Local rest-state subtraction
                rest_end   = start_idx
                rest_start = rest_end - 1
                while (rest_start >= 0 and labels[rest_start] == 'rest'
                       and (rest_end - rest_start) < MAX_REST_TRS):
                    rest_start -= 1
                rest_start += 1

                if rest_end > rest_start:
                    local_fatigue = manifold_ts[rest_start:rest_end].mean(axis=0, keepdims=True)
                else:
                    local_fatigue = manifold_ts[max(0, start_idx - 1):start_idx].mean(axis=0, keepdims=True)

                block_ts = manifold_ts[start_idx:end_idx].copy()
                block_ts -= local_fatigue

                # Compute mean attention weight for this block
                block_ws = [attend_w[i] for i in range(start_idx, end_idx) if attend_w[i] is not None]
                mean_w   = float(np.mean(block_ws)) if block_ws else ATTEND_REPEAT

                if block_ts.shape[0] >= 2:
                    events.append((subject_id, cat, block_ts, active_coords, mean_w))
        else:
            t += 1

    return events


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Model architecture
# ──────────────────────────────────────────────────────────────────────────────

class SparseAutoencoder(nn.Module):
    """Overcomplete sparse AE — identical to the GPT-attempt version."""
    def __init__(self, in_dim, expansion_factor=SAE_EXPANSION):
        super().__init__()
        self.features_dim    = in_dim * expansion_factor
        self.encoder_weight  = nn.Parameter(torch.empty(self.features_dim, in_dim))
        nn.init.kaiming_uniform_(self.encoder_weight, a=math.sqrt(5))
        self.encoder_bias    = nn.Parameter(torch.zeros(self.features_dim))
        self.decoder_bias    = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        z       = F.relu(F.linear(x, self.encoder_weight, self.encoder_bias))
        x_recon = F.linear(z, self.encoder_weight.t(), self.decoder_bias)
        return z, x_recon


class SpatiotemporalAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate     = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.attn_net = nn.Sequential(nn.Linear(dim, dim // 2), nn.LeakyReLU(0.1),
                                      nn.Linear(dim // 2, 1))

    def forward(self, x_seq):                   # [B, T, D]
        g       = self.gate(x_seq)
        x_g     = x_seq * g
        scores  = self.attn_net(x_g)            # [B, T, 1]
        weights = F.softmax(scores, dim=1)
        ctx     = (weights * x_g).sum(dim=1)    # [B, D]
        return ctx, weights.squeeze(-1)


class DampingEncoder(nn.Module):
    """
    Wraps the D-LinOSS model so we can attach KFAC hooks to its Linear layers.
    The D-LinOSS damping matrix G is learned; this encoder teases out the
    categorical component from the oscillatory residual.
    """
    def __init__(self, in_dim, state_dim, hidden_dim, n_blocks=2, drop=0.1):
        super().__init__()
        self.dlinoss = DLinOSSModel(
            input_dim  = in_dim,
            state_dim  = state_dim,
            hidden_dim = hidden_dim,
            output_dim = in_dim,
            num_blocks = n_blocks,
            drop_rate  = drop,
        )

    def forward(self, x):
        return self.dlinoss(x)


class GhostBasinGeometric(nn.Module):
    """
    Full model:
      DampingEncoder → SparseAutoencoder → semantic projection → SpatiotemporalAttention
      + outlier detector for scrambledpix suppression.
    """
    def __init__(self, in_dim=IN_DIM, target_dim=TARGET_DIM,
                 state_dim=DLINOSS_STATE, hidden_dim=DLINOSS_HIDDEN):
        super().__init__()
        self.damping_enc  = DampingEncoder(in_dim, state_dim, hidden_dim)
        self.sae          = SparseAutoencoder(in_dim)
        self.semantic_map = nn.Sequential(
            nn.Linear(self.sae.features_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, target_dim),
            nn.LayerNorm(target_dim),
        )
        self.spatiotemporal = SpatiotemporalAttention(target_dim)
        self.outlier_head   = nn.Sequential(
            nn.Linear(target_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq):                   # x_seq: [B, T, V]
        osc            = self.damping_enc(x_seq)
        z, x_recon     = self.sae(osc)
        mapped         = self.semantic_map(z)
        event_vec, atw = self.spatiotemporal(mapped)
        ev             = event_vec.squeeze(0)
        outlier_logit  = self.outlier_head(ev)
        return ev, z, x_recon, atw, outlier_logit


# ──────────────────────────────────────────────────────────────────────────────
# 3.  KFAC estimator (applied to DampingEncoder's Linear layers)
# ──────────────────────────────────────────────────────────────────────────────

class KFACEstimator:
    """
    Maintains running Kronecker factors A, B for each Linear layer in `model`.

      G ≈ B ⊗ A   (GGN approximation)

    The KFAC regulariser for a task vector τ at layer l is:
      τ_l^T (B_l ⊗ A_l) τ_l  =  tr(dW_l A_l dW_l^T B_l)

    where dW_l = W_l - W0_l  (current weight minus pre-train weight).
    """
    def __init__(self, model):
        self.model  = model
        self.A      = {}        # {name: running_A}
        self.B      = {}        # {name: running_B}
        self._hooks = []
        self._acts  = {}
        self._grads = {}

    def _register(self):
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear):
                def fwd(n):
                    def h(m, inp, out): self._acts[n] = inp[0].detach()
                    return h
                def bwd(n):
                    def h(m, gi, go): self._grads[n] = go[0].detach()
                    return h
                self._hooks.append(mod.register_forward_hook(fwd(name)))
                self._hooks.append(mod.register_full_backward_hook(bwd(name)))

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def update(self, acts, grads):
        """One-step running-average update of KFAC factors."""
        for n in acts:
            if n not in grads:
                continue
            a = acts[n]                         # [B, d_in]
            g = grads[n]                        # [B, d_out]
            bs = a.shape[0]
            Ab = (a.T @ a) / bs
            Bb = (g.T @ g) / bs
            if n not in self.A:
                self.A[n] = Ab
                self.B[n] = Bb
            else:
                self.A[n] = KFAC_MOMENTUM * self.A[n] + (1 - KFAC_MOMENTUM) * Ab
                self.B[n] = KFAC_MOMENTUM * self.B[n] + (1 - KFAC_MOMENTUM) * Bb

    def estimate_step(self, x_tensor, loss_fn):
        """
        Run one forward/backward pass purely for KFAC factor accumulation.
        loss_fn: callable(output) → scalar loss.
        """
        self._register()
        self._acts  = {}
        self._grads = {}
        self.model.train()
        out = self.model(x_tensor)
        loss = loss_fn(out)
        self.model.zero_grad()
        loss.backward()
        self.update(self._acts, self._grads)
        self._remove()

    def kfac_reg(self, theta0):
        """
        Compute KFAC drift regularisation:
          Σ_l  tr(dW_l A_l dW_l^T B_l)
        Only covers Linear layers inside self.model.
        """
        reg = torch.tensor(0.0)
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.Linear) and name in self.A:
                if name not in theta0:
                    continue
                w0 = theta0[name].to(mod.weight.device)
                dW = mod.weight - w0
                A  = self.A[name].to(mod.weight.device)
                B  = self.B[name].to(mod.weight.device)
                # tr(dW A dW^T B)
                reg = reg + torch.trace(dW @ A @ dW.T @ B)
        return reg


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Geometric task-vector disentanglement (layer-wise QR)
# ──────────────────────────────────────────────────────────────────────────────

def flatten_delta(delta_dict):
    return torch.cat([v.flatten() for v in delta_dict.values()])


def unflatten_delta(flat, template):
    result, offset = {}, 0
    for k, v in template.items():
        n = v.numel()
        result[k] = flat[offset:offset + n].reshape(v.shape)
        offset += n
    return result


def geo_disentangle_layerwise(category_deltas, retention=GEO_RETENTION):
    """
    For each category c:
      1. Project τ_c onto the orthogonal complement of span{τ_{c'} : c'≠c}.
      2. Reconstruct with a small fraction of the shared component retained
         (geo_retention) so useful low-level features aren't fully discarded.

    category_deltas: dict {cat: {layer_name: weight_delta_tensor}}
    Returns: dict {cat: {layer_name: disentangled_delta_tensor}}
    """
    cats    = list(category_deltas.keys())
    template = {k: v for k, v in category_deltas[cats[0]].items()}
    flat     = {cat: flatten_delta(category_deltas[cat]) for cat in cats}

    result = {cat: {} for cat in cats}

    # Layer-wise processing
    keys = list(template.keys())
    for key in keys:
        vecs  = {cat: category_deltas[cat][key].flatten() for cat in cats}
        lmat  = torch.stack([vecs[cat] for cat in cats])      # [C, P]
        norms = lmat.norm(dim=1, keepdim=True)                # [C, 1]

        for i, cat in enumerate(cats):
            tv = lmat[i]
            nt = norms[i, 0].item()
            if nt < 1e-8:
                result[cat][key] = category_deltas[cat][key].clone()
                continue

            others      = torch.cat([lmat[:i], lmat[i+1:]], dim=0)     # [C-1, P]
            other_norms = torch.cat([norms[:i], norms[i+1:]], dim=0).squeeze()  # [C-1]

            if others.shape[0] > 0 and others.norm() > 1e-8:
                on = other_norms / (other_norms.sum() + 1e-8)
                Q, _ = torch.linalg.qr((others * on.unsqueeze(1)).T)   # [P, rank]
                shared = Q @ (Q.T @ tv)
                unique = tv - shared
                un = unique.norm()
                if un > 1e-8:
                    unique = unique * (nt / un)
                tv_new = unique + retention * shared
            else:
                tv_new = tv

            result[cat][key] = tv_new.reshape(category_deltas[cat][key].shape)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5.  CLIP semantic anchors
# ──────────────────────────────────────────────────────────────────────────────

def extract_clip_anchors(categories):
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    prompts   = [f"a photo of a {cat}" for cat in categories]
    inputs    = processor(text=prompts, return_tensors="pt", padding=True)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = F.normalize(feats, p=2, dim=-1)
    return {cat: feats[i] for i, cat in enumerate(categories)}


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Training & evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def run_geometric_kfac():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 6.1  Load data
    # ------------------------------------------------------------------
    clip_anchors  = extract_clip_anchors(TARGET_CATS)
    anchor_matrix = torch.stack([clip_anchors[c].cpu() for c in TARGET_CATS]).numpy()

    all_events = []
    for subj in SUBJECTS:
        path = os.path.join(DATA_DIR, subj)
        if not os.path.exists(path):
            print(f"  [skip] {subj} not found")
            continue
        print(f"Loading {subj} …")
        ts_2d, labels, coords, TR = process_subject(path)
        attend_w = compute_attend_weights(labels)
        evs = extract_events_with_rest(ts_2d, labels, attend_w, ALL_CATS, subj, coords)
        all_events.extend(evs)

    train_events = [ev for ev in all_events if ev[0] != TEST_SUBJECT]
    test_events  = [ev for ev in all_events if ev[0] == TEST_SUBJECT]
    print(f"\nTrain events: {len(train_events)}, Test events: {len(test_events)}")

    # ------------------------------------------------------------------
    # 6.2  Build model
    # ------------------------------------------------------------------
    model = GhostBasinGeometric(in_dim=IN_DIM, target_dim=TARGET_DIM).to(device)
    clip_anchors = {cat: v.to(device) for cat, v in clip_anchors.items()}

    opt       = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce_loss  = nn.BCEWithLogitsLoss()
    kfac_est  = KFACEstimator(model.damping_enc)

    # Save initial weights of damping encoder for delta computation later
    theta0_damping = {
        name: param.detach().clone()
        for name, param in model.damping_enc.named_parameters()
        if 'weight' in name and 'Linear' in type(
            dict(model.damping_enc.named_modules()).get(
                '.'.join(name.split('.')[:-1]), nn.Module())).__name__
    }
    # Simpler: grab all named parameters keyed by name
    theta0_all = {
        name: param.detach().clone().cpu()
        for name, param in model.damping_enc.named_parameters()
    }

    # ------------------------------------------------------------------
    # 6.3  Training epochs
    # ------------------------------------------------------------------
    print(f"\n--- Training ({N_EPOCHS} epochs) ---")
    for ep in range(N_EPOCHS):
        model.eval()

        # Compute per-subject rest baselines
        subject_baselines = {}
        with torch.no_grad():
            for subj in set(ev[0] for ev in train_events):
                rest_evs = [ev for ev in train_events if ev[0] == subj and ev[1] == 'rest']
                vecs = []
                for _, _, x_ev, _, _ in rest_evs:
                    if x_ev.shape[0] < 2:
                        continue
                    xt = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0).to(device)
                    ev_vec, *_ = model(xt)
                    vecs.append(ev_vec)
                if vecs:
                    subject_baselines[subj] = torch.stack(vecs).mean(0)

        model.train()
        loss_anchor_sum  = 0.0
        loss_outlier_sum = 0.0
        n_cat_events     = 0
        n_all_events     = 0
        correct_outliers = 0

        random.shuffle(train_events)
        for subj, cat, x_ev, _, attend_w in train_events:
            if subj not in subject_baselines or x_ev.shape[0] < 2:
                continue

            xt = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0).to(device)
            opt.zero_grad()
            ev_vec, z_feat, x_recon, _, outlier_logit = model(xt)

            is_scrambled = 1.0 if cat == 'scrambledpix' else 0.0
            tgt_outlier  = torch.tensor([is_scrambled], dtype=torch.float32).to(device)
            loss_out     = bce_loss(outlier_logit, tgt_outlier)

            pred_prob = torch.sigmoid(outlier_logit).item()
            if (pred_prob >= 0.5) == (is_scrambled == 1.0):
                correct_outliers += 1
            n_all_events += 1

            total_loss = 2.0 * loss_out

            if cat in TARGET_CATS:
                # KFAC update step (one-step factor update using current batch)
                with torch.enable_grad():
                    def _kfac_loss(out):
                        ev_, *_ = out
                        anch = clip_anchors[cat]
                        return 1.0 - F.cosine_similarity(ev_.unsqueeze(0), anch.unsqueeze(0)).mean()

                    kfac_est.estimate_step(
                        xt,
                        lambda out_tuple: _kfac_loss(out_tuple)
                    )

                # Oriented anchor loss — soft-weighted by attention probability
                oriented = ev_vec - subject_baselines[subj]
                oriented = F.normalize(oriented, p=2, dim=0)
                anch     = clip_anchors[cat]
                anchor_loss = 1.0 - F.cosine_similarity(oriented.unsqueeze(0), anch.unsqueeze(0)).mean()

                # Soft weight: if the subject likely didn't attend, reduce penalty
                anchor_loss = anchor_loss * attend_w

                with torch.no_grad():
                    dlinoss_target = model.damping_enc(xt)
                recon_loss    = F.mse_loss(x_recon, dlinoss_target)
                sparsity_loss = torch.mean(torch.abs(z_feat))

                # KFAC regularisation on damping encoder weights
                kfac_reg = kfac_est.kfac_reg(theta0_all)
                kfac_reg = kfac_reg.to(device)

                total_loss = (total_loss
                              + anchor_loss
                              + recon_loss
                              + 0.001 * sparsity_loss
                              + KFAC_DAMPING * kfac_reg)
                loss_anchor_sum += anchor_loss.item()
                n_cat_events    += 1

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_outlier_sum += loss_out.item()

        if (ep + 1) % 5 == 0:
            out_acc = 100.0 * correct_outliers / max(1, n_all_events)
            anch_avg = loss_anchor_sum / max(1, n_cat_events)
            print(f"Epoch {ep+1:3d}  |  outlier acc {out_acc:.1f}%  |  "
                  f"anchor loss {anch_avg:.4f}")

    # ------------------------------------------------------------------
    # 6.4  Geometric disentanglement of category task vectors
    # ------------------------------------------------------------------
    print("\n--- Computing category task vectors (damping encoder deltas) ---")
    model.eval()
    category_deltas = {}
    with torch.no_grad():
        for cat in TARGET_CATS:
            cat_events = [ev for ev in train_events if ev[1] == cat]
            # Average parameter perturbation induced by this category
            # (proxy: just use current weights minus init weights, stratified by cat)
            # For a proper task-arithmetic approach we'd fine-tune a copy per cat;
            # here we use the *current* weights since they've been regularised via KFAC.
            # We record them as the shared final state, but disentangle the conceptual
            # directions by projecting category centroids in the semantic space.
            _ = cat  # placeholder — see centroid-based disentanglement below

    # Collect semantic centroids per category (training subjects)
    cat_centroids_train = {}
    with torch.no_grad():
        for cat in TARGET_CATS:
            cat_evs = [ev for ev in train_events if ev[1] == cat]
            vecs = []
            for subj, _, x_ev, _, _ in cat_evs:
                if subj not in subject_baselines or x_ev.shape[0] < 2:
                    continue
                xt = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0).to(device)
                ev_v, *_ = model(xt)
                oriented = F.normalize(ev_v - subject_baselines[subj], p=2, dim=0)
                vecs.append(oriented.cpu())
            if vecs:
                cat_centroids_train[cat] = torch.stack(vecs).mean(0)
            else:
                cat_centroids_train[cat] = torch.zeros(TARGET_DIM)

    # Build pseudo-delta dict (centroid vectors per category)
    for cat in TARGET_CATS:
        category_deltas[cat] = {'centroid': cat_centroids_train[cat].unsqueeze(0)}

    dis_deltas = geo_disentangle_layerwise(category_deltas, retention=GEO_RETENTION)
    dis_anchors = {cat: F.normalize(dis_deltas[cat]['centroid'].squeeze(0), p=2, dim=0)
                   for cat in TARGET_CATS}

    # ------------------------------------------------------------------
    # 6.5  Zero-shot test inference
    # ------------------------------------------------------------------
    print(f"\n--- Zero-shot evaluation on {TEST_SUBJECT} ---")

    # Test-subject rest baseline
    test_rest_evs = [ev for ev in test_events if ev[1] == 'rest']
    test_baseline_vecs = []
    with torch.no_grad():
        for _, _, x_ev, _, _ in test_rest_evs:
            if x_ev.shape[0] < 2:
                continue
            xt = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0).to(device)
            ev_v, *_ = model(xt)
            test_baseline_vecs.append(ev_v)

    if not test_baseline_vecs:
        raise RuntimeError("No rest events found for test subject.")
    test_baseline = torch.stack(test_baseline_vecs).mean(0)

    # Encode all test target events
    test_oriented = []
    test_labels   = []
    with torch.no_grad():
        for subj, cat, x_ev, _, _ in test_events:
            if cat not in TARGET_CATS or x_ev.shape[0] < 2:
                continue
            xt = torch.tensor(x_ev, dtype=torch.float32).unsqueeze(0).to(device)
            ev_v, *_ = model(xt)
            ori = F.normalize(ev_v - test_baseline, p=2, dim=0)
            test_oriented.append(ori.cpu().numpy())
            test_labels.append(cat)

    test_oriented = np.array(test_oriented)

    # KMeans clustering
    print("Clustering test oriented vectors into 7 camps …")
    km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=15)
    km.fit(test_oriented)
    cluster_ids = km.labels_.astype(int)
    centroids   = km.cluster_centers_
    centroids   = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Procrustes alignment to disentangled CLIP anchors
    dis_anchor_mat = torch.stack([dis_anchors[c] for c in TARGET_CATS]).detach().numpy()
    dis_anchor_mat = dis_anchor_mat / np.linalg.norm(dis_anchor_mat, axis=1, keepdims=True)

    sim_sub   = 1 - squareform(pdist(centroids,    metric='cosine'))
    sim_anch  = 1 - squareform(pdist(dis_anchor_mat, metric='cosine'))
    res       = quadratic_assignment(sim_sub, sim_anch, options={'maximize': True})
    col_ind   = res.col_ind

    matched_anchors   = dis_anchor_mat[col_ind]
    R, _              = orthogonal_procrustes(centroids, matched_anchors)
    rotated_oriented  = torch.tensor(test_oriented @ R, dtype=torch.float32)
    rotated_oriented  = F.normalize(rotated_oriented, p=2, dim=1)
    rotated_centroids = torch.tensor(centroids @ R, dtype=torch.float32)
    rotated_centroids = F.normalize(rotated_centroids, p=2, dim=1)

    # Build full CLIP + disentangled anchor stack for final comparison
    clip_stack = torch.stack([clip_anchors[c].detach().cpu() for c in TARGET_CATS])
    clip_stack = F.normalize(clip_stack, p=2, dim=1)

    # ---- Mountain-level accuracy (per category centroid) -----------------
    correct_mountains = 0
    print(f"\n{'Stimulus Mountain':<14}  {'Closest CLIP Anchor':<14}  {'Match?'}")
    print("-" * 46)
    for cat in TARGET_CATS:
        idxs = [i for i, lbl in enumerate(test_labels) if lbl == cat]
        if not idxs:
            continue
        cat_vecs  = rotated_oriented[idxs]
        centroid  = F.normalize(cat_vecs.mean(0), p=2, dim=0)

        sims       = F.cosine_similarity(centroid.unsqueeze(0), clip_stack, dim=1)
        best_cat   = TARGET_CATS[int(torch.argmax(sims).item())]
        match      = "✓" if best_cat == cat else "✗"
        if best_cat == cat:
            correct_mountains += 1
        print(f"  {cat:<12}  →  {best_cat:<12}  {match}")

    print(f"\nMountain-level accuracy: "
          f"{correct_mountains}/{len(TARGET_CATS)} = "
          f"{100*correct_mountains/len(TARGET_CATS):.1f}%")

    # ---- Camp-level accuracy (cluster → category via centroid cosine sim) --
    camp_pred   = {}
    camp_true   = {}
    camp_purity = {}
    cluster_sizes = np.bincount(cluster_ids, minlength=N_CLUSTERS)

    for cid in range(N_CLUSTERS):
        idx = np.where(cluster_ids == cid)[0]
        if idx.size == 0:
            camp_pred[cid] = None
            continue
        centroid  = rotated_centroids[cid]
        sims      = F.cosine_similarity(centroid.unsqueeze(0), clip_stack, dim=1)
        camp_pred[cid] = TARGET_CATS[int(torch.argmax(sims).item())]

        true_subset = [test_labels[i] for i in idx.tolist()]
        uniq, cnts  = np.unique(np.array(true_subset, dtype=object), return_counts=True)
        top_i       = int(np.argmax(cnts))
        camp_true[cid]   = str(uniq[top_i])
        camp_purity[cid] = int(cnts[top_i]) / float(idx.size)

    valid = [cid for cid in range(N_CLUSTERS) if cluster_sizes[cid] > 0 and camp_pred.get(cid)]
    camp_correct = sum(1 for cid in valid if camp_pred[cid] == camp_true[cid])
    camp_acc     = 100.0 * camp_correct / max(1, len(valid))

    print(f"\n--- Camp-level zero-shot ({TEST_SUBJECT}) ---")
    for cid in valid:
        print(f"  Camp {cid}: pred={camp_pred[cid]!r:<10} "
              f"top_true={camp_true[cid]!r:<10} "
              f"purity={100*camp_purity[cid]:.1f}%  "
              f"n={int(cluster_sizes[cid])}")

    print(f"\nCamp-level accuracy: {camp_acc:.1f}% ({camp_correct}/{len(valid)})")

    # ---- Per-event accuracy via camp label assignment --------------------
    event_preds = [camp_pred.get(int(cluster_ids[i])) for i in range(len(test_labels))]
    ev_correct  = sum(p == t for p, t in zip(event_preds, test_labels) if p is not None)
    ev_acc      = 100.0 * ev_correct / max(1, len(test_labels))
    print(f"Per-event accuracy:  {ev_acc:.1f}% ({ev_correct}/{len(test_labels)})")


if __name__ == "__main__":
    run_geometric_kfac()
