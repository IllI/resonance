"""
clip_dlinoss_mapper.py
───────────────────────────────────────────────────────────────
D-LinOSS-powered JourneyCLIPMapper — all 6 subjects.

The JourneyCLIPMapper concept (from map_clip_superposition.py):
  fMRI journey → CLIP 512D space

This version replaces the LSTM with D-LinOSS oscillator encoding,
uses F-test voxel selection (same as Haxby pipeline), and runs
all 6 subjects with cross-subject analysis.

The CLIP target space is where:
  - Language models encode "a [category]" as text
  - Stable Diffusion / visual generative models encode the image prior
  - Human cultural programming of the concept converges

The Haxby categories (all 8) are human semantic constructs —
even face and cat are culturally mediated categories, not biological
universals. The ghost basin is the convergent electromagnetic trace
of brains running the same shared cultural hallucination.

CLIP is the right target because it was jointly trained on
(image, text) pairs produced by the same cultural programming
that shaped the subjects' category knowledge.
"""

import os, sys, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import f_oneway
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from run_visual_dlinoss import load_haxby_subject, build_block_labels, extract_brain_voxels
from dlinoss.dlinoss_torch import DLinOSSModel
from transformers import CLIPProcessor, CLIPModel

DATA_DIR = r"C:\Users\cityz\IllI\newer_all\data\openneuro\ds000105"
SUBJECTS  = [f'sub-{i}' for i in range(1, 7)]
CATS = sorted(['bottle','cat','chair','face','house','scissors','scrambledpix','shoe'])
K = len(CATS)

# Cognitive state descriptors — the "orientation" of the hallucination
STATES = [
    "a clear, focused mental image of a",
    "a standard mental representation of a",
    "a low energy, tired mental image of a",
    "a noisy, distracted thought of a",
]


# ── CLIP target extraction ───────────────────────────────────

def load_clip_targets(device='cpu'):
    print("  Loading CLIP ViT-B/32 (cultural superposition space)...", end="", flush=True)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    texts, keys = [], []
    for cat in CATS:
        label = "scrambled nonsense static" if cat == 'scrambledpix' else cat
        for state in STATES:
            texts.append(f"{state} {label}")
            keys.append((cat, state))

    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = F.normalize(feats, dim=-1)

    # Also compute mean embedding per category (the stable superposition centroid)
    cat_centroids = {}
    for cat in CATS:
        idxs = [i for i, (c, s) in enumerate(keys) if c == cat]
        cat_centroids[cat] = feats[idxs].mean(dim=0)
    cat_centroids = {c: F.normalize(v, dim=0) for c, v in cat_centroids.items()}

    clip_dict = {keys[i]: feats[i] for i in range(len(keys))}
    print(" done")
    return clip_dict, cat_centroids, model, processor


# ── D-LinOSS Journey Mapper ──────────────────────────────────

class DLinOSSCLIPMapper(nn.Module):
    """
    Replaces the LSTM with D-LinOSS oscillator.

    The D-LinOSS oscillator captures the temporal "journey" of the
    BOLD signal — the damped oscillatory dynamics that encode the
    category hallucination building up over the block.

    Architecture:
      x[T, V] → delta_amp → W → DLinOSS → mean pool → DNN → CLIP[512]
    """
    def __init__(self, in_dim, osc_dim=96, target_dim=512):
        super().__init__()

        # Delta amplification (from original design — keeps subtle patterns)
        self.delta_amp = nn.Parameter(torch.ones(in_dim))

        # Spatial projection to oscillator input dim
        self.W = nn.Linear(in_dim, osc_dim)

        # D-LinOSS oscillator (temporal journey)
        self.dlinoss = DLinOSSModel(
            input_dim=osc_dim, state_dim=32, hidden_dim=osc_dim,
            output_dim=osc_dim, num_blocks=2, drop_rate=0.1
        )

        # Map oscillator output to CLIP space
        self.dnn = nn.Sequential(
            nn.Linear(osc_dim, osc_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(osc_dim * 2, osc_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(osc_dim * 2, target_dim),
        )

    def forward(self, x_seq):
        """
        x_seq: [T, V] — BOLD block trajectory (T timepoints, V voxels)
        Returns: [512] normalized CLIP vector
        """
        # Delta amplification — highlight subtle patterns
        x = x_seq * self.delta_amp.unsqueeze(0)   # [T, V]

        # Spatial anchoring
        x = self.W(x)                              # [T, osc_dim]

        # D-LinOSS temporal journey — captures oscillatory dynamics
        x = self.dlinoss(x.unsqueeze(0))[0]        # [T, osc_dim]

        # Mean pool over time (the settled representation)
        journey = x.mean(dim=0)                    # [osc_dim]

        # Project to CLIP space
        clip_vec = self.dnn(journey)               # [512]

        return F.normalize(clip_vec, dim=0)


# ── Training + evaluation ────────────────────────────────────

def train_mapper(events_train, clip_dict, in_dim, n_epochs=50,
                 lr=5e-4, device='cpu'):
    model = DLinOSSCLIPMapper(in_dim=in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

    for epoch in range(n_epochs):
        random.shuffle(events_train)
        loss_sum = 0
        for cat, x_ev in events_train:
            x_ev = x_ev.to(device)
            pred = model(x_ev)

            # Contrastive: maximize similarity to best-matching state of true category
            best_sim, best_target = -1.0, None
            for (tc, _), tclip in clip_dict.items():
                if tc == cat:
                    sim = F.cosine_similarity(pred.unsqueeze(0),
                                              tclip.unsqueeze(0).to(device)).item()
                    if sim > best_sim:
                        best_sim = sim
                        best_target = tclip.to(device)

            if best_target is None:
                continue

            loss = 1.0 - F.cosine_similarity(pred.unsqueeze(0),
                                              best_target.unsqueeze(0)).mean()

            # Negative sampling: push away from OTHER categories
            neg_loss = 0.0
            for neg_cat in CATS:
                if neg_cat == cat:
                    continue
                for (nc, _), nc_clip in clip_dict.items():
                    if nc == neg_cat:
                        neg_sim = F.cosine_similarity(
                            pred.unsqueeze(0), nc_clip.unsqueeze(0).to(device))
                        neg_loss += torch.clamp(neg_sim - 0.3, min=0).mean()
                        break  # one neg per category

            total_loss = loss + 0.1 * neg_loss
            opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"      epoch {epoch+1:3d}: loss={loss_sum/max(len(events_train),1):.4f}")

    return model


def evaluate(model, events_test, clip_dict, cat_centroids, device='cpu'):
    model.eval()
    correct_cat = 0
    correct_state = 0
    per_cat = defaultdict(lambda: [0, 0])  # [correct, total]
    states_used = defaultdict(int)

    with torch.no_grad():
        for true_cat, x_ev in events_test:
            pred = model(x_ev.to(device))

            # Find best match across all cat×state combinations
            best_sim, best_cat, best_state = -1.0, None, None
            for (cat, state), tclip in clip_dict.items():
                sim = F.cosine_similarity(pred.unsqueeze(0),
                                          tclip.unsqueeze(0).to(device)).item()
                if sim > best_sim:
                    best_sim = sim
                    best_cat = cat
                    best_state = state

            per_cat[true_cat][1] += 1
            if best_cat == true_cat:
                correct_cat += 1
                per_cat[true_cat][0] += 1
                states_used[best_state] += 1

    acc = correct_cat / len(events_test) if events_test else 0
    return acc, per_cat, states_used


def make_events(ts_z, all_labels, run_bounds, bold_runs, top_idx):
    """Extract block event sequences with F-test voxel selection."""
    events = []
    for ri in range(len(bold_runs)):
        s, e = run_bounds[ri], run_bounds[ri + 1]
        rl = all_labels[s:e]
        data = ts_z[top_idx, s:e].T  # [T_run, V]

        current = None
        blk_start = 0
        for i, lbl in enumerate(rl):
            if lbl != current:
                if current in CATS and i - blk_start >= 3:
                    seg = torch.tensor(
                        data[blk_start:i], dtype=torch.float32)
                    events.append((current, seg))
                current = lbl
                blk_start = i
        if current in CATS and len(rl) - blk_start >= 3:
            seg = torch.tensor(data[blk_start:], dtype=torch.float32)
            events.append((current, seg))

    return events


# ── Main ─────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("  D-LinOSS CLIP MAPPER — CULTURAL SUPERPOSITION ALIGNMENT")
    print("  fMRI electromagnetic journey → CLIP shared hallucination space")
    print("=" * 70)
    print("""
  All 8 categories are human-cultural constructs. Even face and cat
  exist in this experiment as culturally mediated categories, not
  biological universals. The ghost basin is the convergent EM trace
  of 6 brains running the same shared cultural hallucination.

  CLIP is the correct target: jointly trained on human images+text,
  it sits at the intersection of visual and linguistic cultural encoding.
""")

    clip_dict, cat_centroids, clip_model, clip_proc = load_clip_targets(device)

    # Show CLIP category geometry (the cultural superposition space)
    print("\n  CLIP cultural superposition nearest neighbors per category:")
    centroids = torch.stack([cat_centroids[c] for c in CATS])
    for i, cat in enumerate(CATS):
        sims = F.cosine_similarity(
            cat_centroids[cat].unsqueeze(0), centroids).cpu().numpy()
        sims[i] = -1
        nearest = CATS[np.argmax(sims)]
        print(f"    {cat:>14} → nearest: {nearest}")

    all_results = []

    for subj in SUBJECTS:
        subj_dir = os.path.join(DATA_DIR, subj)
        if not os.path.exists(subj_dir):
            continue

        print(f"\n{'─'*60}\n  {subj}\n{'─'*60}")
        t0 = time.time()

        # Load + F-test voxel selection
        bold_runs, events_runs, TR = load_haxby_subject(subj_dir)
        bold_concat = np.concatenate(bold_runs, axis=-1)
        ts_2d, mask, coords = extract_brain_voxels(bold_concat, percentile=20)
        V, T_total = ts_2d.shape

        mu = ts_2d.mean(axis=1, keepdims=True)
        sd = ts_2d.std(axis=1, keepdims=True); sd[sd < 1e-8] = 1.0
        ts_z = np.clip((ts_2d - mu) / sd, -5, 5)

        all_labels = []; run_bounds = [0]
        for events, bold in zip(events_runs, bold_runs):
            all_labels.extend(build_block_labels(events, bold.shape[-1], TR))
            run_bounds.append(run_bounds[-1] + bold.shape[-1])
        all_labels = all_labels[:T_total]

        cat_pats = defaultdict(list)
        for ri in range(len(bold_runs)):
            s, e = run_bounds[ri], run_bounds[ri + 1]
            rl = all_labels[s:e]
            for cat in CATS:
                cm = [i for i, l in enumerate(rl) if l == cat]
                if len(cm) >= 2:
                    cat_pats[cat].append(ts_z[:, s:e][:, cm].mean(axis=1))
        groups = [np.stack(v) for v in cat_pats.values()]
        f_stats = np.zeros(V)
        for v in range(V):
            try:
                fs, _ = f_oneway(*[g[:, v] for g in groups])
                if not np.isnan(fs): f_stats[v] = fs
            except: pass
        top500 = np.argsort(f_stats)[-500:]

        # Events: 70% train, 30% test (temporal split)
        all_events = make_events(ts_z, all_labels, run_bounds, bold_runs, top500)
        split = int(len(all_events) * 0.7)
        train_ev = all_events[:split]
        test_ev  = all_events[split:]
        print(f"  {len(train_ev)} train events, {len(test_ev)} test events")

        # Train
        n_epochs = 50
        print(f"  Training D-LinOSS CLIP mapper ({n_epochs} epochs)...")
        mapper = train_mapper(train_ev, clip_dict, in_dim=500,
                              n_epochs=n_epochs, device=device)

        # Evaluate
        acc, per_cat, states_used = evaluate(mapper, test_ev, clip_dict,
                                              cat_centroids, device)
        print(f"\n  Zero-shot CLIP accuracy: {acc*100:.1f}% "
              f"({int(acc*len(test_ev))}/{len(test_ev)})  "
              f"[chance=12.5%]")

        print(f"  Per-category:")
        for cat in CATS:
            c, t = per_cat[cat]
            print(f"    {cat:>14}: {c}/{t}")

        if states_used:
            print(f"  Cognitive states detected in correct matches:")
            for st, n in sorted(states_used.items(), key=lambda x: -x[1]):
                short = st.split(' a ')[-1].split(' mental')[0]
                print(f"    {short}: {n}")

        print(f"  Time: {time.time()-t0:.1f}s")
        all_results.append({'subject': subj, 'acc': acc, 'n_test': len(test_ev)})

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS: fMRI journey → cultural CLIP superposition")
    print(f"{'='*70}")
    print(f"\n  Subject    CLIP Acc    N test   vs chance(12.5%)")
    print(f"  {'─'*45}")
    for r in all_results:
        delta = r['acc'] * 100 - 12.5
        print(f"  {r['subject']}     {r['acc']*100:>6.1f}%    "
              f"{r['n_test']:>6}     {delta:>+.1f}%")
    avg = np.mean([r['acc'] for r in all_results])
    print(f"\n  Average: {avg*100:.1f}%  [chance=12.5%]")
    print(f"""
  Interpretation:
  If accuracy > chance, the D-LinOSS temporal dynamics of BOLD
  signals contain enough structure to be oriented toward the correct
  region of CLIP space — the cultural superposition of the category.
  This would mean: the electromagnetic hallucination happening in
  the brain is pointing toward the same semantic attractor that
  AI models (language + vision) have learned from human culture.
""")


if __name__ == "__main__":
    main()
