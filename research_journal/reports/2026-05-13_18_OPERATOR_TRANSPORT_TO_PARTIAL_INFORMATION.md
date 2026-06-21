# May 13–18: Operator Transport, Prediction, Control, and Partial Information

This range contains 68 commits. Scientific evidence and TPU-launcher evolution are treated separately.

## Program C / Session 2 — May 13

Commit `386724609f` corrected Bell-reference, normalization, complex-dtype, and optimizer errors in a full teleportation simulation. Calibration exposed an omitted off-diagonal `T_yz=-0.497`, invalidating a diagonal-PTM assumption. The committed JSON reports average fidelity 0.75843 at the selected point, exact 0.5 at `chi*t=pi`, and close numerical/theory agreement. The newly exposed `T_yz` drove later operator-space features and PTM-aligned recovery.

Assessment: high-confidence simulation and exact-null control.

## D-LinOSS operator-space bridge and Programs E/F — May 14

Commit `ee7f5ae7af` changed D-LinOSS from scalar classification to a 34-dimensional operator-space reconstruction using full PTM features, singular/eigenvalues, rank, gradients, `V_Q`, and singularity proximity. No separate result artifact was committed.

Program E (`4c74b4fefa`) hid fidelity and labels and reported blind enrichment of transport-like operator strata. Program F (`13957752ae`, `0a282cc73c`) attacked the smooth-trajectory explanation with scrambling, spectrum-preserving randomized orientations, separable rank-1 samples, and held-out Hamiltonians. The full run reported enrichment 3.043 with 95% CI [2.887, 3.177]. A high-fidelity randomized-orientation control classified 100% as transport, appropriately showing that the method tracks invariant PTM spectrum rather than OAT orientation.

Assessment: F is one of the stronger D-LinOSS validation families, but it validates blind invariant-spectrum stratification—not source-specific quantum identification.

## Programs G/G2 — operational recovery

Commits `d244e909c7` and `b3309725ff` fixed a partial-trace bug, verified Bell-pair fidelity 1, and replaced naive laboratory-basis correction with SVD-derived PTM alignment. G recorded standard fidelity 0.624 versus optimized 0.769, with the singular null near 0.5. G2 reported PTM-aligned recovery matching the optimized value to 0.001.

A record conflict remains: G2 attributed the earlier 0.719 value to an optimizer local minimum and reported about 0.770, while the May 15 editorial paper deliberately retained 0.719. This needs a clean rerun and definition audit.

Assessment: strong evidence that basis alignment matters; unresolved numerical lineage for the headline `N=4` value.

## Programs H/H2/H3 — finite-size scaling

Commits `4ad6696813`, `4e43b60f6a`, `156df3dbd3`, and `8819cc67d5` successively expanded and corrected scaling estimates. Early exponents were superseded as grid floors and sampling limits became visible. H3 used a denser adaptive grid and reported recoverable enhancement `0.285*N^-0.468` (`R^2=0.992`) while concurrence decayed much faster; enhancement remained above 0.01 through `N=256`, and the exact null remained 0.5.

Assessment: good evidence that the chosen recoverability metric persists beyond detectable pairwise concurrence over the simulated range. It does not establish a nonzero thermodynamic-limit advantage.

## Programs I0/I+ — representation stability

Commits `8595188408`, `a472974eeb`, `f1d90d0999`, `be16d91f1c`, and `ae87f52c48` compared PTM recovery fronts with concurrence, mutual information, entropy, and operator spreading at multiple sizes. XXZ observables converged; XY showed selective PTM lag; Ising propagated MI/operator-spreading signals while PTM and entropy recovery fronts remained null. The I+ four-representation coefficient of variation was 0.084 for XXZ, 0.365 for XY, and 1.042 for Ising.

Assessment: high-value negative differentiation—information propagation is not equivalent to organized recoverability.

## Programs J/K — blind prediction and causal qualification

Program J (`117e0fdba5`) predicted above-classical recoverability from entropy and MI alone. Precision was excellent on XY/XXZ but poor on Ising; temporal ordering was unresolved at coarse resolution.

Program K (`daadd699aa`, `a286587dd6`, `396692fec7`, `c1748205d9`, `8ea6475f8b`, `d8060e0303`, `a57311928b`, `04c2e91bd7`) added adaptive grids, operator-spreading features, frozen thresholds, basis scrambling, disorder, and adversarial models. Cross-model precision reached 0.920 at `N=10` and 0.951 at `N=12`, with recall around 0.6. However, a stricter v2 artifact reports causal fraction zero for XXZ and model-level causal verdicts FAIL. Commit subjects saying “causal ordering confirmed” are therefore overstated: proximity/simultaneity is supported, strict precedence is not. The full `N=14` campaign was repeatedly preempted and incomplete.

Assessment: strong blind-prediction result, failed strict-causality claim.

## Program N — adaptive controller calibration, May 16–17

The N-series iteratively calibrated response Jacobians, susceptibility gates, sparsity, stable-basin definitions, null switches, and disorder vetoes. Several mechanisms were tried and retired: `dx_norm` hit a shot-noise floor; an N6 null switch was ineffective; an N8 front-velocity veto was not W3-specific. The documented N9 hierarchy favored clean XXZ, sparsely favored W1, approached abstention on W3, and did nothing on OAT.

N10a has the strongest raw evidence: 50 W3 seeds gave mean lifetime change -0.384 with 95% CI [-0.771, +0.003], only 7/50 positive, and 43/50 with no gates fired. This is an intervention-neutral boundary, not a successful controller regime.

Assessment: valuable controller-abstention calibration; many earlier numerical summaries are Markdown-only.

## Program O — endpoint encoded-state fidelity, May 17–18

Commits `d8ec209c8a` and `741a2a32a9` shifted from transport lifetime to endpoint fidelity across multiple probe states. The preregistration expected gains in XXZ/W1 and near-zero effects in W3/OAT. The actual static-versus-agnostic run reported positive mean gains in all four families, including large OAT and W3 gains.

Two corrections are essential:

- The commit subject says 57/60 passed; the committed JSON contains 51 true and 9 false entries.
- W3 and OAT strongly contradict the preregistered near-zero expectation. This is evidence against the N-derived endpoint-fidelity hierarchy, not simple confirmation of universal rescue.

Assessment: high-confidence raw comparison, scientifically surprising and deserving independent replication.

## Program P — partial information recovery, May 18

Commits `8f119f2f38` and `7a01701e43` implemented hidden-phase encoding, degradation channels, reconstruction, and circular-error/information metrics. No Program P result artifact was committed in this interval.

Assessment: implemented/planned only.

## Infrastructure lineage

Launcher changes explain missing or delayed runs but are not scientific results. The important operational sequence was:

- unsupported `--best-effort` reverted to `--spot`;
- premature SSH verification and incompatible flags reverted to the established Windows `y`-pipe flow;
- version-adaptive v4 JAX installation reverted because no suitable TPU wheel existed;
- a 15-minute PROVISIONING requeue rule was removed after repeatedly sending requests to the back of the queue;
- preemption detection, cleanup, v5litepod sizing, unbuffered output, and null-state guards were added.

These iterations particularly explain the incomplete K `N=14` campaign.

## Range-level conclusion

The strongest strands are adversarial invariant-spectrum stratification (F), PTM-aligned recovery (G), persistence beyond pairwise concurrence within the tested finite range (H3), representation-specific propagation (I), and blind prediction with a failed strict-causality interpretation (K). N10a documents useful abstention. Program O is a replication priority because its artifact both contradicts its preregistered regime hierarchy and disagrees with its own commit subject's pass count.
