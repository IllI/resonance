# May 19–31: From Adaptive Recovery to Narrow Semantic-State Transport

The 27 commits in this range follow the lineage `Q/R/S1 -> T/U/V/V2/W -> X/Y/Z/AA -> AB/AD/AE/AF -> AG/AH/AI/AJ -> AK/AL/AM/AN -> AO/AP -> AQ`. The claim progressively narrows as controls expose non-intervention, saturation, chance decoding, target leakage, and conditioning effects.

## Q/R/S1 — fidelity, distinguishability, and echo recoverability

Commit `c14592beef` compared blind controls on hidden states. Q found that adaptive control often reduced ordinary endpoint fidelity. R shifted to paired-state distinguishability and preservation, sometimes improving information metrics while lowering fidelity. S1 reported strong echo recoverability in W3 and an OAT preservation index above one while fidelity fell. This supports controller-induced deformation that remains reversible under the chosen echo—not generic high-fidelity transmission. A failover log named by the summary is absent.

## T/U — spatial transport and hardware-like noise

Commits `f574328481`, `0c8ecc3e07`, `9fb89aab28`, and `e3334a98a7` moved to a spatial Alice/Bob grid, local observables, CPTP noise, crosstalk, and sparse control. T's representative cells favored sparse predictive recovery, but probe-sparse and sparse-predictive outputs were identical and samples were only `n=3`. U improved some XXZ/W3 endpoint metrics; OAT's best sparse arm used zero interventions, and Ising remained unrecoverable. Preservation and fidelity rankings often disagreed.

Assessment: useful transport infrastructure and abstention lesson; underpowered for general scientific claims.

## V/V2/W — entanglement-assisted transfer and trigger inactivity

Commits `5797fea695`, `38e11531d4`, `2e270d441a`, and `154755ace4` tested OOD Haar-state transfer and intervention boundaries. V's vector-adaptive arm fired zero gates and merely matched static entanglement fidelity; random control often had better Bloch similarity. Many V2 threshold arms were identical because they also fired zero gates. W corrected seed coupling, trigger scaling, and intervention reporting; later geometry-aware effects were small and mixed, while DD was consistently harmful.

Assessment: high-confidence diagnostic of inactive triggers; weak evidence for beneficial intervention.

## X/Y/Z/AA — MERA logical recovery and observability

Commits `e21d5e07da`, `b48122174b`, `2bdc8d4fd7`, and `831e9d5814` introduced MERA logical encoding, depth scaling, layer-resolved observables, and renormalized predictors. Logical fidelity remained near two-qubit chance (~0.25); geometry-versus-random differences were around 1e-3; and no stable depth hierarchy emerged. AA found that renormalized entropy/ZZ observables better predicted later recovery, but every tested controller's mean fidelity change versus random was negative.

Assessment: predictive observability improved more convincingly than active recovery. This distinction drives the rest of the range.

## AB/AD/AE/AF — semantic alphabets and moving frames

AB (`4774156642`) replaced raw states with a four-Bell-state alphabet. Accuracy stayed near chance even though reported mutual information was unexpectedly high, weakening the semantic-decoding claim. No identifiable Program AC artifact exists. AD (`6fad43ce28`) and AE (`644654bdb9`) implemented sequential and co-moving frame methods, but AD has no committed result and AE's committed summary is zero bytes despite a commit subject claiming a full TPU sweep. AF (`33b8753232`) contains real forecast data, but improvements are inconsistent and most accuracies remain around 0.2–0.3.

Assessment: method evolution is clear; evidence for semantic transport is weak in this stage. The empty AE artifact is a provenance red flag.

## AG/AH — continuous latent geometry and normal modes

Commits `aff39a4277`, `4e9bb6306f`, `67e273024b`, `691b2ed1cf`, and `bcd6ad84df` moved to 15-dimensional logical Bloch/Gell-Mann trajectories, diffusion/curvature, Koopman models, RNN residuals, and TPU `pmap`. AG completed computation but crashed during dictionary serialization; some values survive only in terminal-derived Markdown. AH substantially improved execution time, but randomized-entropy ablations sometimes outperformed the intended entropy-specific mechanism.

Assessment: meaningful computational capability; weak mechanism specificity.

## AI/AJ — predictive hydrodynamics and all-gates saturation

AI has code but no distinct committed result. AJ (`3faa8e0652`) swept forecast horizons and reduced a latent-diffusion metric, but every horizon produced essentially identical outputs because the controller fired 160/160 gates. EU/US artifacts reproduce the same behavior.

Assessment: an always-on limit, not a forecast-horizon phase diagram. This directly motivated trigger and budget calibration.

## AK/AL/AM/AN — working-tree gaps and topology controls

AK and AM have no committed experiment artifact in this date range; their code/results are `LIVE`. AL (`b96555ed12`) claims a recompilation fix, but its source still labels itself Program AJ and its live recurrence score includes a nonsensical value near 375,000,064. AN's code/results are live; only a later prose analysis was committed in `b91596dc79`. AN reported high held-out cosine, but topology shuffles often performed equally or better, while the free arm was suspiciously robust.

Assessment: low reproducibility for AK–AM; AN is a useful negative/control result but needs raw-artifact preservation.

## AO — folded-field recovery

Commit `0663c20318` added harder Markov, block-permutation, phase, and random-Crofton controls. Remote-analysis prose records a completed focused run: free remained strongest or competitive, IBM-lite led at moderate noise, filament stabilization improved at higher noise, and folded decoding achieved only a tiny positive worst-control lift (+0.0026). Block permutation remained hardest.

Assessment: mixed diagnostic, not decisive topology recovery. The evidence suggests geometric smoothing more than reliable topology inversion.

## AP/AP-S/AP-T/AP-U — tunnel-eigen unfolding and locked calibration

Commits `5b1f8fb88a` and `b91596dc79` introduced tunnel-affinity eigenprojection, strong controls, alpha bracketing, adaptive alpha, and finally locked random/fold-in keys. AP-S initially suggested a noise-dependent alpha crossover; AP-T did not reproduce it cleanly. AP-U identified differing keys as the source of drift. After locking, fixed alpha 0.65 remained the operating point; adaptive variants were not promoted.

Assessment: AP-U's separation of calibration from promotion is the strongest methodological contribution of the late-May sequence, though production evidence survives mainly as Markdown transcriptions of remote stdout.

## AQ — predefined semantic subspaces and payload correction

Commits `a580b962cc` and `b91596dc79` narrowed the goal to transport of predefined occupation/transition/FFT/recurrence semantics.

- AQ-0/0b established a small noise-free motif gate; free often had the best margin while alpha 0.55 improved some semantic metrics.
- AQ-1a-TF's apparent transformation match failed across seeds and was reclassified as fragile endpoint association.
- AQ-1c initially failed. Recalibration then exposed a serious payload bug: marker sequences had replaced the actual source motifs, invalidating earlier operator-identifiability results as semantic transport.
- Corrected AQ-1c recovered the real source strongly, but operator and derived-target recovery remained at chance (0.25).
- AQ-SEQ-0 reported perfect source/transition metrics on only four noise-free transitions; direct next-state recovery was 0.75.
- AQ-IMG-0 recovered 16 motif classes with strong semantic-layer metrics but only 16.89 dB pixel PSNR.
- AQ-IMG-1-LITE fell to motif accuracy 0.375 and PSNR 9.72 dB; its one-example roster and changed target confounded the intended comparison.

Assessment: the durable result is recovery of a tiny predefined source-state/motif vocabulary under narrow conditions. Operator transport, noisy generalization, high-resolution image recovery, and fair residual comparisons remain unresolved.

## Missing-evidence register

- AC: no identifiable artifact.
- AD and AI: code without committed results.
- AE: committed result path is empty.
- AK/AM: live-only code and results.
- AL: live summary contains an apparently broken metric.
- AN/AO/AP/production AQ: strongest evidence often survives as live JSON or Markdown copied from remote stdout rather than committed raw artifacts.

## Range-level conclusion

The most defensible innovation is methodological: progressively distinguishing fidelity from recoverability, prediction from intervention, fixed labels from moving frames, and calibration from promotion. The most important negative results are zero-gate controller “wins,” all-gates saturation, chance logical decoding, randomized-ablation competitiveness, the empty AE artifact, and AQ's payload bug. Those corrections explain why the program arrives in June with a much narrower and more scientifically honest semantic-state claim.
