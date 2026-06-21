# May 8–12: From OAT Fidelity Claim to Narrow IBM Witness Validation

This range contains 89 commits. `RF` denotes a committed numerical/raw artifact; `CM` denotes a claim in prose or a commit subject. Later corrections are preserved rather than silently folded into the original claim.

## 1. Initial OAT teleportation calculation — May 8

Commit `7e696a1` simulated cross-half OAT boundary pairs and reported fidelity above the classical 2/3 limit using `F=(2+C)/3`. The JSON supports the generated curves, but the formula was later shown invalid for the non-X boundary states at `N>=4`. Commit `d9b6de3` gave corrected examples: 0.719 at `N=4` rather than 0.770. This family is foundational as a trajectory generator, but its original teleportation interpretation is superseded.

## 2. MoQE/D-LinOSS framework discriminator — May 8–9

Commits `50104c9`, `ade331d`, `edc2b48`, `bd726f3`, `2ef5827`, `a2ca6cf`, `9bfe855`, and `ec4b4aa` built a learner/observer intended to distinguish OAT from separable, thermal, Lindblad, MBL, SYK, Heisenberg, and speculative Penrose-like signatures.

The first classifier scored 4/6 and misclassified OAT as MBL. A two-layer revision overgeneralized the Penrose probe. The project then explicitly recognized its first training generators as circular hand-crafted “cartoons” and reran against simulated OAT output. A claimed 4/4 confirmation included a numerically marginal bi-twistor criterion; multi-seed validation was only 2/3 fully confirmed, while decoherence weakened or removed the classification. Its strongest feature also inherited the later-falsified fidelity formula.

Assessment: useful early discriminator diagnostics, low support for intrinsic quantum-geometry or Penrose interpretations. These failures triggered the move to linear witnesses, BIC/modal fingerprints, and eventually recovery geometry.

## 3. Boundary audit, witness, and hybrid controller — May 9

Commits `a176e4c`, `a36f72c`, `bfbfd44`, and `7885e15` audited the reduced state, modeled dephasing, introduced a linear entanglement witness, and built a JILA–TPU controller. Direct Horodecki calculations found CHSH violation only at `N=2`. A synthetic witness decayed exactly according to its prescribed Lindblad generator (`b=4`, `K_eff=1`), validating the generator/model pairing rather than experimental data. The controller's witness sign convention was later corrected. A missing CSV `N` header was also repaired.

Assessment: high confidence in synthetic outputs; medium-to-low confidence in experimental projections.

## 4. Calibration probes and paper consolidation — May 10

Commits `5cd5b7c`, `51a56fe`, `494472c`, and `a281709` explored IBM-like noise, SYK, and MBL calibration. Despite “real hardware” wording, the IBM backend was `qiskit_aer_fake` and Sycamore data were synthetic. Short IBM-like windows confused Lindblad with SYK; averaged OTOC was insufficient; later exact-diagonalization retarded Green's functions gave a cleaner SYK modal signature. The MBL artifact itself retained a top-level SYK winner, so “all probes calibrated” was overstated.

The accompanying paper/restructure sequence (`a957790` through `e07dc0f`) improved organization but still embedded the incorrect X-state/fidelity theorem.

Assessment: scientifically useful failure analysis and infrastructure consolidation, not hardware evidence.

## 5. Corrected state and D-LinOSS fingerprint library — May 11–12

Commits `371d251`, `6bebc98`, `a44addd`, `6633efa`, `2251d1a`, `0e9cac8`, and `430cfa2` built a cross-framework fingerprint library and tested BIC-selected modal rank. `6bebc98` corrected the exact OAT reduced state and established that it is not an X-state. Run 1 archived 89 records. Adversarial tests exposed random-telegraph false positives and incomplete cavity-QED recovery; OAT and SYK both selected rank 1 and required decay-rate information, while XXZ selected a much larger rank.

Assessment: D-LinOSS became a preliminary morphology/model-selection tool, not a universal quantum classifier. Adversarial failures materially bound generality.

## 6. Direct falsifications and PTM correction — May 12

Commits `d9b6de3`, `d8b59d5`, `c881940`, `37325f6`, `f357a12`, `5d157c9`, and `d322b30` form the decisive correction sequence:

- Direct optimization falsified `f_max=(1+C)/2` and `F_opt=(2+C)/3` for every tested `N>=4`.
- The simple `Rz(chi*t/2)`/Berry-phase explanation failed.
- An “operational proof” from channel tomography was immediately corrected after a factor-of-two normalization error: the old 0.823 average fidelity became 0.661 for Rz-only at `N=4`, while full SU(2) optimization gave 0.719.
- Exact PTM identities were derived, including `T_xx=cos^(N-2)(chi*t/2)` and zero `T_yy/T_zz` in the stated setting.
- Dephasing removed the claimed advantage in the tested model.

Assessment: this pivot from scalar fidelity to exact PTM and recovery geometry is one of the most important pieces of the project story.

## 7. Recovery geometry, criticality, and pre-IBM hardening — May 12

Commits `955c1ab` through `b7ae5df` developed local-unitary recovery landscapes, super-classical basin volume `V_Q`, stiffness `kappa_Q`, a singularity near `chi*t=pi`, PTM rank-flow language, Dicke extensions, and hardware-noise simulations.

Recorded results included finite `V_Q` in selected OAT regions, zero in several controls, rapidly declining detectability with system size, and a strong simulated relationship between recoverable gain and `V_Q`. However, key corrections matter: an initial exponent beta=0.866 became 1.000 after identifying a crossover artifact; the `kappa_Q` peak had 44–47% coefficient of variation and precise peak/ratio claims were retracted; larger-N exponent extraction hit numerical-resolution limits; and D-LinOSS clustering did not uniquely isolate the singular class.

Two valuable explicit negatives also appear: dual classical TPUs cannot be entangled, and proposed fMRI/CaMKII analogies were unverified.

Assessment: strong hypothesis-generating simulation family with analytically solid PTM identities, but optimizer/sampling-sensitive geometry and exponent claims require artifact-level replication.

## 8. IBM hardware witness validation — May 12

Commits `a90b2c6`, `f7515a5`, `016a0c8`, `3e25016`, `33e1b3e`, `a8a14f6`, `616e49f`, and `8f60e78` narrowed the hardware question to whether `T_xx` follows an attenuated `cos^2(chi*t/2)` curve with a null at `pi`.

The dry run corrected the Hamiltonian and measured-qubit mapping. Run 1 archived calibrated values 0.500, 0.434, and -0.060 and a preregistered 12.47-sigma separation, but the null was 2.1 sigma from zero and compiler routing shifted effective angles. Run 2 fixed layout and disabled optimization, producing a nine-point curve with `R^2=0.9856` and a null 0.43 sigma from zero. Run 3 used a disjoint layout, producing `R^2=0.9758` and a null -0.12 sigma from zero.

Assessment: high confidence in the archived counts, fitted witness curves, and nulls. The experiment measured one PTM component, not full process tomography; “full rank collapse hardware-confirmed” is therefore stronger than the direct evidence. The defensible result is consistency of an attenuated `T_xx` witness curve across two layouts, not teleportation.

## 9. Approach C large-N asymptotics — May 12

Commit `c71f019` revisited the falsified finite-N formula as a possible asymptotic approximation. The archived fit reported discrepancy approximately `0.049*N^-0.093`, but the `N=1024` point was unreliable and larger-N points mixed direct optimization with a purity-corrected estimate. This suggests a convergence question; it does not prove the asymptotic statement.

## Range-level conclusion

The credible story is not the original teleportation claim. It is the rapid self-correction from an invalid X-state shortcut, through adversarial failures and normalization fixes, to a much narrower result with unusually good provenance: an IBM `N=4` witness curve consistent with the analytic `T_xx` form on two layouts. The simulation-level recovery geometry may still contain publishable ideas, but it is evidentially less mature and should be audited separately.
