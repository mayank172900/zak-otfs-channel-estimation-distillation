# Reproduction Gap

## Matches closely

- Repository structure and end-to-end scripts exist and run in smoke mode.
- The support derivation reproduces the paper’s `27 x 43` support image.
- The CNN parameter count is exactly `245473`.
- Dense and matrix-free operator paths are cross-validated by tests.
- The GS/effective-channel fast path now matches a slow reference path closely on single-path and random multipath checks.
- The support-domain operator matches the paper's Eq. (4) matrix construction exactly on controlled cases.
- Full-mode training/evaluation plumbing is now resumable and full-run ready via on-the-fly dataset manifests and adaptive BER evaluation.

## Matches qualitatively only

- The code reproduces the paper’s pipeline stages and figure types.
- The alias/effective decomposition is now qualitatively reasonable instead of catastrophically wrong.
- Perfect-CSI BER is now in the correct low-BER regime on some corrected probes, but smoke-mode averages are still too noisy to treat as anchor evidence.

## Remains uncertain or mismatched

- The absolute NMSE and BER anchor values are still not proven in this checkpoint because the expensive full runs have not been executed after the final support-domain and adaptive-evaluation fixes.
- The conventional estimator remains the main numerical concern; it is improved but still not yet demonstrated to be within tolerance.
- CNN performance remains unresolved until full corrected training is run.
- The current smoke results are not sufficient evidence for the paper’s quantitative claims.
# Update (2026-03-28)

- The main code-level BER blocker addressed in this patch is the MMSE detector ridge scaling: `lambda = MN * N0 / E_d = 1 / rho_d`.
- A second pipeline blocker addressed here is that `numerics.effective_channel_method` now genuinely propagates through frame simulation, so fast and reference paths can be evaluated separately.
- Final full-run CSV/JSON/PNG artifacts still need to be regenerated on the target Windows CUDA machine; use [WINDOWS_RUNBOOK.md](/Users/goodday/Documents/Sem6/MLC/Paper/WINDOWS_RUNBOOK.md).
