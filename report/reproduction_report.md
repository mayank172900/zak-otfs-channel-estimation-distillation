# Reproduction Report

## 1. Objective

Reproduce the end-to-end Zak-OTFS superimposed spread-pilot pipeline from the paper using a runnable, tested, and documented codebase.

## 2. Paper summary in my own words

The paper estimates the Zak-OTFS effective DD-domain channel by reading off the received/spread-pilot cross-ambiguity on a twisted support lattice. The conventional read-off is corrupted by aliasing from the spread-pilot self-ambiguity, by data interference, and by noise. The proposed method treats the support crop as a 2D image and uses a shared single-channel CNN, applied separately to real and imaginary parts, to improve the estimate before pilot cancellation and MMSE detection.

## 3. Equation-to-code mapping table

| Paper item | Code |
| --- | --- |
| Eq. (1) data DD waveform | `src/zakotfs/waveform.py:data_waveform` |
| Eq. (2) spread pilot and chirp filter | `src/zakotfs/pulses.py:chirp_spreading_filter`, `src/zakotfs/waveform.py:spread_pilot` |
| Eq. (3) superposition | `src/zakotfs/waveform.py:superimposed_frame` |
| Eq. (4) effective channel operator | `src/zakotfs/operators.py:build_dense_heff_matrix`, `src/zakotfs/operators.py:apply_heff_operator` |
| Eq. (5) and Eq. (7) cross-ambiguity decomposition | `src/zakotfs/ambiguity.py:cross_ambiguity`, `src/zakotfs/diagnostics.py:error_decomposition` |
| Eq. (6) twisted lattice support condition | `src/zakotfs/lattice.py:lattice_condition`, `src/zakotfs/lattice.py:derive_support_geometry` |
| Eq. (8) conventional read-off | `src/zakotfs/estimators.py:read_off_estimator`, `src/zakotfs/estimators.py:threshold_readoff` |
| Eq. (9) CNN re-embedding | `src/zakotfs/evaluation.py:estimate_channels`, `src/zakotfs/lattice.py:embed_support_image` |
| Eq. (10) CNN loss | `src/zakotfs/training.py:_loss_fn` |

## 4. Implementation choices

- The repository is built around `src/zakotfs/` with config-driven scripts under `scripts/`.
- A dense effective-channel matrix builder is provided for validation, and a matrix-free apply/adjoint pair is used for iterative MMSE.
- The physical channel uses the Vehicular-A profile with fractional delays and Dopplers.
- The GS pulse is implemented as a sampled localization kernel with the matched-filter phase convention stated in the paper.
- The conventional estimator stores both raw and thresholded read-off outputs.
- The CNN architecture matches the paper and its parameter count is verified exactly.

## 5. Assumptions and ambiguities

Detailed items are logged in [ASSUMPTIONS.md](/Users/goodday/Documents/Sem6/MLC/Paper/ASSUMPTIONS.md). The main unresolved technical choice is the sampled effective-channel model for the fractional-delay fractional-Doppler GS-pulse case, because the short paper does not expand the exact closed-form discrete `h_eff[k,l]`.

## 6. Validation tests

`PYTHONPATH=src pytest -q` passes.

Covered checks:

- energy normalization of data and spread pilot
- support derivation to `27 x 43`
- cross-ambiguity indexing sanity
- CNN parameter count `245473`
- dense versus matrix-free operator consistency
- dense versus iterative MMSE consistency
- deterministic reproducibility under fixed seeds

## 7. Reproduced figures

Generated smoke-run figures:

- [fig2_heatmaps_smoke.png](/Users/goodday/Documents/Sem6/MLC/Paper/report/figures/fig2_heatmaps_smoke.png)
- [fig3_lattice_smoke.png](/Users/goodday/Documents/Sem6/MLC/Paper/report/figures/fig3_lattice_smoke.png)
- [fig4_cnn_support_smoke.png](/Users/goodday/Documents/Sem6/MLC/Paper/report/figures/fig4_cnn_support_smoke.png)
- [nmse_vs_pdr_smoke.png](/Users/goodday/Documents/Sem6/MLC/Paper/report/figures/nmse_vs_pdr_smoke.png)
- [nmse_vs_snr_smoke.png](/Users/goodday/Documents/Sem6/MLC/Paper/report/figures/nmse_vs_snr_smoke.png)
- [ber_vs_pdr_smoke.png](/Users/goodday/Documents/Sem6/MLC/Paper/report/figures/ber_vs_pdr_smoke.png)
- [ber_vs_snr_smoke.png](/Users/goodday/Documents/Sem6/MLC/Paper/report/figures/ber_vs_snr_smoke.png)
- [error_decomposition_smoke.png](/Users/goodday/Documents/Sem6/MLC/Paper/report/figures/error_decomposition_smoke.png)

## 8. Quantitative comparison to paper

Smoke-run comparison against the paper anchors:

| Figure anchor | Paper | Reproduced smoke | Gap |
| --- | --- | --- | --- |
| NMSE vs PDR at `10 dB`, conventional | `1.75e-2` | `5.388e-2` | improved but still outside tolerance |
| NMSE vs PDR at `10 dB`, CNN | `1.85e-3` | `1.1437e+0` | very large |
| BER vs PDR at `5 dB`, BPSK, conventional | `2.5e-2` | `1.14211e-1` | improved but still outside tolerance |
| BER vs PDR at `5 dB`, BPSK, CNN | `4.7e-4` | `4.86922e-1` | very large |
| BER vs PDR at `5 dB`, BPSK, perfect | `3e-4` | `0.0` | optimistic due tiny smoke sample |
| BER vs SNR at `18 dB`, BPSK, conventional | `2.5e-2` | `1.18570e-1` | improved but still outside tolerance |
| BER vs SNR at `18 dB`, BPSK, CNN | `4.7e-4` | `4.93461e-1` | very large |
| BER vs SNR at `18 dB`, BPSK, perfect | `2.6e-4` | `0.0` | optimistic due tiny smoke sample |
| BER vs SNR at `18 dB`, 8-QAM, conventional | `1.6e-1` | `2.68381e-1` | improved but still outside tolerance |
| BER vs SNR at `18 dB`, 8-QAM, CNN | `4.0e-2` | `4.96803e-1` | very large |
| BER vs SNR at `18 dB`, 8-QAM, perfect | `2.5e-2` | `7.265e-3` | inconsistent direction |

The current build is therefore only a structural reproduction, not yet a numerically matched reproduction.

## 9. Error decomposition analysis

For one representative smoke frame, the support-set energy decomposition was:

- effective channel term: `7.20`
- alias term: `7.20`
- data interference term: `2.28`
- noise term: `0.032`

Inside the support set, aliasing and effective-channel energy are now comparable rather than wildly imbalanced. This is qualitatively consistent with the paper’s motivation, but the current GS/effective-channel model is still not accurate enough to close the remaining quantitative gap.

## 10. Robustness analysis

The trained network was trained only at `15 dB` data SNR and then evaluated across the smoke SNR sweep. The CNN curves remain better than the conventional NMSE curves in the current implementation, but both remain far from the paper-level accuracy, so the robustness claim is only weakly supported qualitatively.

## 11. What matched well

- Repository structure, scripts, configs, and logging are in place.
- The CNN architecture and parameter count match the paper exactly.
- The support derivation matches the paper values `Delta_k = 27`, `Delta_l = 43`.
- Dense and matrix-free operator paths agree numerically.
- The pilot-only support read-off identity now holds numerically.
- The conventional estimator moved from catastrophic failure to the correct order of magnitude.

## 12. What did not match perfectly

- The reproduced NMSE and BER values are far from the paper anchors.
- The BER curves do not yet show the paper’s reported near-perfect-CSI behavior for the CNN-aided path.
- The U-shaped BER-vs-PDR effect is not yet reproduced convincingly.
- The smoke CNN remains poor because the current smoke training regime is tiny and not representative of the paper.

## 13. Suspected reasons for the remaining gap

- The spread pilot and unwrapped read-off were previously wrong; fixing them removed the worst failure mode but did not finish the reproduction.
- The effective-channel approximation for GS pulse shaping is still too crude.
- The remaining gap now appears to be dominated by GS/effective-channel fidelity rather than the ambiguity/read-off plumbing.
- The smoke training set is too small to evaluate the CNN seriously.
- The smoke confidence intervals are dominated by very small sample counts.

## 14. Recommendations for next-step novelty work

- Replace the approximate sampled GS effective-channel kernel with a closer derivation from the continuous matched-filter model.
- Add sensitivity sweeps for pilot location, gain normalization, and alternate 8-QAM variants.
- Precompute and cache operator kernels to make full-scale training and evaluation feasible.
- Add a reference slow path directly from a more faithful continuous/DD discretization for a small number of frames.

## 15. Reproducibility instructions

```bash
pip install -r requirements.txt
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/reproduce_all.py --mode smoke
PYTHONPATH=src python scripts/reproduce_fig2_fig3_fig4.py --mode smoke
PYTHONPATH=src python scripts/eval_error_decomposition.py --mode smoke
```
# Update (2026-03-28)

- Code changes for the MMSE ridge fix, config-driven fast/reference effective-channel selection, perfect-CSI PDR diagnostics, and BER-vs-SNR modulation-separated plotting are in this patch set.
- Run instructions are documented in [WINDOWS_RUNBOOK.md](/Users/goodday/Documents/Sem6/MLC/Paper/WINDOWS_RUNBOOK.md).
- Root-cause summary is documented in [ROOT_CAUSE_NOTE.md](/Users/goodday/Documents/Sem6/MLC/Paper/ROOT_CAUSE_NOTE.md).
