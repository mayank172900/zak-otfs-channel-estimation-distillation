# Physics Novelty Phase 1-5

Phase 1 builds the dataset and scaffolding for the physics-aware novelty path under `physics_novelty/`.

Phase 2 adds the first trainable model: a lightweight residual CNN that refines `H_base` and predicts a shared uncertainty map.

Phase 3 adds the actual physics-consistency loss based on the Zak-OTFS support-domain pilot/read-off forward operator.

Phase 4 adds confidence-guided channel usage for detection, so the refined estimate does not have to be trusted uniformly across the support image.

Phase 5 adds the full experiment pipeline for paper-style NMSE and BER sweeps, including method comparisons, output saving, and smoke/full evaluation configs.

Phase 1 materializes per-sample support-domain tensors for later phases:

- `H_obs`: raw support observation from the read-off estimator
- `H_true`: true effective support-domain channel
- `H_thr`: thresholded conventional support estimate
- `H_base`: frozen baseline CNN estimate
- `H_phys_true`: pilot/read-off forward-consistency target generated from `H_true`
- scalar metadata: `pdr_db`, `sample_index`, `sample_seed`, `data_snr_db`, `E_p`, `rho_d`, `rho_p`, `noise_variance`

Outputs follow the repo’s dataset style:

- manifest JSON
- memmap `.npy` arrays, with complex tensors stored as separate real and imaginary files

Run generation from the repo root:

```bash
python physics_novelty/scripts/generate_phase1_dataset.py --config physics_novelty/configs/phase1_train.yaml
python physics_novelty/scripts/generate_phase1_dataset.py --config physics_novelty/configs/phase1_val.yaml
python physics_novelty/scripts/generate_phase1_dataset.py --config physics_novelty/configs/phase1_smoke.yaml --force
```

Generated manifests are written under `physics_novelty/results/phase1_datasets/`.

Phase 2 model summary:

- inputs: `H_obs`, `H_thr`, `H_base`
- channel layout: `H_obs_re`, `H_obs_im`, `H_thr_re`, `H_thr_im`, `H_base_re`, `H_base_im`
- outputs: `Delta_H_re`, `Delta_H_im`, `S`
- final estimate: `H_hat = H_base + Delta_H`
- confidence output: `C = exp(-S)`

Phase 2 training uses:

- reconstruction loss: normalized support-domain squared error between `H_hat` and `H_true`
- uncertainty-aware loss: `mean(exp(-S) * |H_hat - H_true|^2 + beta * S)`
- residual regularization: `||Delta_H||^2 / (||H_base||^2 + eps)` to keep refinement anchored to the strong frozen baseline
- physics loss: not enabled yet

Train from the repo root after generating Phase 1 train and val manifests:

```bash
python physics_novelty/scripts/train_phase2_model.py --config physics_novelty/configs/phase2_train.yaml
python physics_novelty/scripts/train_phase2_model.py --config physics_novelty/configs/phase2_smoke.yaml
```

Phase 2 checkpoints and history are written under `physics_novelty/logs/`.

Phase 3 physics target:

- `H_phys_true = G(H_true)`
- `G(H)` means: apply the support-domain channel `H` to the spread pilot, then run the same support-window read-off used by the baseline estimator
- the stored target comes from Phase 1 and the training-time operator uses the same equations

Phase 3 loss:

- `L_rec = ||H_hat - H_true||^2 / (||H_true||^2 + eps)`
- `L_unc = mean(exp(-S) * |H_hat - H_true|^2 + beta * S)`
- `L_delta = ||Delta_H||^2 / (||H_base||^2 + eps)`
- `L_phys = ||G(H_hat) - H_phys_true||^2 / (||H_phys_true||^2 + eps)`
- `L_total = L_rec + lambda_unc * L_unc + lambda_delta * L_delta + lambda_phys * L_phys`

Phase 3 training command:

```bash
python physics_novelty/scripts/train_phase3_model.py --config physics_novelty/configs/phase3_train.yaml
python physics_novelty/scripts/train_phase3_model.py --config physics_novelty/configs/phase3_smoke.yaml
```

The Phase 3 operator is hybrid but differentiable in training:

- the exact linear map for `G(H)` is precomputed once from the baseline equations
- training applies that fixed map with Torch complex matmul
- gradients flow to `H_hat`, but the operator itself is treated as a fixed constant rather than a learned or dynamically rebuilt graph

Phase 4 fusion rule:

- default conservative blend: `H_use = W * H_hat + (1 - W) * H_base`
- with `C = exp(-S)` from the model output
- and `W = clamp(confidence_scale * C, 0, 1)`
- `H_hat` is the refined support estimate and `H_base` is the frozen baseline CNN estimate
- default safety settings in `phase4_base.yaml` use `confidence_scale = 0.25` and `confidence_threshold = 0.8`, so the detector stays biased toward `H_base` unless the model is clearly confident

Available Phase 4 channel usage modes:

- `baseline`: detect with `H_base`
- `refined`: detect with `H_hat`
- `blended`: detect with the default `H_use`
- `thresholded`: detect with a hard confidence mask that switches between `H_hat` and `H_base`

Phase 4 also exposes the effective fusion weights used at inference so fallback behavior can be analyzed directly.

Phase 4 keeps detection inside the baseline path:

- run the Phase 3 model to get `H_hat` and confidence
- select the support-domain channel according to the configured mode
- perform pilot cancellation and MMSE detection with that chosen support estimate

Phase 4 configuration lives in:

- `physics_novelty/configs/phase4_base.yaml`
- `physics_novelty/configs/phase4_train.yaml`
- `physics_novelty/configs/phase4_smoke.yaml`

These outputs feed Phase 5 directly by making one-frame and single-point comparisons between `baseline`, `refined`, and `blended` detection modes easy before running full sweeps.

Phase 5 evaluated methods:

- `conventional`: thresholded read-off estimate
- `baseline`: frozen baseline CNN estimate `H_base`
- `refined`: novelty model estimate `H_hat`
- `blended`: confidence-guided fused estimate `H_use`
- `perfect`: true effective support-domain channel
- optional `thresholded`: hard confidence switch between `H_hat` and `H_base`

Phase 5 sweeps:

- NMSE vs PDR
- NMSE vs SNR
- BER vs PDR
- BER vs SNR

Phase 5 commands:

```bash
python physics_novelty/scripts/run_phase5_nmse.py --config physics_novelty/configs/phase5_smoke.yaml
python physics_novelty/scripts/run_phase5_ber.py --config physics_novelty/configs/phase5_smoke.yaml
python physics_novelty/scripts/run_phase5_full.py --config physics_novelty/configs/phase5_full.yaml
python physics_novelty/scripts/benchmark_physics_operator.py --config physics_novelty/configs/phase3_smoke.yaml
```

Evaluation outputs are written under `physics_novelty/results/` and follow the paper-style naming scheme:

- `phase5_nmse_vs_pdr_<mode>.csv|json|png`
- `phase5_nmse_vs_snr_<mode>.csv|json|png`
- `phase5_ber_vs_pdr_<mode>.csv|json|png`
- `phase5_ber_vs_snr_<mode>.csv|json|png`

Phase 5 remains intentionally conservative:

- sweep loops are correct-first rather than CUDA-optimized
- evaluation uses the existing baseline detector as the downstream reference path
- checkpoint loading is explicit so evaluation can stay isolated from training layout changes

The operator benchmark writes `phase5_physics_operator_benchmark.json` under `physics_novelty/results/` and is intended to support the efficiency side of the paper story by comparing:

- the direct NumPy/estimator-style forward physics target generation
- the cached linear-operator Torch application used during Phase 3 training

Deferred beyond Phase 5:

- faster batched BER evaluation and CUDA-heavy acceleration
- larger ablation grids and paper-ready table assembly
- final manuscript packaging and figure curation
