# Sure-Shot Paper Plan

## Final direction

This paper should be framed as:

- a reproducible Zak-OTFS superimposed-pilot benchmark and reimplementation
- an efficient cached physics-consistent training operator
- a lightweight reliability-aware refinement module with conservative fallback
- a tradeoff study across accuracy, efficiency, and safety

This paper should **not** be framed as:

- a universal BER improvement over the frozen baseline CNN
- an exact reproduction of the authors' hidden implementation
- a claim that the refined model is the best detector-facing estimator in all regimes

## Why this is the safest scope

1. No official repository exists for the original paper.
2. The main baseline pipeline is already reproducible and tested.
3. The cached physics operator has a measurable efficiency gain.
4. The conservative blend is much safer than the earlier naive blend.
5. The frozen baseline CNN still wins overall on smoke accuracy, so the novelty should be positioned as a safe refinement/tradeoff method, not a replacement baseline.

## Current evidence

### Verified engineering state

- `PYTHONPATH=src:physics_novelty/src pytest physics_novelty/tests -q`
- Result: `22 passed`

### Verified efficiency result

From `physics_novelty/results/phase5_physics_operator_benchmark.json`:

- naive direct physics target: about `139.95 ms`
- cached Torch operator: about `15.26 ms`
- speedup: about `9.17x`

### Verified smoke accuracy trend

From `physics_novelty/results/phase5_*_smoke.json`:

- `baseline` remains the strongest overall method
- `refined` remains unsafe to use directly
- `blended` is much safer after conservative fusion, but still generally below `baseline`

This supports a safety/tradeoff paper, not a "state-of-the-art improvement" paper.

## Final method story

### Baseline anchor

Use the frozen baseline CNN estimate `H_base` as the trusted anchor.

### Novelty module

The novelty module predicts:

- residual correction `Delta_H`
- uncertainty map `S`
- confidence `C = exp(-S)`

and forms:

- `H_hat = H_base + Delta_H`

### Safe detection-time fusion

Use conservative fusion:

- `W = clamp(confidence_scale * C, 0, 1)`
- `H_use = W * H_hat + (1 - W) * H_base`

Default safety settings:

- `confidence_scale = 0.25`
- `confidence_threshold = 0.8`

### Training losses

- reconstruction loss `L_rec`
- uncertainty-aware loss `L_unc`
- residual regularization `L_delta`
- physics-consistency loss `L_phys`

Total:

- `L = L_rec + lambda_unc * L_unc + lambda_delta * L_delta + lambda_phys * L_phys`

## Exact claims we can make

1. We provide a reproducible Zak-OTFS benchmark and implementation pipeline in the absence of an official repository.
2. We derive and use a cached linear physics operator that substantially accelerates physics-consistent training relative to direct forward target generation.
3. We introduce a lightweight reliability-aware refinement module with conservative fallback to the frozen baseline.
4. We analyze the tradeoff between refinement aggressiveness, confidence-guided fallback, and downstream BER/NMSE behavior.

## Claims we should avoid

1. "Our method outperforms the baseline CNN across the board."
2. "We exactly reproduce the original authors' implementation."
3. "The refined estimate should replace the baseline estimate."

## Minimum final experiment package

### Figure group A: baseline anchor

- baseline repo NMSE/BER reference curves
- conventional vs baseline vs perfect

### Figure group B: efficiency

- naive physics operator runtime
- cached operator runtime
- speedup bar/table

### Figure group C: refinement tradeoff

- `baseline` vs `refined` vs `blended`
- NMSE vs PDR
- NMSE vs SNR
- BER vs PDR
- BER vs SNR

### Figure group D: reliability/safety

- confidence map examples
- fusion-weight examples
- case studies where blending prevents catastrophic refined errors

## Paper structure

1. Introduction
2. Reproducible Zak-OTFS baseline
3. Efficient physics-consistent operator
4. Lightweight reliability-aware refinement
5. Experimental setup
6. Efficiency and tradeoff results
7. Reliability analysis
8. Conclusion

## Next execution order

1. Keep the current conservative defaults.
2. Use the cached-operator benchmark as a main novelty result.
3. Run medium/full evaluations only for the final selected plots and tables.
4. Write the paper around reproducibility, efficiency, and safe refinement.
5. Do CUDA-heavy optimization only after the paper results are fixed.
