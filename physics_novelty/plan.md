# Physics Novelty Plan

## Objective

Build a new novelty pipeline under `physics_novelty/` on top of the existing Zak-OTFS baseline. The novelty is a physics-consistent, uncertainty-aware channel estimator that refines the baseline CNN estimate in the support domain and improves downstream detection robustness.

This folder is intentionally isolated from:

- `src/zakotfs/` (main reproduction baseline)
- `novelty_paper/` (older adapter-based novelty path)

The baseline remains the reference implementation. All new novelty code should live under `physics_novelty/`.

## Core Idea

The paper CNN treats the support-domain cross-ambiguity read-off as an image denoising problem:

- input: noisy support image
- output: cleaned support-domain channel image

Our novelty extends this in three ways:

1. Start from the strong baseline CNN estimate instead of replacing it blindly.
2. Predict uncertainty or confidence along with the refined channel.
3. Enforce physics consistency using the Zak-OTFS pilot/read-off forward path.

The intended model output is:

- residual correction to the baseline estimate
- uncertainty map

The intended final estimate is:

`H_hat = H_base + Delta_H`

with confidence derived from uncertainty:

`C = exp(-S)`

where:

- `H_obs` is the raw observed support image
- `H_true` is the true support-domain effective channel
- `H_base` is the frozen baseline CNN estimate
- `Delta_H` is the learned correction
- `S` is the learned uncertainty map
- `C` is the confidence map

## Why This Direction

This is the safest and strongest novelty path for this repo because:

- there is no official code release for the paper
- the baseline is already credible on NMSE but still imperfect on BER
- small adapter-style tweaks are too weak as a publication story
- uncertainty and physics consistency directly address the baseline's main weakness: robust detection under modeling ambiguity

## Scientific Positioning

The paper should not claim exact reproduction of the authors' private implementation.

The defensible framing is:

- we build on a documented reimplementation of the published Zak-OTFS pipeline
- we propose a physics-consistent uncertainty-aware estimator on top of that baseline
- we evaluate against the reproduced conventional and baseline CNN estimators

## Folder Layout

Target structure:

- `physics_novelty/README.md`
- `physics_novelty/plan.md`
- `physics_novelty/configs/`
- `physics_novelty/scripts/`
- `physics_novelty/src/zakotfs_physics/`
- `physics_novelty/tests/`
- `physics_novelty/results/`
- `physics_novelty/logs/`

Suggested source modules:

- `params.py`
- `dataset.py`
- `physics_targets.py`
- `model.py`
- `training.py`
- `evaluation.py`
- `utils.py`
- `adapter.py` or `baseline_bridge.py`

## Data Contract

Each Phase 1 sample should contain the tensors needed for later phases.

Required tensors:

- `H_obs_re`, `H_obs_im`
- `H_true_re`, `H_true_im`
- `H_thr_re`, `H_thr_im`
- `H_base_re`, `H_base_im`
- `pdr_db`

Optional if cleanly available in Phase 1:

- alias prior real and imaginary
- synthesized pilot-only read-off target
- extra metadata arrays

Preferred manifest fields:

- generator type
- split
- size
- shape
- seed
- SNR
- PDR values
- baseline checkpoint path
- effective channel method
- file names for all arrays

Save format should follow current repo style:

- manifest JSON
- memmap `.npy` tensors

## Physics Target

We need a forward-consistency target for later training.

Conceptual operator:

`G(H) = ReadOff( ApplySupport( H, sqrt(E_p) * x_s ) )`

where:

- `H` is a support-domain channel image
- `x_s` is the spread pilot
- `ApplySupport` uses the support-domain channel operator
- `ReadOff` means the same support-window ambiguity read-off used by the baseline estimator

For the true channel:

`H_phys_true = G(H_true)`

This should be generated in Phase 1 and stored for later use if practical. If storing it is too expensive or redundant, Phase 1 can store enough metadata to regenerate it deterministically in Phase 2, but the preferred design is to materialize it.

## Model Design

Phase 2 model should be small and stable, not over-designed.

Recommended first model:

- input channels: raw observation, thresholded estimate, baseline estimate, optional alias prior
- shallow convolutional residual network
- outputs:
  - `Delta_H_re`
  - `Delta_H_im`
  - `S`

Possible output layouts:

- 3 channels: `Delta_H_re`, `Delta_H_im`, shared uncertainty
- 4 channels: `Delta_H_re`, `Delta_H_im`, `S_re`, `S_im`

Start with 3 channels unless a strong reason appears.

## Training Math

### Reconstruction Loss

Primary NMSE-style loss:

`L_rec = ||H_hat - H_true||_F^2 / (||H_true||_F^2 + eps)`

### Uncertainty Loss

Use heteroscedastic regression:

`L_unc = mean( exp(-S_i) * |H_hat_i - H_true_i|^2 + beta * S_i )`

This prevents meaningless confidence maps.

### Physics Consistency Loss

Use the forward operator defined above:

`L_phys = ||G(H_hat) - G(H_true)||_F^2 / (||G(H_true)||_F^2 + eps)`

Alternative form if stored target is named `H_phys_true`:

`L_phys = ||G(H_hat) - H_phys_true||_F^2 / (||H_phys_true||_F^2 + eps)`

### Total Loss

Initial training objective:

`L_total = L_rec + lambda_unc * L_unc + lambda_phys * L_phys`

Recommended starting weights:

- `lambda_unc = 0.1`
- `lambda_phys = 0.2`
- `beta = 0.01`

These are only initial values and must be ablated.

## Detection Strategy

The new estimate should not be trusted uniformly.

Recommended inference rule:

`H_use = C .* H_hat + (1 - C) .* H_base`

Interpretation:

- high confidence: trust the new refined estimate
- low confidence: fall back toward the stable baseline CNN estimate

Then run:

- pilot cancellation using `H_use`
- MMSE detection using `H_use`

This is the most practical path to BER gains.

## Phase Plan

### Phase 1: Dataset and Scaffolding

Deliverables:

- `physics_novelty/` folder structure
- config-driven dataset generator
- baseline checkpoint bridge
- manifest plus memmap dataset format
- tests for determinism, shapes, and consistency
- short README

Output tensors should support all later phases without redesign.

### Phase 2: Model and Training

Deliverables:

- residual uncertainty-aware model
- training loop
- checkpointing
- loss implementation for `L_rec` and `L_unc`
- optional first cut of `L_phys`

Start with a stable trainable system before overcomplicating the model.

### Phase 3: Physics Consistency Integration

Deliverables:

- exact implementation of `G(H)`
- physics target handling
- `L_phys` integration
- ablation between:
  - no physics loss
  - physics loss enabled

This phase is the novelty core.

### Phase 4: Confidence-Guided Detection

Deliverables:

- confidence-to-weight mapping
- blended estimate `H_use`
- BER evaluation using new estimator
- ablation:
  - raw `H_hat`
  - blended `H_use`
  - confidence thresholding variants

This is where BER improvement should become visible.

### Phase 5: Evaluation and Ablations

Required evaluations:

- NMSE vs PDR
- NMSE vs SNR
- BER vs PDR
- BER vs SNR

Required ablations:

- conventional
- baseline CNN
- residual-only refinement
- uncertainty-only variant
- physics-loss variant
- full method

Optional robustness studies:

- train on fast, test on reference
- mixed effective-channel methods
- reduced training data
- pilot mismatch sensitivity

## Phase 1 Implementation Guidance

Phase 1 should inspect and reuse:

- `src/zakotfs/dataset.py`
- `src/zakotfs/evaluation.py`
- `src/zakotfs/estimators.py`
- `src/zakotfs/operators.py`
- `src/zakotfs/channel.py`
- `src/zakotfs/training.py`
- `novelty_paper/src/zakotfs_novelty/adapter.py`

Preferred engineering style:

- thin wrappers around baseline functions
- deterministic seeds
- config-driven scripts
- manifest-based dataset generation
- minimal code duplication

Avoid:

- modifying `src/zakotfs/` unless strictly necessary
- building a model in Phase 1
- mixing this work into `novelty_paper/`

## Test Plan

Phase 1 tests should cover:

- folder-local config loading
- manifest generation correctness
- array shapes and dtypes
- deterministic reproduction with fixed seed
- baseline consistency:
  - stored `H_obs` matches the frame generation path
  - stored `H_true` matches the frame generation path
  - stored `H_base` matches frozen baseline inference

Later-phase tests should cover:

- model output shapes
- confidence range behavior
- physics operator consistency
- regression checks for evaluation outputs

## Risks and Caveats

Main technical risks:

- the baseline itself uses a reimplemented effective-channel model because the paper has no official code
- BER remains more fragile than NMSE
- uncertainty can collapse if the regularization is weak
- physics loss can become too expensive if implemented naively

Mitigations:

- keep the novelty tied to the documented reimplementation
- start with support-domain operators only
- reuse cached pilot/read-off targets where possible
- add ablations early

## Paper Framing

Potential method description:

"We propose a physics-consistent uncertainty-aware refinement network for support-domain channel estimation in Zak-OTFS with superimposed spread pilot. Unlike prior image-only enhancement, the proposed method predicts both a residual channel correction and an uncertainty map, and is trained with a forward-consistency loss derived from the pilot/read-off operator. The uncertainty is further used to guide downstream detection."

Potential contribution list:

1. A new uncertainty-aware support-domain refinement method for Zak-OTFS channel estimation.
2. A physics-consistency objective tied to the Zak-OTFS pilot/read-off forward model.
3. Confidence-guided detection using uncertainty-weighted channel estimates.
4. A reproducible novelty pipeline on top of a documented reimplementation baseline.

## Success Criteria

Minimum success:

- NMSE better than baseline CNN on at least part of the sweep
- confidence maps are non-degenerate
- physics loss is stable and meaningful

Strong success:

- BER improvement over baseline CNN
- more stable behavior across PDR and SNR
- robustness to fast/reference effective-channel variation

## Immediate Next Step

The next chat should implement only Phase 1:

- create `physics_novelty/` package structure
- dataset generator
- configs
- tests
- README

No model training in Phase 1.
