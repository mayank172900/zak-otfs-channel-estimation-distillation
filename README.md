# Zak-OTFS Channel Estimation Reproduction and Distillation

This repository contains a full research workflow around the paper **"Zak-OTFS with Superimposed Spread Pilot: CNN-Aided Channel Estimation"**.

It includes:

- a baseline reimplementation of the Zak-OTFS system and CNN estimator
- an exploratory `physics_novelty/` branch for physics-aware refinement
- a final `distill_novelty/` branch for lightweight teacher-student compression
- an IEEE paper draft in `ieee_paper/` based on the final distillation results

The final project direction is an **accuracy-efficiency tradeoff study**: a strong baseline CNN is distilled into compact student models (`Lite-L`, `Lite-M`, `Lite-S`) and evaluated on parameter count, inference latency, NMSE, and BER.

## Repository Layout

```text
.
├── src/zakotfs/                 # Baseline Zak-OTFS implementation
├── configs/                     # Baseline configs
├── scripts/                     # Baseline generation, training, and evaluation scripts
├── tests/                       # Baseline test suite
├── report/                      # Reproduction report and notes
├── novelty_paper/               # Early novelty branch (kept for reference)
├── physics_novelty/             # Physics-aware refinement pipeline
├── distill_novelty/             # Final distillation pipeline
├── ieee_paper/                  # IEEE-format paper draft, figures, and tables
├── results/                     # Baseline compact result artifacts
├── logs/                        # Baseline logs and metadata
└── Zak-OTFS_with_*.pdf          # Source paper PDF
```

## Main Components

### Baseline Reimplementation

The baseline in `src/zakotfs/` includes:

- Zak-OTFS DD-domain signal construction
- superimposed spread pilot generation
- Vehicular-A fractional delay and fractional Doppler channel simulation
- conventional cross-ambiguity read-off estimation
- CNN-aided enhancement on real/imaginary support images
- pilot cancellation and LMMSE data detection

Useful baseline entry points:

- `scripts/smoke_test.py`
- `scripts/generate_dataset.py`
- `scripts/reproduce_all.py`
- `run_all.sh`

### Physics-Aware Refinement

`physics_novelty/` contains a complete phased pipeline for:

- Phase 1 dataset materialization
- uncertainty-aware residual refinement
- physics-consistency loss
- confidence-guided detection
- evaluation and ablations

This branch is preserved as an experimental path and documented in:

- `physics_novelty/README.md`
- `physics_novelty/plan.md`
- `physics_novelty/paper_plan.md`

### Distilled Lightweight Students

`distill_novelty/` is the final novelty direction and the one used in the paper draft.

It contains:

- compact student CNNs: `Lite-L`, `Lite-M`, `Lite-S`
- teacher-student distillation training
- benchmarking for parameter count and inference latency
- evaluation scripts for:
  - NMSE vs PDR
  - NMSE vs SNR
  - BER vs PDR
  - BER vs SNR

Key files:

- `distill_novelty/README.md`
- `distill_novelty/src/zakotfs_distill/model.py`
- `distill_novelty/src/zakotfs_distill/training.py`
- `distill_novelty/src/zakotfs_distill/evaluation.py`
- `distill_novelty/scripts/train_student.py`
- `distill_novelty/scripts/run_distill_bundle.py`

### IEEE Paper Draft

The current IEEE-style paper draft is in:

- `ieee_paper/main.tex`
- `ieee_paper/main.pdf`
- `ieee_paper/references.bib`
- `ieee_paper/paper_status.md`

The paper uses `Lite-M` as the headline model and presents:

- the baseline reproduction context
- the distillation method
- efficiency/accuracy tradeoffs
- figures and tables generated from the actual experiment outputs

## Quick Start

### Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Baseline Tests

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/smoke_test.py
```

### Distillation Tests

```bash
PYTHONPATH=src:physics_novelty/src:distill_novelty/src pytest distill_novelty/tests -q
```

## Reproducing the Distillation Results

### 1. Generate Phase 1 distillation-ready data

```bash
python physics_novelty/scripts/generate_phase1_dataset.py --config physics_novelty/configs/phase1_train.yaml
python physics_novelty/scripts/generate_phase1_dataset.py --config physics_novelty/configs/phase1_val.yaml
```

### 2. Train the main student (`Lite-M` recommended)

```bash
python distill_novelty/scripts/train_student.py --config distill_novelty/configs/distill_lite_m.yaml
```

### 3. Run the full evaluation bundle

```bash
python distill_novelty/scripts/run_distill_bundle.py --config distill_novelty/configs/distill_lite_m.yaml
```

### Optional variants

```bash
python distill_novelty/scripts/train_student.py --config distill_novelty/configs/distill_train.yaml
python distill_novelty/scripts/run_distill_bundle.py --config distill_novelty/configs/distill_eval_full.yaml

python distill_novelty/scripts/train_student.py --config distill_novelty/configs/distill_lite_s.yaml
python distill_novelty/scripts/run_distill_bundle.py --config distill_novelty/configs/distill_lite_s.yaml
```

## Final Distillation Summary

Headline model:

- **Lite-M**

Tradeoff summary from the final full runs:

- Teacher CNN: `245,473` params
- Lite-L: `40,049` params
- Lite-M: `22,789` params
- Lite-S: `6,137` params

Key outcome:

- `Lite-M` provides the strongest overall accuracy-efficiency balance
- `Lite-L` is the higher-fidelity compact student
- `Lite-S` is the aggressive compression point with visible degradation

Compact experiment artifacts are stored in:

- `distill_novelty/results/`
- `ieee_paper/figures/`
- `ieee_paper/tables/`

## Tracked vs Generated Artifacts

Tracked:

- source code
- configs
- tests
- compact JSON/CSV/PNG result artifacts
- paper sources and compiled PDF
- reports and notes

Not tracked:

- multi-GB generated dataset memmaps (`*.npy`)
- training checkpoints and temporary logs
- local caches and archive bundles

This keeps the repository reproducible and still practical to clone and push.

## Important Documents

- Baseline reproduction report: `report/reproduction_report.md`
- Assumptions: `ASSUMPTIONS.md`
- Reproduction gap summary: `REPRO_GAP.md`
- Root-cause note: `ROOT_CAUSE_NOTE.md`
- Physics novelty plan: `physics_novelty/plan.md`
- IEEE paper draft: `ieee_paper/main.pdf`

## Source Context

This project builds on the paper:

- `Zak-OTFS_with_Superimposed_Spread_Pilot_CNN-Aided_Channel_Estimation.pdf`

Because no official reference implementation accompanied the paper, the baseline here is a **reproducible reimplementation**, not a claim of access to the authors' internal code.
