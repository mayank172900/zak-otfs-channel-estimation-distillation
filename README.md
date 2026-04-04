# Zak-OTFS Channel Estimation and Distillation

This repository contains a compact public version of the project built around **"Zak-OTFS with Superimposed Spread Pilot: CNN-Aided Channel Estimation"**.

It keeps only the essentials:

- the baseline Zak-OTFS reimplementation
- the final `distill_novelty/` teacher-student compression pipeline
- configs, scripts, and tests needed to reproduce the code path

## Repository Layout

```text
.
├── configs/             # Baseline configs
├── distill_novelty/     # Final distilled-student pipeline
├── scripts/             # Baseline utility / training / eval scripts
├── src/zakotfs/         # Baseline Zak-OTFS implementation
├── tests/               # Baseline tests
├── README.md
├── requirements.txt
├── pytest.ini
└── Zak-OTFS_with_Superimposed_Spread_Pilot_CNN-Aided_Channel_Estimation.pdf
```

## What Is Included

### Baseline

The baseline under `src/zakotfs/` includes:

- Zak-OTFS DD-domain signal construction
- superimposed spread pilot generation
- Vehicular-A fractional delay / Doppler simulation
- conventional read-off channel estimation
- CNN-aided support enhancement
- pilot cancellation and LMMSE detection

### Final Novelty

`distill_novelty/` is the final novelty path used in this project.

It contains:

- compact student CNNs: `Lite-L`, `Lite-M`, `Lite-S`
- teacher-student distillation training
- benchmark scripts for parameter count and inference latency
- evaluation scripts for NMSE and BER sweeps

The main final model is:

- **Lite-M**

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Baseline Tests

```bash
PYTHONPATH=src pytest -q
```

## Distillation Tests

```bash
PYTHONPATH=src:distill_novelty/src pytest distill_novelty/tests -q
```

## Distillation Workflow

The distillation configs expect Phase 1 support manifests to be available under:

- `distill_novelty/artifacts/phase1_datasets/train_phase1.json`
- `distill_novelty/artifacts/phase1_datasets/val_phase1.json`
- `distill_novelty/artifacts/phase1_datasets/smoke_phase1.json`

Each manifest should provide support-domain arrays for:

- `H_obs`
- `H_base`
- `H_true`

### Train Lite-M

```bash
python distill_novelty/scripts/train_student.py --config distill_novelty/configs/distill_lite_m.yaml
```

### Run Full Distillation Evaluation

```bash
python distill_novelty/scripts/run_distill_bundle.py --config distill_novelty/configs/distill_lite_m.yaml
```

Generated outputs are written at runtime under:

- `distill_novelty/results/`
- `distill_novelty/logs/`

## Final Models

- Teacher CNN: `245,473` params
- Lite-L: `40,049` params
- Lite-M: `22,789` params
- Lite-S: `6,137` params

Recommended final tradeoff model:

- **Lite-M**

## Notes

- Large generated datasets, checkpoints, and local logs are intentionally not tracked.
- This repository is a reproducible reimplementation and extension of the published paper, not an official author repository.
