# Zak-OTFS Channel Estimation Project

This repository contains our reimplementation and extension of the paper **"Zak-OTFS with Superimposed Spread Pilot: CNN-Aided Channel Estimation."**

The project has three main parts:

- a baseline Zak-OTFS pipeline under `src/zakotfs/`
- a distilled student pipeline under `distill_novelty/`
- two written deliverables:
  - the IEEE-style paper under `ieee_paper/`
  - the longer project report under `project_report/`

This is not an official repository from the original paper authors. It is our own implementation, evaluation pipeline, and compression study built on top of the published method.

## Repository Layout

```text
.
├── configs/                 # Baseline system configs
├── distill_novelty/         # Distilled student models, training, and evaluation
├── ieee_paper/              # IEEE-style paper source and compiled PDF
├── project_report/          # Course/project report source and compiled PDF
├── scripts/                 # Baseline helper scripts
├── src/zakotfs/             # Baseline Zak-OTFS implementation
├── tests/                   # Baseline tests
├── README.md
├── requirements.txt
├── pytest.ini
└── Zak-OTFS_with_Superimposed_Spread_Pilot_CNN-Aided_Channel_Estimation.pdf
```

## What The Baseline Covers

The baseline implementation includes:

- Zak-OTFS DD-domain signal construction
- superimposed spread-pilot generation
- Vehicular-A channel simulation with fractional delay and Doppler
- conventional support-domain read-off estimation
- CNN-aided support refinement
- pilot cancellation and LMMSE detection

The baseline code lives in `src/zakotfs/` and is the teacher path used for the later distillation work.

## What The Novelty Adds

The final novelty path in this repo is the distilled student pipeline under `distill_novelty/`.

It includes:

- three compact student models: `Lite-L`, `Lite-M`, and `Lite-S`
- teacher-student distillation training
- parameter-count and latency benchmarking
- NMSE and BER evaluation scripts

The main final model is **Lite-M**, which gave the best overall tradeoff in our experiments.

Final model sizes:

- Teacher CNN: `245,473` parameters
- Lite-L: `40,049`
- Lite-M: `22,789`
- Lite-S: `6,137`

## Papers And Reports

The repo also includes the writing that goes with the code:

- `ieee_paper/`
  - IEEE-style paper source
  - compiled `main.pdf`
- `project_report/`
  - expanded project report
  - compiled `project_report.pdf`

These two documents are written for different purposes. The IEEE paper is the compact research-style version. The project report is the longer explanatory version.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

Baseline tests:

```bash
PYTHONPATH=src pytest -q
```

Distillation tests:

```bash
PYTHONPATH=src:distill_novelty/src pytest distill_novelty/tests -q
```

## Distillation Workflow

The distillation configs expect support manifests under:

- `distill_novelty/artifacts/phase1_datasets/train_phase1.json`
- `distill_novelty/artifacts/phase1_datasets/val_phase1.json`
- `distill_novelty/artifacts/phase1_datasets/smoke_phase1.json`

Each manifest should provide:

- `H_obs`
- `H_base`
- `H_true`

Train the main distilled model:

```bash
python distill_novelty/scripts/train_student.py --config distill_novelty/configs/distill_lite_m.yaml
```

Run the main evaluation bundle:

```bash
python distill_novelty/scripts/run_distill_bundle.py --config distill_novelty/configs/distill_lite_m.yaml
```

Runtime outputs are written under:

- `distill_novelty/results/`
- `distill_novelty/logs/`

## Notes

- Large generated datasets, checkpoints, and temporary logs are intentionally not tracked.
- The original source paper PDF is included for reference.
- The repository contains our reproduced pipeline and extension work; it should not be read as the original authors' own codebase.
