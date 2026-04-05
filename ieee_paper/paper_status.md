# Paper Status

## Current state

- `main.tex` updated with the new structural-compressibility framing
- `main.pdf` rebuilt successfully in this working copy
- page count is now **6 pages**
- author block anonymized for blind review
- new figure included:
  - `figures/support_structure_overview.pdf`

## What was newly added in this working copy

### Structural-insight addition

The manuscript is no longer positioned as "distillation alone." It now argues:

- under the fast effective-channel construction used by the repo experiments, the twisted-support channel image is a deterministic chirp times a sum of path-wise separable atoms,
- after dechirping, the true support image is rank-bounded by path count,
- this explains why the teacher CNN is overparameterized and why Lite-M compression works.

This addition is grounded in new artifacts generated in this working copy:

- `results/structure/support_structure_fast.json`
- `results/structure/support_structure_reference.json`
- `results/structure/support_structure_observed.json`
- `results/structure/support_lowrank_projection.json`

### New figure/table generation support

Updated:

- `scripts/generate_figures.py`
- `scripts/generate_tables.py`

The new scripts:

- regenerate the structural figure/table from `results/structure/`
- skip legacy distillation-JSON-dependent outputs cleanly if those artifacts are absent in this working copy

### New generated paper assets

Figures:

- `figures/support_singular_decay.pdf`
- `figures/support_cumulative_energy.pdf`
- `figures/support_lowrank_projection_nmse.pdf`
- `figures/support_structure_overview.pdf`

Table:

- `tables/support_structure.tex`

## Key numerical evidence used in the new framing

All of the following come from the new structural-analysis JSON files generated in this working copy.

### Fast-model dechirped true support

- median tail energy beyond rank 6: `1.12e-18`
- median rank@99% energy: `2`
- median rank@99.9% energy: `3`

### Reference-model dechirped true support

- median tail energy beyond rank 6: `1.35e-09`
- median rank@99% energy: `2`
- median rank@99.9% energy: `3`

### Raw observed read-off support (SNR = 15 dB, PDR = 5 dB)

- median tail energy beyond rank 6: `1.25e-01`
- median rank@99% energy: `20`
- median rank@99.9% energy: `26`

### Thresholded read-off support (same operating point)

- median tail energy beyond rank 6: `0.00e+00`
- median rank@99% energy: `2`
- median rank@99.9% energy: `2`

### Simple low-rank projection diagnostic

At SNR = 15 dB and PDR = 5 dB:

- raw read-off median NMSE: `3.31e-01`
- rank-2 dechirp/SVD/rechirp median NMSE: `5.70e-02`
- thresholded read-off median NMSE: `5.84e-02`

This is used as supporting evidence that the task is fundamentally low-dimensional after dechirping. It is **not** claimed as a replacement for the learned CNN.

## What was retained from the restored paper workspace

The paper still contains the pre-existing distillation performance story:

- teacher / Lite-L / Lite-M / Lite-S performance figures
- complexity and operating-point comparisons

Those assets were already present in the restored `ieee_paper/` workspace when this update began. In this working copy, the underlying `distill_novelty/results/*.json` files are not available, so these specific performance curves/tables were **not regenerated here**.

That is why:

- the paper's legacy distillation figures are retained as restored assets,
- while the new structural analysis was generated fresh in this working copy.

## Claims intentionally kept narrow

- Exact low-rank claim: **only** for the fast effective-channel model implemented in `src/zakotfs/channel.py`
- Reference-model claim: **empirically approximately low-rank**, not exact
- No teacher/Lite-M output-spectrum claim, because checkpoints were not present in this working copy
- No universal claim about all Zak-OTFS formulations

## Compilation status

Built with:

```bash
tectonic --keep-logs main.tex
```

Result:

- `main.pdf` generated successfully
- 6 pages
- no compilation errors

Remaining warnings:

- 5 unique minor `Underfull \hbox` warnings remain in `main.log`
- these are cosmetic column-fitting warnings, not mathematical or reference errors
- the heuristic `proof_check.py` report still flags `tab:complexity` and `tab:operating_points` because it does not follow `\input{tables/...}`; compilation confirms these table references resolve in the PDF

## Verification run

Executed:

```bash
PYTHONPATH=src pytest tests/test_support_structure.py -q
PYTHONPATH=src pytest tests -q
PYTHONPATH=src python scripts/analyze_support_compressibility.py --config configs/system.yaml --fast-samples 128 --reference-samples 24 --observed-samples 96 --projection-samples 96
python ieee_paper/scripts/generate_figures.py
python ieee_paper/scripts/generate_tables.py
tectonic --keep-logs main.tex
```

Observed:

- new support-structure tests passed
- full root test suite passed (`25 passed`)
- structural JSON/CSV/figure artifacts were created successfully
- paper rebuilt successfully

## Revision note

The latest structural-novelty pass strengthened the manuscript by adding:

- a formal proposition for the fast-model dechirped rank bound,
- sharper contribution wording centered on explaining compressibility,
- a stronger not-an-artifact defense,
- a more prominent low-rank SVD baseline discussion,
- and a clearer future direction toward explicitly low-rank architectures.

This made the paper stronger scientifically, but it also increased the draft from 5 to 6 pages. If a strict 5-page limit is required later, the cleanest next cut is to trim a lower-value legacy table/paragraph rather than weakening the new structural argument.

## Remaining manual items before submission

- Re-verify bibliography metadata before submission
- If desired, regenerate or recover the archived distillation JSON artifacts so the legacy performance plots/tables can also be reproduced from script in this working copy
- Final co-author proofread
