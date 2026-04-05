# IEEE Paper: Lightweight Distilled CNN for Zak-OTFS Channel Estimation

## Building the paper

### Prerequisites
- pdflatex (TeX Live)
- bibtex
- Python 3.8+ with `matplotlib` and `numpy` (for regenerating figures/tables)
- `IEEEtran.cls` and `IEEEtran.bst` (included in this folder)

### Quick build

```bash
cd ieee_paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### Regenerate figures and tables from JSON results

```bash
python scripts/generate_figures.py
python scripts/generate_tables.py
```

This reads from `../distill_novelty/results/*.json` and writes to `figures/` and `tables/`.

## File layout

```
ieee_paper/
  main.tex            - LaTeX source
  references.bib      - Bibliography
  IEEEtran.cls        - IEEE document class
  IEEEtran.bst        - IEEE bibliography style
  main.pdf            - Compiled PDF
  figures/             - Generated PDF/PNG figures
  tables/              - Generated LaTeX table snippets
  scripts/
    generate_figures.py  - JSON -> matplotlib figures
    generate_tables.py   - JSON -> LaTeX tables
  paper_status.md      - Status notes
  README.md            - This file
```
