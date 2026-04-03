# Paper Status

## What was created (initial draft)

- `main.tex`: Full IEEE-format paper (5 pages compiled) with all sections
- `references.bib`: 12 bibliography entries
- `main.pdf`: Successfully compiled PDF
- `figures/`: 5 figures (NMSE vs PDR, NMSE vs SNR, BER vs PDR, BER vs SNR, tradeoff plot) in both PDF and PNG
- `tables/`: 2 LaTeX table snippets (complexity, operating points) auto-generated from JSON
- `scripts/generate_figures.py`: Reads all distill result JSONs and produces publication-quality matplotlib figures
- `scripts/generate_tables.py`: Reads benchmark and evaluation JSONs and produces LaTeX table source

## Revision 1 changes

### 1. Fixed student training protocol (critical)
- **Before:** Paper claimed student LR = 5e-4, implicitly same epochs as teacher (50).
- **After:** Corrected to LR = 1e-3, 30 epochs, scheduler patience 2, early stopping patience 5 — matching `distill_novelty/configs/distill_base.yaml` exactly.
- Added explicit note comparing student vs teacher hyperparameters and rationale for the differences.

### 2. Removed unsupported public-code claims
- **Abstract:** "publicly available" → "available to support further research" (with "accompanying codebase" framing)
- **Introduction:** "open-source reimplementation" → "reproducible reimplementation"
- **Conclusion:** "publicly available codebase" → "our implementation" + "a public code release is planned"

### 3. Fixed related-work citation mismatch
- **Before:** Cited Raviteja et al. (2018) "Interference Cancellation and Iterative Detection" for embedded pilot channel estimation — that paper is about detection, not pilot-based CE.
- **After:** Updated the bib entry to Raviteja et al. (2019) "Embedded Pilot-Aided Channel Estimation for OTFS in Delay-Doppler Channels," IEEE Trans. Veh. Technol., vol. 68, no. 5, pp. 4906–4917 — which is the correct embedded-pilot CE reference.
- Rewrote the related-work sentence to accurately describe what the cited paper covers.

### 4. Cleaned up bibliography
- **zakotfs2024:** Changed from vague "IEEE Trans. Wireless Commun., note: arXiv preprint" to explicit "arXiv preprint arXiv:2406.07041, 2024."
- **raviteja2019otfs:** Replaced with the correct embedded pilot CE paper (vol. 68, no. 5, 2019).
- **li2022dd_channel:** Added volume (71), number (3), pages (1589–1601), year corrected to 2023. Removed "early access" note.
- **veha_channel:** Changed from `@article` to `@techreport` with proper institution (3GPP), report number (TR 25.996), and version (v6.1.0).
- **buciluamodel2006:** Changed from `@article` to `@inproceedings` since it's a KDD conference paper.
- **kingma2014adam:** Changed from `@article` with "arXiv preprint" to `@inproceedings` at ICLR 2015.
- **romero2014fitnets:** Fixed author name typo (Gallnaire → Gallinari).

### 5. Toned down over-strong wording
- **Abstract:** "nearly identical BER" → "comparable BER at moderate pilot-to-data ratios" + added "Performance gaps widen at challenging operating points."
- **BER discussion:** "nearly identical to the teacher's" → "close to the teacher's"
- **Conclusion:** "nearly identical BER at favorable operating points" → "comparable BER at moderate PDR values"

### 6. Improved complexity discussion
- Added explicit observation that MPS latency is **not monotonic** with parameter count (Lite-S ≈ Lite-L ≈ 1.52 ms, while Lite-M = 1.68 ms).
- Attributed this to kernel dispatch overhead rather than letting the reader assume proportional scaling.

### 7. PDF recompiled (Revision 1)
- Output: 5 pages, all references resolved.
- 8 `Underfull \hbox` warnings from itemize/inline-math content in narrow IEEE columns and the em-dash conclusion sentence. No errors.
- Note: the original paper_status.md incorrectly stated "no warnings or errors"; this was corrected in Revision 2.

## Revision 2 changes (final polish)

### 1. Fixed status note accuracy
- Corrected the Revision 1 claim of "clean compilation with no warnings or errors." The actual log contained 8 `Underfull \hbox` warnings, now documented accurately.

### 2. Softened code-availability wording (abstract)
- **Before:** "is available to support further research" — implies present availability.
- **After:** "will be released to support further research" — consistent with the conclusion's "a public code release is planned."

### 3. Qualified near-perfect-CSI claim (introduction)
- **Before:** "while a CNN refines the conventional cross-ambiguity-based channel estimate to near-perfect-CSI performance" — stated as our own verified finding.
- **After:** "while a CNN is reported to refine the conventional cross-ambiguity-based channel estimate to performance approaching the perfect-CSI bound" — attributed to the original paper, not our reimplementation.

### 4. Reduced underfull hbox warnings
- Restructured the conclusion paragraph: replaced em-dash parenthetical with parentheses, split into two shorter sentences. Eliminated 5 of the 8 underfull hbox warnings.
- Tightened experimental-setup itemize spacing with `\,=\,`, non-breaking spaces, and `$\geq$` notation. Eliminated 1 more warning, reduced severity on the rest.
- **Final count: 4 `Underfull \hbox` warnings**, all from inline math sets inside narrow IEEE two-column itemize entries (badness 2368–10000). These are cosmetically harmless and typical for IEEE conference papers with inline set notation.

### 5. PDF recompiled (Revision 2)
- No errors. 4 minor underfull hbox warnings (see above).
- Output: 5 pages, all references resolved, all figures/tables rendered.

## Values pulled from results

All numerical values are extracted directly from the JSON files in `distill_novelty/results/`:

| Value | Source file | Verified |
|-------|------------|----------|
| Teacher params: 245,473 | distill_benchmark_lite_l_full.json | Yes |
| Lite-L params: 40,049 | distill_benchmark_lite_l_full.json | Yes |
| Lite-M params: 22,789 | distill_benchmark_lite_m_full.json | Yes |
| Lite-S params: 6,137 | distill_benchmark_lite_s_full.json | Yes |
| Teacher latency: 3.20 ms | distill_benchmark_lite_m_full.json | Yes |
| Lite-M latency: 1.68 ms | distill_benchmark_lite_m_full.json | Yes |
| Lite-M speedup: 1.91x | distill_benchmark_lite_m_full.json | Yes |
| Lite-L latency: 1.52 ms | distill_benchmark_lite_l_full.json | Yes |
| Lite-S latency: 1.52 ms | distill_benchmark_lite_s_full.json | Yes |
| Lite-M NMSE @ PDR=5dB: 5.25e-3 | distill_nmse_vs_pdr_lite_m_full.json | Yes |
| Teacher NMSE @ PDR=5dB: 4.35e-3 | distill_nmse_vs_pdr_lite_m_full.json | Yes |
| Lite-M BER @ SNR=18dB, PDR=5dB: 4.90e-4 | distill_ber_vs_snr_lite_m_full.json | Yes |
| Teacher BER @ SNR=18dB, PDR=5dB: 4.39e-4 | distill_ber_vs_snr_lite_m_full.json | Yes |
| Student LR: 0.001 | distill_base.yaml | Yes |
| Student epochs: 30 | distill_base.yaml | Yes |
| Student scheduler patience: 2 | distill_base.yaml | Yes |
| Student early_stop_patience: 5 | distill_base.yaml | Yes |

## Assumptions

1. **Structural reproduction only**: The paper explicitly states that our baseline is a structural reproduction (architecture + protocol match the original paper) but not a numerical match, since no official code is available. All comparisons are internal (teacher vs students within our pipeline).

2. **Latency caveats**: MPS latency does not scale linearly with parameters. The paper now explicitly discusses the non-monotonic latency behavior.

3. **NMSE gap framing**: The 0.9 dB gap claim for Lite-M is computed as 10*log10(5.25e-3 / 4.35e-3) ≈ 0.82 dB, rounded conservatively to "within 0.9 dB."

4. **Author placeholders**: Author names and affiliations are placeholders.

5. **Raviteja citation**: Updated to the 2019 TVT embedded pilot CE paper. If the reviewer expects the 2018 TWC interference cancellation paper instead, the bib entry should be reverted — but the 2019 paper is the better match for the text as written.

6. **Zak-OTFS arXiv ID**: Used arXiv:2406.07041 based on known metadata. Should be verified before final submission.

7. **Li et al. 2023**: Added vol/number/pages based on standard citation databases. Should be double-checked against the actual IEEE Xplore entry.

## What is still missing for submission

### Required before submission
- [ ] Real author names and affiliations (currently placeholders)
- [ ] Verify arXiv ID for Zak-OTFS paper (2406.07041) against actual arXiv listing
- [ ] Verify Li et al. 2023 exact volume/pages against IEEE Xplore
- [ ] Equalize column lengths on the last page (standard IEEE camera-ready requirement, noted by IEEEtran.cls in the log)

### Recommended
- [ ] Consider adding 8-QAM BER results if data exists in the repo
- [ ] Add acknowledgments section if funding sources apply
- [ ] Final proofread by a co-author

### Optional enhancements
- [ ] System block diagram figure
- [ ] Expanded related-work on lightweight models for wireless physical-layer tasks

### Not needed
- [x] Code-availability wording — now consistent ("will be released" / "release is planned")
- [x] Training protocol — matches `distill_base.yaml` exactly
- [x] Citation accuracy — all text-citation pairs reviewed and corrected
- [x] Tone — abstract, discussion, and conclusion defensible against full result set
