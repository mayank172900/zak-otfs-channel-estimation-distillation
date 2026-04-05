#!/usr/bin/env python3
"""Generate publication-quality IEEE paper figures from available JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DISTILL_RESULTS = ROOT / "distill_novelty" / "results"
STRUCTURE_RESULTS = ROOT / "results" / "structure"
FIGURES = ROOT / "ieee_paper" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1.4,
        "lines.markersize": 5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    }
)


MARKERS = {"conventional": "s", "teacher": "o", "lite_l": "^", "lite_m": "D", "lite_s": "v", "perfect": "*"}
COLORS = {
    "conventional": "#d62728",
    "teacher": "#1f77b4",
    "lite_l": "#2ca02c",
    "lite_m": "#ff7f0e",
    "lite_s": "#9467bd",
    "perfect": "#7f7f7f",
}
LABELS = {
    "conventional": "Conventional",
    "teacher": "Teacher CNN",
    "lite_l": "Lite-L (40k)",
    "lite_m": "Lite-M (23k)",
    "lite_s": "Lite-S (6k)",
    "perfect": "Perfect CSI",
}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract(records, method_key, x_key, y_key, fixed=None):
    pts = []
    for record in records:
        if record["method"] != method_key:
            continue
        if fixed and any(record.get(key) != value for key, value in fixed.items()):
            continue
        pts.append((record[x_key], record[y_key]))
    pts.sort()
    if not pts:
        return np.array([]), np.array([])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def _plot_metric_vs_x(ax, records, x_key, y_key, methods, fixed=None, ylabel="", xlabel="", logy=True):
    for method in methods:
        x, y = _extract(records, method, x_key, y_key, fixed)
        if len(x) == 0:
            continue
        ax.plot(
            x,
            y,
            marker=MARKERS.get(method, "o"),
            color=COLORS.get(method, "k"),
            label=LABELS.get(method, method),
            markerfacecolor="none" if method == "conventional" else COLORS.get(method, "k"),
        )
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="best")


def _merge_student_records(base_records, variant, variant_records):
    merged = list(base_records)
    for record in variant_records:
        entry = dict(record)
        if entry["method"] == "student":
            entry["method"] = variant
            merged.append(entry)
    return merged


def _maybe_generate_performance_figure(output_name: str, result_stem: str, title: str, x_key: str, y_key: str, methods, fixed=None) -> bool:
    variant_files = [DISTILL_RESULTS / f"{result_stem}_{variant}_full.json" for variant in ["lite_l", "lite_m", "lite_s"]]
    if not all(path.exists() for path in variant_files):
        print(f"  [SKIP] {output_name}: distillation result JSONs are not available in this working copy")
        return False

    base = _load_json(DISTILL_RESULTS / f"{result_stem}_lite_m_full.json")
    recs = list(base["records"])
    for variant in ["lite_l", "lite_m", "lite_s"]:
        payload = _load_json(DISTILL_RESULTS / f"{result_stem}_{variant}_full.json")
        recs = _merge_student_records(recs, variant, payload["records"])

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    _plot_metric_vs_x(ax, recs, x_key, y_key, methods, fixed=fixed, ylabel=y_key.upper(), xlabel=x_key.replace("_", " ").title())
    ax.set_title(title, fontsize=9)
    fig.savefig(FIGURES / f"{output_name}.pdf")
    fig.savefig(FIGURES / f"{output_name}.png")
    plt.close(fig)
    print(f"  [OK] {output_name}")
    return True


def fig_support_structure_overview() -> bool:
    fast = _load_json(STRUCTURE_RESULTS / "support_structure_fast.json")
    reference = _load_json(STRUCTURE_RESULTS / "support_structure_reference.json")
    observed = _load_json(STRUCTURE_RESULTS / "support_structure_observed.json")
    projection = _load_json(STRUCTURE_RESULTS / "support_lowrank_projection.json")
    if not all(item is not None for item in [fast, reference, observed, projection]):
        print("  [SKIP] support_structure_overview: structure result JSONs are missing")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7))

    x = np.arange(1, len(fast["singular_energy"]["median"]) + 1)
    for label, y, color in [
        ("True (fast)", fast["singular_energy"]["median"], "#1f77b4"),
        ("True (reference)", reference["singular_energy"]["median"], "#2ca02c"),
        ("Raw read-off", observed["series"]["h_obs"]["singular_energy"]["median"], "#d62728"),
        ("Thresholded", observed["series"]["h_thr"]["singular_energy"]["median"], "#ff7f0e"),
    ]:
        axes[0].semilogy(x, y, marker="o", color=color, label=label)
    axes[0].set_xlabel("Singular-value index")
    axes[0].set_ylabel("Normalized energy")
    axes[0].grid(True, which="both", ls=":", alpha=0.4)
    axes[0].legend(loc="best")

    ranks = np.array(sorted(int(rank) for rank in projection["projection_nmse_by_rank"].keys()), dtype=int)
    medians = np.array([projection["projection_nmse_by_rank"][str(rank)]["median"] for rank in ranks], dtype=float)
    p25 = np.array([projection["projection_nmse_by_rank"][str(rank)]["p25"] for rank in ranks], dtype=float)
    p75 = np.array([projection["projection_nmse_by_rank"][str(rank)]["p75"] for rank in ranks], dtype=float)
    axes[1].plot(ranks, medians, marker="o", color="#1f77b4", label="Raw read-off + rank-$r$ SVD")
    axes[1].fill_between(ranks, p25, p75, color="#1f77b4", alpha=0.18)
    axes[1].axhline(projection["raw_observation_nmse"]["median"], color="#d62728", linestyle="--", label="Raw read-off")
    axes[1].axhline(projection["thresholded_nmse"]["median"], color="#ff7f0e", linestyle=":", label="Thresholded")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Rank $r$")
    axes[1].set_ylabel("NMSE vs. true support")
    axes[1].grid(True, which="both", ls=":", alpha=0.4)
    axes[1].legend(loc="best")

    fig.savefig(FIGURES / "support_structure_overview.pdf")
    fig.savefig(FIGURES / "support_structure_overview.png")
    plt.close(fig)
    print("  [OK] support_structure_overview")
    return True


if __name__ == "__main__":
    print("Generating figures...")
    fig_support_structure_overview()
    _maybe_generate_performance_figure(
        "nmse_vs_pdr",
        "distill_nmse_vs_pdr",
        "NMSE vs PDR (SNR = 15 dB, BPSK)",
        "pdr_db",
        "nmse",
        ["conventional", "teacher", "lite_l", "lite_m", "lite_s"],
        fixed={"data_snr_db": 15.0},
    )
    _maybe_generate_performance_figure(
        "nmse_vs_snr",
        "distill_nmse_vs_snr",
        "NMSE vs SNR (PDR = 5 dB, BPSK)",
        "data_snr_db",
        "nmse",
        ["conventional", "teacher", "lite_l", "lite_m", "lite_s"],
        fixed={"pdr_db": 5.0},
    )
    _maybe_generate_performance_figure(
        "ber_vs_pdr",
        "distill_ber_vs_pdr",
        "BER vs PDR (SNR = 18 dB, BPSK)",
        "pdr_db",
        "ber",
        ["conventional", "teacher", "lite_l", "lite_m", "lite_s", "perfect"],
        fixed={"data_snr_db": 18.0, "modulation": "bpsk"},
    )
    _maybe_generate_performance_figure(
        "ber_vs_snr",
        "distill_ber_vs_snr",
        "BER vs SNR (PDR = 5 dB, BPSK)",
        "data_snr_db",
        "ber",
        ["conventional", "teacher", "lite_l", "lite_m", "lite_s", "perfect"],
        fixed={"pdr_db": 5.0, "modulation": "bpsk"},
    )
    print("Done. Figures in:", FIGURES.resolve())
