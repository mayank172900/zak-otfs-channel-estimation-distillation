#!/usr/bin/env python3
"""Generate publication-quality figures for IEEE paper from JSON results."""

import json
import os
import sys
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterSciNotation

# ── paths ──────────────────────────────────────────────────────────────
RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "distill_novelty", "results")
FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES, exist_ok=True)

# ── style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
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
})

MARKERS = {"conventional": "s", "teacher": "o", "lite_l": "^", "lite_m": "D", "lite_s": "v", "perfect": "*"}
COLORS = {"conventional": "#d62728", "teacher": "#1f77b4", "lite_l": "#2ca02c", "lite_m": "#ff7f0e", "lite_s": "#9467bd", "perfect": "#7f7f7f"}
LABELS = {"conventional": "Conventional", "teacher": "Teacher CNN", "lite_l": "Lite-L (40k)", "lite_m": "Lite-M (23k)", "lite_s": "Lite-S (6k)", "perfect": "Perfect CSI"}

def load_json(fname):
    with open(os.path.join(RESULTS, fname)) as f:
        return json.load(f)

def extract(records, method_key, x_key, y_key, fixed=None):
    """Extract x,y arrays for a given method from records, optionally filtering by fixed conditions."""
    pts = []
    for r in records:
        if r["method"] != method_key:
            continue
        if fixed:
            skip = False
            for k, v in fixed.items():
                if r.get(k) != v:
                    skip = True
                    break
            if skip:
                continue
        pts.append((r[x_key], r[y_key]))
    pts.sort()
    if not pts:
        return np.array([]), np.array([])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def plot_metric_vs_x(ax, records, x_key, y_key, methods, fixed=None, ylabel="", xlabel="", logy=True):
    for m in methods:
        x, y = extract(records, m, x_key, y_key, fixed)
        if len(x) == 0:
            continue
        ax.plot(x, y, marker=MARKERS.get(m, "o"), color=COLORS.get(m, "k"), label=LABELS.get(m, m), markerfacecolor="none" if m == "conventional" else COLORS.get(m, "k"))
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(loc="best")


# ── merge all student data into unified records ─────────────────────────
def merge_student_records(base_records, variant, variant_records):
    """Replace 'student' method label with variant name in variant_records and append to base."""
    merged = list(base_records)
    for r in variant_records:
        rc = dict(r)
        if rc["method"] == "student":
            rc["method"] = variant
            merged.append(rc)
    return merged


# ═══════════════════════════════════════════════════════════════════════
# 1. NMSE vs PDR
# ═══════════════════════════════════════════════════════════════════════
def fig_nmse_vs_pdr():
    base = load_json("distill_nmse_vs_pdr_lite_m_full.json")["records"]  # has teacher + conventional
    recs = list(base)
    for var in ["lite_l", "lite_m", "lite_s"]:
        d = load_json(f"distill_nmse_vs_pdr_{var}_full.json")["records"]
        recs = merge_student_records(recs, var, d)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fixed = {"data_snr_db": 15.0}
    plot_metric_vs_x(ax, recs, "pdr_db", "nmse", ["conventional", "teacher", "lite_l", "lite_m", "lite_s"],
                     fixed=fixed, ylabel="NMSE", xlabel="PDR (dB)")
    ax.set_title("NMSE vs PDR (SNR = 15 dB, BPSK)", fontsize=9)
    fig.savefig(os.path.join(FIGURES, "nmse_vs_pdr.pdf"))
    fig.savefig(os.path.join(FIGURES, "nmse_vs_pdr.png"))
    plt.close(fig)
    print("  [OK] nmse_vs_pdr")


# ═══════════════════════════════════════════════════════════════════════
# 2. NMSE vs SNR
# ═══════════════════════════════════════════════════════════════════════
def fig_nmse_vs_snr():
    base = load_json("distill_nmse_vs_snr_lite_m_full.json")["records"]
    recs = list(base)
    for var in ["lite_l", "lite_m", "lite_s"]:
        d = load_json(f"distill_nmse_vs_snr_{var}_full.json")["records"]
        recs = merge_student_records(recs, var, d)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fixed = {"pdr_db": 5.0}
    plot_metric_vs_x(ax, recs, "data_snr_db", "nmse", ["conventional", "teacher", "lite_l", "lite_m", "lite_s"],
                     fixed=fixed, ylabel="NMSE", xlabel="Data SNR (dB)")
    ax.set_title("NMSE vs SNR (PDR = 5 dB, BPSK)", fontsize=9)
    fig.savefig(os.path.join(FIGURES, "nmse_vs_snr.pdf"))
    fig.savefig(os.path.join(FIGURES, "nmse_vs_snr.png"))
    plt.close(fig)
    print("  [OK] nmse_vs_snr")


# ═══════════════════════════════════════════════════════════════════════
# 3. BER vs PDR
# ═══════════════════════════════════════════════════════════════════════
def fig_ber_vs_pdr():
    base = load_json("distill_ber_vs_pdr_lite_m_full.json")["records"]
    recs = list(base)
    for var in ["lite_l", "lite_m", "lite_s"]:
        d = load_json(f"distill_ber_vs_pdr_{var}_full.json")["records"]
        recs = merge_student_records(recs, var, d)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fixed = {"data_snr_db": 18.0, "modulation": "bpsk"}
    plot_metric_vs_x(ax, recs, "pdr_db", "ber",
                     ["conventional", "teacher", "lite_l", "lite_m", "lite_s", "perfect"],
                     fixed=fixed, ylabel="BER", xlabel="PDR (dB)")
    ax.set_title("BER vs PDR (SNR = 18 dB, BPSK)", fontsize=9)
    fig.savefig(os.path.join(FIGURES, "ber_vs_pdr.pdf"))
    fig.savefig(os.path.join(FIGURES, "ber_vs_pdr.png"))
    plt.close(fig)
    print("  [OK] ber_vs_pdr")


# ═══════════════════════════════════════════════════════════════════════
# 4. BER vs SNR
# ═══════════════════════════════════════════════════════════════════════
def fig_ber_vs_snr():
    base = load_json("distill_ber_vs_snr_lite_m_full.json")["records"]
    recs = list(base)
    for var in ["lite_l", "lite_m", "lite_s"]:
        d = load_json(f"distill_ber_vs_snr_{var}_full.json")["records"]
        recs = merge_student_records(recs, var, d)

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    fixed = {"pdr_db": 5.0, "modulation": "bpsk"}
    plot_metric_vs_x(ax, recs, "data_snr_db", "ber",
                     ["conventional", "teacher", "lite_l", "lite_m", "lite_s", "perfect"],
                     fixed=fixed, ylabel="BER", xlabel="Data SNR (dB)")
    ax.set_title("BER vs SNR (PDR = 5 dB, BPSK)", fontsize=9)
    fig.savefig(os.path.join(FIGURES, "ber_vs_snr.pdf"))
    fig.savefig(os.path.join(FIGURES, "ber_vs_snr.png"))
    plt.close(fig)
    print("  [OK] ber_vs_snr")


# ═══════════════════════════════════════════════════════════════════════
# 5. Tradeoff scatter (params vs NMSE)
# ═══════════════════════════════════════════════════════════════════════
def fig_tradeoff():
    params = {"teacher": 245473, "lite_l": 40049, "lite_m": 22789, "lite_s": 6137}
    # NMSE at PDR=5 dB, SNR=15 dB
    nmse_pdr5 = {}
    for var in ["lite_l", "lite_m", "lite_s"]:
        d = load_json(f"distill_nmse_vs_pdr_{var}_full.json")["records"]
        for r in d:
            if r["method"] == "student" and r["pdr_db"] == 5.0:
                nmse_pdr5[var] = r["nmse"]
                break
    d = load_json("distill_nmse_vs_pdr_lite_m_full.json")["records"]
    for r in d:
        if r["method"] == "teacher" and r["pdr_db"] == 5.0:
            nmse_pdr5["teacher"] = r["nmse"]

    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for m in ["teacher", "lite_l", "lite_m", "lite_s"]:
        ax.scatter(params[m], nmse_pdr5[m], marker=MARKERS[m], color=COLORS[m], s=60, label=LABELS[m], zorder=5)
        ax.annotate(LABELS[m], (params[m], nmse_pdr5[m]), textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("NMSE (PDR = 5 dB, SNR = 15 dB)")
    ax.set_title("Accuracy-Efficiency Tradeoff", fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.savefig(os.path.join(FIGURES, "tradeoff.pdf"))
    fig.savefig(os.path.join(FIGURES, "tradeoff.png"))
    plt.close(fig)
    print("  [OK] tradeoff")


if __name__ == "__main__":
    print("Generating figures...")
    fig_nmse_vs_pdr()
    fig_nmse_vs_snr()
    fig_ber_vs_pdr()
    fig_ber_vs_snr()
    fig_tradeoff()
    print("Done. Figures in:", os.path.abspath(FIGURES))
