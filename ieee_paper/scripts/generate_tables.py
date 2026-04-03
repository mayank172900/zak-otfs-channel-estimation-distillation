#!/usr/bin/env python3
"""Generate LaTeX table snippets from JSON results for IEEE paper."""

import json
import os

RESULTS = os.path.join(os.path.dirname(__file__), "..", "..", "distill_novelty", "results")
TABLES = os.path.join(os.path.dirname(__file__), "..", "tables")
os.makedirs(TABLES, exist_ok=True)


def load_json(fname):
    with open(os.path.join(RESULTS, fname)) as f:
        return json.load(f)


# ══════════════════════════════════════════════════════════════
# Table 1: Model Complexity Comparison
# ══════════════════════════════════════════════════════════════
def table_complexity():
    benchmarks = {}
    for var in ["lite_l", "lite_m", "lite_s"]:
        benchmarks[var] = load_json(f"distill_benchmark_{var}_full.json")

    rows = []
    # Teacher row
    b = benchmarks["lite_l"]  # teacher stats are same in all
    rows.append(("Teacher CNN", b["teacher_params"], b["teacher_mean_ms"], "1.00$\\times$"))
    # Student rows
    for var, label in [("lite_l", "Lite-L"), ("lite_m", "Lite-M"), ("lite_s", "Lite-S")]:
        b = benchmarks[var]
        rows.append((label, b["student_params"], b["student_mean_ms"],
                      f"{b['student_speedup_vs_teacher']:.2f}$\\times$"))

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Model complexity comparison. Latency measured on Apple MPS with batch size 1, averaged over 50 iterations.}")
    lines.append("\\label{tab:complexity}")
    lines.append("\\begin{tabular}{lrrr}")
    lines.append("\\toprule")
    lines.append("Model & Params & Latency (ms) & Speedup \\\\")
    lines.append("\\midrule")
    for name, params, lat, spd in rows:
        lines.append(f"{name} & {params:,} & {lat:.2f} & {spd} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out = os.path.join(TABLES, "complexity.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] {out}")


# ══════════════════════════════════════════════════════════════
# Table 2: Key operating-point summary
# ══════════════════════════════════════════════════════════════
def table_operating_points():
    # Collect NMSE at PDR=5dB SNR=15dB and BER at PDR=5dB SNR=18dB for all methods
    nmse_data = {}
    ber_data = {}

    for var in ["lite_l", "lite_m", "lite_s"]:
        d = load_json(f"distill_nmse_vs_pdr_{var}_full.json")["records"]
        for r in d:
            if r["pdr_db"] == 5.0 and r["data_snr_db"] == 15.0:
                if r["method"] == "student":
                    nmse_data[var] = r["nmse"]
                elif r["method"] == "teacher":
                    nmse_data["teacher"] = r["nmse"]
                elif r["method"] == "conventional":
                    nmse_data["conventional"] = r["nmse"]

    for var in ["lite_l", "lite_m", "lite_s"]:
        d = load_json(f"distill_ber_vs_snr_{var}_full.json")["records"]
        for r in d:
            if r["pdr_db"] == 5.0 and r["data_snr_db"] == 18.0 and r["modulation"] == "bpsk":
                if r["method"] == "student":
                    ber_data[var] = r["ber"]
                elif r["method"] == "teacher":
                    ber_data["teacher"] = r["ber"]
                elif r["method"] == "conventional":
                    ber_data["conventional"] = r["ber"]
                elif r["method"] == "perfect":
                    ber_data["perfect"] = r["ber"]

    params = {"teacher": 245473, "lite_l": 40049, "lite_m": 22789, "lite_s": 6137, "conventional": "---", "perfect": "---"}

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Key operating-point comparison at PDR = 5\\,dB. NMSE at SNR = 15\\,dB; BER at SNR = 18\\,dB with BPSK.}")
    lines.append("\\label{tab:operating_points}")
    lines.append("\\begin{tabular}{lrll}")
    lines.append("\\toprule")
    lines.append("Method & Params & NMSE & BER \\\\")
    lines.append("\\midrule")

    for key, label in [("conventional", "Conventional"), ("teacher", "Teacher CNN"),
                       ("lite_l", "Lite-L"), ("lite_m", "Lite-M"), ("lite_s", "Lite-S"),
                       ("perfect", "Perfect CSI")]:
        p = params.get(key, "---")
        p_str = f"{p:,}" if isinstance(p, int) else p
        n = nmse_data.get(key, None)
        n_str = f"{n:.4e}" if n is not None else "---"
        b = ber_data.get(key, None)
        b_str = f"{b:.4e}" if b is not None else "---"
        lines.append(f"{label} & {p_str} & {n_str} & {b_str} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out = os.path.join(TABLES, "operating_points.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [OK] {out}")


if __name__ == "__main__":
    print("Generating tables...")
    table_complexity()
    table_operating_points()
    print("Done.")
