#!/usr/bin/env python3
"""Generate LaTeX table snippets from available JSON results for the IEEE paper."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DISTILL_RESULTS = ROOT / "distill_novelty" / "results"
STRUCTURE_RESULTS = ROOT / "results" / "structure"
TABLES = ROOT / "ieee_paper" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def table_complexity() -> bool:
    benchmarks = {variant: _load_json(DISTILL_RESULTS / f"distill_benchmark_{variant}_full.json") for variant in ["lite_l", "lite_m", "lite_s"]}
    if not all(benchmarks.values()):
        print("  [SKIP] complexity.tex: distillation benchmark JSONs are not available in this working copy")
        return False

    rows = []
    teacher = benchmarks["lite_l"]
    rows.append(("Teacher CNN", teacher["teacher_params"], teacher["teacher_mean_ms"], "1.00$\\times$"))
    for variant, label in [("lite_l", "Lite-L"), ("lite_m", "Lite-M"), ("lite_s", "Lite-S")]:
        payload = benchmarks[variant]
        rows.append((label, payload["student_params"], payload["student_mean_ms"], f"{payload['student_speedup_vs_teacher']:.2f}$\\times$"))

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Model complexity comparison. Latency measured on Apple MPS with batch size 1, averaged over 50 iterations.}",
        "\\label{tab:complexity}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model & Params & Latency (ms) & Speedup \\\\",
        "\\midrule",
    ]
    lines.extend(f"{name} & {params:,} & {lat:.2f} & {spd} \\\\" for name, params, lat, spd in rows)
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES / "complexity.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  [OK] complexity.tex")
    return True


def table_operating_points() -> bool:
    nmse_data: dict[str, float] = {}
    ber_data: dict[str, float] = {}
    available = True

    for variant in ["lite_l", "lite_m", "lite_s"]:
        nmse_payload = _load_json(DISTILL_RESULTS / f"distill_nmse_vs_pdr_{variant}_full.json")
        ber_payload = _load_json(DISTILL_RESULTS / f"distill_ber_vs_snr_{variant}_full.json")
        if nmse_payload is None or ber_payload is None:
            available = False
            break
        for record in nmse_payload["records"]:
            if record["pdr_db"] == 5.0 and record["data_snr_db"] == 15.0:
                key = variant if record["method"] == "student" else record["method"]
                nmse_data[key] = record["nmse"]
        for record in ber_payload["records"]:
            if record["pdr_db"] == 5.0 and record["data_snr_db"] == 18.0 and record["modulation"] == "bpsk":
                key = variant if record["method"] == "student" else record["method"]
                ber_data[key] = record["ber"]

    if not available:
        print("  [SKIP] operating_points.tex: distillation evaluation JSONs are not available in this working copy")
        return False

    params = {"teacher": 245473, "lite_l": 40049, "lite_m": 22789, "lite_s": 6137, "conventional": "---", "perfect": "---"}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Key operating-point comparison at PDR = 5\\,dB. NMSE at SNR = 15\\,dB; BER at SNR = 18\\,dB with BPSK.}",
        "\\label{tab:operating_points}",
        "\\begin{tabular}{lrll}",
        "\\toprule",
        "Method & Params & NMSE & BER \\\\",
        "\\midrule",
    ]
    for key, label in [
        ("conventional", "Conventional"),
        ("teacher", "Teacher CNN"),
        ("lite_l", "Lite-L"),
        ("lite_m", "Lite-M"),
        ("lite_s", "Lite-S"),
        ("perfect", "Perfect CSI"),
    ]:
        p = params.get(key, "---")
        p_str = f"{p:,}" if isinstance(p, int) else p
        n_str = f"{nmse_data[key]:.4e}" if key in nmse_data else "---"
        b_str = f"{ber_data[key]:.4e}" if key in ber_data else "---"
        lines.append(f"{label} & {p_str} & {n_str} & {b_str} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES / "operating_points.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  [OK] operating_points.tex")
    return True


def table_support_structure() -> bool:
    fast = _load_json(STRUCTURE_RESULTS / "support_structure_fast.json")
    reference = _load_json(STRUCTURE_RESULTS / "support_structure_reference.json")
    observed = _load_json(STRUCTURE_RESULTS / "support_structure_observed.json")
    if not all(item is not None for item in [fast, reference, observed]):
        print("  [SKIP] support_structure.tex: structure JSONs are missing")
        return False

    rows = [
        ("True support (fast)", fast["tail_beyond_path_count"]["median"], fast["rank_for_energy"]["0.99"]["median"], fast["rank_for_energy"]["0.999"]["median"]),
        ("True support (reference)", reference["tail_beyond_path_count"]["median"], reference["rank_for_energy"]["0.99"]["median"], reference["rank_for_energy"]["0.999"]["median"]),
        ("Raw read-off", observed["series"]["h_obs"]["tail_beyond_path_count"]["median"], observed["series"]["h_obs"]["rank_for_energy"]["0.99"]["median"], observed["series"]["h_obs"]["rank_for_energy"]["0.999"]["median"]),
        ("Thresholded read-off", observed["series"]["h_thr"]["tail_beyond_path_count"]["median"], observed["series"]["h_thr"]["rank_for_energy"]["0.99"]["median"], observed["series"]["h_thr"]["rank_for_energy"]["0.999"]["median"]),
    ]

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Dechirped structural-compressibility summary. Tail energy is measured beyond the physical path count $P=6$.}",
        "\\label{tab:support_structure}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Support image & Tail $> P$ & Rank@99\\% & Rank@99.9\\% \\\\",
        "\\midrule",
    ]
    for label, tail, rank99, rank999 in rows:
        lines.append(f"{label} & {tail:.2e} & {rank99:.1f} & {rank999:.1f} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    (TABLES / "support_structure.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  [OK] support_structure.tex")
    return True


if __name__ == "__main__":
    print("Generating tables...")
    table_support_structure()
    table_complexity()
    table_operating_points()
    print("Done. Tables in:", TABLES.resolve())
