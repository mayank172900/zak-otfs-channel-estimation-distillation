from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK_PATH = Path("novelty_paper/FB_LARA_kaggle_single_file.ipynb")
BASE_DIR = Path("novelty_paper")
PATTERNS = ["src/zakotfs_novelty/*.py", "configs/*.yaml", "tests/*.py", "pytest.ini"]


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def collect_payloads() -> dict[str, str]:
    payloads: dict[str, str] = {}
    for pattern in PATTERNS:
        for path in sorted(BASE_DIR.glob(pattern)):
            payloads[path.relative_to(BASE_DIR).as_posix()] = path.read_text(encoding="utf-8")
    return payloads


def build_notebook(payloads: dict[str, str]) -> dict:
    payload_json = json.dumps(payloads, ensure_ascii=False)

    intro = """
# FB-LARA Kaggle Single-File Notebook

This notebook is self-contained. It writes the embedded `zakotfs_novelty` package, configs, and tests into the Kaggle working directory, then runs the FB-LARA novelty pipeline.

Upload to Kaggle:
- `full_cnn_best.pt` or `train_history_full.zip`
- optionally `train_full.json` / `val_full.json` plus their `.npy` memmaps from the 480k baseline dataset

If the baseline train/val manifests are present, adapter dataset generation reuses them directly and skips the slow full-frame resimulation path.
"""

    run_order = """
## Run Order

1. Run the setup/config cell and confirm the uploaded checkpoint was found.
2. Run the embed/write cell once.
3. Optionally run the novelty tests.
4. Generate the adapter datasets.
5. Train `generic` and `fb_lara`.
6. Run NMSE first.
7. Run BER only if you want the full ablation curves.
"""

    cell_setup = """
from pathlib import Path
import os
import sys
import zipfile
import subprocess

import torch

KAGGLE_INPUT_ROOT = Path('/kaggle/input')
WORK_ROOT = Path('/kaggle/working/fb_lara_single')
WORK_ROOT.mkdir(parents=True, exist_ok=True)
EXTRACT_ROOT = WORK_ROOT / 'extracted_inputs'
EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)

AUTO_EXTRACT_ZIPS = True
RUN_TESTS = False
RUN_DATASET_BUILD = True
RUN_TRAIN_GENERIC = True
RUN_TRAIN_FB_LARA = True
RUN_EVAL_NMSE = True
RUN_EVAL_BER = False
DEVICE_OVERRIDE = 'cuda' if torch.cuda.is_available() else 'cpu'
ADAPTER_TRAIN_SIZE = 48000
ADAPTER_VAL_SIZE = 12000

def _search_direct(name: str):
    if not KAGGLE_INPUT_ROOT.exists():
        return None
    matches = sorted(KAGGLE_INPUT_ROOT.rglob(name))
    return matches[0] if matches else None

def _search_zip(name: str):
    if not AUTO_EXTRACT_ZIPS or not KAGGLE_INPUT_ROOT.exists():
        return None
    for zip_path in sorted(KAGGLE_INPUT_ROOT.rglob('*.zip')):
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = [member for member in zf.namelist() if Path(member).name == name]
            if not members:
                continue
            out_dir = EXTRACT_ROOT / zip_path.stem
            if not out_dir.exists() or not any(out_dir.rglob(name)):
                print(f'[setup] extracting {zip_path} to {out_dir}')
                zf.extractall(out_dir)
            matches = sorted(out_dir.rglob(name))
            if matches:
                return matches[0]
    return None

def find_artifact(name: str):
    direct = _search_direct(name)
    return direct if direct is not None else _search_zip(name)

BASELINE_CHECKPOINT = find_artifact('full_cnn_best.pt')
BASELINE_TRAIN_MANIFEST = find_artifact('train_full.json')
BASELINE_VAL_MANIFEST = find_artifact('val_full.json')

print({'device': DEVICE_OVERRIDE, 'checkpoint': str(BASELINE_CHECKPOINT) if BASELINE_CHECKPOINT else None})
print({'train_manifest': str(BASELINE_TRAIN_MANIFEST) if BASELINE_TRAIN_MANIFEST else None})
print({'val_manifest': str(BASELINE_VAL_MANIFEST) if BASELINE_VAL_MANIFEST else None})

assert BASELINE_CHECKPOINT is not None, 'Upload full_cnn_best.pt or a zip containing it.'
"""

    cell_embed = f"""
from pathlib import Path
import sys

FILE_PAYLOADS = {payload_json}

for rel_path, text in FILE_PAYLOADS.items():
    out_path = WORK_ROOT / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding='utf-8')

if str(WORK_ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(WORK_ROOT / 'src'))

print(f'[setup] wrote {{len(FILE_PAYLOADS)}} embedded files under {{WORK_ROOT}}')
"""

    cell_config = """
from zakotfs_novelty.params import load_config

def apply_runtime_overrides(cfg, *, train_cfg: bool):
    cfg.raw['device'] = DEVICE_OVERRIDE
    cfg.raw.setdefault('backbone', {})['checkpoint_path'] = str(BASELINE_CHECKPOINT)
    cfg.raw['paths'] = {
        'results_dir': str((WORK_ROOT / 'results').resolve()),
        'logs_dir': str((WORK_ROOT / 'logs').resolve()),
        'report_dir': str((WORK_ROOT / 'report').resolve()),
    }
    if train_cfg:
        cfg.raw['adapter_dataset']['train_size_total'] = int(ADAPTER_TRAIN_SIZE)
        cfg.raw['adapter_dataset']['val_size_total'] = int(ADAPTER_VAL_SIZE)
        if BASELINE_TRAIN_MANIFEST is not None and BASELINE_VAL_MANIFEST is not None:
            cfg.raw['adapter_dataset']['baseline_manifest_paths'] = {
                'train': str(BASELINE_TRAIN_MANIFEST),
                'val': str(BASELINE_VAL_MANIFEST),
            }
        else:
            cfg.raw['adapter_dataset']['baseline_manifest_paths'] = {}
    return cfg

train_cfg = apply_runtime_overrides(load_config(WORK_ROOT / 'configs' / 'adapter_train.yaml'), train_cfg=True)
eval_cfg = apply_runtime_overrides(load_config(WORK_ROOT / 'configs' / 'adapter_eval.yaml'), train_cfg=False)

print('[setup] runtime train config ready')
print(train_cfg.raw['adapter_dataset'])
print('[setup] runtime eval device:', eval_cfg.raw['device'])
"""

    cell_tests = """
if RUN_TESTS:
    env = dict(os.environ)
    env['PYTHONPATH'] = str(WORK_ROOT / 'src')
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests', '--rootdir', str(WORK_ROOT), '-c', str(WORK_ROOT / 'pytest.ini')], cwd=WORK_ROOT, env=env, check=True)
else:
    print('[tests] skipped')
"""

    cell_dataset = """
from zakotfs_novelty.adapter import generate_adapter_dataset

train_adapter_manifest = WORK_ROOT / 'results' / 'adapter_datasets' / 'train_adapter.json'
val_adapter_manifest = WORK_ROOT / 'results' / 'adapter_datasets' / 'val_adapter.json'

if RUN_DATASET_BUILD:
    train_adapter_manifest = generate_adapter_dataset(train_cfg, 'train', force=False)
    val_adapter_manifest = generate_adapter_dataset(train_cfg, 'val', force=False)

print({'train_adapter_manifest': str(train_adapter_manifest), 'val_adapter_manifest': str(val_adapter_manifest)})
"""

    cell_train = """
from pathlib import Path
from zakotfs_novelty.training import train_adapter

adapter_ckpts = {}
if RUN_TRAIN_GENERIC:
    adapter_ckpts['generic'] = train_adapter(train_cfg, Path(train_adapter_manifest), Path(val_adapter_manifest), adapter_kind='generic', mode='full')
if RUN_TRAIN_FB_LARA:
    adapter_ckpts['fb_lara'] = train_adapter(train_cfg, Path(train_adapter_manifest), Path(val_adapter_manifest), adapter_kind='fb_lara', mode='full')

print({key: str(value) for key, value in adapter_ckpts.items()})
"""

    cell_load = """
from pathlib import Path
from zakotfs_novelty.adapter import load_frozen_backbone
from zakotfs_novelty.evaluation import evaluate_ber_vs_pdr, evaluate_ber_vs_snr, evaluate_nmse_vs_pdr, evaluate_nmse_vs_snr, save_eval_outputs
from zakotfs_novelty.training import load_adapter_checkpoint

def apply_available_methods(cfg, adapters):
    available = ['conventional', 'cnn'] + sorted(adapters.keys())
    available_with_perfect = available + ['perfect']
    for section in ['nmse_vs_pdr', 'nmse_vs_snr']:
        cfg.raw['evaluation'][section]['methods'] = [method for method in cfg.raw['evaluation'][section]['methods'] if method in available]
    for section in ['ber_vs_pdr', 'ber_vs_snr']:
        cfg.raw['evaluation'][section]['methods'] = [method for method in cfg.raw['evaluation'][section]['methods'] if method in available_with_perfect]

backbone, backbone_device = load_frozen_backbone(eval_cfg)
loaded_adapters = {}
for kind in ['generic', 'fb_lara']:
    if kind in adapter_ckpts:
        loaded_adapters[kind] = load_adapter_checkpoint(eval_cfg, Path(adapter_ckpts[kind]), adapter_kind=kind)
    else:
        ckpt_path = WORK_ROOT / 'logs' / 'checkpoints' / f'full_{kind}_best.pt'
        if ckpt_path.exists():
            loaded_adapters[kind] = load_adapter_checkpoint(eval_cfg, ckpt_path, adapter_kind=kind)

apply_available_methods(eval_cfg, loaded_adapters)
print({'backbone_device': str(backbone_device), 'adapters': list(loaded_adapters.keys())})
"""

    cell_nmse = """
if RUN_EVAL_NMSE:
    nmse_pdr = evaluate_nmse_vs_pdr(eval_cfg, backbone, loaded_adapters, mode='full')
    save_eval_outputs(nmse_pdr, 'nmse', 'adapter_nmse_vs_pdr_full', eval_cfg)
    nmse_snr = evaluate_nmse_vs_snr(eval_cfg, backbone, loaded_adapters, mode='full')
    save_eval_outputs(nmse_snr, 'nmse', 'adapter_nmse_vs_snr_full', eval_cfg)
else:
    print('[eval] NMSE skipped')
"""

    cell_ber = """
if RUN_EVAL_BER:
    ber_pdr = evaluate_ber_vs_pdr(eval_cfg, backbone, loaded_adapters, mode='full')
    save_eval_outputs(ber_pdr, 'ber', 'adapter_ber_vs_pdr_full', eval_cfg)
    ber_snr = evaluate_ber_vs_snr(eval_cfg, backbone, loaded_adapters, mode='full')
    save_eval_outputs(ber_snr, 'ber', 'adapter_ber_vs_snr_full', eval_cfg)
else:
    print('[eval] BER skipped')
"""

    cell_artifacts = """
results_dir = WORK_ROOT / 'results'
logs_dir = WORK_ROOT / 'logs'
print('[artifacts] results')
for path in sorted(results_dir.rglob('*')):
    if path.is_file():
        print(path)
print('[artifacts] logs')
for path in sorted(logs_dir.rglob('*')):
    if path.is_file():
        print(path)
"""

    return {
        "cells": [
            md(intro),
            md(run_order),
            code(cell_setup),
            code(cell_embed),
            code(cell_config),
            code(cell_tests),
            code(cell_dataset),
            code(cell_train),
            code(cell_load),
            code(cell_nmse),
            code(cell_ber),
            code(cell_artifacts),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    payloads = collect_payloads()
    notebook = build_notebook(payloads)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
    print(NOTEBOOK_PATH)
    print(NOTEBOOK_PATH.stat().st_size)


if __name__ == "__main__":
    main()
