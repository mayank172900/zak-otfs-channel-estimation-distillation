# FB-LARA Novelty Pipeline

This folder isolates the novelty implementation from the reproduced baseline.

Core pieces:

- `src/zakotfs_novelty/`: copied baseline package with FB-LARA additions
- `configs/adapter_train.yaml`: small residual-learning dataset and adapter training recipe
- `configs/adapter_train_mps.yaml`: same, pinned to `mps`
- `configs/adapter_eval.yaml`: ablation evaluation config
- `configs/adapter_eval_mps.yaml`: same, pinned to `mps`
- `scripts/generate_adapter_dataset.py`: precompute 8-channel FB-LARA features
- `scripts/train_adapter.py`: train `generic` or `fb_lara`
- `scripts/eval_adapter_ablations.py`: run ablation NMSE/BER curves
- `scripts/reproduce_adapter.py`: end-to-end adapter pipeline

Default backbone checkpoint:

- `../logs/checkpoints/full_cnn_best.pt`

Override it only if your baseline checkpoint lives elsewhere.

MPS quick start from the repo root:

```bash
export PYTHONPATH=novelty_paper/src
python3 novelty_paper/scripts/check_device.py
python3 -m pytest -q novelty_paper/tests --rootdir novelty_paper -c novelty_paper/pytest.ini
python3 novelty_paper/scripts/train_adapter.py --config novelty_paper/configs/adapter_train_mps.yaml --mode full --adapter-kind generic
python3 novelty_paper/scripts/train_adapter.py --config novelty_paper/configs/adapter_train_mps.yaml --mode full --adapter-kind fb_lara
python3 novelty_paper/scripts/eval_adapter_ablations.py --config novelty_paper/configs/adapter_eval_mps.yaml --mode full --sections nmse_pdr nmse_snr
```
