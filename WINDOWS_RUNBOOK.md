# Windows Runbook

## Environment

Use your existing PyTorch CUDA environment on Windows 10 with the RTX 3050 laptop GPU.

```powershell
cd <PATH_TO_UNZIPPED_REPO>
python -m pip install --upgrade pip
python -m pip install -r requirements_no_torch.txt
$env:PYTHONPATH = "$PWD\src"
```

Do not reinstall `torch` from `requirements.txt` if you already have a CUDA-enabled PyTorch environment. Verify CUDA first:

```powershell
python scripts\check_cuda.py
```

If that script fails, stop there and fix the PyTorch CUDA environment first. The reproduction commands below assume `torch.cuda.is_available()` is `True`.

## Tests

```powershell
pytest -q
```

## Diagnostics Before Full Training

Fast path:

```powershell
python scripts\diagnose_effective_channel.py --config configs\eval_fast_cuda.yaml --num-random 64
python scripts\diagnose_perfect_csi_pdr.py --config configs\eval_fast_cuda.yaml --modulation bpsk --data-snr-db 18 --pdr-db 0 5 20 25 30 35
```

Reference path:

```powershell
python scripts\diagnose_effective_channel.py --config configs\eval_reference_cuda.yaml --num-random 64
python scripts\diagnose_perfect_csi_pdr.py --config configs\eval_reference_cuda.yaml --modulation bpsk --data-snr-db 18 --pdr-db 0 5 20 25 30 35
```

## Full Reproduction

Fast path:

```powershell
python scripts\windows_reproduce.py --train-config configs\train_fast_cuda.yaml --eval-config configs\eval_fast_cuda.yaml --mode full
```

Reference path:

```powershell
python scripts\windows_reproduce.py --train-config configs\train_reference_cuda.yaml --eval-config configs\eval_reference_cuda.yaml --mode full
```

## Artifacts To Check

- `results_fast\*.csv`
- `results_fast\*.json`
- `results_fast\*.png`
- `results_reference\*.csv`
- `results_reference\*.json`
- `results_reference\*.png`
- `logs_fast\checkpoints\full_epoch_*.pt`
- `logs_fast\checkpoints\full_cnn_best.pt`
- `logs_fast\train_history_full.json`
- `logs_reference\checkpoints\full_epoch_*.pt`
- `logs_reference\checkpoints\full_cnn_best.pt`
- `logs_reference\train_history_full.json`

## Paper Anchor Checks

At 10 dB PDR NMSE:

- conventional near `1.75e-2`
- CNN near `1.85e-3`

At 18 dB data SNR, 5 dB PDR, BPSK BER:

- conventional near `2.5e-2`
- CNN near `4.7e-4`
- perfect near `3.0e-4`

At 18 dB data SNR, 5 dB PDR, BER vs SNR anchor:

- BPSK: perfect `2.6e-4`, CNN `4.7e-4`, conventional `2.5e-2`
- 8-QAM: perfect `2.5e-2`, CNN `4.0e-2`, conventional `1.6e-1`
