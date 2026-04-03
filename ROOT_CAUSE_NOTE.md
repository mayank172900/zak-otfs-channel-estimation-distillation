# Root Cause Note

The highest-priority blocker was the MMSE detector ridge term. The solver had been regularized with raw `N0`, but the detector's unknown DD data vector has per-entry variance `E_d / (MN)` under this repo's waveform normalization. The correct LMMSE ridge is therefore `MN * N0 / E_d = 1 / rho_d`. That scaling error makes the detector behave much closer to an under-regularized ZF solve, which directly inflates perfect-CSI BER and then contaminates the conventional and CNN-assisted BER curves.

The second blocker was configuration drift in the effective-channel path. `simulate_frame()` was hard-wired to the fast support construction, so changing `numerics.effective_channel_method` did not actually change the evaluation/training pipeline. That prevented a clean fast-vs-reference comparison for the fractional delay/fractional Doppler channel model.

The plotting issue was separate but misleading: BER-vs-SNR plots grouped only by method, so BPSK and 8-QAM traces collapsed into the same three lines. That hid modulation-specific gaps even when the CSV data was distinct.

This patch set fixes the MMSE ridge scaling, routes the effective-channel method through config-aware support generation, adds deterministic diagnostics for perfect-CSI PDR invariance and fast-vs-reference mismatch, and separates modulation in BER-vs-SNR plotting. The final paper-faithful figures and anchor numbers still need to be regenerated on the target Windows CUDA machine.
