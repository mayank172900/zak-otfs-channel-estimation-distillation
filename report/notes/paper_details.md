# Paper Extraction Notes

## Core equations

- Data DD waveform: paper Eq. (1)
- Spread pilot DD waveform: paper Eq. (2)
- Superimposed frame: paper Eq. (3)
- Matrix model: paper Eq. (4)
- Cross-ambiguity decomposition: paper Eq. (5) and Eq. (7)
- Self-ambiguity lattice condition: paper Eq. (6)
- Conventional read-off estimator: paper Eq. (8)
- CNN-enhanced estimate re-embedding: paper Eq. (9)
- CNN loss: paper Eq. (10)

## Fixed parameters

- `nu_p = 30 kHz`
- `tau_p = 1 / nu_p`
- `M = 31`
- `N = 37`
- `q = 3`
- `T = N * tau_p`
- `B = M * nu_p`
- GS pulse: `alpha_tau = alpha_nu = 0.044`, `Omega_tau = Omega_nu = 1.0278`
- Receiver matched filter: `w_rx(tau, nu) = conj(w_tx(-tau, -nu)) * exp(j 2 pi tau nu)`

## Channel

- Vehicular-A, `P = 6`
- `nu_max = 815 Hz`
- `tau_max = 2.51 us`
- Delays `[0, 0.31, 0.71, 1.09, 1.73, 2.51] us`
- Relative powers `[0, -1, -9, -10, -15, -20] dB`
- Fractional delays and Dopplers
- Effective-channel matrix summation indices `m,n in {-1,0,1}`

## Support geometry

- Support centered at `(0,0)`
- For `M=31, N=37, q=3`, the paper states `Delta_k = 27`, `Delta_l = 43`
- Therefore `k in [-13, 13]`, `l in [-21, 21]`
- Support image size `27 x 43`

## CNN

- Shared real-valued network used separately on real and imaginary inputs
- Conv 1: `1 -> 64`, kernel `27x27`, ReLU
- Conv 2: `64 -> 32`, kernel `9x9`, ReLU
- Conv 3: `32 -> 32`, kernel `5x5`, ReLU
- Conv 4: `32 -> 1`, kernel `15x15`, linear
- Trainable parameter count must be `245473`

## Training

- SNR `15 dB`
- PDR list `[0, 5, 20, 25, 30, 35] dB`
- Train size `480000`
- Validation/test size `120000`
- Batch size `64`
- Epochs `50`
- LR `5e-4`
- Adam
- `ReduceLROnPlateau(factor=0.8, patience=3, min_lr=1e-6)`
- Save best validation loss checkpoint

## Evaluation

- NMSE vs PDR: BPSK, data SNR `15 dB`, PDR sweep `[-5,0,5,10,15,20,25]`
- NMSE vs SNR: BPSK, PDR `5 dB`, SNR sweep `[0,3,6,9,12,15,18,21,24]`
- BER vs PDR: BPSK, data SNR `18 dB`, MMSE, compare conventional/CNN/perfect
- BER vs SNR: PDR `5 dB`, MMSE, BPSK and 8-QAM, compare conventional/CNN/perfect
