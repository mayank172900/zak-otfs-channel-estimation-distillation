from __future__ import annotations

from pathlib import Path

from zakotfs.dataset import simulate_frame
from zakotfs.params import load_config


def main() -> None:
    cfg = load_config(Path("configs/system.yaml"))
    import numpy as np

    frame = simulate_frame(cfg, "bpsk", data_snr_db=15.0, pdr_db=5.0, rng=np.random.default_rng(cfg.seed))
    print(frame.support_input.shape, frame.support_true.shape, frame.y_dd.shape)


if __name__ == "__main__":
    main()
