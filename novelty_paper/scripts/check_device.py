from __future__ import annotations

import torch


def main() -> None:
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"mps_available={hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
