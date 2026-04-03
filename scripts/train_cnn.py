from __future__ import annotations

from _common import common_parser, load_cfg
from zakotfs.dataset import generate_dataset
from zakotfs.training import train_cnn


def main() -> None:
    parser = common_parser()
    parser.set_defaults(config="configs/train.yaml")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    print(f"[script] train_cnn config={args.config} mode={args.mode}", flush=True)
    train_path = generate_dataset(cfg, "train", mode=args.mode, force=False)
    val_path = generate_dataset(cfg, "val", mode=args.mode, force=False)
    ckpt = train_cnn(cfg, train_path=train_path, val_path=val_path, mode=args.mode)
    print(f"[script] training complete checkpoint={ckpt}", flush=True)
    print(ckpt)


if __name__ == "__main__":
    main()
