from __future__ import annotations

from _common import common_parser, load_cfg
from zakotfs.dataset import generate_dataset


def main() -> None:
    parser = common_parser()
    parser.add_argument("--split", choices=["train", "val"], required=True)
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    print(f"[script] generate_dataset config={args.config} mode={args.mode} split={args.split}", flush=True)
    path = generate_dataset(cfg, split=args.split, mode=args.mode, force=True)
    print(f"[script] dataset artifact={path}", flush=True)
    print(path)


if __name__ == "__main__":
    main()
