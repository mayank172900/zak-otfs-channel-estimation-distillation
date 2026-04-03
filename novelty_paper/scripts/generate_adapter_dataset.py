from __future__ import annotations

from _common import common_parser, load_cfg
from zakotfs_novelty.adapter import generate_adapter_dataset


def main() -> None:
    parser = common_parser("novelty_paper/configs/adapter_train.yaml")
    parser.add_argument("--split", choices=["train", "val"], required=True)
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    if args.backbone_checkpoint:
        cfg.raw.setdefault("backbone", {})["checkpoint_path"] = str(args.backbone_checkpoint)
    path = generate_adapter_dataset(cfg, split=args.split, force=bool(args.force))
    print(f"[script] adapter_dataset={path}", flush=True)
    print(path)


if __name__ == "__main__":
    main()
