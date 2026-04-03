from __future__ import annotations

from _common import common_parser, load_cfg
from zakotfs_novelty.adapter import generate_adapter_dataset
from zakotfs_novelty.training import train_adapter


def main() -> None:
    parser = common_parser("novelty_paper/configs/adapter_train.yaml")
    parser.add_argument("--adapter-kind", choices=["generic", "fb_lara"], default="fb_lara")
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--force-dataset", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    if args.backbone_checkpoint:
        cfg.raw.setdefault("backbone", {})["checkpoint_path"] = str(args.backbone_checkpoint)
    train_path = generate_adapter_dataset(cfg, "train", force=bool(args.force_dataset))
    val_path = generate_adapter_dataset(cfg, "val", force=bool(args.force_dataset))
    ckpt = train_adapter(cfg, train_path=train_path, val_path=val_path, adapter_kind=args.adapter_kind, mode=args.mode)
    print(f"[script] adapter_checkpoint={ckpt}", flush=True)
    print(ckpt)


if __name__ == "__main__":
    main()
