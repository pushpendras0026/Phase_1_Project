#!/usr/bin/env python3
"""03_train_lora.py — fine-tune SAM2 with LoRA on BTCV organ slices.

Usage:
    python scripts/03_train_lora.py --organ 6
    python scripts/03_train_lora.py --organ 1 --qv-only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix #9: seed before ANY torch / numpy import
from src.utils.seed import set_seed
set_seed(42)

import yaml

from src.data.dataset import BTCVSliceDataset
from src.engine.predictor import build_predictor
from src.train.lora import add_lora_peft, inject_lora_qv, trainable_report
from src.train.train import run_training


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune SAM2 on BTCV")
    parser.add_argument("--organ",   type=int, required=True, help="BTCV organ label id")
    parser.add_argument("--qv-only", action="store_true",
                        help="Use custom Q&V-only LoRA (Option B) instead of peft")
    parser.add_argument("--config",  default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # ── Build model and inject LoRA ──────────────────────────────────────────
    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])
    model     = predictor.model

    if args.qv_only:
        inject_lora_qv(model.image_encoder,
                       r=cfg["train"]["rank"], alpha=cfg["train"]["alpha"])
    else:
        model.image_encoder = add_lora_peft(
            model.image_encoder,
            r=cfg["train"]["rank"],
            alpha=cfg["train"]["alpha"],
        )

    trainable_report(model)

    # ── Dataset (exclude val cases) ──────────────────────────────────────────
    val_set  = set(cfg["split"]["val_cases"])
    img_dir  = Path(cfg["paths"]["raw_images"])
    lbl_dir  = Path(cfg["paths"]["raw_labels"])
    all_cases = [p.stem for p in sorted(img_dir.glob("*.nii"))
                 if p.stem not in val_set]

    print(f"Training cases: {len(all_cases)}  |  organ={args.organ}")
    dataset = BTCVSliceDataset(
        all_cases, args.organ,
        str(img_dir), str(lbl_dir),
        image_size=cfg["model"]["image_size"],
    )
    print(f"Dataset slices: {len(dataset)}")

    if len(dataset) == 0:
        print("No slices found for this organ in training cases. Exiting.")
        return

    # ── Train ────────────────────────────────────────────────────────────────
    save_path = Path(cfg["paths"]["checkpoints"]) / f"lora_organ{args.organ}.pt"
    run_training(cfg, model, dataset, args.organ, str(save_path))


if __name__ == "__main__":
    main()
