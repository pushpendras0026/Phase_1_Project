#!/usr/bin/env python3
"""04_evaluate.py — evaluate zero-shot vs LoRA Dice + HD95 on the val split.

Usage:
    python scripts/04_evaluate.py --organ 6
    python scripts/04_evaluate.py --organ 6 --lora checkpoints/lora_organ6.pt
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Fix #9: seed before any torch imports
from src.utils.seed import set_seed
set_seed(42)

import yaml

from src.eval.evaluate import evaluate_organ, print_table


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAM2 segmentation on BTCV val split")
    parser.add_argument("--organ",  type=int, required=True, help="BTCV organ label id")
    parser.add_argument("--lora",   default=None, help="LoRA adapter .pt (auto-detected if absent)")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Auto-detect LoRA adapter for this organ if not provided
    lora_path = args.lora
    if lora_path is None:
        auto = Path(cfg["paths"]["checkpoints"]) / f"lora_organ{args.organ}.pt"
        if auto.exists():
            lora_path = str(auto)
            print(f"Auto-detected LoRA adapter: {lora_path}")
        else:
            print("No LoRA adapter found — running zero-shot only.")

    print(f"\nEvaluating organ={args.organ}  val cases={cfg['split']['val_cases']}\n")
    rows = evaluate_organ(cfg, args.organ, lora_path)

    if not rows:
        print("No results — check that val cases exist in the data directory.")
        return

    summary = print_table(rows, args.organ)

    # Save per-case CSV
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"eval_organ{args.organ}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Per-case CSV saved -> {csv_path}")


if __name__ == "__main__":
    main()
