#!/usr/bin/env python3
"""02_run_inference.py — SAM2 bidirectional propagation on one BTCV case.

Usage:
    python scripts/02_run_inference.py --case img0001 --organ 6
    python scripts/02_run_inference.py --case img0001 --organ 6 --lora checkpoints/lora_organ6.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
import numpy as np

from src.data.nifti_io import load_volume, apply_hu_window
from src.data.volume_to_frames import write_frames_png
from src.data.prompts import bbox_from_mask, best_start_slice
from src.engine.predictor import build_predictor, init_state
from src.engine.propagate import propagate_bidirectional
from src.utils.viz import save_overlay_gif


def main():
    parser = argparse.ArgumentParser(description="SAM2 zero-shot / LoRA inference")
    parser.add_argument("--case",   required=True, help="e.g. img0001")
    parser.add_argument("--organ",  type=int, required=True, help="BTCV organ label id")
    parser.add_argument("--lora",   default=None, help="path to LoRA adapter .pt")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    img_path   = Path(cfg["paths"]["raw_images"]) / f"{args.case}.nii"
    lbl_path   = Path(cfg["paths"]["raw_labels"]) / f"{args.case.replace('img', 'label')}.nii"
    frames_dir = Path(cfg["paths"]["processed"]) / args.case
    out_dir    = Path("outputs") / args.case
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write PNG frames if not already on disk
    if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
        print("Writing PNG frames...")
        write_frames_png(str(img_path), str(frames_dir))

    vol, _, spacing = load_volume(str(img_path))
    lbl, _, _       = load_volume(str(lbl_path))
    vol_u8 = apply_hu_window(vol)                        # (H,W,Z) uint8

    try:
        start_z = best_start_slice(lbl, args.organ)
    except ValueError as e:
        print(e); return

    gt_slice = (lbl[:, :, start_z] == args.organ).astype(np.uint8)
    bbox = bbox_from_mask(gt_slice, pad=4)
    if bbox is None:
        print(f"Organ {args.organ} not found in {args.case}."); return

    print(f"Building predictor  start_z={start_z}  bbox={bbox}")
    predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])

    if args.lora:
        import torch
        ckpt = torch.load(args.lora, map_location="cuda")
        predictor.model.image_encoder.load_state_dict(ckpt, strict=False)
        print(f"Loaded LoRA adapter from {args.lora}")

    state = init_state(predictor, str(frames_dir))

    # Resize masks back to original volume spatial dimensions
    target_hw = (vol.shape[0], vol.shape[1])
    pred3d = propagate_bidirectional(predictor, state, start_z, bbox,
                                     target_hw=target_hw)

    npy_path = out_dir / f"mask_organ{args.organ}.npy"
    np.save(str(npy_path), pred3d)
    print(f"Saved mask  shape={pred3d.shape}  -> {npy_path}")

    gt3d = (lbl == args.organ).astype(bool)
    gif_path = out_dir / f"overlay_organ{args.organ}.gif"
    save_overlay_gif(vol_u8, pred3d, str(gif_path), gt3d)
    print(f"Saved GIF -> {gif_path}")


if __name__ == "__main__":
    main()
