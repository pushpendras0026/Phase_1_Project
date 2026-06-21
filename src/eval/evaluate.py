# [Week 5] — implement the TODO functions below and replace this stub.
# Template: week_5/src/eval/evaluate.py
"""evaluate.py — zero-shot vs LoRA comparison over the validation split."""
import numpy as np
from pathlib import Path

from src.data.nifti_io import load_volume, apply_hu_window
from src.data.prompts import bbox_from_mask, best_start_slice
from src.data.volume_to_frames import write_frames_png
from src.engine.predictor import build_predictor, init_state
from src.engine.propagate import propagate_bidirectional
from src.eval.metrics import dice_score, hd95
from src.utils.viz import save_overlay_gif


def _load_predictor(cfg, lora_path=None):
    """Build predictor and optionally load a LoRA adapter.

    TODO:
    1. predictor = build_predictor(cfg["model"]["cfg"], cfg["model"]["ckpt"])
    2. If lora_path is given and the file exists:
           import torch
           adapter = torch.load(lora_path, map_location="cuda")
           predictor.model.image_encoder.load_state_dict(adapter, strict=False)
    3. Return predictor
    """
    pass


def evaluate_organ(cfg, organ_id, lora_path=None):
    """Run inference on every val case and return per-case result dicts.

    Each result dict has keys: case, dsc_zs, hd95_zs, dsc_lora, hd95_lora.

    TODO:
    1. val_cases = cfg["split"]["val_cases"]
       img_dir   = Path(cfg["paths"]["raw_images"])
       lbl_dir   = Path(cfg["paths"]["raw_labels"])

    2. Build predictors:
           pred_zs   = _load_predictor(cfg)
           pred_lora = _load_predictor(cfg, lora_path) if lora_path else None

    3. For each case in val_cases:
       a. Construct paths:
              img_path   = img_dir / f"{case}.nii"
              lbl_path   = lbl_dir / f"{case.replace('img', 'label')}.nii"
              frames_dir = Path(cfg["paths"]["processed"]) / case
              out_dir    = Path("outputs") / case
              out_dir.mkdir(parents=True, exist_ok=True)
          Skip if either file does not exist.

       b. Write PNG frames if not already done:
              if not frames_dir.exists() or not any(frames_dir.glob("*.png")):
                  write_frames_png(str(img_path), str(frames_dir))

       c. Load volume and labels:
              vol, _, spacing = load_volume(str(img_path))
              lbl, _, _       = load_volume(str(lbl_path))
              vol_u8          = apply_hu_window(vol)
              gt3d            = (lbl == organ_id).astype(bool)
          Skip this case if gt3d has no True values (organ absent).

       d. Find prompt:
              start_z = best_start_slice(lbl, organ_id)
              gt_slice = (lbl[:, :, start_z] == organ_id).astype(np.uint8)
              bbox = bbox_from_mask(gt_slice, pad=4)
          Skip if bbox is None.

       e. Zero-shot inference:
              state_zs = init_state(pred_zs, str(frames_dir))
              mask_zs  = propagate_bidirectional(pred_zs, state_zs, start_z, bbox,
                                                 target_hw=(vol.shape[0], vol.shape[1]))
              dsc_zs = dice_score(mask_zs, gt3d)
              hd_zs  = hd95(mask_zs, gt3d, spacing)
          Save GIF: save_overlay_gif(vol_u8, mask_zs, str(out_dir / f"zeroshot_{organ_id}.gif"), gt3d)

       f. LoRA inference (if pred_lora is not None):
              state_lo = init_state(pred_lora, str(frames_dir))
              mask_lo  = propagate_bidirectional(pred_lora, state_lo, start_z, bbox,
                                                 target_hw=(vol.shape[0], vol.shape[1]))
              dsc_lo = dice_score(mask_lo, gt3d)
              hd_lo  = hd95(mask_lo, gt3d, spacing)
          Save GIF. If no LoRA, set dsc_lo = hd_lo = float("nan").

       g. Append result dict and print a one-line summary.

    4. Return the list of result dicts.

    Args:
        cfg       : config dict loaded from configs/default.yaml
        organ_id  : BTCV integer label for the target organ
        lora_path : path to saved LoRA adapter .pt file, or None for zero-shot only

    Returns:
        List of dicts with keys: case, dsc_zs, hd95_zs, dsc_lora, hd95_lora
    """
    pass


def print_table(rows, organ_id):
    """Print a formatted summary table and return aggregated metrics.

    TODO:
    1. Compute mean and std for each metric across all rows (skip NaN values):
           dsc_zs_mean, dsc_zs_std   = mean and std of [r["dsc_zs"]   for r in rows]
           dsc_lora_mean, dsc_lora_std = ...
           hd_zs_mean, hd_zs_std     = ...
           hd_lora_mean, hd_lora_std  = ...

    2. Print a table with columns: Metric | Zero-shot | LoRA
       Example format:
           ============================================================
             Organ 6  |  6 val cases
           ============================================================
             Metric         Zero-shot           LoRA
             --------------------------------------------------
             DSC            0.912 ± 0.023       0.941 ± 0.018
             HD95 (mm)      4.2   ± 1.1         2.9   ± 0.8
           ============================================================

    3. Return a dict with keys:
       organ, dsc_zs_mean, dsc_zs_std, dsc_lora_mean, dsc_lora_std,
       hd95_zs_mean, hd95_zs_std, hd95_lora_mean, hd95_lora_std

    Args:
        rows     : list of result dicts from evaluate_organ()
        organ_id : integer organ label (for display)

    Returns:
        Dict of aggregated metric values
    """
    pass
