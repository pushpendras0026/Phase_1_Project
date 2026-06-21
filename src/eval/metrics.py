# [Week 4] — implement the TODO functions below and replace this stub.
# Template: week_4/src/eval/metrics.py
import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance


def dice_score(pred3d, gt3d, eps=1e-6):
    """Volumetric Dice Similarity Coefficient between two 3D binary masks.

    DSC = 2 × |A ∩ B| / (|A| + |B|)
    Range: 0 (no overlap) to 1 (perfect overlap).
    Published baselines: liver ~0.93, spleen ~0.94, pancreas ~0.7–0.8.

    TODO:
    1. Cast to bool:   p = pred3d.astype(bool)  ;  g = gt3d.astype(bool)
    2. Intersection:   inter = (p & g).sum()
    3. Return:         float((2 * inter + eps) / (p.sum() + g.sum() + eps))

    Args:
        pred3d : 3D bool/uint8 numpy array (H, W, Z) — predicted organ mask
        gt3d   : 3D bool/uint8 numpy array (H, W, Z) — ground-truth organ mask
        eps    : smoothing constant (prevents division by zero for empty volumes)

    Returns:
        Dice score as a Python float
    """
    pass


def hd95(pred3d, gt3d, spacing):
    """95th-percentile Hausdorff Distance in millimetres via MONAI.

    HD95 measures boundary error — how far off the predicted surface is from
    the ground truth at the 95th percentile (robust to a few outlier voxels).
    Lower is better. Typical values: liver <5 mm, pancreas <20 mm.

    TODO:
    1. Convert to float32 torch tensors with batch and channel dims:
           p = torch.tensor(pred3d.astype(np.uint8), dtype=torch.float32)[None, None]
           g = torch.tensor(gt3d.astype(np.uint8),   dtype=torch.float32)[None, None]
           # Shape: (1, 1, H, W, Z)

    2. MONAI expects spacing in (z, y, x) order for 3D data.
       Our volume is (H, W, Z) so spacing = (sx, sy, sz) from load_volume().
       Reorder:  sp_zyx = (float(spacing[2]), float(spacing[1]), float(spacing[0]))

    3. Compute:
           dist = compute_hausdorff_distance(p, g, percentile=95, spacing=sp_zyx)

    4. Return float(dist.squeeze())

    IMPORTANT: always pass spacing — without it the result is in voxels, not mm.

    Args:
        pred3d  : 3D bool numpy array (H, W, Z)
        gt3d    : 3D bool numpy array (H, W, Z)
        spacing : voxel spacing in mm as (sx, sy, sz) from nifti_io.load_volume()

    Returns:
        HD95 distance in millimetres as a Python float
    """
    pass
