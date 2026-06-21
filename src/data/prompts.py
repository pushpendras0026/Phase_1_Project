# [Week 2] — implement the TODO functions below and replace this stub.
# Template: week_2/src/data/prompts.py
import numpy as np
from scipy import ndimage


def bbox_from_mask(mask2d, pad=4):
    """Compute the bounding box of a binary 2D mask with pixel padding.

    TODO:
    1. Find pixel coordinates where mask is True:
           ys, xs = np.where(mask2d)
    2. If no True pixels: return None  (organ is absent in this slice)
    3. Compute box with padding (clamp to image boundaries):
           H, W = mask2d.shape
           x0 = max(int(xs.min()) - pad, 0)
           x1 = min(int(xs.max()) + pad, W - 1)
           y0 = max(int(ys.min()) - pad, 0)
           y1 = min(int(ys.max()) + pad, H - 1)
    4. Return np.array([x0, y0, x1, y1], dtype=np.float32)

    This box is passed directly to SAM 2's add_new_points_or_box().

    Args:
        mask2d : 2D binary array (H, W) — True/1 where the organ exists
        pad    : pixels to expand the box on each side

    Returns:
        float32 array [x0, y0, x1, y1], or None if mask is entirely empty
    """
    pass


def best_start_slice(vol_mask, organ_id, k=5):
    """Find the best axial slice to place the initial SAM 2 prompt.

    Using the single largest-area slice can fail on branchy anatomy (e.g. pancreas)
    where the organ appears as multiple disconnected blobs — SAM 2 struggles to lock
    on at initialisation. This function picks the most compact, single-blob slice.

    TODO:
    1. Create binary mask:  m = (vol_mask == organ_id)   →  shape (H, W, Z)
    2. Compute per-slice area:
           areas = m.reshape(-1, m.shape[2]).sum(axis=0)  →  shape (Z,)
    3. Find slices with area > 0:
           nonzero = np.where(areas > 0)[0]
       If none found: raise ValueError(f"Organ id {organ_id} not found in mask volume.")
    4. Get top-k slice indices by area:
           top_k = nonzero[np.argsort(areas[nonzero])[::-1][:k]]
    5. Define a scoring function for each candidate slice z:
       a. Count connected components:
              lbl, n_comp = ndimage.label(m[:, :, z])
       b. Estimate perimeter (pixels on the boundary):
              eroded    = ndimage.binary_erosion(m[:, :, z])
              perimeter = np.sum(m[:, :, z] ^ eroded)
       c. Compactness: areas[z] / (perimeter + 1e-6)
       d. Score tuple: (n_comp, -compactness)
          — sort ascending: fewer blobs wins, then more compact wins as tiebreaker
    6. Return int(min(top_k, key=score_function))

    Args:
        vol_mask : 3D integer label array (H, W, Z) — BTCV organ label map
        organ_id : integer label for the target organ (e.g. 6 = liver)
        k        : number of top-area slices to evaluate

    Returns:
        Integer z-index of the best starting slice
    """
    pass
