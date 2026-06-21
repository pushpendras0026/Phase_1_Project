# [Week 3] — implement the TODO functions below and replace this stub.
# Template: week_3/src/engine/propagate.py
import numpy as np
from PIL import Image


def propagate_bidirectional(predictor, state, start_idx, bbox, obj_id=1,
                             target_hw=None):
    """Propagate a bounding-box prompt forward then backward through all slices.

    This is the core of the project: the Z axis is treated as "time" and SAM 2's
    memory attention tracks the organ outward from the prompt slice in both directions.

    TODO:
    1. Reset any previous state:
           predictor.reset_state(state)

    2. Add the bounding box prompt at the chosen start slice:
           predictor.add_new_points_or_box(
               state,
               frame_idx=start_idx,
               obj_id=obj_id,
               box=bbox,          # np.float32 array [x0, y0, x1, y1]
           )

    3. Initialise a dict to collect per-slice masks:  masks = {}

    4. Propagate FORWARD (start_idx → last slice):
           for f_idx, _obj_ids, logits in predictor.propagate_in_video(
                   state, start_frame_idx=start_idx, reverse=False):
               pred = (logits[0] > 0.0).cpu().numpy().squeeze()  # (H, W) bool
               # If target_hw is given and pred.shape != target_hw, resize with PIL NEAREST
               masks[f_idx] = pred

    5. Propagate BACKWARD (start_idx → slice 0):
           Same call with reverse=True — SAM 2 handles the backward direction natively.
           Store results in the same masks dict (backward pass fills slices < start_idx).

    6. Stack into a 3D volume:
           Z = max(masks.keys()) + 1
           blank = np.zeros_like(masks[start_idx])
           return np.stack([masks.get(z, blank) for z in range(Z)], axis=-1)
           # Result shape: (H, W, Z)  dtype: bool

    Resize hint (step 4 & 5):
        SAM 2 outputs masks at image_size × image_size (e.g. 1024×1024).
        If target_hw is given, resize each pred back to original volume resolution:
            pil = Image.fromarray(pred.astype(np.uint8)).resize(
                      (target_hw[1], target_hw[0]), Image.NEAREST)
            pred = np.array(pil).astype(bool)

    Args:
        predictor  : SAM2VideoPredictor from build_predictor()
        state      : inference state from init_state()
        start_idx  : axial slice index where the box prompt is placed
        bbox       : float32 array [x0, y0, x1, y1] — bounding box in PNG frame coords
        obj_id     : object ID integer (use 1 for single-organ segmentation)
        target_hw  : optional (H, W) to resize masks back to original volume resolution

    Returns:
        bool numpy array of shape (H, W, Z) — True where the organ is predicted
    """
    pass
