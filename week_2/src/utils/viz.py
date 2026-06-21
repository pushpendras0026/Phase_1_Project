import numpy as np
import imageio.v2 as imageio


def save_overlay_gif(vol_u8, pred_mask3d, out_path, gt_mask3d=None, duration=0.08):
    """Save an animated GIF that scrolls through all axial slices with mask overlay.

    Colour convention:
      Green  = ground-truth mask  (gt_mask3d)
      Red    = predicted mask     (pred_mask3d)
      Where both overlap, red is drawn on top.

    TODO:
    1. Build a list of RGB frames — one per axial slice:
       For z in range(vol_u8.shape[2]):
         a. Make a 3-channel copy of the greyscale slice:
                rgb = np.repeat(vol_u8[:, :, z:z+1], 3, axis=2).copy()
                # shape (H, W, 3) uint8
         b. If gt_mask3d is given, blend green into GT pixels:
                gt = gt_mask3d[:, :, z].astype(bool)
                rgb[gt] = (rgb[gt] * 0.5 + np.array([0, 200, 0]) * 0.5).astype(np.uint8)
         c. Blend red into predicted pixels (drawn on top):
                pred = pred_mask3d[:, :, z].astype(bool)
                rgb[pred] = (rgb[pred] * 0.5 + np.array([220, 60, 60]) * 0.5).astype(np.uint8)
         d. Append rgb to the frames list
    2. Save:  imageio.mimsave(out_path, frames, duration=duration, loop=0)

    Args:
        vol_u8      : (H, W, Z) uint8 greyscale CT volume
        pred_mask3d : (H, W, Z) bool  predicted organ mask — shown in red
        out_path    : output .gif file path
        gt_mask3d   : (H, W, Z) bool  ground-truth mask — shown in green (optional)
        duration    : seconds per frame in the GIF (default 0.08 ≈ 12 fps)
    """
    pass
