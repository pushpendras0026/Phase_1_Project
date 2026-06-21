import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb
from src.data.prompts import bbox_from_mask


class BTCVSliceDataset(Dataset):
    """PyTorch Dataset — yields one 2D axial slice per item for LoRA fine-tuning.

    Each item is a tuple: (rgb_slice, gt_mask, bbox)
      rgb_slice : float32 numpy array  (H, W, 3)  values in [0, 1]
      gt_mask   : uint8  numpy array   (H, W)     binary — 1 where organ is present
      bbox      : float32 numpy array  (4,)        [x0, y0, x1, y1] in pixel coords

    Only slices that contain the target organ are included.
    Images and masks are resized to image_size × image_size.
    """

    def __init__(self, cases, organ_id, image_dir, label_dir, image_size=1024):
        """Build the list of all valid (image_path, label_path, slice_z) samples.

        TODO:
        1. Store self.organ_id = organ_id  and  self.image_size = image_size
        2. Build self.samples = []
        3. For each case name in cases:
           a. Construct paths:
                  img_path = Path(image_dir) / f"{case}.nii"
                  lbl_path = Path(label_dir) / f"{case.replace('img', 'label')}.nii"
           b. Skip if either file does not exist
           c. Load labels only (to find which slices contain the organ):
                  lbl_vol, _, _ = load_volume(str(lbl_path))
           d. For each z in range(lbl_vol.shape[2]):
                  if (lbl_vol[:, :, z] == organ_id).any():
                      self.samples.append((str(img_path), str(lbl_path), z))

        Args:
            cases      : list of case name strings, e.g. ["img0001", "img0002"]
            organ_id   : BTCV integer label for target organ (e.g. 6 = liver)
            image_dir  : path to folder containing image .nii files
            label_dir  : path to folder containing label .nii files
            image_size : resize images/masks to this square size (use 512 if 1024 OOMs)
        """
        self.organ_id = organ_id
        self.image_size = image_size
        self.samples = []
        # TODO: populate self.samples as described above
        pass

    def __len__(self):
        """Return the total number of valid slices across all cases."""
        return len(self.samples)

    def __getitem__(self, i):
        """Load and return one (rgb_slice, gt_mask, bbox) tuple.

        TODO:
        1. Unpack:   img_path, lbl_path, z = self.samples[i]
        2. Load volume and labels:
               vol, _, _ = load_volume(img_path)
               lbl, _, _ = load_volume(lbl_path)
        3. Extract and window the image slice:
               slice_u8 = apply_hu_window(vol[:, :, z])   →  (H, W) uint8
               rgb = to_rgb(slice_u8)                      →  (H, W, 3) uint8
        4. Extract the binary ground-truth mask:
               gt = (lbl[:, :, z] == self.organ_id).astype(np.uint8)   →  (H, W)
        5. Resize both to image_size × image_size using PIL:
               s = self.image_size
               rgb_r = np.array(Image.fromarray(rgb).resize((s, s), Image.BILINEAR))
               gt_r  = np.array(Image.fromarray(gt).resize((s, s),  Image.NEAREST))
        6. Compute bounding box on the RESIZED mask:
               bbox = bbox_from_mask(gt_r, pad=4)
               If bbox is None (can happen after resize):
                   bbox = np.array([0., 0., float(s-1), float(s-1)], dtype=np.float32)
        7. Convert image to float [0, 1]:
               rgb_f = rgb_r.astype(np.float32) / 255.0
        8. Return (rgb_f, gt_r, bbox)

        Returns:
            rgb_f : float32 array (image_size, image_size, 3) in [0, 1]
            gt_r  : uint8  array (image_size, image_size)    binary mask
            bbox  : float32 array (4,)                       [x0, y0, x1, y1]
        """
        pass
