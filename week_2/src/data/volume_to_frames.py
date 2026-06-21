import os
import imageio.v2 as imageio

from src.data.nifti_io import load_volume, apply_hu_window, to_rgb


def write_frames_png(case_path, out_dir):
    """Save every axial slice of a CT volume as a lossless PNG file.

    TODO:
    1. Load the volume:   vol, _, _ = load_volume(case_path)
    2. Apply HU window:   vol_u8 = apply_hu_window(vol)   →  (H, W, Z) uint8
    3. Create output dir: os.makedirs(out_dir, exist_ok=True)
    4. Loop z in range(vol_u8.shape[2]):
       a. Get one slice:    vol_u8[:, :, z]             →  shape (H, W)
       b. Convert to RGB:   to_rgb(slice)               →  shape (H, W, 3)
       c. Save as PNG:      imageio.imwrite(f"{out_dir}/{z:05d}.png", rgb_slice)
          - Filename MUST be 5-digit zero-padded: 00000.png, 00001.png, ...
          - Use PNG not JPEG — JPEG compression damages thin organ edges
    5. Return vol_u8.shape[2]  (number of slices written)

    NOTE: Place data/processed/ on an SSD.
    30 volumes × ~250 slices × 1024² PNG is several GB of data.
    An HDD makes the data loader slower than the GPU.

    Args:
        case_path : path to the .nii or .nii.gz image file
        out_dir   : folder to write PNG frames into, e.g. data/processed/img0001/

    Returns:
        Number of PNG files written (= number of axial slices)
    """
    pass


def frames_to_arrays(case_path):
    """Return all axial slices as a list of in-memory (H, W, 3) uint8 arrays.

    TODO:
    1. Load and window the volume (same as write_frames_png steps 1-2)
    2. For each z: convert slice to RGB with to_rgb() and append to a list
    3. Return the list

    Use this instead of write_frames_png when disk space is tight.
    No files are written — frames live entirely in RAM.

    Args:
        case_path: path to the .nii or .nii.gz image file

    Returns:
        List of (H, W, 3) uint8 arrays, one per axial slice
    """
    pass
