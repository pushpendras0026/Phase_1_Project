from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def build_predictor(cfg, ckpt, device="cuda"):
    """Build a SAM 2 video predictor for bidirectional slice propagation.

    TODO:
    1. Call:  build_sam2_video_predictor(cfg, ckpt, device=device)
    2. Return the predictor object

    Model size -> (cfg, ckpt) mapping (values come from configs/default.yaml):
      tiny      : "sam2.1_hiera_t.yaml"   |  "checkpoints/sam2.1_hiera_tiny.pt"
      small     : "sam2.1_hiera_s.yaml"   |  "checkpoints/sam2.1_hiera_small.pt"
      base_plus : "sam2.1_hiera_b+.yaml"  |  "checkpoints/sam2.1_hiera_base_plus.pt"

    Args:
        cfg    : SAM 2 config name (yaml file name, not full path)
        ckpt   : path to the downloaded checkpoint .pt file
        device : "cuda" or "cpu"

    Returns:
        SAM2VideoPredictor instance
    """
    pass


def build_image_predictor(cfg, ckpt, device="cuda"):
    """Build a SAM 2 image predictor for single-slice fine-tuning (Week 4).

    TODO:
    1. model = build_sam2(cfg, ckpt, device=device)
    2. Return SAM2ImagePredictor(model)

    The image predictor is used in train.py — it exposes the image encoder,
    prompt encoder, and mask decoder as separate callable components.

    Args:
        cfg    : SAM 2 config name
        ckpt   : path to checkpoint .pt file
        device : "cuda" or "cpu"

    Returns:
        SAM2ImagePredictor instance
    """
    pass


def init_state(predictor, frames_dir):
    """Initialise the SAM 2 video predictor state from a folder of PNG frames.

    TODO:
    1. Call:
           predictor.init_state(
               video_path=frames_dir,
               offload_video_to_cpu=True,   # MANDATORY on 8 GB VRAM
               offload_state_to_cpu=True,   # MANDATORY on 8 GB VRAM
           )
    2. Return the inference state dict

    Both offload flags MUST be True on the RTX 4060 (8 GB).
    Without them, the 250-slice memory bank fills the GPU and crashes.

    Args:
        predictor  : SAM2VideoPredictor from build_predictor()
        frames_dir : path to folder containing 00000.png, 00001.png, ... slices

    Returns:
        inference_state dict — pass this to add_new_points_or_box and propagate_in_video
    """
    pass
