# [Week 4] — implement the TODO functions below and replace this stub.
# Template: week_4/src/train/train.py
"""train.py — AMP + gradient-accumulation LoRA fine-tuning loop.

forward_prompted uses the SAM2Base internal path:
    forward_image  ->  _prepare_backbone_features  ->
    sam_prompt_encoder  ->  sam_mask_decoder

This bypasses SAM2ImagePredictor.set_image() which runs under torch.no_grad()
— calling the encoder directly keeps LoRA gradients alive during training.
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from src.train.losses import total_loss

# SAM 2 uses ImageNet normalisation constants
_MEAN = torch.tensor([0.485, 0.456, 0.406])
_STD  = torch.tensor([0.229, 0.224, 0.225])


def _normalize(img_t):
    """Normalise a (B, 3, H, W) float [0, 1] tensor to ImageNet stats."""
    mean = _MEAN.to(img_t.device).view(1, 3, 1, 1)
    std  = _STD.to(img_t.device).view(1, 3, 1, 1)
    return (img_t - mean) / std


def forward_prompted(model, img_t_norm, box_batch, image_size):
    """Run one SAM 2 forward pass and return per-sample mask logits.

    TODO:
    1. Encode the image (LoRA adapters live here — must stay in grad context):
           backbone_out = model.forward_image(img_t_norm)
           _, vision_feats, _, feat_sizes = model._prepare_backbone_features(backbone_out)

    2. Reshape vision features from (H*W, B, C) → (B, C, H, W):
           B = img_t_norm.shape[0]
           feats = [
               f.permute(1, 2, 0).view(B, -1, *sz)
               for f, sz in zip(vision_feats[::-1], feat_sizes[::-1])
           ]
           feats = feats[::-1]           # restore finest-first order
           image_embed    = feats[-1]    # (B, C, h, w) — coarsest, main embed
           high_res_feats = feats[:-1]   # list of finer scale features

    3. Encode the bounding box prompt:
           box_t = box_batch.to(img_t_norm.device).view(B, 1, 4)   # (B, 1, 4)
           sparse_emb, dense_emb = model.sam_prompt_encoder(
               points=None, boxes=box_t, masks=None
           )

    4. Decode to get low-resolution mask logits:
           dec_kwargs = dict(
               image_embeddings=image_embed,
               image_pe=model.sam_prompt_encoder.get_dense_pe(),
               sparse_prompt_embeddings=sparse_emb,
               dense_prompt_embeddings=dense_emb,
               multimask_output=False,
               repeat_image=False,
           )
           if high_res_feats:
               dec_kwargs["high_res_features"] = high_res_feats
           dec_out = model.sam_mask_decoder(**dec_kwargs)
           low_res_masks = dec_out[0]   # (B, 1, h, w)

    5. Upsample to image_size × image_size and remove channel dim:
           return F.interpolate(
               low_res_masks, (image_size, image_size),
               mode="bilinear", align_corners=False,
           ).squeeze(1)                  # (B, H, W)

    NOTE: Refer to external/sam2/ source code for exact API signatures.

    Args:
        model       : SAM2Base (predictor.model) with LoRA injected
        img_t_norm  : (B, 3, H, W) float, ImageNet-normalised
        box_batch   : (B, 4) float, xyxy pixel coords at image_size scale
        image_size  : spatial size used for resizing (from cfg["model"]["image_size"])

    Returns:
        (B, H, W) logit tensor with gradient attached
    """
    pass


def lora_state_dict(model):
    """Extract only LoRA adapter weights (A / B matrices and peft lora_ keys).

    TODO:
    1. Iterate model.state_dict().items()
    2. Keep keys containing ".A", ".B", "lora_A", or "lora_B"
    3. Return the filtered dict

    This is what gets saved to checkpoints/lora_{organ}.pt after training.
    """
    pass


def run_training(cfg, model, dataset, organ_id, save_path):
    """AMP + gradient-accumulation training loop.

    Fix #3: accum_steps accumulates micro-batches before each optimiser step.
    Fix #6: bfloat16 avoids NaN/Inf that float16 triggers in ViT attention.
    Fix #7: zero_grad(set_to_none=True) removes tensors instead of zeroing them.

    TODO:
    1. Build DataLoader:
           loader = DataLoader(dataset, batch_size=cfg["train"]["micro_batch"],
                               shuffle=True, num_workers=0, pin_memory=False)

    2. Build optimiser on LoRA params only:
           trainable_params = [p for p in model.parameters() if p.requires_grad]
           opt = torch.optim.AdamW(trainable_params, lr=cfg["train"]["lr"])

    3. Build AMP scaler:
           scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"])

    4. opt.zero_grad(set_to_none=True)   ← Fix #7

    5. For each epoch in range(cfg["train"]["epochs"]):
       model.train()
       For each step, (img, gt, box) in enumerate(loader):
         a. Move to GPU:
                device = next(model.parameters()).device
                img_t  = _normalize(img.permute(0,3,1,2).to(device))  # (B,3,H,W)
                gt     = gt.to(device, dtype=torch.float32)
                box    = box.to(device)
         b. Forward with bfloat16 autocast:         ← Fix #6
                with torch.cuda.amp.autocast(enabled=cfg["train"]["amp"],
                                             dtype=torch.bfloat16):
                    logits = forward_prompted(model, img_t, box, cfg["model"]["image_size"])
                    loss   = total_loss(logits, gt) / cfg["train"]["accum_steps"]
                    # Divide BEFORE backward so gradient magnitude is correct
         c. scaler.scale(loss).backward()
         d. Every accum_steps micro-batches:         ← Fix #3
                if (step + 1) % cfg["train"]["accum_steps"] == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

    6. After the full training loop, flush any leftover gradient:
           if n_steps % accum_steps != 0:
               scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)

    7. Save LoRA adapter:
           Path(save_path).parent.mkdir(parents=True, exist_ok=True)
           torch.save(lora_state_dict(model), save_path)
           print(f"LoRA adapter saved -> {save_path}")

    8. Return the list of per-epoch average losses (for plotting / logging)

    Args:
        cfg       : config dict loaded from configs/default.yaml
        model     : SAM2Base with LoRA injected (predictor.model)
        dataset   : BTCVSliceDataset instance
        organ_id  : integer BTCV label (used for bookkeeping only)
        save_path : file path to save the LoRA adapter, e.g. "checkpoints/lora_liver.pt"

    Returns:
        List of float — mean loss per epoch
    """
    pass
