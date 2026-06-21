# [Week 4] — implement the TODO functions below and replace this stub.
# Template: week_4/src/train/lora.py
import torch
import torch.nn as nn


# ── Option A: peft on the fused qkv projection (easy path — start here) ─────

def add_lora_peft(image_encoder, r=8, alpha=16):
    """Inject LoRA adapters into the image encoder using the peft library.

    This is the EASY PATH — use this first to get a working training loop.
    It adapts Q, K, and V together via the fused qkv linear layer.

    TODO:
    1. from peft import LoraConfig, get_peft_model
    2. config = LoraConfig(
           r=r,
           lora_alpha=alpha,
           target_modules=["qkv"],
           lora_dropout=0.05,
           bias="none",
       )
    3. return get_peft_model(image_encoder, config)

    After calling this, run trainable_report() — trainable % should be well under 1%.

    Args:
        image_encoder : SAM 2 Hiera ViT image encoder (predictor.model.image_encoder)
        r             : LoRA rank (default 8)
        alpha         : LoRA alpha — effective scale = alpha / r (default 16 → scale 2)

    Returns:
        image_encoder wrapped with LoRA adapters (in-place and returned)
    """
    pass


# ── Option B: custom Q & V only — no peft dependency (stretch goal) ──────────

class LoRALinear(nn.Module):
    """Low-rank adapter that targets only Q and V, leaving K unchanged.

    The Hiera attention block uses a single fused nn.Linear for QKV with
    out_features = 3 × in_features (layout: [Q | K | V] along output dim).
    This wrapper adds a low-rank correction to Q and V only by zeroing the K slice.

    TODO — implement __init__:
    1. super().__init__()
    2. self.base = base
    3. Freeze all base parameters:
           for p in base.parameters(): p.requires_grad = False
    4. self.r = r
    5. self.scale = alpha / r
    6. self.qv_only = qv_only
    7. self.dim = base.in_features      # size of Q, K, or V each
    8. self.A = nn.Parameter(torch.empty(r, base.in_features))
    9. self.B = nn.Parameter(torch.zeros(base.out_features, r))
       # B is zero → LoRA starts as identity (no change to output at step 0)
    10. nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    TODO — implement forward:
    1. Compute delta:  delta = (x @ self.A.t()) @ self.B.t() * self.scale
    2. If self.qv_only: zero the K slice (middle third of fused output):
           delta = delta.clone()
           delta[..., self.dim : 2 * self.dim] = 0.0
    3. Return self.base(x) + delta
    """

    def __init__(self, base: nn.Linear, r=8, alpha=16, qv_only=True):
        super().__init__()
        # TODO: implement as described above
        pass

    def forward(self, x):
        # TODO: implement as described above
        pass


def inject_lora_qv(model, r=8, alpha=16):
    """Replace every attention QKV linear with a LoRALinear (Q & V only).

    TODO:
    1. for _name, mod in model.named_modules():
    2.     if hasattr(mod, "attn") and hasattr(mod.attn, "qkv"):
               mod.attn.qkv = LoRALinear(mod.attn.qkv, r, alpha, qv_only=True)
           elif hasattr(mod, "qkv") and isinstance(mod.qkv, nn.Linear):
               mod.qkv = LoRALinear(mod.qkv, r, alpha, qv_only=True)
    3. return model

    Args:
        model : SAM 2 image encoder (Hiera ViT)
        r     : LoRA rank
        alpha : LoRA alpha

    Returns:
        model with LoRALinear replacing all qkv layers (modified in-place)
    """
    pass


def trainable_report(model):
    """Print the number and fraction of trainable parameters.

    TODO:
    1. trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    2. total     = sum(p.numel() for p in model.parameters())
    3. print(f"Trainable: {trainable:,} / {total:,} = {100 * trainable / total:.3f}%")

    Expected output after LoRA injection: trainable well under 1% of total.
    """
    pass
