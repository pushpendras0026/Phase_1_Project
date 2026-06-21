# [Week 4] — implement the TODO functions below and replace this stub.
# Template: week_4/src/train/losses.py
import torch
import torch.nn.functional as F


def dice_loss(logits, targets, eps=1e-6):
    """Soft Dice loss between predicted logits and binary targets.

    Dice loss penalises poor spatial overlap — more informative than BCE alone
    for segmentation tasks where the organ is a small fraction of the image.

    TODO:
    1. Convert logits to probabilities:  p = torch.sigmoid(logits)
    2. Intersection:  inter = (p * targets).sum()
    3. Union:         union = p.sum() + targets.sum()
    4. Dice coefficient:  (2 * inter + eps) / (union + eps)
    5. Dice LOSS = 1 - dice_coefficient
       (0 = perfect overlap, 1 = completely wrong)
    6. Return the scalar loss tensor

    Args:
        logits  : raw model output tensor (any shape), NOT sigmoid-activated
        targets : binary ground-truth tensor, same shape as logits (float32, 0 or 1)
        eps     : smoothing constant to avoid division by zero

    Returns:
        Scalar Dice loss tensor (with gradient)
    """
    pass


def total_loss(logits, targets):
    """Combined Dice + Binary Cross-Entropy loss.

    TODO:
    1. loss_dice = dice_loss(logits, targets)
    2. loss_bce  = F.binary_cross_entropy_with_logits(logits, targets)
    3. return loss_dice + loss_bce

    Dice   — optimises global spatial overlap
    BCE    — optimises pixel-level classification probability
    Together they are more stable and less sensitive to class imbalance.

    Args:
        logits  : raw model output (not sigmoid-activated)
        targets : binary float32 ground truth, same shape as logits

    Returns:
        Scalar combined loss tensor (with gradient)
    """
    pass
