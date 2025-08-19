import torch
import torch.nn as nn
import torch.nn.functional as F
from qkal.legendre import legendre_targets_from_y
from qkal.config import QKALReconstructionConfig
class QKALLoss(nn.Module):
    """
    MSE between predicted Legendre coefficients and the Legendre targets
    computed from u in [0, 1].

    Args:
        degree (int): Highest Legendre degree. Expects preds shape (B, degree+1).
        reduction (str): 'mean' | 'sum' | 'none'
    """
    def __init__(self, config: QKALReconstructionConfig):
        super().__init__()

        self.config = config
        self.degree = self.config.degree
        self.reduction = self.config.reduction

        if self.reduction not in ("mean", "sum", "none"):
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")

    def forward(self, preds: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: (B, K) predicted coefficients, where K == degree + 1
            u:     (B,)   targets in [0,1]
        """
        # Build targets (B, K) from u
        targets = legendre_targets_from_y(u, self.degree)

        # Ensure device/dtype match
        targets = targets.to(device=preds.device, dtype=preds.dtype)

        # Shape sanity check
        if preds.ndim != 2:
            raise ValueError(f"preds must be 2D (B,K), got {preds.shape}")

        if preds.size(1) != self.degree + 1:
            raise ValueError(f"preds second dim must be degree+1={self.degree + 1}, got {preds.size(1)}")

        return F.mse_loss(preds, targets, reduction=self.reduction)
