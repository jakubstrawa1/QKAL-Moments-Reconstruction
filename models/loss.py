import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class SimpleCoefficientLoss(nn.Module):
    """
    Generic squared error loss between predicted coefficients and
    basis-expanded targets.

    Args:
        degree (int): highest degree/order (preds expected shape (B, degree+1))
        basis_fn (Callable): maps (y, degree) -> (B, degree+1) tensor
        reduction (str): 'mean' | 'sum' | 'none'
    """

    def __init__(self, config, basis_fn: Callable):
        super().__init__()
        self.degree = config.degree
        self.basis_fn = basis_fn

    def forward(self, preds: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        targets = self.basis_fn(y, self.degree).to(device=preds.device, dtype=preds.dtype)

        if preds.ndim != 2:
            raise ValueError(f"preds must be 2D (B,K), got {preds.shape}")
        if preds.size(1) != self.degree + 1:
            raise ValueError(f"preds second dim must be {self.degree+1}, got {preds.size(1)}")

        return F.mse_loss(preds, targets)