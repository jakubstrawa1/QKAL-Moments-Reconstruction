import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.legendre import compute_scaled_legendre_polynomials


class QKAL_Layer(nn.Module):
    """
    Same QKAL Layer as in the original paper
    https://arxiv.org/html/2507.13393v1
    """
    def __init__(self, in_f: int, out_f: int, degree: int):
        super().__init__()
        self.degree = degree
        self.poly_w = nn.Parameter(
            torch.empty(out_f, in_f * (degree + 1))
        )
        nn.init.kaiming_uniform_(self.poly_w, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the QKAL_Layer.

        This method computes the elementwise Gaussian CDF transform of the
        input, evaluates scaled Legendre polynomials up to the specified
        degree, flattens the polynomial basis features, and applies a learned
        linear mapping to produce output features.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_f).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_f).
        """
        u = 0.5 * (1 + torch.erf(x / math.sqrt(2)))
        P = compute_scaled_legendre_polynomials(u, self.degree)
        P_flat = P.view(x.size(0), -1)
        return F.linear(P_flat, self.poly_w)