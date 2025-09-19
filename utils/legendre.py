import torch
import math

def compute_scaled_legendre_polynomials(x: torch.Tensor, order: int) -> torch.Tensor:
    """
   P_k(2x-1) dla k=0..order.
    Skalowanie: sqrt(2k+1).
    Wejście:
      x: (..., N) w [0,1]
    Wyjście:
      (..., N, order+1)
    """
    t = 2.0 * x - 1.0
    P0 = torch.ones_like(t)
    if order == 0:
        return P0[..., None]
    P1 = t
    P = [P0, P1]
    for n in range(1, order):
        P_next = ((2.0 * n + 1.0) * t * P[-1] - n * P[-2]) / (n + 1.0)
        P.append(P_next)
    P = torch.stack(P, dim=-1)  # (..., N, order+1)
    degrees = torch.arange(order + 1, device=x.device, dtype=x.dtype)
    scale = torch.sqrt(2.0 * degrees + 1.0)
    return P * scale

def legendre_targets_from_y(u_y: torch.Tensor, degree: int):
    """
    u_y: (B,) w [0,1]
    """
    P = compute_scaled_legendre_polynomials(u_y.unsqueeze(1), degree)  # (B,1,K)
    return P.squeeze(1)  # (B,K)

def legendre_targets_from_y_minmax(
    y: torch.Tensor,
    degree: int,
    y_min: torch.Tensor,
    y_max: torch.Tensor,
    gaussianize: bool = True,
) -> torch.Tensor:
    u = (y.view(-1) - y_min) / (y_max - y_min + 1e-12)
    u = u.clamp(0.0, 1.0)
    if gaussianize:
        u = 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0)))
    return compute_scaled_legendre_polynomials(u, degree)  # (B, K)

def fit_y_minmax_from_tensor(y_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    y_min = torch.min(y_train)
    y_max = torch.max(y_train)
    y_max = torch.maximum(y_max, y_min + torch.tensor(1e-8, device=y_train.device))
    return y_min, y_max
