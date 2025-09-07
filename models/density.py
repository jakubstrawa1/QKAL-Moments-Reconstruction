from torch import Tensor
import math
import numpy as np
from utils.config import ReconstructionConfig, CalibrationMode
from utils.legendre import compute_scaled_legendre_polynomials
import torch, torch.nn.functional as F


def _model_device_dtype(model):
    p = next(model.parameters(), None)
    if p is None:
        return torch.device("cpu"), torch.float32
    return p.device, p.dtype

def _to_torch(arr, device, dtype):
    if isinstance(arr, np.ndarray):
        t = torch.from_numpy(arr)
    elif isinstance(arr, Tensor):
        t = arr
    else:
        t = torch.as_tensor(arr)
    return t.to(device=device, dtype=dtype)

def empirical_cdf(y_data: Tensor, y_grid: Tensor) -> Tensor:
    y_data = y_data.view(1, -1)
    y_grid = y_grid.view(-1, 1)
    return (y_data <= y_grid).float().mean(dim=1)

def inverse_cdf(u: Tensor, y_grid: Tensor, cdf_vals: Tensor) -> Tensor:
    u = u.clamp(0, 1)
    idx = torch.searchsorted(cdf_vals, u, right=True)
    idx = idx.clamp(1, len(y_grid)-1)
    x0 = cdf_vals[idx-1]; x1 = cdf_vals[idx]
    y0 = y_grid[idx-1];   y1 = y_grid[idx]
    w = (u - x0) / (x1 - x0 + 1e-12)
    return y0 + w * (y1 - y0)

def kde_pdf(y_query: Tensor, y_data: Tensor, bandwidth: float = None) -> Tensor:
    yq = y_query.view(-1, 1)      # [M,1]
    yd = y_data.view(1, -1)       # [1,N]
    N = yd.shape[1]
    if bandwidth is None:
        std = yd.std().clamp_min(1e-12)
        bandwidth = 1.06 * std * (N ** (-1/5))
    u = (yq - yd) / (bandwidth + 1e-12)
    kern = torch.exp(-0.5 * u**2) / (math.sqrt(2*math.pi) * (bandwidth + 1e-12))
    return kern.mean(dim=1).view(-1)

@torch.no_grad()
def density_from_model(
    model,
    x_batch,
    y_data,
    config: ReconstructionConfig
):
    device, dtype = _model_device_dtype(model)
    x_batch = _to_torch(x_batch, device, dtype)
    y_data  = _to_torch(y_data,  device, dtype)

    rho_u_batch, u_grid = predict_density_u(
        model,
        x_batch,
        nn_grid=config.grid_size,
        mode=config.calibration_mode,
        degree=config.degree)

    u_grid = u_grid.to(device=device, dtype=dtype)
    rho_u_batch = rho_u_batch.to(device=device, dtype=dtype)

    y_min, y_max = y_data.min(), y_data.max()
    pad = 0.05 * (y_max - y_min + 1e-12)

    y_grid = torch.linspace(
        y_min - pad,
        y_max + pad,
        config.grid_size,
        device=device,
        dtype=dtype)

    cdf_vals = empirical_cdf(y_data, y_grid)
    y_of_u = inverse_cdf(u_grid, y_grid, cdf_vals)
    p_marg_y = kde_pdf(y_of_u, y_data, bandwidth=config.kde_bandwidth)

    du = 1.0 / config.grid_size
    dy_du = torch.gradient(y_of_u, spacing=(u_grid,))[0].abs().clamp_min(1e-12)

    f_y_tilde = rho_u_batch * p_marg_y.unsqueeze(0)

    if config.normalize_density:
        Z = torch.sum(f_y_tilde * dy_du.unsqueeze(0), dim=1) * du
        f_y_batch = f_y_tilde / (Z.unsqueeze(1) + 1e-12)
    else:
        f_y_batch = f_y_tilde

    return y_of_u, f_y_batch, u_grid, rho_u_batch, dy_du, p_marg_y


@torch.no_grad()
def predict_density_u(model, x: torch.Tensor, nn_grid: int, mode: CalibrationMode, degree: int):
    """
    Reconstruct rho(u|x) on u_k grid (u_k) = (k-0.5)/nn_grid in CDF space.
    """
    m = model.forward(x)
    device, dtype = m.device, m.dtype

    u_grid = torch.arange(1, nn_grid + 1, device=device, dtype=dtype)
    u_grid = (u_grid - 0.5) / nn_grid

    P = compute_scaled_legendre_polynomials(u_grid[None, :], degree).squeeze(0)

    rho_raw = torch.matmul(m, P.t())
    rho_raw = rho_raw.clamp(min=-30.0, max=30.0)

    a = getattr(model, "cal_a", None)
    b = getattr(model, "cal_b", None)
    c = getattr(model, "cal_c", None)
    a = a.to(dtype) if isinstance(a, torch.Tensor) else torch.tensor(1.0, device=device, dtype=dtype)
    b = b.to(dtype) if isinstance(b, torch.Tensor) else torch.tensor(1e-3, device=device, dtype=dtype)
    c = c.to(dtype) if isinstance(c, torch.Tensor) else torch.tensor(1.0, device=device, dtype=dtype)

    if mode == CalibrationMode.CLAMP:
        rho_pos = torch.clamp(rho_raw, min=1e-3)
    elif mode == CalibrationMode.CALIBRATED_SOFTPLUS:

        rho_pos = a * torch.logaddexp(torch.log(b + 1e-12), c * rho_raw)
    else:
        rho_pos = F.softplus(rho_raw)

    rho_pos = torch.nan_to_num(rho_pos, nan=0.0, posinf=1e6, neginf=0.0)

    denom = rho_pos.mean(dim=1, keepdim=True) + 1e-12
    rho_u = rho_pos / denom
    rho_u = torch.nan_to_num(rho_u, nan=1.0, posinf=1.0, neginf=0.0)

    return rho_u, u_grid
