import torch
from torch import Tensor
import math
import numpy as np
from qkal.config import QKALReconstructionConfig

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
def density_from_qkal(
    model,
    x_batch,
    y_data,
    config: QKALReconstructionConfig
):
    device, dtype = _model_device_dtype(model)
    x_batch = _to_torch(x_batch, device, dtype)
    y_data  = _to_torch(y_data,  device, dtype)

    rho_u_batch, u_grid = model.predict_density_u(x_batch, nn_grid=config.grid_size, mode=config.calibration_mode)
    u_grid = u_grid.to(device=device, dtype=dtype)
    rho_u_batch = rho_u_batch.to(device=device, dtype=dtype)

    y_min, y_max = y_data.min(), y_data.max()
    pad = 0.05 * (y_max - y_min + 1e-12)
    y_grid = torch.linspace(y_min - pad, y_max + pad, config.grid_size, device=device, dtype=dtype)
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
