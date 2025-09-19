from torch import Tensor
import math
import numpy as np
from utils.config import ReconstructionConfig, CalibrationMode
from utils.legendre import compute_scaled_legendre_polynomials
import torch, torch.nn.functional as F

EPS = 1e-12

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

def _kde_bandwidth(y_data: Tensor, rule: str, mult: float) -> Tensor:
    N = y_data.numel()
    std = y_data.std().clamp_min(1e-12)
    if rule == "silverman":
        base = 1.06 * std
    elif rule == "scott":
        base = std
    else:
        q75 = torch.quantile(y_data, 0.75)
        q25 = torch.quantile(y_data, 0.25)
        iqr = (q75 - q25).clamp_min(1e-12)
        robust_sigma = torch.minimum(std, iqr / 1.34)
        base = robust_sigma
    bw = base * (N ** (-1/5)) * float(mult)
    return torch.clamp(bw, min=1e-6)

def kde_pdf(y_query: Tensor, y_data: Tensor, bandwidth: float | Tensor = None,
            rule: str = "silverman", mult: float = 1.0) -> Tensor:
    yq = y_query.view(-1, 1)
    yd = y_data.view(1, -1)
    if bandwidth is None:
        bandwidth = _kde_bandwidth(y_data.view(-1), rule=rule, mult=mult)
    else:
        bandwidth = torch.as_tensor(bandwidth, device=y_data.device, dtype=y_data.dtype)
        bandwidth = torch.clamp(bandwidth, min=1e-6)
    u = (yq - yd) / (bandwidth + EPS)
    kern = torch.exp(-0.5 * u**2) / (math.sqrt(2*math.pi) * (bandwidth + EPS))
    return kern.mean(dim=1).view(-1)

@torch.no_grad()
def density_from_model(
    model,
    x_batch,
    y_data_raw,
    config: ReconstructionConfig,
    *,
    qt=None,
    space: str = "raw",
):
    device, dtype = _model_device_dtype(model)
    x_batch = _to_torch(x_batch, device, dtype)
    y_data_raw = _to_torch(y_data_raw, device, dtype)

    rho_u_batch, u_grid = predict_density_u(
        model, x_batch, nn_grid=config.grid_size,
        mode=config.calibration_mode, degree=config.degree
    )
    u_grid = u_grid.to(device=device, dtype=dtype)
    rho_u_batch = rho_u_batch.to(device=device, dtype=dtype)

    # u -> y
    if qt is not None:
        u_np = u_grid.detach().cpu().numpy().reshape(-1, 1)
        y_np = qt.inverse_transform(u_np).ravel()
        y_of_u = torch.from_numpy(y_np).to(device=device, dtype=dtype)
    else:
        y_min, y_max = y_data_raw.min(), y_data_raw.max()
        y_of_u = y_min + (y_max - y_min) * u_grid

    if space == "raw":
        s_of_u = y_of_u
        s_data = y_data_raw
        dy_du = torch.gradient(y_of_u, spacing=(u_grid,))[0].abs().clamp_min(1e-12)
        J = dy_du
    elif space == "log1p":
        s_of_u = torch.log1p(y_of_u.clamp_min(-1.0 + 1e-9))
        s_data = torch.log1p(y_data_raw.clamp_min(-1.0 + 1e-9))
        dy_du = torch.gradient(y_of_u, spacing=(u_grid,))[0].abs().clamp_min(1e-12)
        dz_dy = 1.0 / (1.0 + y_of_u.clamp_min(-1.0 + 1e-9))
        J = (dy_du * dz_dy).abs().clamp_min(1e-12)
    else:
        raise ValueError("space must be 'raw' or 'log1p'")

    p_marg_s = kde_pdf(
        s_of_u, s_data,
        bandwidth=config.kde_bandwidth,
        rule=config.kde_bw_rule,
        mult=config.kde_bw_mult
    )

    f_s_tilde = rho_u_batch * p_marg_s.unsqueeze(0)

    if config.normalize_density:
        du = 1.0 / config.grid_size
        Z = torch.sum(f_s_tilde * J.unsqueeze(0), dim=1) * du
        f_s_batch = f_s_tilde / (Z.unsqueeze(1) + EPS)
    else:
        f_s_batch = f_s_tilde

    return s_of_u, f_s_batch, u_grid, rho_u_batch, J, p_marg_s

@torch.no_grad()
def predict_density_u(model, x: torch.Tensor, nn_grid: int, mode: CalibrationMode, degree: int):
    m = model.forward(x)
    device, dtype = m.device, m.dtype

    u_grid = torch.arange(1, nn_grid + 1, device=device, dtype=dtype)
    u_grid = (u_grid - 0.5) / nn_grid

    P = compute_scaled_legendre_polynomials(u_grid[None, :], degree).squeeze(0)  # (nn, K)
    rho_raw = torch.matmul(m, P.t()).clamp(min=-30.0, max=30.0)

    a = getattr(model, "cal_a", None)
    b = getattr(model, "cal_b", None)
    c = getattr(model, "cal_c", None)
    a = a.to(dtype) if isinstance(a, torch.Tensor) else torch.tensor(1.0, device=device, dtype=dtype)
    b = b.to(dtype) if isinstance(b, torch.Tensor) else torch.tensor(1e-3, device=device, dtype=dtype)
    c = c.to(dtype) if isinstance(c, torch.Tensor) else torch.tensor(1.0, device=device, dtype=dtype)

    if mode == CalibrationMode.CLAMP:
        rho_pos = torch.clamp(rho_raw, min=1e-3)
    elif mode == CalibrationMode.CALIBRATED_SOFTPLUS:
        rho_pos = a * torch.logaddexp(torch.log(b + EPS), c * rho_raw)
    else:
        rho_pos = F.softplus(rho_raw)

    rho_pos = torch.nan_to_num(rho_pos, nan=0.0, posinf=1e6, neginf=0.0)
    denom = rho_pos.mean(dim=1, keepdim=True) + EPS
    rho_u = rho_pos / denom
    rho_u = torch.nan_to_num(rho_u, nan=1.0, posinf=1.0, neginf=0.0)

    return rho_u, u_grid
