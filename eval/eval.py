from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from models.density import density_from_model

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))

@torch.no_grad()
def eval_nll(model, X: np.ndarray, y: np.ndarray, config, batch_size: int = 2048) -> float:
    device = next(model.parameters()).device
    dl = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32)),
                      torch.from_numpy(y.astype(np.float32))),
        batch_size=batch_size, shuffle=False
    )
    nll_sum, n_obs = 0.0, 0
    eps = 1e-12

    for xb, yb in dl:
        xb = xb.to(device)
        y_np = yb.numpy()

        y_of_u, f_y_batch, _, _, _, _ = density_from_model(model, xb, yb, config)
        y_grid = y_of_u.detach().cpu().numpy()     # (B, nn) lub (nn,)
        fy     = f_y_batch.detach().cpu().numpy()  # (B, nn) lub (nn,)

        # Ujednolicenie kształtów: (B, nn) dla fy; y_grid: (B, nn) lub (nn,)
        if fy.ndim == 1:
            # Jedno-elem. batch: zrób "batch" = 1
            fy = fy[None, :]
            y_np = y_np.reshape(1)
        # normalizacja gęstości
        area = np.trapz(fy, x=y_grid, axis=1) if np.ndim(y_grid) == 2 else np.trapz(fy, x=y_grid, axis=1)
        fy = fy / np.clip(area[:, None], eps, None)

        # Interpolacja
        if np.ndim(y_grid) == 1:
            # wspólna siatka dla wszystkich próbek
            for i in range(len(y_np)):
                fi = np.interp(y_np[i], y_grid, fy[i], left=fy[i, 0], right=fy[i, -1])
                nll_sum += -np.log(max(fi, eps))
        else:
            # siatka per-próbka
            for i in range(len(y_np)):
                fi = np.interp(y_np[i], y_grid[i], fy[i], left=fy[i, 0], right=fy[i, -1])
                nll_sum += -np.log(max(fi, eps))
        n_obs += len(y_np)

    return nll_sum / max(n_obs, 1)

@torch.no_grad()
def predict_mean_from_density(model, X: np.ndarray, y_like: np.ndarray, config,
                              batch_size: int = 2048) -> np.ndarray:
    """
    E[Y|x] liczona numerycznie z zrekonstruowanej gęstości (trapezy).
    """
    device = next(model.parameters()).device
    dl = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32)),
                      torch.from_numpy(y_like.astype(np.float32))),
        batch_size=batch_size, shuffle=False
    )
    outs = []
    for xb, yb in dl:
        y_of_u, f_y_batch, *_ = density_from_model(model, xb.to(device), yb.to(device), config)
        y_grid = y_of_u.detach().cpu().numpy()    # (B, nn)
        fy     = f_y_batch.detach().cpu().numpy() # (B, nn)
        area = np.trapz(fy, x=y_grid, axis=1)
        fy = fy / np.clip(area[:, None], 1e-12, None)
        Ey = np.trapz(y_grid * fy, x=y_grid, axis=1)
        outs.append(Ey)
    return np.concatenate(outs, axis=0)

def baseline_lr_metrics(X_tr, y_tr, X_te, y_te):
    """
    Baseline: regresja liniowa + Gaussian NLL (homoscedastyczny szum).
    """
    lr = LinearRegression().fit(X_tr, y_tr)
    mu_tr = lr.predict(X_tr)
    mu_te = lr.predict(X_te)

    sigma2 = float(np.var(y_tr - mu_tr, ddof=1))
    sigma2 = max(sigma2, 1e-12)

    nll_te = float(np.mean(0.5*np.log(2*np.pi*sigma2) + (y_te - mu_te)**2/(2*sigma2)))
    return {"rmse": rmse(y_te, mu_te), "mae": mae(y_te, mu_te), "nll": nll_te, "model": lr}

@torch.no_grad()
def pit_and_coverage(model, X: np.ndarray, y: np.ndarray, config,
                     qs=(0.1, 0.5, 0.9), batch_size: int = 2048):
    device = next(model.parameters()).device
    dl = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32)),
                      torch.from_numpy(y.astype(np.float32))),
        batch_size=batch_size, shuffle=False
    )
    us = []
    cover_counts = {float(t): 0 for t in qs}
    n_total = 0

    for xb, yb in dl:
        xb = xb.to(device); yb = yb.to(device)
        y_of_u, f_y_batch, u_grid, *_ = density_from_model(model, xb, yb, config)
        yg = y_of_u.detach().cpu().numpy()
        fy = f_y_batch.detach().cpu().numpy()

        if fy.ndim == 1:
            fy = fy[None, :]
        B, nn = fy.shape

        if np.ndim(yg) == 1:
            # wspólna siatka
            cdf = np.cumsum((fy[:, 1:] + fy[:, :-1]) * (yg[1:] - yg[:-1])[None, :] / 2.0, axis=1)
            cdf = np.hstack([np.zeros((B, 1)), cdf])
        else:
            # siatka per-próbka
            cdf = np.cumsum((fy[:, 1:] + fy[:, :-1]) * (yg[:, 1:] - yg[:, :-1]) / 2.0, axis=1)
            cdf = np.hstack([np.zeros((B, 1)), cdf])
        cdf = np.clip(cdf, 0.0, 1.0)

        y_np = yb.detach().cpu().numpy()
        # PIT
        if np.ndim(yg) == 1:
            for i in range(B):
                us.append(np.interp(y_np[i], yg, cdf[i], left=0.0, right=1.0))
        else:
            for i in range(B):
                us.append(np.interp(y_np[i], yg[i], cdf[i], left=0.0, right=1.0))

        # odwrócenie CDF
        taus = np.array(sorted(qs), dtype=float)
        if np.ndim(yg) == 1:
            for i in range(B):
                qi = np.interp(taus, cdf[i], yg, left=yg[0], right=yg[-1])
                for t, qv in zip(taus, qi):
                    cover_counts[float(t)] += float(y_np[i] <= qv)
        else:
            for i in range(B):
                qi = np.interp(taus, cdf[i], yg[i], left=yg[i, 0], right=yg[i, -1])
                for t, qv in zip(taus, qi):
                    cover_counts[float(t)] += float(y_np[i] <= qv)

        n_total += B

    u = np.asarray(us)
    u_sorted = np.sort(u)
    grid = (np.arange(1, len(u_sorted)+1) - 0.5) / len(u_sorted)
    ks = float(np.max(np.abs(u_sorted - grid)))
    coverage = {t: cover_counts[t] / n_total for t in cover_counts}
    return {
        "pit_mean": float(np.mean(u)),
        "pit_var":  float(np.var(u)),
        "pit_ks":   ks,
        "coverage": coverage,
        "u": u
    }
