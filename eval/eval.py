# eval/eval.py
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from models.density import density_from_model

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))

def _transform_y(y: np.ndarray, space: str) -> np.ndarray:
    if space == "raw":
        return y.astype(np.float32)
    elif space == "log1p":
        return np.log1p(y).astype(np.float32)
    else:
        raise ValueError(f"Unknown space='{space}', use 'raw' or 'log1p'.")

def _inv_on_mean(z_grid: np.ndarray, fy: np.ndarray, space: str) -> np.ndarray:
    """
    Liczy E[Y|x] z f w wybranej przestrzeni:
    - raw:   ∫ y f_Y(y|x) dy
    - log1p: ∫ (exp(z)-1) f_Z(z|x) dz
    """
    if space == "raw":
        Ey = np.trapezoid(z_grid * fy, x=z_grid, axis=1)
    elif space == "log1p":
        Ey = np.trapezoid((np.expm1(z_grid)) * fy, x=z_grid, axis=1)
    else:
        raise ValueError
    return Ey

@torch.no_grad()
def eval_nll(model, X: np.ndarray, y: np.ndarray, config,
             batch_size: int = 2048, space: str = "raw",
             y_ref_for_space: np.ndarray | None = None) -> float:
    """
    NLL liczone NA TEŚCIE (X,y), ale rekonstrukcja gęstości używa
    referencyjnego rozkładu Y (train/val) przekazanego jako y_ref_for_space.
    """
    device = next(model.parameters()).device
    y_eval = _transform_y(y, space)

    # Referencyjne Y do ECDF/KDE (train/val) w tej samej przestrzeni co 'space'
    if y_ref_for_space is None:
        y_ref_for_space = y_eval
    y_ref_t = torch.from_numpy(y_ref_for_space.astype(np.float32)).to(device)

    dl = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32)),
                      torch.from_numpy(y_eval.astype(np.float32))),
        batch_size=batch_size, shuffle=False
    )

    nll_sum, n_obs = 0.0, 0
    eps = 1e-12

    for xb, yb in dl:
        xb = xb.to(device)

        # Gęstość na siatce zbudowanej z y_ref_for_space
        y_of_u, f_y_batch, *_ = density_from_model(model, xb, y_ref_t, config)

        y_grid = y_of_u.detach().cpu().numpy()    # (nn,)
        fy     = f_y_batch.detach().cpu().numpy() # (B, nn)
        if fy.ndim == 1:
            fy = fy[None, :]
            y_np = yb.cpu().numpy().reshape(1)
        else:
            y_np = yb.cpu().numpy()

        # dopięcie normalizacji (ostrożnościowo)
        area = np.trapezoid(fy, x=y_grid, axis=1)
        fy = fy / np.clip(area[:, None], eps, None)

        # interpolacja gęstości w punktach obserwacji
        for i in range(len(y_np)):
            fi = np.interp(y_np[i], y_grid, fy[i], left=fy[i, 0], right=fy[i, -1])
            nll_sum += -np.log(max(fi, eps))
        n_obs += len(y_np)

    return nll_sum / max(n_obs, 1)

@torch.no_grad()
def predict_mean_from_density(model, X: np.ndarray, y_like: np.ndarray, config,
                              batch_size: int = 2048, space: str = "raw",
                              y_ref_for_space: np.ndarray | None = None) -> np.ndarray:
    """
    Zwraca E[Y|x]. Jeśli space='log1p', liczymy gęstość w z=log(1+y),
    a następnie wracamy na skalę raw: ∫ (exp(z)-1) f_Z(z|x) dz.
    Siatka oparta o referencyjne Y (train/val) przekazane jako y_ref_for_space.
    """
    device = next(model.parameters()).device
    y_eval = _transform_y(y_like, space)

    if y_ref_for_space is None:
        y_ref_for_space = y_eval
    y_ref_t = torch.from_numpy(y_ref_for_space.astype(np.float32)).to(device)

    dl = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32)),
                      torch.from_numpy(y_eval.astype(np.float32))),
        batch_size=batch_size, shuffle=False
    )
    outs = []
    for xb, yb in dl:
        y_of_u, f_y_batch, *_ = density_from_model(model, xb.to(device), y_ref_t, config)
        z_grid = y_of_u.detach().cpu().numpy()    # (nn,)
        fy     = f_y_batch.detach().cpu().numpy() # (B, nn)

        area = np.trapezoid(fy, x=z_grid, axis=1)
        fy = fy / np.clip(area[:, None], 1e-12, None)

        Ey = _inv_on_mean(z_grid, fy, space=space)
        outs.append(Ey)
    return np.concatenate(outs, axis=0)

def baseline_lr_metrics(X_tr, y_tr, X_te, y_te, space: str = "raw"):
    """
    Baseline: regresja liniowa + homoscedastyczny Gaussian NLL
    (w wybranej przestrzeni), a RMSE/MAE zawsze na skali raw.
    """
    lr = LinearRegression().fit(X_tr, y_tr)
    mu_tr = lr.predict(X_tr)
    mu_te = lr.predict(X_te)

    if space == "raw":
        resid_tr = y_tr - mu_tr
        sigma2 = float(np.var(resid_tr, ddof=1))
        sigma2 = max(sigma2, 1e-12)
        nll_te = float(np.mean(0.5*np.log(2*np.pi*sigma2) + (y_te - mu_te)**2/(2*sigma2)))
    elif space == "log1p":
        z_tr = np.log1p(y_tr); z_mu_tr = np.log1p(np.clip(mu_tr, 0, None))
        z_te = np.log1p(y_te); z_mu_te = np.log1p(np.clip(mu_te, 0, None))
        sigma2 = float(np.var(z_tr - z_mu_tr, ddof=1))
        sigma2 = max(sigma2, 1e-12)
        nll_te = float(np.mean(0.5*np.log(2*np.pi*sigma2) + (z_te - z_mu_te)**2/(2*sigma2)))
    else:
        raise ValueError("space must be 'raw' or 'log1p'")

    return {"rmse": rmse(y_te, mu_te), "mae": mae(y_te, mu_te), "nll": nll_te, "model": lr}
