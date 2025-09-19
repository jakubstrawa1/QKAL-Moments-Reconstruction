from __future__ import annotations
from copy import deepcopy
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils.legendre import legendre_targets_from_y
from models.loss import SimpleCoefficientLoss
from utils.config import ReconstructionConfig
from sklearn.preprocessing import QuantileTransformer

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 128,
    val_ratio: float = 0.2,
    seed: int = 42
):

    rng = np.random.default_rng(seed)
    idx = np.arange(len(y)); rng.shuffle(idx)
    split = int((1.0 - val_ratio) * len(y))
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, y_tr = X[tr_idx].astype(np.float32), y[tr_idx].astype(np.float32)
    X_va, y_va = X[va_idx].astype(np.float32), y[va_idx].astype(np.float32)

    qt = QuantileTransformer(
        n_quantiles=min(1000, len(y_tr)),
        output_distribution="uniform",
        subsample=10_000,
        random_state=seed,
    )
    # u in [0,1]
    u_tr = qt.fit_transform(y_tr.reshape(-1, 1)).astype(np.float32).ravel()
    u_va = qt.transform(y_va.reshape(-1, 1)).astype(np.float32).ravel()

    tr_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(u_tr))
    va_ds = TensorDataset(torch.from_numpy(X_va), torch.from_numpy(u_va))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    return tr_dl, va_dl, qt


def train_one_model(model, tr_dl, va_dl, cfg: ReconstructionConfig, loss: str = "mse"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def basis(u: torch.Tensor, degree: int):
        return legendre_targets_from_y(u, degree)
    criterion = SimpleCoefficientLoss(cfg, basis_fn=basis)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    if loss == "huber":
        base_crit = torch.nn.HuberLoss(reduction="none")
        def compute_loss(preds, y):
            targets = basis(y, cfg.degree).to(device=preds.device, dtype=preds.dtype)
            return base_crit(preds, targets).mean()
    else:
        def compute_loss(preds, y):
            return criterion(preds, y)

    grad_clip = 0.0

    def run_epoch(dl, train: bool):
        model.train(train)
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            preds = model(xb)
            loss_val = compute_loss(preds, yb)
            if train:
                opt.zero_grad(set_to_none=True)
                loss_val.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
            tot += float(loss_val.item()) * xb.size(0)
            n   += xb.size(0)
        return tot / max(n, 1)

    best_val, best_state = float("inf"), None
    history = {"train": [], "val": []}
    for ep in range(1, cfg.epochs + 1):
        tr = run_epoch(tr_dl, True)
        va = run_epoch(va_dl, False)
        history["train"].append(tr)
        history["val"].append(va)
        if va < best_val and np.isfinite(va):
            best_val = va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"[{ep:02d}/{cfg.epochs}] train={tr:.5f} val={va:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return {"model": model, "best_val": best_val, "history": history}
