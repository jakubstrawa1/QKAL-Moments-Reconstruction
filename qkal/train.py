from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
from qkal.model import QKAL
from qkal.loss import QKALLoss
from qkal.config import QKALReconstructionConfig

def train_qkal_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    config: QKALReconstructionConfig) -> Tuple[torch.nn.Module, QuantileTransformer]:
    """
    Minimal training for QKAL on (X, y).

    Args:
        X: (N, D) float32/float64 features.
        y: (N,) float targets.
        degree, hidden_dim, epochs, batch_size, lr, seed, device: usual knobs.

    Returns:
        model: best-val QKAL model.
        qt: fitted QuantileTransformer (maps y -> uniform u in [0, 1]).
    """
    assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "Bad shapes"
    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Targets -> uniform quantiles u
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(y)),
        output_distribution="uniform",
        subsample=10_000,
        random_state=config.seed,
    )
    u = qt.fit_transform(y.reshape(-1, 1)).astype(np.float32).ravel()

    # Deterministic 80/20 split
    N = len(y)
    idx = np.arange(N)
    rng.shuffle(idx)
    split = int(0.8 * N)
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, u_tr = X[tr_idx].astype(np.float32), u[tr_idx]
    X_va, u_va = X[va_idx].astype(np.float32), u[va_idx]

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(u_tr)),
                       batch_size=config.batch_size, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(u_va)),
                       batch_size=config.batch_size, shuffle=False)

    # Model/opt
    in_dim = X.shape[1]
    model = QKAL(config).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = QKALLoss(config)

    def run_epoch(dl, train: bool) -> float:
        model.train(train)
        tot, n = 0.0, 0
        for xb, ub in dl:
            xb, ub = xb.to(dev), ub.to(dev)
            preds = model(xb)
            loss = criterion(preds, ub)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            bs = xb.size(0)
            tot += loss.item() * bs
            n += bs
        return tot / max(n, 1)

    best_val, best_state = float("inf"), None
    for ep in range(1, config.epochs + 1):
        tr = run_epoch(tr_dl, train=True)
        va = run_epoch(va_dl, train=False)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        print(f"[{ep:02d}/{config.epochs}] train={tr:.5f} val={va:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, qt
