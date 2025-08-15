"""
Training Module
"""

#import qkal


import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import QuantileTransformer
import numpy as np

def train_qkal_from_arrays(X: np.ndarray, y: np.ndarray, degree=6, hidden_dim=256,
                           epochs=20, batch_size=256, lr=3e-3, device=None):
    """
    X: (N, D) float
    y: (N,) float (wartość ciągła)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    qt = QuantileTransformer(n_quantiles=min(1000, len(y)), output_distribution='uniform', subsample=10_000, random_state=42)
    u = qt.fit_transform(y.reshape(-1, 1)).astype(np.float32).reshape(-1)

    N = len(y)
    idx = np.arange(N)
    np.random.default_rng(42).shuffle(idx)
    split = int(0.8 * N)
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr, u_tr = X[tr_idx].astype(np.float32), u[tr_idx].astype(np.float32)
    X_va, u_va = X[va_idx].astype(np.float32), u[va_idx].astype(np.float32)

    tr_dl = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(u_tr)), batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(u_va)), batch_size=batch_size, shuffle=False)

    in_dim = X.shape[1]
    model = QKAL(degree=degree, in_dim=in_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    def run_epoch(dl, train=True):
        model.train(train)
        tot, n = 0.0, 0
        for xb, ub in dl:
            xb, ub = xb.to(device), ub.to(device)
            preds = model(xb)                                # (B, K)
            #model x -> Kan(x) -> z. [l1,l2,l3,l4,l5] -> momenty

            targets = legendre_targets_from_y(ub, degree)    # (B, K) F_norm(y_true) = ub.  F.legendre(ub) ... n degree.  [l1, l2, l3, ln]

            loss = F.mse_loss(preds, targets)

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            tot += loss.item() * xb.size(0)
            n += xb.size(0)
        return tot / max(n,1)

    best, best_state = float("inf"), None
    for ep in range(1, epochs+1):
        tr = run_epoch(tr_dl, train=True)
        va = run_epoch(va_dl, train=False)
        if va < best:
            best, best_state = va, {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        print(f"[{ep:02d}/{epochs}] train={tr:.5f}  val={va:.5f}")

    if best_state:
        model.load_state_dict(best_state)

    return model, qt
