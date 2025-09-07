from __future__ import annotations
from torch.nn.utils import clip_grad_norm_
from typing import Tuple, Type
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.loss import SimpleCoefficientLoss
from utils.config import ReconstructionConfig
from utils.legendre import legendre_targets_from_y


def train_model_from_arrays(
    model_cls: Type[torch.nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    config: ReconstructionConfig
) -> Tuple[torch.nn.Module, QuantileTransformer]:

    assert X.ndim == 2 and y.ndim == 1 and len(X) == len(y), "Bad shapes"

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = (dev.type == "cuda")
    pin = use_cuda

    # --- train/val split (80/20) ---
    N = len(y)
    idx = np.arange(N); rng.shuffle(idx)
    split = int(0.8 * N)
    tr_idx, va_idx = idx[:split], idx[split:]

    X_tr = X[tr_idx].astype(np.float32)
    X_va = X[va_idx].astype(np.float32)

    # --- QuantileTransformer ---
    qt = QuantileTransformer(
        n_quantiles=min(1000, len(tr_idx)),
        output_distribution="uniform",
        subsample=10_000,
        random_state=config.seed,
    )
    u_tr = qt.fit_transform(y[tr_idx].reshape(-1, 1)).astype(np.float32).ravel()
    u_va = qt.transform(      y[va_idx].reshape(-1, 1)).astype(np.float32).ravel()

    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(u_tr)),
        batch_size=config.batch_size, shuffle=True, pin_memory=pin
    )
    va_dl = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(u_va)),
        batch_size=config.batch_size, shuffle=False, pin_memory=pin
    )

    # --- model/opt/loss ---
    model = model_cls(config).to(dev)
    opt = torch.optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    criterion = SimpleCoefficientLoss(config, basis_fn=legendre_targets_from_y)

    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    patience = 10
    stale = 0
    best_val = float("inf")
    best_state = None

    def run_epoch(dl, is_train: bool) -> float:
        model.train(is_train)
        total, count = 0.0, 0
        for xb, ub in dl:
            xb = xb.to(dev, non_blocking=pin)
            ub = ub.to(dev, non_blocking=pin).float()

            with torch.set_grad_enabled(is_train), torch.amp.autocast('cuda', enabled=use_cuda):
                preds = model(xb)
                loss = criterion(preds, ub)
                if loss.dim() != 0:
                    loss = loss.mean()

            if is_train:
                opt.zero_grad(set_to_none=True)
                if use_cuda:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    clip_grad_norm_(model.parameters(), max_norm=1.0)
                    opt.step()

            bs = xb.size(0)
            total += float(loss.detach().cpu()) * bs
            count += bs

        return total / max(count, 1)

    # --- early stopping ---
    for ep in range(1, config.epochs + 1):
        tr = run_epoch(tr_dl, True)
        va = run_epoch(va_dl, False)
        scheduler.step(va)

        improved = va < best_val - 1e-6
        if improved:
            best_val = va
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        print(f"[{ep:02d}/{config.epochs}] train={tr:.5f} val={va:.5f} best={best_val:.5f}")
        if stale >= patience:
            print(f"Early stop @ epoch {ep} (best val={best_val:.5f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, qt
