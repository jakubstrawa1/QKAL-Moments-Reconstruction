import numpy as np
import pytest
import torch

from utils.config import QKALReconstructionConfig
from qkal.train import train_qkal_from_arrays
from qkal.eval import eval_nll, predict_mean_from_density, rmse, mae


try:
    from qkal.eval import pit_and_coverage
    HAS_PIT = True
except Exception:
    HAS_PIT = False


def tiny_config(input_dim: int) -> QKALReconstructionConfig:

    cfg = QKALReconstructionConfig()

    cfg.input_dim = input_dim

    if hasattr(cfg, "degree"): cfg.degree = 2
    if hasattr(cfg, "hidden_dim"): cfg.hidden_dim = 32
    if hasattr(cfg, "batch_size"): cfg.batch_size = 256
    if hasattr(cfg, "epochs"): cfg.epochs = 2
    if hasattr(cfg, "learning_rate"): cfg.learning_rate = 1e-3
    if hasattr(cfg, "weight_decay"): cfg.weight_decay = 1e-4

    if hasattr(cfg, "nn_grid_train"): cfg.nn_grid_train = 128
    if hasattr(cfg, "nn_grid_eval"):  cfg.nn_grid_eval  = 256
    if hasattr(cfg, "seed"): cfg.seed = 0
    return cfg


def make_synthetic(n=800, d=5, noise=0.5, seed=0):
    """Proste dane do testu: y = Xw + eps."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w = np.array([1.2, -0.7, 0.5, 0.0, 0.9], dtype=np.float32)[:d]
    y = (X @ w + rng.normal(scale=noise, size=n)).astype(np.float32)
    return X, y


@pytest.mark.parametrize("n,d", [(800, 5)])
def test_eval_nll_and_mean_are_finite(n, d):
    torch.manual_seed(0)
    X, y = make_synthetic(n=n, d=d, noise=0.5, seed=0)

    cfg = tiny_config(d)
    model, _ = train_qkal_from_arrays(X, y, cfg)

    # NLL powinno być skończone
    nll = eval_nll(model, X[:200], y[:200], cfg)
    assert np.isfinite(nll), f"NLL is not finite: {nll}"

    # Predykcja punktowa z gęstości
    yhat = predict_mean_from_density(model, X[:200], y[:200], cfg)
    assert yhat.shape == (200,)
    assert np.isfinite(yhat).all()

    r = rmse(y[:200], yhat)
    m = mae(y[:200], yhat)
    assert np.isfinite(r) and np.isfinite(m)


@pytest.mark.skipif(not HAS_PIT, reason="pit_and_coverage not available in qkal.eval")
def test_pit_coverage_reasonable():
    torch.manual_seed(0)
    X, y = make_synthetic(n=800, d=5, noise=0.7, seed=1)
    cfg = tiny_config(X.shape[1])
    model, _ = train_qkal_from_arrays(X, y, cfg)

    stats = pit_and_coverage(model, X[:300], y[:300], cfg, qs=(0.1, 0.5, 0.9))
    # podstawowe sanity checki
    assert 0.0 <= stats["pit_mean"] <= 1.0
    assert 0.0 <= stats["pit_var"]  <= 1.0
    assert 0.0 <= stats["pit_ks"]   <= 1.0
    for t, cv in stats["coverage"].items():
        assert 0.0 <= cv <= 1.0, f"Coverage out of [0,1] for q={t}: {cv}"
