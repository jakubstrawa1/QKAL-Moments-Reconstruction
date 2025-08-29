import math
import torch
import pytest

from qkal.legendre import compute_scaled_legendre_polynomials, legendre_targets_from_y


def test_shapes_general():
    torch.manual_seed(0)
    x = torch.rand(8, 3)  # dowolny kształt
    K = 5
    P = compute_scaled_legendre_polynomials(x, K - 1)
    assert P.shape == (8, 3, K)  # (..., K)


def test_known_values_at_center():
    # x=0.5 -> t=0. P0=1, P1=0, P2=-1/2 (po skali: *sqrt(5) => -sqrt(5)/2)
    x = torch.tensor([0.5], dtype=torch.float64)  # [0,1]
    P = compute_scaled_legendre_polynomials(x, order=2).squeeze(0)  # (K,)
    assert torch.allclose(P[0], torch.tensor(1.0, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(P[1], torch.tensor(0.0, dtype=torch.float64), atol=1e-12)
    expected_P2 = -0.5 * math.sqrt(5.0)
    assert torch.allclose(P[2], torch.tensor(expected_P2, dtype=torch.float64), atol=1e-12)


def test_orthonormality_empirical():
    # Empirycznie: E[phi_m(x) phi_n(x)] ≈ delta_mn dla x~U(0,1)
    torch.manual_seed(0)
    N = 4000
    K = 5
    x = torch.rand(N, 1, dtype=torch.float64)  # [N,1]
    P = compute_scaled_legendre_polynomials(x, K - 1)  # [N,1,K]
    Phi = P.squeeze(1)  # [N,K]
    G = (Phi.T @ Phi) / N  # przybliżenie E[..]
    I = torch.eye(K, dtype=torch.float64)
    assert torch.allclose(G, I, atol=0.05), f"Gram ≈ I? max|G-I|={float((G-I).abs().max())}"


def test_gradients_exist():
    # Sprawdź, że przepływ gradientów działa
    x = torch.rand(10, 2, requires_grad=True)
    P = compute_scaled_legendre_polynomials(x, order=3)
    loss = P.pow(2).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_legendre_targets_from_y_matches_compute():
    u = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float32)  # w [0,1]
    K = 4
    T = legendre_targets_from_y(u, degree=K - 1)         # (B,K)
    P = compute_scaled_legendre_polynomials(u.unsqueeze(1), K - 1).squeeze(1)
    assert torch.allclose(T, P, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_works():
    device = torch.device("cuda")
    x = torch.rand(16, 2, device=device)
    P = compute_scaled_legendre_polynomials(x, order=3)
    assert P.is_cuda and P.shape == (16, 2, 4)
