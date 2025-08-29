import torch
import pytest
from qkal.layer import QKAL_Layer

@pytest.mark.parametrize("B,in_f,out_f,degree", [
    (8, 5, 7, 0),
    (8, 5, 7, 2),
    (4, 3, 4, 3),
])
def test_output_shape_and_finiteness(B, in_f, out_f, degree):
    torch.manual_seed(0)
    layer = QKAL_Layer(in_f=in_f, out_f=out_f, degree=degree)
    x = torch.randn(B, in_f, dtype=torch.float32)
    y = layer(x)
    assert y.shape == (B, out_f)
    assert torch.isfinite(y).all()

def test_degree_zero_outputs_are_constant_wrt_x():
    torch.manual_seed(0)
    B, in_f, out_f, degree = 6, 5, 7, 0
    layer = QKAL_Layer(in_f, out_f, degree)
    x1 = torch.randn(B, in_f)
    x2 = torch.randn(B, in_f) * 3.14 + 2.0
    y1 = layer(x1)
    y2 = layer(x2)

    assert torch.allclose(y1, y2, atol=1e-6)

def test_backward_grads_exist_and_finite():
    torch.manual_seed(0)
    B, in_f, out_f, degree = 10, 6, 4, 3
    layer = QKAL_Layer(in_f, out_f, degree)
    x = torch.randn(B, in_f, requires_grad=True)
    y = layer(x)
    loss = (y ** 2).mean()
    loss.backward()

    assert layer.poly_w.grad is not None
    assert torch.isfinite(layer.poly_w.grad).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

def test_extreme_inputs_do_not_nan():
    torch.manual_seed(0)
    B, in_f, out_f, degree = 5, 4, 3, 2
    layer = QKAL_Layer(in_f, out_f, degree)
    x = (torch.randn(B, in_f) * 1e6)
    y = layer(x)
    assert torch.isfinite(y).all()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_device_works():
    torch.manual_seed(0)
    device = torch.device("cuda")
    B, in_f, out_f, degree = 8, 5, 7, 2
    layer = QKAL_Layer(in_f, out_f, degree).to(device)
    x = torch.randn(B, in_f, device=device)
    y = layer(x)
    assert y.is_cuda and y.shape == (B, out_f)
