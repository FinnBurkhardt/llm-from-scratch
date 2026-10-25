import torch
import torch.nn as nn

from scratch_llm.rmsnorm import RMSNorm


def test_output_shape():

    model = RMSNorm(hidden_size=128)

    x = torch.randn(8, 20, 128)

    y = model(x)

    assert y.shape == x.shape


def test_reference():

    torch.manual_seed(42)

    x = torch.randn(8, 20, 128)

    custom = RMSNorm(128, eps=1e-6)

    reference = nn.RMSNorm(128, eps=1e-6)

    y_custom = custom(x)
    y_reference = reference(x)

    torch.testing.assert_close(
        y_custom,
        y_reference,
        rtol=1e-5,
        atol=1e-5
    )


def test_gradients():

    model = RMSNorm(128)

    x = torch.randn(
        4, 10, 128,
        requires_grad=True
    )

    y = model(x)

    loss = y.square().mean()

    loss.backward()

    assert x.grad is not None
    assert model.gamma.grad is not None

    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(model.gamma.grad).all()


def test_zero_input():

    model = RMSNorm(128)

    x = torch.zeros(4, 10, 128)

    y = model(x)

    assert torch.isfinite(y).all()

    torch.testing.assert_close(
        y,
        torch.zeros_like(y)
    )


def test_parameter_count():

    model = RMSNorm(128)

    num_parameters = sum(
        p.numel() for p in model.parameters()
    )

    assert num_parameters == 128
