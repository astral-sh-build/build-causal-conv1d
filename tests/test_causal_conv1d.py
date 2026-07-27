from importlib.metadata import version

from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from causal_conv1d.causal_conv1d_interface import (
    causal_conv1d_ref,
    causal_conv1d_update_ref,
)
import pytest
import torch


@pytest.fixture(scope="module")
def device() -> torch.device:
    assert torch.cuda.is_available(), "The tests must run on a CUDA GPU"
    return torch.device("cuda")


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    elif actual.dtype == torch.float16:
        rtol, atol = 3e-3, 5e-3
    else:
        rtol, atol = 3e-4, 1e-3
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_published_cuda_wheel(device: torch.device) -> None:
    assert version("causal-conv1d") == "1.6.2.post1+cu.12.8.torch.2.10"
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert torch.cuda.get_device_name(device)


@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_causal_convolution(
    device: torch.device,
    dtype: torch.dtype,
    activation: str | None,
) -> None:
    torch.manual_seed(0)
    source = torch.randn((2, 32, 64), device=device, dtype=dtype)
    weight = torch.randn((32, 4), device=device)
    bias = torch.randn(32, device=device)

    actual = causal_conv1d_fn(source, weight, bias, activation=activation)
    expected = causal_conv1d_ref(source, weight, bias, activation=activation)

    assert_close(actual, expected)


def test_initial_and_final_states(device: torch.device) -> None:
    torch.manual_seed(0)
    source = torch.randn((2, 64, 32), device=device).transpose(1, 2)
    initial_states = torch.randn((2, 3, 32), device=device).transpose(1, 2)
    weight = torch.randn((32, 4), device=device)
    bias = torch.randn(32, device=device)

    actual, final_states = causal_conv1d_fn(
        source,
        weight,
        bias,
        initial_states=initial_states,
        return_final_states=True,
    )
    expected, expected_final_states = causal_conv1d_ref(
        source,
        weight,
        bias,
        initial_states=initial_states,
        return_final_states=True,
    )

    assert_close(actual, expected)
    assert_close(final_states, expected_final_states)


@pytest.mark.parametrize("activation", [None, "silu"])
def test_streaming_update(device: torch.device, activation: str | None) -> None:
    torch.manual_seed(0)
    source = torch.randn((2, 32), device=device)
    state = torch.randn((2, 3, 32), device=device).transpose(1, 2)
    expected_state = state.clone()
    weight = torch.randn((32, 4), device=device)
    bias = torch.randn(32, device=device)

    actual = causal_conv1d_update(
        source,
        state,
        weight,
        bias,
        activation=activation,
    )
    expected = causal_conv1d_update_ref(
        source,
        expected_state,
        weight,
        bias,
        activation=activation,
    )

    assert_close(actual, expected)
    assert_close(state, expected_state)


def test_causal_convolution_backward(device: torch.device) -> None:
    torch.manual_seed(0)
    source = torch.randn((2, 32, 64), device=device, requires_grad=True)
    weight = torch.randn((32, 4), device=device, requires_grad=True)
    bias = torch.randn(32, device=device, requires_grad=True)
    expected_source = source.detach().clone().requires_grad_()
    expected_weight = weight.detach().clone().requires_grad_()
    expected_bias = bias.detach().clone().requires_grad_()
    gradient = torch.randn_like(source)

    actual = causal_conv1d_fn(source, weight, bias, activation="silu")
    expected = causal_conv1d_ref(
        expected_source,
        expected_weight,
        expected_bias,
        activation="silu",
    )
    actual.backward(gradient)
    expected.backward(gradient)

    assert source.grad is not None
    assert weight.grad is not None
    assert bias.grad is not None
    assert expected_source.grad is not None
    assert expected_weight.grad is not None
    assert expected_bias.grad is not None
    assert_close(source.grad, expected_source.grad)
    torch.testing.assert_close(
        weight.grad,
        expected_weight.grad,
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        bias.grad,
        expected_bias.grad,
        rtol=1e-3,
        atol=1e-3,
    )
