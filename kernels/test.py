import torch
import torch.nn.functional as F

from cos_norm import cosine_norm_forward_naive, cosine_norm_backward_naive
from resid_connect import resid_connect_fwd_naive, resid_connect_bwd_naive

import torch._dynamo
torch._dynamo.config.suppress_errors = True

def test_resid_connect_backward(M, N, dtype, device=torch.device('cuda')):
    """
    Tests the manual backward implementation against PyTorch's autograd
    
    Args:
        M: batch size * sequence length
        N: model dimension
        dtype: data type (float16 or float32)
        device: device to run on
    """
    # Create input tensors
    h = torch.randn((M, N), dtype=dtype, device=device, requires_grad=False)
    h_eigen = torch.randn((M, N), dtype=dtype, device=device, requires_grad=False)
    alpha = torch.randn(N, dtype=dtype, device=device, requires_grad=False)
    h_auto = h.clone().detach().requires_grad_(True), 
    h_eigen_auto = h_eigen.clone().detach().requires_grad_(True), 
    alpha_auto = alpha.clone().detach().requires_grad_(True)
    
    # Forward pass naive
    output = resid_connect_fwd_naive(h, h_eigen, alpha)

    # forward pass autograd
    h_eigen_normed = h_eigen_auto / torch.norm(h_eigen_auto, p=2, dim=1, keepdim=True).clamp(min=1e-12) # TODO error
    out_pre_norm = h_auto + alpha_auto * (h_eigen_normed - h_auto)
    output_auto = out_pre_norm / torch.norm(out_pre_norm, p=2, dim=1, keepdim=True).clamp(min=1e-12)

    # Random gradient for backpropagation
    grad_output = torch.randn((M, N), dtype=dtype, device=device)

    # Backward passes
    dLdh_manual, dLdh_eigen_manual, dLdalpha_manual = resid_connect_bwd_naive(
        h, h_eigen, alpha, grad_output
    )
    output_auto.backward(grad_output)
    dLdh_auto = h.grad.clone().detach()
    dLdh_eigen_auto = h_auto.grad.clone().detach()
    dLdalpha_auto = alpha.grad.clone().detach()
    
    # Compare gradients
    atol, rtol = (1e-2, 1e-2) if dtype == torch.float16 else (1e-5, 1e-5)
    
    # Test gradients for h
    try:
        torch.testing.assert_close(dLdh_auto, dLdh_manual, atol=atol, rtol=rtol)
        print(f"✓ Gradient for h passed (M={M}, N={N}, dtype={dtype})")
    except AssertionError as e:
        print(f"✗ Gradient for h failed (M={M}, N={N}, dtype={dtype})")
        raise e
    
    # Test gradients for h_eigen
    try:
        torch.testing.assert_close(dLdh_eigen_auto, dLdh_eigen_manual, atol=atol, rtol=rtol)
        print(f"✓ Gradient for h_eigen passed (M={M}, N={N}, dtype={dtype})")
    except AssertionError as e:
        print(f"✗ Gradient for h_eigen failed (M={M}, N={N}, dtype={dtype})")
        raise e
    
    # Test gradients for alpha
    try:
        torch.testing.assert_close(dLdalpha_auto, dLdalpha_manual, atol=atol, rtol=rtol)
        print(f"✓ Gradient for alpha passed (M={M}, N={N}, dtype={dtype})")
    except AssertionError as e:
        print(f"✗ Gradient for alpha failed (M={M}, N={N}, dtype={dtype})")
        raise e


if __name__ == "__main__":
    # Test with various dimensions and data types
    test_resid_connect_backward(64, 64, torch.float32)
    test_resid_connect_backward(256, 768, torch.float32)
    test_resid_connect_backward(256, 768, torch.float16)
    test_resid_connect_backward(128, 4096, torch.float32)
    test_resid_connect_backward(128, 4096, torch.float16)