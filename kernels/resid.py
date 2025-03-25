import math

import torch
#print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

# lets us cache a ton of different kernels when benchmarking
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # Increase from default of 8

import triton
import triton.language as tl

from torch.library import triton_op, wrap_triton

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

from cos_norm import cosine_norm_forward_naive, cosine_norm_backward_naive

@torch.compile
def resid_fwd_naive(h, h_eigen, alpha):
    """
    h & h_eigen: shape (M, N) 
    alpha: shape (N)
    """
    with torch.no_grad():
        h_eigen = cosine_norm_forward_naive(h_eigen)
        h = cosine_norm_forward_naive(h + alpha * (h_eigen - h))
    return h

@torch.compile
def resid_bwd_naive(h, h_eigen, alpha, grad_output):
    """
    Computes gradients for residual connection with cosine normalization
    
    Args:
        h: original input, shape (M, N)
        h_eigen: eigenspace input, shape (M, N)
        alpha: scaling factor, shape (N)
        grad_output: gradient from upstream, shape (M, N)
    
    Returns:
        grad_h: gradient for h
        grad_h_eigen: gradient for h_eigen
        grad_alpha: gradient for alpha
    """
    with torch.no_grad():
        # Forward pass computations we need
        h_eigen_normed = cosine_norm_forward_naive(h_eigen)
        residual = h + alpha * (h_eigen_normed - h)
        
        # Backward pass through the final normalization
        grad_residual = cosine_norm_backward_naive(residual, grad_output)
        
        # Gradients for the residual connection components
        grad_h = grad_residual * (1 - alpha)
        grad_h_eigen_normed = grad_residual * alpha
        
        # Backward pass through the first normalization (h_eigen)
        grad_h_eigen = cosine_norm_backward_naive(h_eigen, grad_h_eigen_normed)
        
        # Compute gradient for alpha - reshape to match alpha's shape (N)
        grad_alpha = torch.sum(grad_residual * (h_eigen_normed - h), dim=0)
    
    return grad_h, grad_h_eigen, grad_alpha


@triton.jit
def resid_fwd_kernel(
    h_ptr, h_eigen_ptr, alpha_ptr, out_ptr,
    stride_M, stride_N,
    N,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    h_ptr += row * stride_M
    h_eigen_ptr += row * stride_M
    out_ptr += row * stride_M
    #norm_ptr += row # TODO add in norm storing for bwd pass later

    cols = tl.arange(0, BLOCK_SIZE) * stride_N # stride since we never asserted x.is_contiguous()
    mask = cols < N

    h = tl.load(h_ptr + cols, mask=mask, other=0.).to(tl.float32)
    h_eigen = tl.load(h_eigen_ptr + cols, mask=mask, other=0.).to(tl.float32)
    alpha = tl.load(alpha_ptr + cols, mask=mask, other=0.).to(tl.float32)

    # compute L_2 norm & normalize h_eigen
    eps: tl.constexpr = 1e-12
    inf: tl.constexpr = 1e12
    h_eigen_normed = h_eigen / tl.clamp(tl.sqrt(tl.sum(h_eigen * h_eigen, axis=0)), eps, inf)

    # movement along hypersphere residual connection
    out_ = h + (alpha * (h_eigen_normed - h))

    # normalize output
    out = out_ / tl.clamp(tl.sqrt(tl.sum(out_ * out_, axis=0)), eps, inf)

    #tl.store(norm_ptr, norm)
    tl.store(out_ptr + cols, out.to(out_ptr.type.element_ty), mask=mask)


# used to derive our block size
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
sram_per_sm = properties["max_shared_mem"]

#@torch.compile(fullgraph=True)
@triton_op("mylib::resid_fwd_triton", mutates_args={})
def resid_fwd_triton(h, h_eigen, alpha):
    assert all([x.device == DEVICE for x in [h, h_eigen, alpha]])
    assert h.shape == h_eigen.shape
    assert h.stride() == h_eigen.stride()
    assert alpha.shape[0] == h.shape[-1]
    M, N = h.reshape(-1, h.shape[-1]).shape

    # this kernel is designed for normalizing vectors that fit in SRAM
    max_entries = sram_per_sm // h.element_size()
    block_size = triton.next_power_of_2(N)
    assert max_entries >= block_size, f"cosine norm kernel only supports vectors up to {max_entries}"
    # H100s have 256kb of SRAM per SM so this would fit a model dimension of 64 thousand at fp32, plenty

    # pre-allocate output
    out = torch.empty_like(h)

    wrap_triton(resid_fwd_kernel)[(M,)](
        h, h_eigen, alpha, out,
        h.stride(-2), h.stride(-1),
        N,
        BLOCK_SIZE=block_size
    )

    return out


@triton.jit
def resid_bwd_kernel(
    h_ptr, dLdh_ptr,
    h_eigen_ptr, dLdh_eigen_ptr,
    alpha_ptr, dLdalpha_ptr,
    out_ptr, dLdout_ptr,
    stride_M, stride_N,
    N,
    BLOCK_SIZE
):
    pass

def resid_bwd_triton(ctx, grad):
    h, h_eigen, alpha, out = ctx.saved_tensors
    block_size = ctx.block_size
    M, N = h.reshape(-1, h.shape[-1]).shape
    
    dLdh = torch.empty_like(h)
    dLdh_eigen = torch.empty_like(h_eigen)
    dLdalpha = torch.empty_like(alpha)
    dLdout = torch.empty_like(out)

    wrap_triton(resid_bwd_kernel)[(M,)](
        h, dLdh, h_eigen, dLdh_eigen, alpha, dLdalpha, out, dLdout,
        h.stride(-2), h.stride(-1),
        N,
        block_size
    )

    return dLdh, dLdh_eigen, dLdalpha

def resid_bwd_triton_setup_context(ctx, inputs, output):
    h, h_eigen, alpha = inputs
    ctx.save_for_backward(h, h_eigen, alpha, output)

    max_entries = sram_per_sm // h.element_size()
    block_size = triton.next_power_of_2(N)
    ctx.block_size = block_size

resid_fwd_triton.register_autograd(resid_bwd_triton, setup_context=resid_bwd_triton_setup_context)











def create_diff_heatmap(expected, actual, M, N, dtype, test_name, atol):
    """Create a heatmap visualization for gradient differences"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Make path more generic to work with both fwd and bwd tests
    heatmap_dir = './kernels'
    heatmap_path = f'{heatmap_dir}/resid_{test_name}_heatmap.png'
    
    # Create directory if it doesn't exist
    os.makedirs(heatmap_dir, exist_ok=True)
    
    # Convert to numpy arrays
    actual_np = actual.detach().cpu().numpy()
    expected_np = expected.detach().cpu().numpy()
    
    # Compute differences and masks
    abs_diff = np.abs(expected_np - actual_np)
    abs_fail_mask = (abs_diff > atol).astype(np.int32)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(abs_fail_mask, cmap="hot", aspect="auto")
    plt.xlabel("Model Dimension")
    plt.ylabel("Sequence Position")
    plt.colorbar()
    plt.title(f"Failed {test_name} Test Heatmap (M={M}, N={N}, dtype={dtype})")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Created error heatmap at {heatmap_path}")


def test_resid_fwd(M, N, dtype, device=DEVICE, atol=1e-3, rtol=1e-3):
    # create data
    h_i = cosine_norm_forward_naive(
        torch.randn((M, N), dtype=dtype, device=device, requires_grad=False)
    )
    h_eigen = torch.randn((M, N), dtype=dtype, device=device, requires_grad=False)
    alpha = torch.randn(N, dtype=dtype, device=device, requires_grad=False)

    # run each
    h_iplus1_naive = resid_fwd_naive(h_i, h_eigen, alpha)
    h_iplus1_triton = resid_fwd_triton(h_i, h_eigen, alpha)

    # test with try/except to catch failures
    import os
    heatmap_path = './kernels/resid_fwd_heatmap.png'
    try:
        torch.testing.assert_close(h_iplus1_naive, h_iplus1_triton, atol=atol, rtol=rtol)
        print(f"✓ passed fwd test (M={M}, N={N}, dtype={dtype})")
        
        # Delete old heatmap if test passes and file exists
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
            
    except AssertionError as e:
        print(f"✗ failed test (M={M}, N={N}, dtype={dtype})")
        # Use the common heatmap function
        create_diff_heatmap(h_iplus1_naive, h_iplus1_triton, M, N, dtype, "fwd", atol)
        
        # Re-raise the exception
        raise e


def test_resid_bwd(M, N, dtype, device=DEVICE, atol=5e-3, rtol=0):
    """
    Tests backward implementations for residual connection:
    1. PyTorch autograd through a differentiable forward function
    2. The naive backward function (resid_bwd_naive)
    3. The triton backward function (resid_bwd_triton)
    """
    # Create data with gradients
    h = torch.randn((M, N), dtype=dtype, device=device, requires_grad=True)
    h_eigen = torch.randn((M, N), dtype=dtype, device=device, requires_grad=True)
    alpha = torch.randn(N, dtype=dtype, device=device, requires_grad=True)
    
    # Gradient from upstream
    grad_output = torch.randn((M, N), dtype=dtype, device=device)
    
    # Method 1: PyTorch autograd backward
    # Create a differentiable version of the forward pass
    def forward_differentiable(h, h_eigen, alpha):
        # Same logic as resid_fwd_naive but without torch.no_grad()
        h_eigen_normed = h_eigen / torch.clamp(torch.norm(h_eigen, dim=-1, keepdim=True), min=1e-12)
        residual = h + alpha * (h_eigen_normed - h)
        out = residual / torch.clamp(torch.norm(residual, dim=-1, keepdim=True), min=1e-12)
        return out
    
    # Clone inputs for autograd
    h_auto = h.clone().detach().requires_grad_(True)
    h_eigen_auto = h_eigen.clone().detach().requires_grad_(True)
    alpha_auto = alpha.clone().detach().requires_grad_(True)
    
    # Forward pass
    output_auto = forward_differentiable(h_auto, h_eigen_auto, alpha_auto)
    
    # Backward pass
    output_auto.backward(grad_output)
    
    grad_h_auto = h_auto.grad
    grad_h_eigen_auto = h_eigen_auto.grad
    grad_alpha_auto = alpha_auto.grad
    
    # Method 2: Naive backward
    grad_h_naive, grad_h_eigen_naive, grad_alpha_naive = resid_bwd_naive(
        h, h_eigen, alpha, grad_output
    )
    
    # Method 3: Triton backward
    #grad_h_triton, grad_h_eigen_triton, grad_alpha_triton = resid_bwd_triton(
    #    h, h_eigen, alpha, grad_output
    #)
    
    # Compare autograd vs naive
    try:
        torch.testing.assert_close(grad_h_auto, grad_h_naive, atol=atol, rtol=rtol)
        torch.testing.assert_close(grad_h_eigen_auto, grad_h_eigen_naive, atol=atol, rtol=rtol)
        torch.testing.assert_close(grad_alpha_auto, grad_alpha_naive, atol=atol, rtol=rtol)
        print(f"✓ passed autograd vs naive bwd test (M={M}, N={N}, dtype={dtype})")
    except AssertionError as e:
        print(f"✗ failed autograd vs naive bwd test (M={M}, N={N}, dtype={dtype})")
        create_diff_heatmap(grad_h_auto, grad_h_naive, M, N, dtype, "autograd_vs_naive", atol)
        raise e
    """
    # Compare naive vs triton
    try:
        torch.testing.assert_close(grad_h_naive, grad_h_triton, atol=atol, rtol=rtol)
        torch.testing.assert_close(grad_h_eigen_naive, grad_h_eigen_triton, atol=atol, rtol=rtol)
        torch.testing.assert_close(grad_alpha_naive, grad_alpha_triton, atol=atol, rtol=rtol)
        print(f"✓ passed naive vs triton test (M={M}, N={N}, dtype={dtype})")
    except AssertionError as e:
        print(f"✗ failed naive vs triton test (M={M}, N={N}, dtype={dtype})")
        create_diff_heatmap(grad_h_naive, grad_h_triton, M, N, dtype, "naive_vs_triton", atol)
        raise e"""


def previous_power_two(n):
    return int(math.log(n, 2))

configs = []
for mode in ["fwd"]:#, "bwd"]:
    for dtype_bytes in [2, 4]:
        configs.append(
            triton.testing.Benchmark(
                x_names=['N'],
                x_vals=[2 ** i for i in range(8, previous_power_two(sram_per_sm // dtype_bytes))], 
                line_arg='provider',
                line_vals=['triton', 'naive'],
                line_names=['Triton', 'naive + torch.compile'],
                styles=[('blue', '-'), ('green', '-')],
                ylabel='GB/s',
                plot_name=f'resid_{mode}_fp{8*dtype_bytes}',
                args={"mode": mode, "dtype_bytes": dtype_bytes}, 
            ))
@triton.testing.perf_report(configs)
def benchmark(N, provider, mode, dtype_bytes, device=DEVICE):
    # create data
    assert dtype_bytes in [2, 4]
    dtype = torch.float16 if dtype_bytes == 2 else torch.float32
    h = torch.randn((32*1024, N), dtype=dtype, device=device, requires_grad=False)
    h_eigen = torch.randn((32*1024, N), dtype=dtype, device=device, requires_grad=False)
    alpha = torch.randn(N, dtype=dtype, device=device, requires_grad=False)

    # confidence itnerval for testing
    quantiles = [0.5, 0.001, 0.999]

    if provider == "triton":
        fn = lambda: resid_fwd_triton(h, h_eigen, alpha)
    if provider == "naive":
        fn = lambda: resid_fwd_naive(h, h_eigen, alpha)
    elif mode == "bwd":
        y = fn()
        dLdy = torch.randn_like(y)
        if provider == "naive":
            fn = lambda: cosine_norm_backward_naive(x, dLdy)
        else:
            fn = lambda: y.backward(dLdy, retain_graph=True)
    
    # benchmark
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    gbps = lambda ms: (4 if mode == "fwd" else 3) * h.numel() * h.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)



if __name__ == "__main__":
    # always run unit-tests
    test_resid_fwd(64, 64, torch.float32)
    test_resid_fwd(64, 64, torch.float16)
    test_resid_fwd(2048, 768, torch.float16)
    test_resid_fwd(2048, 8192, torch.float16)
    test_resid_fwd(2048, 768, torch.float32)
    test_resid_fwd(2048, 8192, torch.float32)

    # Run backward tests
    test_resid_bwd(64, 64, torch.float32)
    test_resid_bwd(64, 64, torch.float16)
    test_resid_bwd(2048, 768, torch.float16)
    test_resid_bwd(2048, 8192, torch.float16)
    test_resid_bwd(2048, 768, torch.float32)
    test_resid_bwd(2048, 8192, torch.float32)

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='./benchmarks/', print_data=False)
    