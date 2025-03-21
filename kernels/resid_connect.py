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

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

from cos_norm import cosine_norm_forward_naive, cosine_norm_backward_naive

@torch.compile
def resid_connect_fwd_naive(h, h_eigen, alpha):
    """
    h & h_eigen: shape (M, N) 
    alpha: shape (N)
    """
    with torch.no_grad():
        h_eigen = cosine_norm_forward_naive(h_eigen)
        h = cosine_norm_forward_naive(h + alpha * (h_eigen - h))
    return h


@torch.compile
def resid_connect_bwd_naive(h, h_eigen, alpha, grad_output):
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
        
        # Compute gradient for alpha
        grad_alpha = torch.sum(grad_residual * (h_eigen_normed - h), dim=(0, 1))
    
    return grad_h, grad_h_eigen, grad_alpha


@triton.jit
def resid_connect_fwd_kernel(
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

@torch.compile(fullgraph=True)
def resid_connect_fwd_triton(h, h_eigen, alpha):
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

    resid_connect_fwd_kernel[(M,)](
        h, h_eigen, alpha, out,
        h.stride(-2), h.stride(-1),
        N,
        BLOCK_SIZE=block_size
    )

    return out


def test_resid_connect(M, N, dtype, device=DEVICE, atol=1e-3, rtol=1e-3):
    # create data
    h_i = cosine_norm_forward_naive(
        torch.randn((M, N), dtype=dtype, device=device, requires_grad=False)
    )
    h_eigen = torch.randn((M, N), dtype=dtype, device=device, requires_grad=False)
    alpha = torch.randn(N, dtype=dtype, device=device, requires_grad=False)

    # run each
    h_iplus1_naive = resid_connect_fwd_naive(h_i, h_eigen, alpha)
    h_iplus1_triton = resid_connect_fwd_triton(h_i, h_eigen, alpha)

    # test with try/except to catch failures
    import os
    heatmap_path = './kernels/resid_connect_heatmap.png'
    try:
        torch.testing.assert_close(h_iplus1_naive, h_iplus1_triton, atol=atol, rtol=rtol)
        print(f"✓ passed test (M={M}, N={N}, dtype={dtype})")
        
        # Delete old heatmap if test passes and file exists
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
            
    except AssertionError as e:
        print(f"✗ failed test (M={M}, N={N}, dtype={dtype})")
        # Only create and save the heatmap visualization if test fails
        import numpy as np
        import matplotlib.pyplot as plt
        # Convert to numpy arrays
        actual = h_iplus1_triton.detach().cpu().numpy()
        expected = h_iplus1_naive.detach().cpu().numpy()
        # Compute differences and masks
        abs_diff = np.abs(expected - actual)
        abs_fail_mask = (abs_diff > atol).astype(np.int32)
        plt.figure(figsize=(8, 6))
        plt.imshow(abs_fail_mask, cmap="hot", aspect="auto")
        plt.xlabel("Model Dimension")
        plt.ylabel("Sequence Position")
        plt.colorbar()
        plt.title(f"Failed Test Heatmap (M={M}, N={N}, dtype={dtype})")
        plt.savefig(heatmap_path)
        plt.close()
        
        # Re-raise the exception
        raise e



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
                plot_name=f'resid_connect_{mode}_fp{8*dtype_bytes}',
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
        fn = lambda: resid_connect_fwd_triton(h, h_eigen, alpha)
    if provider == "naive":
        fn = lambda: resid_connect_fwd_naive(h, h_eigen, alpha)
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
    test_resid_connect(64, 64, torch.float32)
    test_resid_connect(2048, 768, torch.float16)
    test_resid_connect(2048, 8192, torch.float16)
    test_resid_connect(2048, 768, torch.float32)
    test_resid_connect(2048, 8192, torch.float32)

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='./benchmarks/', print_data=False)
    