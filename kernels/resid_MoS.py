import math

import torch
#print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.library import triton_op, wrap_triton

# lets us cache a ton of different kernels when benchmarking
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # Increase from default of 8

# fixes issue w frankenstein not keeping autograd graph during benchmark
torch._functorch.config.donated_buffer=False

#torch.set_float32_matmul_precision('high')

import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

#@torch.compile
def resid_MoS_naive(h_prev, h_eigen, alpha):
    """
    h_prev & h_eigen: shape (B*N, D) 
    alpha: shape (D, R) where R << D
    """
    assert h_prev.shape[-1] == h_eigen.shape[-1] and h_prev.shape[-1] == alpha.shape[0]
    h_eigen_normed = h_eigen / torch.norm(h_eigen, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    h_grad = h_eigen_normed - h_prev

    probs = F.softmax(h_prev @ alpha, dim=-1) #(B*N, D) @ (D, R) -> (B*N, R)
    scales = probs.unsqueeze(1) * alpha.unsqueeze(0) # (B*N, 1, R) * (1, D, R) -> (B*N, D, R)
    scale = torch.sum(scales, dim=-1) # (B*N, D)

    h_adj = h_prev + scale * h_grad
    h_out = h_adj / torch.norm(h_adj, p=2, dim=-1, keepdim=True).clamp(min=1e-12)
    return h_out


#@torch.compile
def resid_MoS_backward_naive(h_prev, h_eigen, alpha, grad_output):
    # pass
    return h_prev, h_eigen, alpha



@triton.jit
def fwd_kernel(
    h_prev_ptr, h_eigen_ptr, h_out_ptr,
    alpha_ptr,
    stride_h_M, stride_h_D,
    stride_alpha_D, stride_alpha_R,
    D: tl.constexpr, R: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr, BLOCK_SIZE_R: tl.constexpr,
):
    eps: tl.constexpr = 1e-12
    inf: tl.constexpr = 1e12

    row = tl.program_id(0)
    h_prev_ptr += row * stride_h_M
    h_eigen_ptr += row * stride_h_M
    h_out_ptr += row * stride_h_M

    offsets_D = tl.arange(0, BLOCK_SIZE_D)
    offsets_R = tl.arange(0, BLOCK_SIZE_R)
    mask_D = offsets_D < D

    h_prev = tl.load(h_prev_ptr + offsets_D * stride_h_D, mask=mask_D, other=0.).to(tl.float32)
    h_eigen = tl.load(h_eigen_ptr + offsets_D * stride_h_D, mask=mask_D, other=0.).to(tl.float32)
    alpha_offsets = offsets_D[:, None] * stride_alpha_D + offsets_R[None, :] * stride_alpha_R
    alpha_mask = mask_D[:, None] & (offsets_R < R)[None, :]
    alpha = tl.load(alpha_ptr + alpha_offsets, mask=alpha_mask, other=0.).to(tl.float32)

    # compute L_2 norm & normalize h_eigen
    h_eigen_normed = h_eigen / tl.clamp(tl.sqrt(tl.sum(h_eigen * h_eigen, axis=0)), eps, inf)

    # our residual "gradient"
    h_grad = h_eigen_normed - h_prev

    # find amount of movement along hypersphere residual connection
    logits = tl.sum(h_prev[:, None] * alpha, axis=0) # (BLOCK_SIZE_R)
    logits_safe = logits - tl.max(logits) # (1)
    probs = tl.exp(logits_safe) / tl.sum(tl.exp(logits_safe)) # (BLOCK_SIZE_R)
    scales = alpha * probs # (BLOCK_SIZE_D, BLOCK_SIZE_R) * (BLOCK_SIZE_R) = (BLOCK_SIZE_D, BLOCK_SIZE_R)
    scale = tl.sum(scales, axis=1) # (BLOCK_SIZE_D)
        # akin to our learning rate 'eta' in actual SGD
    
    # residual connection, aka our movement along the hypersphere
    resid = h_prev + scale * h_grad

    # normalize & store output
    h_out = resid / tl.clamp(tl.sqrt(tl.sum(resid * resid, axis=0)), eps, inf)
    tl.store(h_out_ptr + offsets_D * stride_h_D, h_out.to(h_out_ptr.type.element_ty), mask=mask_D)

# used to derive our block size
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
sram_per_sm = properties["max_shared_mem"]

@triton_op("mylib::resid_MoS_triton", mutates_args={})
def resid_MoS_triton(
    h_prev: torch.Tensor, 
    h_eigen: torch.Tensor, 
    alpha: torch.Tensor,
) -> torch.Tensor:
    assert h_prev.shape == h_eigen.shape
    assert h_prev.stride() == h_eigen.stride()
    assert h_prev.shape[-1] == h_eigen.shape[-1] and h_prev.shape[-1] == alpha.shape[0]
    M, D = h_prev.reshape(-1, h_prev.shape[-1]).shape
        # M = B*N
    R = alpha.shape[1]

    # this kernel is designed for normalizing vectors that fit in SRAM
    block_size_R = triton.next_power_of_2(R)
    block_size_D = triton.next_power_of_2(D)
    max_entries = (sram_per_sm // h_prev.element_size()) // block_size_R
    assert max_entries >= block_size_D, f"resid MoS kernel only supports vectors up to {max_entries}"
    # NOTE: this restriction is a good bit smaller than that of our other kernels

    # pre-allocate output
    h_out = torch.empty_like(h_prev)

    wrap_triton(fwd_kernel)[(M,)](
        h_prev, h_eigen, h_out, 
        alpha,
        h_prev.stride(-2), h_prev.stride(-1),
        alpha.stride(0), alpha.stride(1),
        D, R,
        BLOCK_SIZE_D=block_size_D, BLOCK_SIZE_R = block_size_R
    )

    return h_out



def create_diff_heatmap(expected, actual, M, N, dtype, heatmap_path, atol):
    """Create a heatmap visualization for gradient differences"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Convert to numpy arrays
    expected_np = expected.detach().cpu().numpy()
    actual_np = actual.detach().cpu().numpy()
    
    # Compute differences and masks
    abs_diff = np.abs(expected_np - actual_np)
    abs_fail_mask = (abs_diff > atol).astype(np.int32)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(abs_fail_mask, cmap="hot", aspect="auto")
    plt.xlabel("Model Dimension")
    plt.ylabel("Sequence Position")
    plt.colorbar()
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Created error heatmap at {heatmap_path}")

def test(M, D, R, dtype, device=DEVICE, atol=1e-3, rtol=0):
    h_prev = torch.randn((M, D), dtype=dtype, device=device, requires_grad=True)
    h_eigen = torch.randn((M, D), dtype=dtype, device=device, requires_grad=True)
    alpha = torch.randn((D, R), dtype=dtype, device=device, requires_grad=True)

    h_prev_triton = h_prev.clone().detach().requires_grad_(True)
    h_eigen_triton = h_eigen.clone().detach().requires_grad_(True)
    alpha_triton = alpha.clone().detach().requires_grad_(True)

    h_out_torch = resid_MoS_naive(h_prev, h_eigen, alpha)
    h_out_triton = resid_MoS_triton(h_prev_triton, h_eigen_triton, alpha_triton)

    ### fwd test
    import os
    test_name = f"h_out_M={M},D={D},R={R}_{dtype}"
    heatmap_path = f'./resid_MoS_{test_name}_heatmap.png'
    try:
        std = h_out_torch.std() # scale all assertions to variance of 1 so that tolerances make sense
        torch.testing.assert_close(h_out_torch / std, h_out_triton / std, atol=atol, rtol=rtol)
        print(f"✓ passed fwd test {test_name}")
        #print(h_out_torch)
        #print(h_out_triton)
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed fwd test {test_name}")
        print(h_out_torch)
        print(h_out_triton)
        create_diff_heatmap(h_out_torch, h_out_triton, M, D, dtype, heatmap_path, atol)
        raise e
    """
    # Gradient from upstream
    dLdh_out = torch.randn((M, D), dtype=dtype, device=device)
    
    # Method 1: torch autograd
    h_out_torch.backward(dLdh_out)
    dLdh_prev = h_prev.grad
    dLdh_eigen = h_eigen.grad
    dLdalpha = alpha.grad
    
    # Method 2: Manual backward
    dLdh_prev_manual, dLdh_eigen_manual, dLdalpha_manual = resid_MoS_backward_naive(
        h_prev, h_eigen, alpha, dLdh_out
    )
    
    # Compare autograd vs manual backward
    test_name = f"dLdh_prev_M={M},D={D},R={R}_{dtype}"
    heatmap_path = f'./resid_MoS_{test_name}_heatmap.png'
    try:
        torch.testing.assert_close(dLdh_prev, dLdh_prev_manual, atol=atol, rtol=rtol)
        print(f"✓ passed backward test on h_prev ({test_name})")
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed backward test on h_prev ({test_name})")
        create_diff_heatmap(dLdh_prev, dLdh_prev_manual, M, D, dtype, heatmap_path, atol)
        raise e
        
    test_name = f"dLdh_eigen_M={M},D={D},R={R}_{dtype}"
    heatmap_path = f'./resid_MoS_{test_name}_heatmap.png'
    try:
        torch.testing.assert_close(dLdh_eigen, dLdh_eigen_manual, atol=atol, rtol=rtol)
        print(f"✓ passed backward test on h_eigen ({test_name})")
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed backward test on h_eigen ({test_name})")
        create_diff_heatmap(dLdh_eigen, dLdh_eigen_manual, M, D, dtype, heatmap_path, atol)
        raise e
        
    test_name = f"M={M},D={D},R={R}_{dtype}"
    try:
        torch.testing.assert_close(dLdalpha, dLdalpha_manual, atol=atol, rtol=rtol)
        print(f"✓ passed backward test on alpha ({test_name})")
    except AssertionError as e:
        print(f"✗ failed backward test on alpha ({test_name})")
        # Alpha has different dimensions, so we can't use the heatmap function directly
        print("Autograd alpha grad:", dLdalpha)
        print("Manual alpha grad:", dLdalpha_manual)
        raise e
        
    print(f"✓ passed all backward tests ({test_name})")
    """

def previous_power_two(n):
    return int(math.log(n, 2))

configs = []
for dtype_bytes in [2, 4]:
    configs.append(
        triton.testing.Benchmark(
            x_names=['D'],
            x_vals=[2 ** i for i in range(7, 11)], 
            line_arg='provider',
            line_vals=['triton', 'naive'],
            line_names=['Triton', 'naive + torch.compile'],
            styles=[('blue', '-'), ('green', '-')],
            ylabel='GB/s',
            plot_name=f'resid_MoS_fwd_fp{8*dtype_bytes}',
            args={"mode": "fwd", "dtype_bytes": dtype_bytes, "R": 11}, 
        ))
    
@triton.testing.perf_report(configs)
def benchmark(D, provider, mode, dtype_bytes, R, device=DEVICE):
    """
    Benchmarks the performance of resid_MoS implementations.
    
    Args:
        D: Dimension size
        provider: Implementation to use ('triton' or 'naive')
        mode: 'fwd' for forward pass (backward not implemented yet)
        dtype_bytes: Number of bytes per element (2 for fp16, 4 for fp32)
        R: Number of mixture components
        device: Device to run on
    
    Returns:
        Performance in GB/s
    """
    # create data
    assert dtype_bytes in [2, 4]
    dtype = torch.float16 if dtype_bytes == 2 else torch.float32
    # Use a reasonable batch size for benchmarking
    M = 16 * 1024
    h_prev = torch.randn((M, D), dtype=dtype, device=device, requires_grad=True)
    h_eigen = torch.randn((M, D), dtype=dtype, device=device, requires_grad=True)
    alpha = torch.randn((D, R), dtype=dtype, device=device, requires_grad=True)

    # confidence interval for testing
    quantiles = [0.5, 0.001, 0.999]

    if provider == "triton":
        fn = lambda: resid_MoS_triton(h_prev, h_eigen, alpha)
    elif provider == "naive":
        fn = lambda: resid_MoS_naive(h_prev, h_eigen, alpha)
    elif mode == "bwd":
        # Placeholder for backward pass benchmarking
        # Will be implemented when backward pass is added
        print("Backward pass not implemented yet")
        return 0, 0, 0
    
    # benchmark
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    
    # Calculate bytes processed for forward pass
    # h_prev (M,D) + h_eigen (M,D) + alpha (D,R) + output (M,D)
    bytes_per_iter = h_prev.numel() * h_prev.element_size() + \
                     h_eigen.numel() * h_eigen.element_size() + \
                     alpha.numel() * alpha.element_size() + \
                     h_prev.numel() * h_prev.element_size()
    
    gbps = lambda ms: bytes_per_iter * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

def measure_memory_usage(func, M, D, R, dtype):
    """Measure peak memory usage of a function"""
    # Reset the memory stats on the current GPU device
    torch.cuda.reset_peak_memory_stats()
    # Run the function
    func(M, D, R, dtype)
    # Synchronize to ensure all asynchronous ops complete
    torch.cuda.synchronize()
    # Get the maximum memory allocated during the operation
    peak_memory = torch.cuda.max_memory_allocated()
    return peak_memory

def run_custom_kernel(M, D, R, dtype):
    """Run the Triton implementation of resid_MoS"""
    h_prev = torch.randn((M, D), dtype=dtype, device=DEVICE, requires_grad=True)
    h_eigen = torch.randn((M, D), dtype=dtype, device=DEVICE, requires_grad=True)
    alpha = torch.randn((D, R), dtype=dtype, device=DEVICE, requires_grad=True)
    _ = resid_MoS_triton(h_prev, h_eigen, alpha)
    # When backward is implemented, uncomment:
    # _.backward(torch.randn_like(_))

def run_pytorch_operator(M, D, R, dtype):
    """Run the naive PyTorch implementation of resid_MoS"""
    h_prev = torch.randn((M, D), dtype=dtype, device=DEVICE, requires_grad=True)
    h_eigen = torch.randn((M, D), dtype=dtype, device=DEVICE, requires_grad=True)
    alpha = torch.randn((D, R), dtype=dtype, device=DEVICE, requires_grad=True)
    _ = resid_MoS_naive(h_prev, h_eigen, alpha)
    # When backward is implemented, uncomment:
    # _.backward(torch.randn_like(_))


if __name__ == "__main__":
    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        # Run performance benchmark first for autotuning
        benchmark.run(save_path='./benchmarks/', print_data=True)
        
        # Then measure memory usage
        print("\nMeasuring memory consumption:")
        print("================================")
        for dtype in [torch.float16, torch.float32]:
            dtype_name = "FP16" if dtype == torch.float16 else "FP32"
            for D in [64, 128, 256, 768, 1024]:
                for R in [6, 11]:
                    M = 8*1024  # Batch size
                    
                    # Measure memory usage for both implementations
                    triton_peak = measure_memory_usage(run_custom_kernel, M, D, R, dtype)
                    naive_peak = measure_memory_usage(run_pytorch_operator, M, D, R, dtype)
                    
                    # Print results
                    print(f"M={M}, D={D}, R={R}, {dtype_name}:")
                    print(f"  Triton: {triton_peak/1024**2:.2f} MB")
                    print(f"  Naive:  {naive_peak/1024**2:.2f} MB")
                    print(f"  Memory saved: {100*(naive_peak-triton_peak)/naive_peak:.2f}%")
                    print("--------------------------")
    else:
        test(64, 48, 6, torch.float32)
        test(1024, 384, 32, torch.float32)
        test(1024, 384, 32, torch.float16)
        test(1024, 384, 11, torch.float32)
        test(1024, 384, 11, torch.float16)