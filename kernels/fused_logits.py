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

import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@torch.compile
def fused_logits_naive(A: torch.Tensor, B: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    assert A.shape[-1] == B.shape[-2]
    assert s.shape[0] == B.shape[-1]
    return (A @ B) * s


autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def fused_logits_fwd(
    a_ptr, b_ptr, c_ptr, s_ptr,
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    stride_s_N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, 
):
    PID = tl.program_id(axis=0) 
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group 
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_K = tl.arange(0, BLOCK_SIZE_K)
    a_offsets = offsets_M[:, None] * stride_a_M + offsets_K[None, :] * stride_a_K
    b_offsets = offsets_K[:, None] * stride_b_K + offsets_N[None, :] * stride_b_N

    # we'll iterate along the K dimension of both A and B to compute a single block of the C matrix
    accumulator = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offsets_K < K - k * BLOCK_SIZE_K
        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0).to(tl.float32) # shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0).to(tl.float32) # shape (BLOCK_SIZE_K, BLOCK_SIZE_N)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    # entry-wise multiply by scale factor
    s = tl.load(s_ptr + offsets_N * stride_s_N, mask=offsets_N < N).to(tl.float32)
    accumulator *= s[None, :]

    # write back the block of the output matrix C with masks
    c_offsets = stride_c_M * offsets_M[:, None] + stride_c_N * offsets_N[None, :]
    c_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) 
    tl.store(c_ptr + c_offsets, accumulator.to(c_ptr.type.element_ty), mask=c_mask) # shape (BLOCK_SIZE_M, BLOCK_SIZE_N)

@triton.jit
def fused_logits_dLdA(
    a_ptr, b_ptr, c_ptr, 
    dLda_ptr, dLdb_ptr, dLdc_ptr, 
    s_ptr, dLds_parts_ptr, dLds_ptr,
    locks_ptr,
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    stride_s_N,
    stride_dLds_parts_LG, stride_dLds_parts_N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, LOCK_GROUP_SIZE: tl.constexpr,
):
    pass

@triton.jit
def fused_logits_dLdB(
    a_ptr, b_ptr, c_ptr, 
    dLda_ptr, dLdb_ptr, dLdc_ptr, 
    s_ptr, dLds_parts_ptr, dLds_ptr,
    locks_ptr,
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    stride_s_N,
    stride_dLds_parts_LG, stride_dLds_parts_N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, LOCK_GROUP_SIZE: tl.constexpr,
):
    pass

@triton.jit
def fused_logits_dLds(
    dLds_parts_ptr, dLds_ptr,
    N, 
    stride_group, stride_N,
    LOCK_GROUP_SIZE: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, 
):
    pass

@triton_op("mylib::logits_fused", mutates_args={})
def fused_logits_triton(A: torch.Tensor, B: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    assert A.shape[-1] == B.shape[-2]
    assert s.shape[0] == B.shape[-1]
    assert s.ndim == 1
    (M, K), (_, N) = A.reshape(-1, A.shape[-1]).shape, B.reshape(-1, B.shape[-1]).shape
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    wrap_triton(fused_logits_fwd)[grid](
        A, B, C, s,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        s.stride(0)
    )
    return C

def fused_logits_triton_bwd(ctx, dLdC):
    A, B, s, C = ctx.saved_tensors
    M, N, K = ctx.M, ctx.N, ctx.k

    dLdA = torch.empty_like(A)
    dLdB = torch.empty_like(B)
    dLds = torch.empty_like(s)

    LOCK_GROUP_SIZE = 64 # total heuristic
    if N <= 8192: GROUP_SIZE = 96
    if N <= 4096: GROUP_SIZE = 128
    if N <= 1024: GROUP_SIZE = 256
    dLds_parts = torch.empty((LOCK_GROUP_SIZE, N), dtype=A.dtype, device=A.device)
    locks = torch.zeros(2 * LOCK_GROUP_SIZE, dtype=torch.int32, device=A.device)

    grid_dLdA = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(K, meta['BLOCK_SIZE_K']),)
    grid_dLdB = lambda meta: (triton.cdiv(K, meta['BLOCK_SIZE_K']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    grid_dLds = (triton.cdiv())

    wrap_triton(fused_logits_dLdA)[grid_dLdA](
        A, B, C, 
        dLdA, dLdB, dLdC, 
        s, dLds_parts, dLds,
        locks,
        M, N, K, 
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1), 
        s.stride(0),
        dLds_parts.stride(0), dLds_parts.stride(1)
    )
    wrap_triton(fused_logits_dLdB)[grid_dLdB](
        A, B, C, 
        dLdA, dLdB, dLdC, 
        s, dLds_parts, dLds,
        locks,
        M, N, K, 
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        C.stride(0), C.stride(1), 
        s.stride(0),
        dLds_parts.stride(0), dLds_parts.stride(1)
    )
    wrap_triton(fused_logits_dLds)[grid_dLds](
        dLds_parts, dLds,
        N, 
        dLds_parts.stride(0), dLds_parts.stride(1),
    )

    return

def fused_logits_triton_setup_ctx(ctx, inputs, output):
    A, B, s = inputs
    ctx.save_for_backward(A, B, s, output)
    (M, K), (_, N) = A.reshape(-1, A.shape[-1]).shape, B.reshape(-1, B.shape[-1]).shape
    ctx.M, ctx.N, ctx.K = M, N, K

fused_logits_triton.register_autograd(fused_logits_triton_bwd, setup_context=fused_logits_triton_setup_ctx)


def create_diff_heatmap(expected, actual, M, N, dtype, test_name, atol):
    """Create a heatmap visualization for gradient differences"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
    
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
    plt.title(f"Failed {test_name} Test Heatmap (M={M}, N={N}, dtype={dtype})")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Created error heatmap at {heatmap_path}")

def test(M, N, K, dtype, device=DEVICE, atol=1e-3, rtol=1e-3):
    A_naive = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    B_naive = torch.randn((K, N), dtype=dtype, device=device, requires_grad=True)
    s_naive = torch.randn((N,), dtype=dtype, device=device, requires_grad=True)

    A_triton = A_naive.clone().detach().requires_grad_(True) 
    B_triton = B_naive.clone().detach().requires_grad_(True)
    s_triton = s_naive.clone().detach().requires_grad_(True)

    C_naive = fused_logits_naive(A_naive, B_naive, s_naive)
    C_triton = fused_logits_triton(A_triton, B_triton, s_triton)

    import os
    test_name = f"C_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(C_naive, C_triton, atol=atol, rtol=rtol)
        print(f"✓ passed fwd test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed test (M={M}, N={N}, K={K}, dtype={dtype})")
        create_diff_heatmap(C_naive, C_triton, M, N, dtype, test_name, atol)
        raise e

    dLdC = torch.randn_like(C_naive)
    C_naive.backward(dLdC)
    C_triton.backward(dLdC)

    test_name = f"dLdA_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(A_naive.grad, A_triton.grad, atol=atol, rtol=rtol)
        print(f"✓ passed dLdA test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLdA test (M={M}, N={N}, K={K}, dtype={dtype})")
        create_diff_heatmap(A_naive.grad, A_triton.grad, M, N, dtype, f"fwd_M={M},N={N}_{dtype}", atol)
        raise e

    test_name = f"dLdB_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(B_naive.grad, B_triton.grad, atol=atol, rtol=rtol)
        print(f"✓ passed dLdB test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLdB test (M={M}, N={N}, K={K}, dtype={dtype})")
        create_diff_heatmap(B_naive.grad, B_triton.grad, M, N, dtype, f"fwd_M={M},N={N}_{dtype}", atol)
        raise e

    test_name = f"dLdB_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(s_naive.grad, s_triton.grad, atol=atol, rtol=rtol)
        print(f"✓ passed dLdB test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLds test (M={M}, N={N}, K={K}, dtype={dtype})")
        create_diff_heatmap(s_naive.grad, s_triton.grad, M, N, dtype, f"fwd_M={M},N={N}_{dtype}", atol)
        raise e

if __name__ == "__main__":
    test(128, 128, 128, torch.float32)
    test(128, 128, 128, torch.float16)
    test(16*1024, 2**15, 768, torch.float32)
    test(16*1024, 2**15, 768, torch.float16)