import math

import torch
#print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.library import triton_op, wrap_triton

# lets us cache a ton of different kernels when benchmarking
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64  # Increase from default of 8

# fixes issue w frankenstein custom ops not keeping autograd graph during benchmark
torch._functorch.config.donated_buffer=False

# takes advantage of hardware
torch.set_float32_matmul_precision('high')

import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@torch.compile
def fused_logits_naive(A: torch.Tensor, B: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    assert A.shape[-1] == B.shape[-2]
    assert s.shape[0] == B.shape[-1]
    return (A @ B) * s

@torch.compile
def fused_logits_naive_bwd(A: torch.Tensor, B: torch.Tensor, s: torch.Tensor, dLdD) -> torch.Tensor:
    C = A @ B
    dLdC = dLdD * s
    dLds = torch.sum(dLdD * C, axis=0)
    dLdA = dLdC @ B.T
    dLdB = A.T @ dLdC
    return dLdA, dLdB, dLds


# for saving the fwd pass' autotuned BLOCK_SIZE_M for use later in the bwd pass
_block_size_m = None

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
    a_ptr, b_ptr, d_ptr, s_ptr,
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_d_M, stride_d_N, 
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
    mask_M = offsets_M < M
    mask_N = offsets_N < N

    # we'll iterate along the K dimension of both A and B to compute a single block of the C matrix
    c = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_K = offsets_K < K - k * BLOCK_SIZE_K
        a_mask = mask_M[:, None] & mask_K[None, :]
        b_mask = mask_K[:, None] & mask_N[None, :]
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0).to(tl.float32) # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0).to(tl.float32) # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        c = tl.dot(a, b, acc=c) # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    # entry-wise multiply by scale factor
    s = tl.load(s_ptr + offsets_N * stride_s_N, mask=mask_N, other=1.).to(tl.float32)
    d = c * s[None, :]

    # write back the block of the output matrix C with masks
    d_offsets = stride_d_M * offsets_M[:, None] + stride_d_N * offsets_N[None, :]
    d_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) 
    tl.store(d_ptr + d_offsets, d.to(d_ptr.type.element_ty), mask=d_mask) # shape (BLOCK_SIZE_M, BLOCK_SIZE_N)

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_N': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 256, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def fused_logits_dLdA(
    a_ptr, b_ptr, d_ptr, 
    dLda_ptr, dLdb_ptr, dLdd_ptr, 
    s_ptr, dLds_parts_ptr, dLds_ptr,
    locks_ptr,
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_d_M, stride_d_N, 
    stride_s_N,
    stride_dLds_parts_LG, stride_dLds_parts_N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, #LOCK_GROUP_SIZE: tl.constexpr,
):
    PID = tl.program_id(axis=0) 
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_PID_along_K = tl.cdiv(K, BLOCK_SIZE_K)
    num_PID_in_group = GROUP_SIZE * num_PID_along_K
    group_id = PID // num_PID_in_group 
    first_PID_in_group_along_M = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M, GROUP_SIZE) 
    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_K = (PID % num_PID_in_group) // group_size_adj

    offsets_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_K = PID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_N = tl.arange(0, BLOCK_SIZE_N)
    d_offsets = offsets_M[:, None] * stride_d_M + offsets_N[None, :] * stride_d_N
    bT_offsets = offsets_N[:, None] * stride_b_N + offsets_K[None, :] * stride_b_K
    mask_M = offsets_M < M
    mask_K = offsets_K < K

    dLda = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_K], dtype=tl.float32)
    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        mask_N = offsets_N < N
        d_mask = mask_M[:, None] & mask_N[None, :]
        bT_mask = mask_N[:, None] & mask_K[None, :]
        dLdd = tl.load(dLdd_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32) # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        bT = tl.load(b_ptr + bT_offsets, mask=bT_mask, other=0.0).to(tl.float32) # (BLOCK_SIZE_N, BLOCK_SIZE_K)
        s = tl.load(s_ptr + offsets_N, mask=mask_N, other=1.) # (BLOCK_SIZE_N)
        dLdc = dLdd * s[None, :] # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        dLda = tl.dot(dLdc, bT, acc=dLda) # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        offsets_N += BLOCK_SIZE_N
        d_offsets += BLOCK_SIZE_N * stride_d_N
        bT_offsets += BLOCK_SIZE_N * stride_b_N

    a_offsets = stride_a_M * offsets_M[:, None] + stride_a_K * offsets_K[None, :]
    a_mask = (offsets_M[:, None] < M) & (offsets_K[None, :] < K) 
    tl.store(dLda_ptr + a_offsets, dLda.to(dLda_ptr.type.element_ty), mask=a_mask) # shape (BLOCK_SIZE_M, BLOCK_SIZE_K)

autotune_configs = [
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_M': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def fused_logits_dLdB(
    a_ptr, b_ptr, d_ptr, 
    dLda_ptr, dLdb_ptr, dLdd_ptr, 
    s_ptr, dLds_parts_ptr, dLds_ptr,
    locks_ptr,
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_d_M, stride_d_N, 
    stride_s_N,
    stride_dLds_parts_LG, stride_dLds_parts_N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, #LOCK_GROUP_SIZE: tl.constexpr,
):
    PID = tl.program_id(axis=0) 
    num_PID_along_K = tl.cdiv(K, BLOCK_SIZE_K)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N
    group_id = PID // num_PID_in_group 
    first_PID_in_group_along_K = group_id * GROUP_SIZE 
    group_size_adj = min(num_PID_along_K - first_PID_in_group_along_K, GROUP_SIZE) 
    PID_K = first_PID_in_group_along_K + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offsets_K = PID_K * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offsets_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offsets_M = tl.arange(0, BLOCK_SIZE_M)
    aT_offsets = offsets_K[:, None] * stride_a_K + offsets_M[None, :] * stride_a_M
    d_offsets = offsets_M[:, None] * stride_d_M + offsets_N[None, :] * stride_d_N
    mask_K = offsets_K < K
    mask_N = offsets_N < N

    s = tl.load(s_ptr + offsets_N * stride_s_N, mask=mask_N, other=1.) # (BLOCK_SIZE_N)
    dLdb = tl.zeros([BLOCK_SIZE_K, BLOCK_SIZE_N], dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BLOCK_SIZE_M)):
        mask_M = offsets_M < M - m * BLOCK_SIZE_M
        aT_mask = mask_K[:, None] & mask_M[None, :]
        d_mask = mask_M[:, None] & mask_N[None, :]
        aT = tl.load(a_ptr + aT_offsets, mask=aT_mask, other=0.0).to(tl.float32) # (BLOCK_SIZE_K, BLOCK_SIZE_M)
        dLdd = tl.load(dLdd_ptr + d_offsets, mask=d_mask, other=0.0).to(tl.float32) # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        dLdc = dLdd * s[None, :] # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        dLdb = tl.dot(aT, dLdc, acc=dLdb) # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        aT_offsets += BLOCK_SIZE_M * stride_a_M
        d_offsets += BLOCK_SIZE_M * stride_d_M

    b_offsets = stride_b_K * offsets_K[:, None] + stride_b_N * offsets_N[None, :]
    b_mask = (offsets_K[:, None] < K) & (offsets_N[None, :] < N) 
    tl.store(dLdb_ptr + b_offsets, dLdb.to(dLdb_ptr.type.element_ty), mask=b_mask) # shape (BLOCK_SIZE_K, BLOCK_SIZE_N)

autotune_configs = [
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def fused_logits_dLds_p1(
    a_ptr, b_ptr, d_ptr, 
    dLda_ptr, dLdb_ptr, dLdd_ptr, 
    s_ptr, dLds_parts_ptr, dLds_ptr,
    locks_ptr,
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_d_M, stride_d_N, 
    stride_s_N,
    stride_dLds_parts_ROW, stride_dLds_parts_N,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr, #LOCK_GROUP_SIZE: tl.constexpr,
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
    mask_M = offsets_M < M
    mask_N = offsets_N < N

    c = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask_K = offsets_K < K - k * BLOCK_SIZE_K
        a_mask = mask_M[:, None] & mask_K[None, :]
        b_mask = mask_K[:, None] & mask_N[None, :]
        a = tl.load(a_ptr + a_offsets, mask=a_mask, other=0.0).to(tl.float32) # (BLOCK_SIZE_M, BLOCK_SIZE_K)
        b = tl.load(b_ptr + b_offsets, mask=b_mask, other=0.0).to(tl.float32) # (BLOCK_SIZE_K, BLOCK_SIZE_N)
        c = tl.dot(a, b, acc=c) # (BLOCK_SIZE_M, BLOCK_SIZE_N)
        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    # compute dLds contribution
    d_offsets = stride_d_M * offsets_M[:, None] + stride_d_N * offsets_N[None, :]
    d_mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N) 
    dLdd = tl.load(dLdd_ptr + d_offsets, mask=d_mask, other=0.).to(tl.float32)
    dLds_part = tl.sum(dLdd * c, axis=0) # (BLOCK_SIZE_N)

    # store it
    dLds_parts_ptr += PID_M * stride_dLds_parts_ROW
    tl.store(
        dLds_parts_ptr + offsets_N * stride_dLds_parts_N, 
        dLds_part.to(dLds_parts_ptr.type.element_ty), 
        mask=mask_N
    )

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_ROWS": BLOCK_SIZE_ROWS, "BLOCK_SIZE_COLS": BLOCK_SIZE_COLS}
        )
        for BLOCK_SIZE_ROWS in [16, 32, 64, 128]
        for BLOCK_SIZE_COLS in [16, 32, 64, 128]
    ],
    key=["ROWS", "COLS"],
)
@triton.jit
def fused_logits_dLds_p2(
    dLds_parts_ptr, dLds_ptr,
    stride_dLds_parts_ROW, stride_dLds_parts_COL,
    stride_dLds_COL,
    ROWS, COLS,
    BLOCK_SIZE_ROWS: tl.constexpr, BLOCK_SIZE_COLS: tl.constexpr
):
    pid = tl.program_id(0)
    offsets_ROW = tl.arange(0, BLOCK_SIZE_ROWS)
    offsets_COL = pid * BLOCK_SIZE_COLS + tl.arange(0, BLOCK_SIZE_COLS)
    mask_COL = offsets_COL < COLS
    dLds_offsets = offsets_ROW[:, None] * stride_dLds_parts_ROW + offsets_COL[None, :] * stride_dLds_parts_COL

    acc = tl.zeros([BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS], dtype=tl.float32)
    for row in range(0, ROWS, BLOCK_SIZE_ROWS):
        mask = (offsets_ROW[:, None] < ROWS) & mask_COL[None, :]
        dLds_part = tl.load(dLds_parts_ptr + dLds_offsets, mask=mask, other=0.)
        acc += dLds_part
        dLds_offsets += BLOCK_SIZE_ROWS
    dLds = tl.sum(acc, axis=0)

    tl.store(dLds_ptr + offsets_COL * stride_dLds_COL, dLds, mask=mask_COL)

@triton_op("mylib::logits_fused", mutates_args={})
def fused_logits_triton(A: torch.Tensor, B: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    A @ B = C
    C * s = D
    """
    global _block_size_m
    
    assert A.shape[-1] == B.shape[-2]
    assert s.shape[0] == B.shape[-1]
    assert s.ndim == 1
    (M, K), (_, N) = A.reshape(-1, A.shape[-1]).shape, B.reshape(-1, B.shape[-1]).shape
    D = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    wrap_triton(fused_logits_fwd)[grid](
        A, B, D, s,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        D.stride(0), D.stride(1),
        s.stride(0)
    )
    # Save config for backward pass
    best_config = getattr(fused_logits_fwd, "best_config", None)
    _block_size_m = best_config.kwargs['BLOCK_SIZE_M']
    return D

def fused_logits_triton_bwd(ctx, dLdD):
    global _block_size_m
    A, B, s, D = ctx.saved_tensors
    M, N, K = ctx.M, ctx.N, ctx.K

    dLdA = torch.empty_like(A)
    dLdB = torch.empty_like(B)
    dLds = torch.zeros_like(s)

    # Use config from forward pass if available
    if _block_size_m is not None:
        max_blocks_m = triton.cdiv(M, _block_size_m)
    else:
        # Fallback to conservative size
        _block_size_m = 32
        max_blocks_m = triton.cdiv(M, 32)
    
    dLds_parts = torch.empty((max_blocks_m, N), dtype=A.dtype, device=A.device)
    locks = torch.zeros(2 * max_blocks_m, dtype=torch.int32, device=A.device)

    grid_dLdA = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(K, meta['BLOCK_SIZE_K']),)
    wrap_triton(fused_logits_dLdA)[grid_dLdA](
        A, B, D, 
        dLdA, dLdB, dLdD, 
        s, dLds_parts, dLds,
        locks,
        M, N, K, 
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        D.stride(0), D.stride(1), 
        s.stride(0),
        dLds_parts.stride(0), dLds_parts.stride(1)
    )

    grid_dLdB = lambda meta: (triton.cdiv(K, meta['BLOCK_SIZE_K']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    wrap_triton(fused_logits_dLdB)[grid_dLdB](
        A, B, D, 
        dLdA, dLdB, dLdD, 
        s, dLds_parts, dLds,
        locks,
        M, N, K,
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        D.stride(0), D.stride(1), 
        s.stride(0),
        dLds_parts.stride(0), dLds_parts.stride(1)
    )
    
    grid_dLds_p1 = lambda meta: (triton.cdiv(M, _block_size_m) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    wrap_triton(fused_logits_dLds_p1)[grid_dLds_p1](
        A, B, D, 
        dLdA, dLdB, dLdD, 
        s, dLds_parts, dLds,
        locks,
        M, N, K,
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        D.stride(0), D.stride(1), 
        s.stride(0),
        dLds_parts.stride(0), dLds_parts.stride(1),
        BLOCK_SIZE_M = _block_size_m
    )
    grid_dLds_p2 = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_ROWS']),)
    wrap_triton(fused_logits_dLds_p2)[grid_dLds_p2](
        dLds_parts, dLds,
        dLds_parts.stride(0), dLds_parts.stride(1),
        dLds.stride(0),
        ROWS=max_blocks_m, COLS=N,
    )

    return dLdA, dLdB, dLds

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

def test(M, N, K, dtype, device=DEVICE, atol=2e-2, rtol=0):
    A_torch = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    B_torch = torch.randn((K, N), dtype=dtype, device=device, requires_grad=True)
    s_torch = torch.randn((N,), dtype=dtype, device=device, requires_grad=True)

    A_triton = A_torch.clone().detach().requires_grad_(True) 
    B_triton = B_torch.clone().detach().requires_grad_(True)
    s_triton = s_torch.clone().detach().requires_grad_(True)

    D_torch = fused_logits_naive(A_torch, B_torch, s_torch)
    D_triton = fused_logits_triton(A_triton, B_triton, s_triton)

    ### fwd test
    import os
    test_name = f"C_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(
            D_torch / D_torch.std(), # scale all assertions to variance of 1 so that tolerances make sense
            D_triton / D_triton.std(), 
            atol=atol, rtol=rtol)
        print(f"✓ passed fwd test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed fwd test (M={M}, N={N}, K={K}, dtype={dtype})")
        print(D_torch)
        print(D_triton)
        create_diff_heatmap(D_torch, D_triton, M, N, dtype, test_name, atol)
        raise e
    
    dLdD = torch.randn_like(D_torch)
    D_torch.backward(dLdD)
    dLdA_naive, dLdB_naive, dLds_naive = fused_logits_naive_bwd(A_torch, B_torch, s_torch, dLdD)

    ### naive implementation bwd tests
    test_name = f"dLdA_naive_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(
            A_torch.grad / A_torch.grad.std(), 
            dLdA_naive / dLdA_naive.std(), 
            atol=atol, rtol=rtol)
        print(f"✓ passed dLdA_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLdA_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        create_diff_heatmap(A_torch.grad, dLdA_naive, M, N, dtype, f"fwd_M={M},N={N}_{dtype}", atol)
        raise e

    test_name = f"dLdB_naive_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(
            B_torch.grad / B_torch.grad.std(), 
            dLdB_naive / dLdB_naive.std(), 
            atol=atol, rtol=rtol)
        print(f"✓ passed dLdB_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLdB_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        create_diff_heatmap(B_torch.grad, dLdB_naive, M, N, dtype, f"fwd_M={M},N={N}_{dtype}", atol)
        raise e
    
    test_name = f"dLds_naive_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(
            s_torch.grad / s_torch.grad.std(), 
            dLds_naive / dLds_naive.std(), 
            atol=atol, rtol=rtol)
        print(f"✓ passed dLds_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLds_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        create_diff_heatmap(s_torch.grad.unsqueeze(0), dLds_naive.unsqueeze(0), 1, N, dtype, f"fwd_N={N}_{dtype}", atol)
        raise e

    
    D_triton.backward(dLdD)

    ### triton implementation bwd tests
    test_name = f"dLdA_triton_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(
            A_torch.grad / A_torch.grad.std(), 
            A_triton.grad / A_triton.grad.std(), 
            atol=atol, rtol=rtol)
        print(f"✓ passed dLdA_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLdA_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        print(A_torch.grad)
        print(A_triton.grad)
        create_diff_heatmap(A_torch.grad, A_triton.grad, M, N, dtype, f"fwd_M={M},N={N}_{dtype}", atol)
        raise e

    test_name = f"dLdB_triton_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(
            B_torch.grad / B_torch.grad.std(), 
            B_triton.grad / B_triton.grad.std(), 
            atol=atol, rtol=rtol)
        print(f"✓ passed dLdB_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLdB_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        print(B_torch.grad)
        print(B_torch.grad)
        create_diff_heatmap(B_torch.grad, B_triton.grad, M, N, dtype, f"fwd_M={M},N={N}_{dtype}", atol)
        raise e

    test_name = f"dLds_triton_M={M},N={N}_{dtype}"
    try:
        torch.testing.assert_close(
            s_torch.grad / s_torch.grad.std(), 
            s_triton.grad / s_triton.grad.std(), 
            atol=atol, rtol=rtol)
        print(f"✓ passed dLds_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./fused_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLds_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        print(s_torch.grad)
        print(s_triton.grad)
        create_diff_heatmap(s_torch.grad.unsqueeze(0), s_triton.grad.unsqueeze(0), 1, N, dtype, f"fwd_N={N}_{dtype}", atol)
        raise e

if __name__ == "__main__":
    test(2, 2, 2, torch.float32)
    test(2, 2, 2, torch.float16)
    test(32, 32, 32, torch.float32)
    test(32, 32, 32, torch.float16)
    test(128, 128, 128, torch.float32)
    test(128, 128, 128, torch.float16)
    test(512, 512, 512, torch.float32)
    test(512, 512, 512, torch.float16)
    # the errors at this point are so rare (0.4%) and relatively small (atol ~1e-1) that idc
    # likely just bc I'm using a 30 series GPU which Triton always gets big floating point errors on accumulation
    test(8*1024, 2**14, 384, torch.float32) 
    test(8*1024, 2**14, 384, torch.float16)