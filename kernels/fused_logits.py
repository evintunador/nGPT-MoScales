import math

import torch
#print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
from torch.library import triton_op, wrap_triton

# lets us cache a ton of different kernels when benchmarking
import torch._dynamo
torch._dynamo.config.cache_size_limit = 256  # Increase from default of 8

# fixes issue w frankenstein custom ops not keeping autograd graph during benchmark
torch._functorch.config.donated_buffer=False

# takes advantage of hardware
torch.set_float32_matmul_precision('high')

import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@torch.compile
def scaled_logits_naive(A: torch.Tensor, B: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    assert A.shape[-1] == B.shape[-2]
    assert s.shape[0] == B.shape[-1]
    return (A @ B) * s

@torch.compile
def scaled_logits_naive_bwd(A: torch.Tensor, B: torch.Tensor, s: torch.Tensor, dLdD) -> torch.Tensor:
    C = A @ B
    dLdC = dLdD * s
    dLds = torch.sum(dLdD * C, axis=0)
    dLdA = dLdC @ B.T
    dLdB = A.T @ dLdC
    return dLdA, dLdB, dLds


# for saving the fwd pass' autotuned block sizes for use later in the bwd pass
_block_size_m = None
_block_size_n = None

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
def scaled_logits_fwd(
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
def scaled_logits_dLdA(
    a_ptr, b_ptr, d_ptr, 
    dLda_ptr, dLdb_ptr, dLdd_ptr, 
    s_ptr,
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_d_M, stride_d_N, 
    stride_s_N,
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
def scaled_logits_dLdB(
    a_ptr, b_ptr, d_ptr, 
    dLda_ptr, dLdb_ptr, dLdd_ptr, 
    s_ptr,
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_d_M, stride_d_N, 
    stride_s_N,
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


@triton.autotune([
        triton.Config(
            {"BLOCK_SIZE_K": BLOCK_SIZE_K, "GROUP_SIZE": 8},
            num_stages=ns, num_warps=nw)
        for BLOCK_SIZE_K in [16, 32, 64]
        for ns in [2,3,5,7]
        for nw in [2, 4, 8, 16]],
    key=["K"],)
@triton.jit
def scaled_logits_dLds_p1(
    a_ptr, b_ptr, d_ptr, 
    dLda_ptr, dLdb_ptr, dLdd_ptr, 
    s_ptr, dLds_parts_ptr, dLds_ptr, debug_ptr, debug2_ptr, debuglocks_ptr,
    M, N, K, 
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_d_M, stride_d_N, 
    stride_s_N,
    stride_dLds_parts_M, stride_dLds_parts_N,
    locks_ptr, num_locks_M, stride_locks_M, stride_locks_N,
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
    dLds_part = tl.sum(dLdd * c, axis=0).to(tl.float32) # (BLOCK_SIZE_N)

    """
    # naive implementation with no locks; equivalent to lock_group_size = 1
    dLds_parts_ptr += PID_M * stride_dLds_parts_M
    tl.store(
        dLds_parts_ptr + offsets_N * stride_dLds_parts_N, 
        dLds_part.to(dLds_parts_ptr.type.element_ty), 
        mask=mask_N
    )
    """
    # add contribution to lock group
    lock_id_M = PID_M  % num_locks_M
    lock_id_N = PID_N
    locks_ptr += lock_id_M * stride_locks_M + lock_id_N * stride_locks_N
    count_ptr = locks_ptr + num_locks_M * stride_locks_M
    dLds_parts_ptr += lock_id_M * stride_dLds_parts_M + lock_id_N * BLOCK_SIZE_N * stride_dLds_parts_N
    offsets_N = tl.arange(0, BLOCK_SIZE_N) * stride_dLds_parts_N

    # extra for debugging
    debuglocks_ptr += lock_id_M * stride_locks_M + lock_id_N * stride_locks_N
    debugcount_ptr = locks_ptr + num_locks_M * stride_locks_M
    debug_ptr += lock_id_M * stride_dLds_parts_M + lock_id_N * BLOCK_SIZE_N * stride_dLds_parts_N
    debug2_ptr += lock_id_M * stride_dLds_parts_M + lock_id_N * BLOCK_SIZE_N * stride_dLds_parts_N


    """
    # check lock
    while tl.atomic_cas(locks_ptr, 0, 1) == 1:
        pass
    # Check if we're the first thread for this lock
    count = tl.load(count_ptr)
    if count == 0: # if so, then set count to 1 so future threads know they're not
        tl.atomic_xchg(count_ptr, 1)
    else: # otherwise, this PID is not first and therefore needs to accumulate
        ### the code we want to run
        #dLds_part += tl.load(dLds_parts_ptr + offsets_N, mask=mask_N).to(tl.float32)

        ### the debugging code that shouldn't even trigger given that we only have one PID
        # this first line does not trigger (debuglocks prints out as all zeros)
        tl.atomic_xchg(debugcount_ptr, 1) 
        # but then these lines do trigger (debug and debug2 tensors filled with values)
        current_values = tl.load(dLds_parts_ptr + offsets_N, mask=mask_N, other=0.0).to(tl.float32)
        tl.store(debug_ptr + offsets_N, 
                current_values.to(debug_ptr.type.element_ty), 
                mask=mask_N)
        tl.store(debug2_ptr + offsets_N, 
                (dLds_part + current_values).to(debug_ptr.type.element_ty), 
                mask=mask_N)
    """

    # i brought some of the code out of the else statement to show that it's this current_values loading that's the issue
    # although of note is the fact that this current_values loading was triggering even inside the else statement when the else statement wasn't getting triggered
    tl.debug_barrier()
    current_values = tl.load(dLds_parts_ptr + offsets_N, mask=mask_N).to(tl.float32)
    tl.debug_barrier()
    tl.store(debug_ptr + offsets_N, 
            current_values.to(debug_ptr.type.element_ty), 
            mask=mask_N)
    tl.store(debug2_ptr + offsets_N, 
            (dLds_part + current_values).to(debug2_ptr.type.element_ty), 
            mask=mask_N)
    # store output
    #tl.atomic_xchg(dLds_parts_ptr + offsets_N, 0.) # clean up
    tl.debug_barrier()
    tl.store(dLds_parts_ptr + offsets_N, 
            dLds_part.to(dLds_parts_ptr.type.element_ty), 
            mask=mask_N)
    # finally release lock
    #tl.atomic_xchg(locks_ptr, 0)
    #"""
    # Use atomic add to avoid race conditions across threads within the block
    #for i in range(BLOCK_SIZE_N):
        #if i < N:
            #tl.atomic_add(dLds_parts_ptr + i * stride_dLds_parts_N, dLds_part[i])

@triton.autotune([
        triton.Config(
            {"BLOCK_SIZE_ROWS": BLOCK_SIZE_ROWS, "BLOCK_SIZE_COLS": BLOCK_SIZE_COLS})
        for BLOCK_SIZE_ROWS in [16, 32, 64, 128]
        for BLOCK_SIZE_COLS in [16, 32, 64, 128]],
    key=["ROWS", "COLS"],)
@triton.jit
def scaled_logits_dLds_p2(
    dLds_parts_ptr, dLds_ptr,
    stride_dLds_parts_ROW, stride_dLds_parts_COL,
    stride_dLds_COL,
    ROWS, COLS,
    BLOCK_SIZE_ROWS: tl.constexpr, BLOCK_SIZE_COLS: tl.constexpr
):
    pid = tl.program_id(0)
    dLds_parts_ptr += pid * BLOCK_SIZE_COLS 
    dLds_ptr += pid * BLOCK_SIZE_COLS 
    offsets_ROW = tl.arange(0, BLOCK_SIZE_ROWS)
    offsets_COL = tl.arange(0, BLOCK_SIZE_COLS)
    mask_COL = pid * BLOCK_SIZE_COLS + offsets_COL < COLS

    dLds = tl.zeros([BLOCK_SIZE_COLS], dtype=tl.float32)
    for _ in range(0, ROWS, BLOCK_SIZE_ROWS):
        dLds_parts_offsets = offsets_ROW[:, None] * stride_dLds_parts_ROW + offsets_COL[None, :] * stride_dLds_parts_COL
        mask = (offsets_ROW[:, None] < ROWS) & mask_COL[None, :]
        dLds_part = tl.load(dLds_parts_ptr + dLds_parts_offsets, mask=mask, other=0.).to(tl.float32) #(BLOCK_SIZE_ROWS, BLOCK_SIZE_COLS)
        dLds += tl.sum(dLds_part, axis=0) # (BLOCK_SIZE_COLS)
        offsets_ROW += BLOCK_SIZE_ROWS

    tl.store(dLds_ptr + offsets_COL * stride_dLds_COL, dLds.to(dLds_ptr.type.element_ty), mask=mask_COL)

@triton_op("mylib::logits_fused", mutates_args={})
def scaled_logits_triton(A: torch.Tensor, B: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """
    A @ B = C
    C * s = D
    """
    global _block_size_m
    global _block_size_n
    assert A.shape[-1] == B.shape[-2]
    assert s.shape[0] == B.shape[-1]
    assert s.ndim == 1
    (M, K), (_, N) = A.reshape(-1, A.shape[-1]).shape, B.reshape(-1, B.shape[-1]).shape
    D = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    wrap_triton(scaled_logits_fwd)[grid](
        A, B, D, s,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        D.stride(0), D.stride(1),
        s.stride(0)
    )
    # Save config for backward pass
    _block_size_m = getattr(scaled_logits_fwd, "best_config", None).kwargs['BLOCK_SIZE_M']
    _block_size_n = getattr(scaled_logits_fwd, "best_config", None).kwargs['BLOCK_SIZE_N']
    return D

def scaled_logits_triton_bwd(ctx, dLdD):
    global _block_size_m
    global _block_size_n
    A, B, s, D = ctx.saved_tensors
    M, N, K = ctx.M, ctx.N, ctx.K

    dLdA = torch.empty_like(A)
    dLdB = torch.empty_like(B)
    dLds = torch.zeros_like(s)

    grid_dLdA = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(K, meta['BLOCK_SIZE_K']),)
    wrap_triton(scaled_logits_dLdA)[grid_dLdA](
        A, B, D, 
        dLdA, dLdB, dLdD, 
        s, 
        M, N, K, 
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        D.stride(0), D.stride(1), 
        s.stride(0),
    )

    grid_dLdB = lambda meta: (triton.cdiv(K, meta['BLOCK_SIZE_K']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),)
    wrap_triton(scaled_logits_dLdB)[grid_dLdB](
        A, B, D, 
        dLdA, dLdB, dLdD, 
        s, 
        M, N, K,
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        D.stride(0), D.stride(1), 
        s.stride(0),
    )

    # Use config from forward pass
    num_pids_M = triton.cdiv(M, _block_size_m)
    num_pids_N = triton.cdiv(N, _block_size_n)
    lock_group_size = 1 # TODO set heuristically
    num_locks_M = triton.cdiv(num_pids_M, lock_group_size)
    print("num_pids_M, num_pids_N, num_locks_M:\n", num_pids_M, num_pids_N, num_locks_M)
    dLds_parts = torch.zeros((num_locks_M, N), dtype=torch.float32, device=A.device)
    locks = torch.zeros((2 * num_locks_M, num_pids_N), dtype=torch.int32, device=A.device)
    print("locks:\n", locks)
    debuglocks = torch.zeros((2 * num_locks_M, num_pids_N), dtype=torch.int32, device=A.device)
    
    debug = torch.zeros((num_locks_M, N), dtype=torch.float32, device=A.device)
    debug2 = torch.zeros((num_locks_M, N), dtype=torch.float32, device=A.device)

    grid_dLds_p1 = lambda meta: (num_pids_M * num_pids_N,)
    wrap_triton(scaled_logits_dLds_p1)[grid_dLds_p1](
        A, B, D, 
        dLdA, dLdB, dLdD, 
        s, dLds_parts, dLds, debug, debug2, debuglocks,
        M, N, K,
        A.stride(0), A.stride(1), 
        B.stride(0), B.stride(1), 
        D.stride(0), D.stride(1), 
        s.stride(0),
        dLds_parts.stride(0), dLds_parts.stride(1),
        locks, num_locks_M, locks.stride(0), locks.stride(1),
        BLOCK_SIZE_M = _block_size_m, BLOCK_SIZE_N = _block_size_n
    )
    print("dLds_parts:\n", dLds_parts)
    print("debug:\n", debug)
    print("debug2:\n", debug2)
    print("locks:\n", locks)
    print("debuglocks:\n", debuglocks)
    print("dLds_parts.sum(0):\n", dLds_parts.sum(0))
    grid_dLds_p2 = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_COLS']),)
    wrap_triton(scaled_logits_dLds_p2)[grid_dLds_p2](
        dLds_parts, dLds,
        dLds_parts.stride(0), dLds_parts.stride(1),
        dLds.stride(0),
        ROWS=num_locks_M, COLS=N,
    )

    return dLdA, dLdB, dLds

def scaled_logits_triton_setup_ctx(ctx, inputs, output):
    A, B, s = inputs
    ctx.save_for_backward(A, B, s, output)
    (M, K), (_, N) = A.reshape(-1, A.shape[-1]).shape, B.reshape(-1, B.shape[-1]).shape
    ctx.M, ctx.N, ctx.K = M, N, K

scaled_logits_triton.register_autograd(scaled_logits_triton_bwd, setup_context=scaled_logits_triton_setup_ctx)





def create_diff_heatmap(expected, actual, M, N, dtype, test_name, atol):
    """Create a heatmap visualization for gradient differences"""
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    heatmap_path = f'./scaled_logits_{test_name}_heatmap.png'
    
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

    D_torch = scaled_logits_naive(A_torch, B_torch, s_torch)
    D_triton = scaled_logits_triton(A_triton, B_triton, s_triton)

    ### fwd test
    import os
    test_name = f"C_M={M},N={N}_{dtype}"
    try:
        std = D_torch.std() # scale all assertions to variance of 1 so that tolerances make sense
        torch.testing.assert_close(D_torch / std, D_triton / std, atol=atol, rtol=rtol)
        print(f"✓ passed fwd test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./scaled_logits_{test_name}_heatmap.png'
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
    dLdA_naive, dLdB_naive, dLds_naive = scaled_logits_naive_bwd(A_torch, B_torch, s_torch, dLdD)

    ### naive implementation bwd tests
    test_name = f"dLdA_naive_M={M},N={N}_{dtype}"
    try:
        std = A_torch.grad.std()
        torch.testing.assert_close(A_torch.grad / std, dLdA_naive / std, atol=atol, rtol=rtol)
        print(f"✓ passed dLdA_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./scaled_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLdA_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        create_diff_heatmap(A_torch.grad, dLdA_naive, M, N, dtype, test_name, atol)
        raise e

    test_name = f"dLdB_naive_M={M},N={N}_{dtype}"
    try:
        std = B_torch.grad.std()
        torch.testing.assert_close(B_torch.grad / std, dLdB_naive / std, atol=atol, rtol=rtol)
        print(f"✓ passed dLdB_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./scaled_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLdB_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        create_diff_heatmap(B_torch.grad, dLdB_naive, M, N, dtype, test_name, atol)
        raise e
    
    test_name = f"dLds_naive_M={M},N={N}_{dtype}"
    try:
        std = s_torch.grad.std()
        torch.testing.assert_close(s_torch.grad / std, dLds_naive / std, atol=atol, rtol=rtol)
        print(f"✓ passed dLds_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./scaled_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLds_naive test (M={M}, N={N}, K={K}, dtype={dtype})")
        create_diff_heatmap(s_torch.grad.unsqueeze(0), dLds_naive.unsqueeze(0), 1, N, dtype, test_name, atol)
        raise e

    
    D_triton.backward(dLdD)

    ### triton implementation bwd tests
    test_name = f"dLdA_triton_M={M},N={N}_{dtype}"
    try:
        std = A_torch.grad.std()
        torch.testing.assert_close(A_torch.grad / std, A_triton.grad / std, atol=atol, rtol=rtol)
        print(f"✓ passed dLdA_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./scaled_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLdA_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        print(A_torch.grad)
        print(A_triton.grad)
        create_diff_heatmap(A_torch.grad, A_triton.grad, M, N, dtype, test_name, atol)
        raise e

    test_name = f"dLdB_triton_M={M},N={N}_{dtype}"
    try:
        std = B_torch.grad.std()
        torch.testing.assert_close(B_torch.grad / std, B_triton.grad / std, atol=atol, rtol=rtol)
        print(f"✓ passed dLdB_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        heatmap_path = f'./scaled_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLdB_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        print(B_torch.grad)
        print(B_torch.grad)
        create_diff_heatmap(B_torch.grad, B_triton.grad, M, N, dtype, test_name, atol)
        raise e

    test_name = f"dLds_triton_M={M},N={N}_{dtype}"
    try:
        std = s_torch.grad.std()
        torch.testing.assert_close(s_torch.grad / std, s_triton.grad / std, atol=atol, rtol=rtol)
        print(f"✓ passed dLds_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        print(s_torch.grad)
        print(s_triton.grad)
        heatmap_path = f'./scaled_logits_{test_name}_heatmap.png'
        if os.path.exists(heatmap_path):
            os.remove(heatmap_path)
            print(f"Deleted old heatmap file: {heatmap_path}")
    except AssertionError as e:
        print(f"✗ failed dLds_triton test (M={M}, N={N}, K={K}, dtype={dtype})")
        print(s_torch.grad)
        print(s_triton.grad)
        create_diff_heatmap(s_torch.grad.unsqueeze(0), s_triton.grad.unsqueeze(0), 1, N, dtype, test_name, atol)
        raise e



configs = []
for mode in ["fwd", "bwd"]:
    for dtype_bytes in [2, 4]:
        configs.append(
            triton.testing.Benchmark(
                x_names=['M', 'N', 'K'],
                x_vals=[2**i for i in range(8, 10)],#15)], 
                line_arg='provider',
                line_vals=['triton', 'torch'],
                line_names=['Triton', 'naive + torch.compile'],
                styles=[('blue', '-'), ('green', '-')],
                ylabel='TFLOPs',
                plot_name=f'scaled_logits_{mode}_fp{8*dtype_bytes}',
                args={"mode": mode, "dtype_bytes": dtype_bytes}, 
            ))
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider, mode, dtype_bytes, device=DEVICE):
    # create data
    assert dtype_bytes in [2, 4]
    dtype = torch.float16 if dtype_bytes == 2 else torch.float32
    A = torch.randn((M, K), dtype=dtype, device=device, requires_grad=True)
    B = torch.randn((K, N), dtype=dtype, device=device, requires_grad=True)
    s = torch.randn((N,), dtype=dtype, device=device, requires_grad=True)

    # confidence itnerval for testing
    quantiles = [0.5, 0.001, 0.999]

    if provider == "triton":
        fn = lambda: scaled_logits_triton(A, B, s)
    if provider == "torch":
        fn = lambda: scaled_logits_naive(A, B, s)
    elif mode == "bwd":
        D = fn()
        dLdD = torch.randn_like(D)
        fn = lambda: D.backward(dLdD, retain_graph=True)
    
    # benchmark
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    flops_per_matmul = 2.0 * M * N * K
    flops_per_mult = M * N * K
    if mode == "fwd":
        total_flops = flops_per_matmul + flops_per_mult
    if mode == "bwd":
        # theoretical if we had instantiated C in VRAM
        total_flops = 2 * flops_per_matmul + flops_per_mult
        # in reality in order to save VRAM our triton implementation does 
        #total_flops = 3 * flops_per_matmul + 3 * flops_per_mult
    tfps = lambda ms: total_flops * 1e-12 / (ms * 1e-3)
    return tfps(ms), tfps(max_ms), tfps(min_ms)



def measure_memory_usage(func, size, dtype):
    # Reset the memory stats on the current GPU device
    torch.cuda.reset_peak_memory_stats()
    # Run the function (forward, backward, etc.)
    func(size, dtype)
    # Synchronize to ensure all asynchronous ops complete
    torch.cuda.synchronize()
    # Get the maximum memory allocated during the operation
    peak_memory = torch.cuda.max_memory_allocated()
    return peak_memory

# Example usage with your custom kernel and PyTorch's operator:
def run_custom_kernel(size, dtype):
    # Replace with a call to your fused kernel
    A = torch.randn((size, size), device=DEVICE, requires_grad=True, dtype=dtype)
    B = torch.randn((size, size), device=DEVICE, requires_grad=True, dtype=dtype)
    s = torch.randn((size,), device=DEVICE, requires_grad=True, dtype=dtype)
    D = scaled_logits_triton(A, B, s)
    D.backward(torch.randn_like(D))

def run_pytorch_operator(size, dtype):
    A = torch.randn((size, size), device=DEVICE, requires_grad=True, dtype=dtype)
    B = torch.randn((size, size), device=DEVICE, requires_grad=True, dtype=dtype)
    s = torch.randn((size,), device=DEVICE, requires_grad=True, dtype=dtype)
    D = scaled_logits_naive(A, B, s)  # similar operation as a baseline
    D.backward(torch.randn_like(D))



if __name__ == "__main__":
    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        # check runtime
        benchmark.run(save_path='./benchmarks/', print_data=False)
        # we do runtime first so that triton's built-in benchmarking tools can worry about autotuning
        # so that the autotuning doesn't mess with our memory estimation

        # measure memory usage
        for dtype in [torch.float16, torch.float32]:
            for size in [2**i for i in range(8, 10)]:#15)]:
                custom_peak = measure_memory_usage(run_custom_kernel, size, dtype)
                pytorch_peak = measure_memory_usage(run_pytorch_operator, size, dtype)
                print(f"dtype={dtype} | M,N,K={size} | Peak memory used by custom kernel: {custom_peak * 1e-6:.1f} Mb")
                print(f"dtype={dtype} | M,N,K={size} | Peak memory used by PyTorch operator: {pytorch_peak * 1e-6:.1f} Mb")
                print(f"Memory saved: {100 * (pytorch_peak - custom_peak) / pytorch_peak:.2f}%")
    else:
        
        # (prolly) no masking
        test(32, 32, 32, torch.float32)
        """
        test(32, 32, 32, torch.float16)
        test(128, 128, 128, torch.float32)
        test(128, 128, 128, torch.float16)
        test(512, 512, 512, torch.float32)
        test(512, 512, 512, torch.float16)
        test(1024, 1024, 1024, torch.float32)
        test(1024, 1024, 1024, torch.float16)
        test(2048, 2048, 2048, torch.float32)
        test(2048, 2048, 2048, torch.float16)

        # masking
        test(2, 2, 2, torch.float32)
        test(2, 2, 2, torch.float16)
        test(39, 39, 39, torch.float32)
        test(39, 39, 39, torch.float16)
        test(2067, 2067, 2067, torch.float32)
        test(2067, 2067, 2067, torch.float16)
        
        # uneven dimensions
        test(1024, 256, 128, torch.float32)
        test(1024, 256, 128, torch.float16)
        test(32, 512, 2024, torch.float32)
        test(32, 512, 2024, torch.float16)
        test(8*1024, 2**14, 384, torch.float32) 
        test(8*1024, 2**14, 384, torch.float16)
    
        
        # uneven dimensions & masking
        test(1080, 267, 129, torch.float32)
        test(1080, 267, 129, torch.float16)
        test(31, 569, 2199, torch.float32)
        test(31, 569, 2199, torch.float16)
        test(8*1024+3, 2**14+42, 383, torch.float16)
        test(8*1024+3, 2**14+42, 383, torch.float32) 
        #"""

