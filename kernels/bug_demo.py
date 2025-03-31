import math
import torch
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

def scaled_logits_dLdA():
    pass

def scaled_logits_dLdB():
    pass

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
    #"""

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