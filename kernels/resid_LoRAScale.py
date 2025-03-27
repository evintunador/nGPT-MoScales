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

try:
    from .cos_norm import cosine_norm_forward_naive, cosine_norm_backward_naive
except Exception as e:
    from cos_norm import cosine_norm_forward_naive, cosine_norm_backward_naive

@torch.compile
def resid_LoRAScale_naive(h_prev, h_eigen, alpha_up, alpha_down, alpha_bias):
    """
    h_prev & h_eigen: shape (M, N) 
    alpha_up: shape (N, R) where R << N
    alpha_down: shape (R, N) 
    alpha_bias: shape (N) 
    """
    assert all([h_prev.shape[-1] == N for N in 
        [h_eigen.shape[-1], alpha_up.shape[0], alpha_down.shape[1], alpha_bias.shape[0]])
    assert alpha_up.shape[1] == alpha_down.shape[0]
    h_eigen_normed = h_eigen / torch.norm(h_eigen, p=2, dim=dim, keepdim=True).clamp(min=1e-12)
    h_grad = h_eigen_normed - h_prev
    h_eta = ((h_prev @ alpha_up) @ alpha_down) + alpha_bias
    h_adj = h_prev + h_eta * h_grad
    h_out = h_adj / torch.norm(h_adj, p=2, dim=dim, keepdim=True).clamp(min=1e-12)
    return h_out



@triton.jit
def fwd_kernel(
    h_prev_ptr, h_eigen_ptr, h_out_ptr,
    alpha_up_ptr, alpha_down_ptr, alpha_bias_ptr,
    stride_h_M, stride_h_N,
    stride_alpha_N, stride_alpha_R,
    N: tl.constexpr, R: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    h_prev_ptr += row * stride_M
    h_eigen_ptr += row * stride_M
    h_out_ptr += row * stride_M

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N

    h_prev = tl.load(h_prev_ptr + cols * stride_h_N, mask=mask, other=0.).to(tl.float32)
    h_eigen = tl.load(h_eigen_ptr + cols * stride_h_N, mask=mask, other=0.).to(tl.float32)

    # compute L_2 norm & normalize h_eigen
    eps: tl.constexpr = 1e-12
    inf: tl.constexpr = 1e12
    h_eigen_normed = h_eigen / tl.clamp(tl.sqrt(tl.sum(h_eigen * h_eigen, axis=0)), eps, inf)

    # movement along hypersphere residual connection
    h_grad = h_eigen_normed - h_prev
    mid = tl.zeros([R], dtype=tl.float32)
    for i in tl.range(R):
        alpha_up_i = tl.load(alpha_up_ptr + cols * stride_alpha_N)
        mid[i] = tl.dot(h_prev, alpha_up_i, acc=mid[i])
    h_eta = tl.zeros([N], dtype=tl.float32)
    rank_rows = tl.arange(0, R)
    for j in tl.range(0, N): # TODO come back & block this by chunks of N/R instead of looping thru N naively
        alpha_down_j = tl.load(alpha_down_ptr + )
    alpha_bias = tl.load(alpha_bias_ptr + cols, mask=mask, other=0.).to(tl.float32)

    #h_grad_step = h_grad @ alpha_up @ alpha_down + alpha_bias
    resid = h + h_grad_step

    # normalize output
    h_out = resid / tl.clamp(tl.sqrt(tl.sum(resid * resid, axis=0)), eps, inf)

    tl.store(h_out_ptr + cols * stride_h_N, h_out.to(h_out_ptr.type.element_ty), mask=mask)

# used to derive our block size
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
sram_per_sm = properties["max_shared_mem"]

#@torch.compile(fullgraph=True)
@triton_op("mylib::resid_fwd_triton", mutates_args={})
def resid_LoRAScale(
    h_prev: torch.Tensor, 
    h_eigen: torch.Tensor, 
    alpha_up: torch.Tensor,
    alpha_down: torch.Tensor,
    alpha_bias: torch.Tensor,
) -> torch.Tensor:
    assert h_prev.shape == h_eigen.shape
    assert h_prev.stride() == h_eigen.stride()
    assert all([h_prev.shape[-1] == N for N in 
        [h_eigen.shape[-1], alpha_up.shape[0], alpha_down.shape[1], alpha_bias.shape[0]])
    assert alpha_up.shape[1] == alpha_down.shape[0]
    M, N = h_prev.reshape(-1, h_prev.shape[-1]).shape
    R = alpha_up.shape[1]

    # this kernel is designed for normalizing vectors that fit in SRAM
    max_entries = sram_per_sm // h_prev.element_size()
    block_size = triton.next_power_of_2(N)
    assert max_entries >= block_size, f"resid LoRAScale kernel only supports vectors up to {max_entries}"
    # H100s have 256kb of SRAM per SM so this would fit a model dimension of 64 thousand at fp32, plenty

    # pre-allocate output
    h_out = torch.empty_like(h_prev)

    wrap_triton(fwd_kernel)[(M,)](
        h_prev, h_eigen, h_out, alpha_up, alpha_down, alpha_bias,
        h_prev.stride(-2), h_prev.stride(-1),
        alpha_up.stride(0), alpha_up.stride(1),
        alpha_down.stride(0), alpha_down.stride(1),
        N, R,
        BLOCK_SIZE=block_size
    )

    return h_out