import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

@torch.compile
def cosine_norm_naive(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Places vectors onto the unit-hypersphere"""
    # calculate the magnitude of the vectors
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-12)
    # divide by the magnitude to place on the unit hypersphere
    return x / norm


@triton.autotune( # decorator figures out what meta-parameters will be most efficient
    [
        triton.Config(
            num_stages=num_stages, num_warps=num_warps,
        )
        for num_stages in [3, 5, 7]
        for num_warps in [4, 8, 16, 32]
    ],
    key=["N"],
)
@triton.jit
def cos_norm_fwd(
    x_ptr,
    y_ptr,
    stride_M, stride_N,
    N,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M

    # compute L_2 norm
    cols = tl.arange(0, BLOCK_SIZE) * stride_N # stride since we never asserted x.is_contiguous()
    mask = cols < N
    x = tl.load(x_ptr + cols, mask=mask, other=0.)
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    eps: tl.constexpr = 1e-12
    y = x / (norm + eps) # TODO does this need to be tl.full()?

    tl.store(y_ptr + cols, y, mask=mask)


properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
sram_per_sm = properties["max_shared_mem"]
# should derive how big a hidden dimension we can do using this

@torch.compile(full_graph=True)
def cosine_norm_triton(x: torch.Tensor) -> torch.Tensor:
    # we know this function will only be used on dim=-1 in nGPT
    M, N = x.reshape(-1, x.shape[-1]).shape

    # this kernel is designed for normalizing vectors that fit in SRAM
    max_entries = sram_per_sm // x.element_size()
    block_size = triton.next_power_of_2(N)
    assert max_entries >= block_size, f"cosine norm kernel only supports vectors up to {max_entries}"
    # H100s have 256kb of SRAM per SM so this would fit a model dimension of 64 thousand at fp32, plenty

    # pre-allocate output
    y = torch.empty_like(x)

    # each row gets it own PID
    cos_norm_fwd[(M,)](
        x, y,
        x.stride(-2), x.stride(-1),
        N,
        BLOCK_SIZE=block_size
    )

    return y


def test_cos_norm(M, N, dtype, device=DEVICE):
    # create data
    x = torch.randn((M, N), dtype=dtype, device=device, requires_grad=True)

    # run each
    y_torch = torch.nn.functional.normalize(x, p=2, dim=1)
    y_naive = cosine_norm_naive(x)
    y_triton = cosine_norm_triton(x)

    # test
    torch.testing.assert_close(y_torch, y_naive, atol=1e-3, rtol=1e-3)
    print("naive passed cos norm")
    torch.testing.assert_close(y_torch, y_triton, atol=1e-3, rtol=1e-3)
    print("triton passed cos norm")


test_cos_norm(2048, 768)
test_cos_norm(2048, 8192)



