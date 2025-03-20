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
    cols = tl.arange(0, BLOCK_SIZE) * stride_N # stride since we never asserted x.is_contiguous()
    mask = cols < N

    # compute L_2 norm
    x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
    norm = tl.sqrt(tl.sum(x * x, axis=0))
    eps: tl.constexpr = 1e-12
    y = x / (norm + eps)

    tl.store(y_ptr + cols, y.to(y_ptr.type.element_ty), mask=mask)

# used to derive our block size
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
sram_per_sm = properties["max_shared_mem"]

#@torch.compile(fullgraph=True) # <- gives error for some reason
#@triton_op("mylib::cos_norm", mutates_args={})
def cosine_norm_triton(x: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    # we know this function will only be used on dim=-1 in nGPT
    M, N = x.reshape(-1, x.shape[-1]).shape

    # this kernel is designed for normalizing vectors that fit in SRAM
    max_entries = sram_per_sm // x.element_size()
    block_size = triton.next_power_of_2(N)
    assert max_entries >= block_size, f"cosine norm kernel only supports vectors up to {max_entries}"
    # H100s have 256kb of SRAM per SM so this would fit a model dimension of 64 thousand at fp32, plenty

    # pre-allocate output or use input tensor for in-place operation
    y = x if inplace else torch.empty_like(x)

    # each row gets it own PID
    cos_norm_fwd[(M,)](
        x, y,
        x.stride(-2), x.stride(-1),
        N,
        BLOCK_SIZE=block_size
    )

    return y

def cosine_norm_triton_(x: torch.Tensor) -> torch.Tensor:
    """In-place version of cosine_norm_triton following PyTorch naming convention"""
    return cosine_norm_triton(x, inplace=True)

def test_cos_norm(M, N, dtype, device=DEVICE):
    # create data
    x = torch.randn((M, N), dtype=dtype, device=device, requires_grad=True)
    x_clone = x.clone()  # for in-place testing

    # run each
    y_torch = torch.nn.functional.normalize(x, p=2, dim=1)
    y_naive = cosine_norm_naive(x)
    y_triton = cosine_norm_triton(x)
    
    # test in-place version
    y_triton_inplace = cosine_norm_triton_(x_clone)

    # test out-of-place results
    torch.testing.assert_close(y_torch, y_naive, atol=1e-3, rtol=1e-3)
    print(f"✓ Naive implementation passed cos norm test (M={M}, N={N}, dtype={dtype})")
    
    torch.testing.assert_close(y_torch, y_triton, atol=1e-3, rtol=1e-3)
    print(f"✓ Triton implementation passed cos norm test (M={M}, N={N}, dtype={dtype})")
    
    # test in-place results
    torch.testing.assert_close(y_torch, y_triton_inplace, atol=1e-3, rtol=1e-3)
    print(f"✓ Triton in-place implementation passed cos norm test (M={M}, N={N}, dtype={dtype})")
    
    # verify in-place operation actually modified the input
    assert x_clone is y_triton_inplace, "In-place operation failed to modify input tensor"
    print(f"✓ Verified in-place operation modified input tensor")


def previous_power_two(n):
    return int(math.log(n, 2))

configs = []
for dtype_bytes in [2, 4]:
    configs.append(
        triton.testing.Benchmark(
            x_names=['N'],
            x_vals=[2 ** i for i in range(1, previous_power_two(sram_per_sm // dtype_bytes))], 
            line_arg='provider',
            line_vals=['triton', 'torch', 'naive'],
            line_names=['Triton', 'torch.nn.functional', 'naive + torch.compile'],
            styles=[('blue', '-'), ('green', '-'), ('red', '-')],
            ylabel='GB/s',
            plot_name=f'cos_norm_fwd_fp{8*dtype_bytes}',
            args={"dtype_bytes": dtype_bytes}, 
        ))
@triton.testing.perf_report(configs)
def benchmark(N, provider, dtype_bytes, device=DEVICE):
    # create data
    assert dtype_bytes in [2, 4]
    dtype = torch.float16 if dtype_bytes == 2 else torch.float32
    x = torch.randn((32*1024, N), dtype=dtype, device=device, requires_grad=False)

    # confidence itnerval for testing
    quantiles = [0.5, 0.001, 0.999]

    def y_fwd():
        if provider == "triton":
            return cosine_norm_triton(x)
            #return torch.ops.mylib.cos_norm.default(x)
        if provider == "torch":
            return torch.nn.functional.normalize(x, p=2, dim=1)
        if provider == "naive":
            return cosine_norm_naive(x)

    # benchmark
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # always run unit-tests
    test_cos_norm(2048, 768, torch.float16)
    test_cos_norm(2048, 8192, torch.float16)
    test_cos_norm(2048, 768, torch.float32)
    test_cos_norm(2048, 8192, torch.float32)

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='./benchmarks/', print_data=False)




