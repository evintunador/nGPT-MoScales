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
def cosine_norm_forward_naive(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Places vectors onto the unit-hypersphere"""
    # calculate the magnitude of the vectors
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-12)
    # divide by the magnitude to place on the unit hypersphere
    return x / norm

@torch.compile
def cosine_norm_backward_naive(x: torch.Tensor, grad_output: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Computes the gradient for cosine normalization.
    
    The gradient is: grad_input = (grad_output - y * sum(y * grad_output)) / norm
    where y is the normalized output.
    """
    norm = torch.norm(x, p=2, dim=dim, keepdim=True).clamp(min=1e-12)
    y = x / norm
    
    # Compute sum(y * grad_output) along specified dimension
    grad_dot_y = torch.sum(y * grad_output, dim=dim, keepdim=True)
    
    # Compute the gradient
    grad_input = (grad_output - y * grad_dot_y) / norm
    
    return grad_input


@triton.jit
def cos_norm_fwd(
    x_ptr,
    y_ptr,
    norm_ptr,
    stride_M, stride_N,
    N,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    x_ptr += row * stride_M
    y_ptr += row * stride_M
    norm_ptr += row
    cols = tl.arange(0, BLOCK_SIZE) * stride_N # stride since we never asserted x.is_contiguous()
    mask = cols < N

    x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)

    # compute L_2 norm & normalize
    eps: tl.constexpr = 1e-12
    inf: tl.constexpr = 1e12
    norm = tl.clamp(tl.sqrt(tl.sum(x * x, axis=0)), eps, inf)
    y = x / norm

    tl.store(norm_ptr, norm)
    tl.store(y_ptr + cols, y.to(y_ptr.type.element_ty), mask=mask)

@triton.jit
def cos_norm_bwd(
    x_ptr, dLdx_ptr, y_ptr, dLdy_ptr, norm_ptr,
    stride_M, stride_N,
    N,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    offset = row * stride_M
    #x_ptr += offset
    dLdx_ptr += offset
    y_ptr += offset
    dLdy_ptr += offset
    norm_ptr += row
    cols = tl.arange(0, BLOCK_SIZE) * stride_N # stride since we never asserted x.is_contiguous()
    mask = cols < N

    #x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)
    y = tl.load(y_ptr + cols, mask=mask, other=0.).to(tl.float32)
    dLdy = tl.load(dLdy_ptr + cols, mask=mask, other=0.).to(tl.float32)
    norm = tl.load(norm_ptr) # fp32

    # compute grad
    #eps: tl.constexpr = 1e-12
    #inf: tl.constexpr = 1e12
    #norm = tl.clamp(tl.sqrt(tl.sum(x * x, axis=0)), eps, inf)
    #y = x / norm
    dLdy_dot_y = tl.sum(y * dLdy, axis=0)
    dLdx = (dLdy - y * dLdy_dot_y) / norm
    
    tl.store(dLdx_ptr + cols, dLdx.to(dLdx_ptr.type.element_ty), mask=mask)



# used to derive our block size
properties = triton.runtime.driver.active.utils.get_device_properties(DEVICE.index)
sram_per_sm = properties["max_shared_mem"]
"""
# Pre-compile the optimized backward function outside the class
@torch.compile
def _cos_norm_backward_torch(y, norm, dLdy):
    grad_dot_y = torch.sum(y * dLdy, dim=-1, keepdim=True)
    dLdx = (dLdy - y * grad_dot_y) / norm.unsqueeze(-1)
    return dLdx
#"""
class _cosine_norm_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dim: int = -1):
        # we know this function will only be used on dim=-1 in nGPT
        M, N = x.reshape(-1, x.shape[-1]).shape

        # this kernel is designed for normalizing vectors that fit in SRAM
        max_entries = sram_per_sm // x.element_size()
        block_size = triton.next_power_of_2(N)
        assert max_entries >= block_size, f"cosine norm kernel only supports vectors up to {max_entries}"
        # H100s have 256kb of SRAM per SM so this would fit a model dimension of 64 thousand at fp32, plenty

        # pre-allocate output & norm storage for bwd pass
        y = torch.empty_like(x)
        norm = torch.empty((M,), device=x.device, dtype=torch.float32, requires_grad=False)

        # each row gets it own PID
        cos_norm_fwd[(M,)](x, y, norm, x.stride(-2), x.stride(-1), N, BLOCK_SIZE=block_size)

        ctx.save_for_backward(x, y, norm)
        ctx.M, ctx.N, ctx.block_size = M, N, block_size
        return y

    @staticmethod
    def backward(ctx, dLdy):
        x, y, norm = ctx.saved_tensors
        M, N, block_size = ctx.M, ctx.N, ctx.block_size
        
        #dLdx = _cos_norm_backward_torch(y, norm, dLdy)
        
        dLdx = torch.empty_like(x)
        cos_norm_bwd[(M,)](
            x, dLdx, y, dLdy, norm,
            x.stride(-2), x.stride(-1),
            N,
            BLOCK_SIZE=block_size
        )
        
        return dLdx, None

cosine_norm_triton = _cosine_norm_triton.apply


@triton.jit
def cos_norm_fwd_inplace(
    x_ptr,
    stride_M, stride_N,
    N,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    x_ptr += row * stride_M
    cols = tl.arange(0, BLOCK_SIZE) * stride_N # stride since we never asserted x.is_contiguous()
    mask = cols < N

    x = tl.load(x_ptr + cols, mask=mask, other=0.).to(tl.float32)

    # compute L_2 norm & normalize
    eps: tl.constexpr = 1e-12
    inf: tl.constexpr = 1e12
    norm = tl.clamp(tl.sqrt(tl.sum(x * x, axis=0)), eps, inf)
    y = x / norm

    tl.store(x_ptr + cols, y.to(x_ptr.type.element_ty), mask=mask)

def cosine_norm_triton_(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """In-place version of cosine_norm_triton forward pass following PyTorch naming convention"""
    # we know this function will only be used on dim=-1 in nGPT
    M, N = x.reshape(-1, x.shape[-1]).shape

    # this kernel is designed for normalizing vectors that fit in SRAM
    max_entries = sram_per_sm // x.element_size()
    block_size = triton.next_power_of_2(N)
    assert max_entries >= block_size, f"cosine norm kernel only supports vectors up to {max_entries}"
    # H100s have 256kb of SRAM per SM so this would fit a model dimension of 64 thousand at fp32, plenty

    # each row gets it own PID
    cos_norm_fwd_inplace[(M,)](x, x.stride(-2), x.stride(-1), N, BLOCK_SIZE=block_size)

    return x


def test_cos_norm(M, N, dtype, device=DEVICE):
    # create data
    x_triton = torch.randn((M, N), dtype=dtype, device=device, requires_grad=True)
    x_torch = x_triton.clone().detach().requires_grad_(True)
    x_naive = x_triton.clone().detach().requires_grad_(True)
    x_inplace = x_triton.clone().detach().requires_grad_(False)
    
    # run each
    y_torch = torch.nn.functional.normalize(x_torch, p=2, dim=1)
    y_naive = cosine_norm_forward_naive(x_naive)
    y_triton = cosine_norm_triton(x_triton)
    y_triton_inplace = cosine_norm_triton_(x_inplace)

    # test each
    torch.testing.assert_close(y_torch, y_naive, atol=1e-3, rtol=1e-3)
    print(f"✓ Naive implementation passed cos norm test (M={M}, N={N}, dtype={dtype})")
    torch.testing.assert_close(y_torch, y_triton, atol=1e-3, rtol=1e-3)
    print(f"✓ Triton implementation passed cos norm test (M={M}, N={N}, dtype={dtype})")
    torch.testing.assert_close(y_torch, y_triton_inplace, atol=1e-3, rtol=1e-3)
    print(f"✓ Triton in-place implementation passed cos norm test (M={M}, N={N}, dtype={dtype})")
    assert x_inplace is y_triton_inplace, "In-place operation failed to modify input tensor"
    print(f"✓ Verified in-place operation modified input tensor")
    
    # Test backward pass
    grad_output = torch.randn_like(x_triton)
    y_torch.backward(grad_output, retain_graph=True)
    torch_grad = x_torch.grad.clone().detach()
    naive_grad = cosine_norm_backward_naive(x_naive.detach(), grad_output, dim=1)
    y_triton.backward(grad_output, retain_graph=True)
    triton_grad = x_triton.grad.clone().detach()
    
    # Compare gradients
    torch.testing.assert_close(torch_grad, naive_grad, atol=1e-3, rtol=1e-3)
    print(f"✓ Naive backward implementation passed gradient test (M={M}, N={N}, dtype={dtype})")
    torch.testing.assert_close(torch_grad, triton_grad, atol=1e-3, rtol=1e-3)
    print(f"✓ Triton backward implementation passed gradient test (M={M}, N={N}, dtype={dtype})")


def previous_power_two(n):
    return int(math.log(n, 2))

configs = []
for mode in ["fwd", "bwd"]:
    for dtype_bytes in [2, 4]:
        configs.append(
            triton.testing.Benchmark(
                x_names=['N'],
                x_vals=[2 ** i for i in range(8, previous_power_two(sram_per_sm // dtype_bytes))], 
                line_arg='provider',
                line_vals=['triton', 'torch', 'naive'] + (['inplace'] if mode == "fwd" else []),
                line_names=['Triton', 'torch.nn.functional', 'naive + torch.compile'] + (['inplace triton'] if mode == "fwd" else []),
                styles=[('blue', '-'), ('green', '-'), ('red', '-')] + ([('orange', '-')] if mode == "fwd" else []),
                ylabel='GB/s',
                plot_name=f'cos_norm_{mode}_fp{8*dtype_bytes}',
                args={"mode": mode, "dtype_bytes": dtype_bytes}, 
            ))
@triton.testing.perf_report(configs)
def benchmark(N, provider, mode, dtype_bytes, device=DEVICE):
    # create data
    assert dtype_bytes in [2, 4]
    dtype = torch.float16 if dtype_bytes == 2 else torch.float32
    x = torch.randn((16*1024, N), dtype=dtype, device=device, requires_grad=True)

    # confidence itnerval for testing
    quantiles = [0.5, 0.001, 0.999]

    if provider == "torch":
        fn = lambda: torch.nn.functional.normalize(x, p=2, dim=1)
    if provider == "naive":
        fn = lambda: cosine_norm_forward_naive(x)
    if provider == "triton":
        fn = lambda: cosine_norm_triton(x)
    if provider == "inplace":
        fn = lambda: cosine_norm_triton_(x)
    elif mode == "bwd":
        y = fn()
        dLdy = torch.randn_like(y)
        fn = lambda: y.backward(dLdy, retain_graph=True)
    
    # benchmark
    ms, min_ms, max_ms = triton.testing.do_bench(fn, quantiles=quantiles)
    gbps = lambda ms: (2 if mode == "fwd" else 3) * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    # always run unit-tests
    test_cos_norm(2048, 768, torch.float32)
    test_cos_norm(2048, 8192, torch.float32)
    test_cos_norm(2048, 768, torch.float16)
    test_cos_norm(2048, 8192, torch.float16)

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='./benchmarks/', print_data=False)




