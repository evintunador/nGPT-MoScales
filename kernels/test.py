import math

import torch
#print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F

import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

import triton
from triton import language as tl

@triton.jit
def add_kernel(
    in_ptr0,
    in_ptr1,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    output = x + y
    tl.store(out_ptr + offsets, output, mask=mask)

@torch.compile(fullgraph=True)
def add_fn(x, y):
    output = torch.zeros_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=4)
    return output

x = torch.randn(4, device="cuda")
y = torch.randn(4, device="cuda")
out = add_fn(x, y)
print(f"Vector addition of\nX:\t{x}\nY:\t{y}\nis equal to\n{out}")