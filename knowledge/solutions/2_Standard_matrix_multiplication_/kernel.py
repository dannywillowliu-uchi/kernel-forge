import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def leaky_relu_kernel(x_ptr, out_ptr, n_elements, negative_slope, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = tl.where(x >= 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super(ModelNew, self).__init__()
        self.negative_slope = negative_slope
        self._out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._out is None or self._out.shape != x.shape:
            self._out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 4096
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        leaky_relu_kernel[grid](x, self._out, n_elements, self.negative_slope, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
        return self._out
