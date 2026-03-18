import torch
import torch.nn as nn
import triton
import triton.language as tl


# v17: 4 chunks per block in normalize, and also try in stats

@triton.jit
def _ln_partial_stats_4x(
    X, Partials,
    N: tl.constexpr,
    TOTAL_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAT_BLOCKS: tl.constexpr,
):
    """Each block computes partial sums for 4 consecutive chunks."""
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    total_s = 0.0
    total_sq = 0.0
    
    for i in range(4):
        chunk = block_idx * 4 + i
        if chunk < TOTAL_CHUNKS:
            col_offset = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = col_offset < N
            x = tl.load(X + row * N + col_offset, mask=mask, other=0.0).to(tl.float32)
            total_s += tl.sum(x, axis=0)
            total_sq += tl.sum(x * x, axis=0)
    
    idx = row * NUM_STAT_BLOCKS + block_idx
    tl.store(Partials + idx * 2, total_s)
    tl.store(Partials + idx * 2 + 1, total_sq)


@triton.jit
def _ln_reduce_stats(
    Partials, Stats,
    N: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    BLOCK_CEIL: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    chunk_ids = tl.arange(0, BLOCK_CEIL)
    mask = chunk_ids < NUM_BLOCKS
    idx = row * NUM_BLOCKS + chunk_ids
    s = tl.load(Partials + idx * 2, mask=mask, other=0.0)
    sq = tl.load(Partials + idx * 2 + 1, mask=mask, other=0.0)
    total_s = tl.sum(s, axis=0)
    total_sq = tl.sum(sq, axis=0)
    mean = total_s / N
    var = total_sq / N - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Stats + row * 2, mean)
    tl.store(Stats + row * 2 + 1, rstd)


@triton.jit
def _ln_normalize_4x(
    X, Y, W, B, Stats,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOTAL_CHUNKS: tl.constexpr,
):
    """Each block normalizes 4 consecutive chunks."""
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    mean = tl.load(Stats + row * 2)
    rstd = tl.load(Stats + row * 2 + 1)
    
    for i in range(4):
        chunk = block_idx * 4 + i
        if chunk < TOTAL_CHUNKS:
            col_offset = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = col_offset < N
            x = tl.load(X + row * N + col_offset, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(W + col_offset, mask=mask, other=1.0).to(tl.float32)
            b = tl.load(B + col_offset, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * w + b
            tl.store(Y + row * N + col_offset, y, mask=mask)


class TritonLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, normalized_shape, eps=1e-5):
        batch_size = 1
        for d in x.shape[:-len(normalized_shape)]:
            batch_size *= d
        
        N = 1
        for d in normalized_shape:
            N *= d
        
        x_flat = x.contiguous().view(batch_size, N)
        
        BLOCK_SIZE = 1024
        total_chunks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE  # 4096
        num_stat_blocks = (total_chunks + 3) // 4  # 1024
        num_norm_blocks = (total_chunks + 3) // 4  # 1024
        
        block_ceil = 1
        while block_ceil < num_stat_blocks:
            block_ceil *= 2
        
        partials = torch.empty(batch_size * num_stat_blocks * 2, device=x.device, dtype=torch.float32)
        stats = torch.empty(batch_size * 2, device=x.device, dtype=torch.float32)
        y_flat = torch.empty_like(x_flat)
        
        _ln_partial_stats_4x[(batch_size, num_stat_blocks)](
            x_flat, partials,
            N=N, TOTAL_CHUNKS=total_chunks,
            BLOCK_SIZE=BLOCK_SIZE, NUM_STAT_BLOCKS=num_stat_blocks,
            num_warps=2,
        )
        
        _ln_reduce_stats[(batch_size,)](
            partials, stats,
            N=N, NUM_BLOCKS=num_stat_blocks, BLOCK_CEIL=block_ceil,
            eps=eps,
            num_warps=4,
        )
        
        w_flat = weight.contiguous().view(-1)
        b_flat = bias.contiguous().view(-1)
        
        _ln_normalize_4x[(batch_size, num_norm_blocks)](
            x_flat, y_flat, w_flat, b_flat, stats,
            N=N, BLOCK_SIZE=BLOCK_SIZE,
            TOTAL_CHUNKS=total_chunks,
            num_warps=4,
        )
        
        return y_flat.view_as(x)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
    
    def forward(self, x):
        return TritonLayerNorm.apply(x, self.weight, self.bias, self.normalized_shape, self.eps)
