import torch
import torch.nn as nn
import triton
import triton.language as tl


# v19: Combine 2-chunk stats + 2-chunk normalize
# Stats: BS=1024, 2 chunks per block = effective 2048 per block
# Normalize: BS=512, 2 chunks per block = effective 1024 per block

@triton.jit
def _ln_partial_stats_2x(
    X, Partials,
    N: tl.constexpr,
    TOTAL_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_STAT_BLOCKS: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    # First chunk
    chunk0 = block_idx * 2
    col0 = chunk0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask0 = col0 < N
    x0 = tl.load(X + row * N + col0, mask=mask0, other=0.0).to(tl.float32)
    s = tl.sum(x0, axis=0)
    sq = tl.sum(x0 * x0, axis=0)
    
    # Second chunk
    chunk1 = block_idx * 2 + 1
    if chunk1 < TOTAL_CHUNKS:
        col1 = chunk1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask1 = col1 < N
        x1 = tl.load(X + row * N + col1, mask=mask1, other=0.0).to(tl.float32)
        s += tl.sum(x1, axis=0)
        sq += tl.sum(x1 * x1, axis=0)
    
    idx = row * NUM_STAT_BLOCKS + block_idx
    tl.store(Partials + idx * 2, s)
    tl.store(Partials + idx * 2 + 1, sq)


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
def _ln_normalize_2x(
    X, Y, W, B, Stats,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOTAL_CHUNKS: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    mean = tl.load(Stats + row * 2)
    rstd = tl.load(Stats + row * 2 + 1)
    
    chunk0 = block_idx * 2
    col0 = chunk0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask0 = col0 < N
    x0 = tl.load(X + row * N + col0, mask=mask0, other=0.0).to(tl.float32)
    w0 = tl.load(W + col0, mask=mask0, other=1.0).to(tl.float32)
    b0 = tl.load(B + col0, mask=mask0, other=0.0).to(tl.float32)
    y0 = (x0 - mean) * rstd * w0 + b0
    tl.store(Y + row * N + col0, y0, mask=mask0)
    
    chunk1 = block_idx * 2 + 1
    if chunk1 < TOTAL_CHUNKS:
        col1 = chunk1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask1 = col1 < N
        x1 = tl.load(X + row * N + col1, mask=mask1, other=0.0).to(tl.float32)
        w1 = tl.load(W + col1, mask=mask1, other=1.0).to(tl.float32)
        b1 = tl.load(B + col1, mask=mask1, other=0.0).to(tl.float32)
        y1 = (x1 - mean) * rstd * w1 + b1
        tl.store(Y + row * N + col1, y1, mask=mask1)


class ModelNew(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = 1e-5
    
    def forward(self, x):
        batch_size = 1
        for d in x.shape[:-len(self.normalized_shape)]:
            batch_size *= d
        
        N = 1
        for d in self.normalized_shape:
            N *= d
        
        x_flat = x.contiguous().view(batch_size, N)
        
        STATS_BS = 1024
        NORM_BS = 512
        
        total_stats_chunks = (N + STATS_BS - 1) // STATS_BS
        num_stat_blocks = (total_stats_chunks + 1) // 2
        
        total_norm_chunks = (N + NORM_BS - 1) // NORM_BS
        num_norm_blocks = (total_norm_chunks + 1) // 2
        
        block_ceil = 1
        while block_ceil < num_stat_blocks:
            block_ceil *= 2
        
        partials = torch.empty(batch_size * num_stat_blocks * 2, device=x.device, dtype=torch.float32)
        stats = torch.empty(batch_size * 2, device=x.device, dtype=torch.float32)
        y_flat = torch.empty_like(x_flat)
        
        _ln_partial_stats_2x[(batch_size, num_stat_blocks)](
            x_flat, partials,
            N=N, TOTAL_CHUNKS=total_stats_chunks,
            BLOCK_SIZE=STATS_BS, NUM_STAT_BLOCKS=num_stat_blocks,
            num_warps=2,
        )
        
        _ln_reduce_stats[(batch_size,)](
            partials, stats,
            N=N, NUM_BLOCKS=num_stat_blocks, BLOCK_CEIL=block_ceil,
            eps=self.eps,
            num_warps=4,
        )
        
        w_flat = self.weight.contiguous().view(-1)
        b_flat = self.bias.contiguous().view(-1)
        
        _ln_normalize_2x[(batch_size, num_norm_blocks)](
            x_flat, y_flat, w_flat, b_flat, stats,
            N=N, BLOCK_SIZE=NORM_BS,
            TOTAL_CHUNKS=total_norm_chunks,
            num_warps=2,
        )
        
        return y_flat.view_as(x)
