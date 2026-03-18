import torch
import torch.nn as nn
import triton
import triton.language as tl


# v12: Multi-chunk-per-block approach
# Each block handles CHUNKS_PER_BLOCK chunks via grid-stride loop
# This reduces total blocks while keeping each block's work parallelizable
# Benefits: fewer blocks = less partial storage, cheaper reduce, less launch overhead

@triton.jit
def _ln_stats_multi(
    X, Partials,
    N: tl.constexpr,
    NUM_BLOCKS_PER_ROW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    CHUNKS_PER_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block_idx = tl.program_id(1)
    
    total_s = 0.0
    total_sq = 0.0
    
    for c in range(CHUNKS_PER_BLOCK):
        chunk = block_idx * CHUNKS_PER_BLOCK + c
        col_offset = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offset < N
        x = tl.load(X + row * N + col_offset, mask=mask, other=0.0).to(tl.float32)
        total_s += tl.sum(x, axis=0)
        total_sq += tl.sum(x * x, axis=0)
    
    idx = row * NUM_BLOCKS_PER_ROW + block_idx
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
def _ln_normalize(
    X, Y, W, B, Stats,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    mean = tl.load(Stats + row * 2)
    rstd = tl.load(Stats + row * 2 + 1)
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
        
        # Each block processes CHUNKS_PER_BLOCK chunks
        CHUNKS_PER_BLOCK = 16  # 4096/16 = 256 blocks per row
        num_blocks_per_row = total_chunks // CHUNKS_PER_BLOCK
        
        block_ceil = 1
        while block_ceil < num_blocks_per_row:
            block_ceil *= 2
        
        partials = torch.empty(batch_size * num_blocks_per_row * 2, device=x.device, dtype=torch.float32)
        stats = torch.empty(batch_size * 2, device=x.device, dtype=torch.float32)
        y_flat = torch.empty_like(x_flat)
        
        # Stats: 256 blocks per row * 16 rows = 4096 blocks
        _ln_stats_multi[(batch_size, num_blocks_per_row)](
            x_flat, partials,
            N=N, NUM_BLOCKS_PER_ROW=num_blocks_per_row,
            BLOCK_SIZE=BLOCK_SIZE, CHUNKS_PER_BLOCK=CHUNKS_PER_BLOCK,
            num_warps=2,
        )
        
        # Reduce: only 256 partials per row
        _ln_reduce_stats[(batch_size,)](
            partials, stats,
            N=N, NUM_BLOCKS=num_blocks_per_row, BLOCK_CEIL=block_ceil,
            eps=eps,
            num_warps=4,
        )
        
        # Normalize: 4096 chunks per row
        w_flat = weight.contiguous().view(-1)
        b_flat = bias.contiguous().view(-1)
        
        _ln_normalize[(batch_size, total_chunks)](
            x_flat, y_flat, w_flat, b_flat, stats,
            N=N, BLOCK_SIZE=BLOCK_SIZE,
            num_warps=2,
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
