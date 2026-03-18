import torch
import torch.nn as nn
import triton
import triton.language as tl


# v13: Persistent kernel approach
# Launch exactly enough CTAs to fill the GPU (e.g., 160 SMs * 2 blocks/SM = 320)
# Each CTA processes multiple tiles in a grid-stride loop
# This eliminates kernel launch overhead for the huge grid

@triton.jit
def _ln_persistent_stats(
    X, Partials,
    N: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Grid-stride loop over all tiles
    for tile_id in range(pid, TOTAL_TILES, NUM_CTAS):
        row = tile_id // TILES_PER_ROW
        chunk = tile_id % TILES_PER_ROW
        
        col_offset = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = col_offset < N
        x = tl.load(X + row * N + col_offset, mask=mask, other=0.0).to(tl.float32)
        s = tl.sum(x, axis=0)
        sq = tl.sum(x * x, axis=0)
        
        idx = row * TILES_PER_ROW + chunk
        tl.store(Partials + idx * 2, s)
        tl.store(Partials + idx * 2 + 1, sq)


@triton.jit
def _ln_reduce_stats(
    Partials, Stats,
    N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    CHUNK_BLOCK: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    chunk_ids = tl.arange(0, CHUNK_BLOCK)
    mask = chunk_ids < NUM_CHUNKS
    idx = row * NUM_CHUNKS + chunk_ids
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
def _ln_persistent_norm(
    X, Y, W, B, Stats,
    N: tl.constexpr,
    batch_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TOTAL_TILES: tl.constexpr,
    NUM_CTAS: tl.constexpr,
    TILES_PER_ROW: tl.constexpr,
):
    pid = tl.program_id(0)
    
    for tile_id in range(pid, TOTAL_TILES, NUM_CTAS):
        row = tile_id // TILES_PER_ROW
        chunk = tile_id % TILES_PER_ROW
        
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
        tiles_per_row = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        total_tiles = batch_size * tiles_per_row
        
        # B200 has 160 SMs, launch enough CTAs
        NUM_CTAS = 640  # 4 blocks per SM
        
        chunk_block = 1
        while chunk_block < tiles_per_row:
            chunk_block *= 2
        
        partials = torch.empty(batch_size * tiles_per_row * 2, device=x.device, dtype=torch.float32)
        stats = torch.empty(batch_size * 2, device=x.device, dtype=torch.float32)
        y_flat = torch.empty_like(x_flat)
        
        _ln_persistent_stats[(NUM_CTAS,)](
            x_flat, partials,
            N=N, batch_size=batch_size,
            BLOCK_SIZE=BLOCK_SIZE, TOTAL_TILES=total_tiles,
            NUM_CTAS=NUM_CTAS, TILES_PER_ROW=tiles_per_row,
            num_warps=2,
        )
        
        _ln_reduce_stats[(batch_size,)](
            partials, stats,
            N=N, NUM_CHUNKS=tiles_per_row, CHUNK_BLOCK=chunk_block,
            eps=eps,
            num_warps=4,
        )
        
        w_flat = weight.contiguous().view(-1)
        b_flat = bias.contiguous().view(-1)
        
        _ln_persistent_norm[(NUM_CTAS,)](
            x_flat, y_flat, w_flat, b_flat, stats,
            N=N, batch_size=batch_size,
            BLOCK_SIZE=BLOCK_SIZE, TOTAL_TILES=total_tiles,
            NUM_CTAS=NUM_CTAS, TILES_PER_ROW=tiles_per_row,
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
