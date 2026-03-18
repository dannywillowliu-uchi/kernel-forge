import torch
import torch.nn as nn
import triton
import triton.language as tl


# v18: Best config from sweep: stats BS=1024/NW=2, norm BS=512/NW=2 (2-chunk)
# Also: pre-allocate buffers to avoid allocation overhead in forward pass

@triton.jit
def _ln_partial_stats(
    X, Partials,
    N: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    chunk = tl.program_id(1)
    col_offset = chunk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offset < N
    x = tl.load(X + row * N + col_offset, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(x, axis=0)
    sq = tl.sum(x * x, axis=0)
    idx = row * NUM_CHUNKS + chunk
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
    
    # Chunk 0
    chunk0 = block_idx * 2
    col0 = chunk0 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask0 = col0 < N
    x0 = tl.load(X + row * N + col0, mask=mask0, other=0.0).to(tl.float32)
    w0 = tl.load(W + col0, mask=mask0, other=1.0).to(tl.float32)
    b0 = tl.load(B + col0, mask=mask0, other=0.0).to(tl.float32)
    y0 = (x0 - mean) * rstd * w0 + b0
    tl.store(Y + row * N + col0, y0, mask=mask0)
    
    # Chunk 1
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
        
        # Pre-compute constants
        self._N = 1
        for d in normalized_shape:
            self._N *= d
        
        self._STATS_BS = 1024
        self._NORM_BS = 512
        self._num_stats_chunks = (self._N + self._STATS_BS - 1) // self._STATS_BS
        self._total_norm_chunks = (self._N + self._NORM_BS - 1) // self._NORM_BS
        self._num_norm_blocks = (self._total_norm_chunks + 1) // 2
        
        self._chunk_block = 1
        while self._chunk_block < self._num_stats_chunks:
            self._chunk_block *= 2
        
        # Buffers allocated lazily
        self._partials = None
        self._stats = None
    
    def forward(self, x):
        batch_size = 1
        for d in x.shape[:-len(self.normalized_shape)]:
            batch_size *= d
        
        N = self._N
        x_flat = x.contiguous().view(batch_size, N)
        
        # Lazy buffer allocation
        needed_partials = batch_size * self._num_stats_chunks * 2
        if self._partials is None or self._partials.shape[0] < needed_partials:
            self._partials = torch.empty(needed_partials, device=x.device, dtype=torch.float32)
        if self._stats is None or self._stats.shape[0] < batch_size * 2:
            self._stats = torch.empty(batch_size * 2, device=x.device, dtype=torch.float32)
        
        y_flat = torch.empty_like(x_flat)
        
        _ln_partial_stats[(batch_size, self._num_stats_chunks)](
            x_flat, self._partials,
            N=N, NUM_CHUNKS=self._num_stats_chunks,
            BLOCK_SIZE=self._STATS_BS,
            num_warps=2,
        )
        
        _ln_reduce_stats[(batch_size,)](
            self._partials, self._stats,
            N=N, NUM_CHUNKS=self._num_stats_chunks,
            CHUNK_BLOCK=self._chunk_block,
            eps=self.eps,
            num_warps=4,
        )
        
        w_flat = self.weight.contiguous().view(-1)
        b_flat = self.bias.contiguous().view(-1)
        
        _ln_normalize_2x[(batch_size, self._num_norm_blocks)](
            x_flat, y_flat, w_flat, b_flat, self._stats,
            N=N, BLOCK_SIZE=self._NORM_BS,
            TOTAL_CHUNKS=self._total_norm_chunks,
            num_warps=2,
        )
        
        return y_flat.view_as(x)
