# Episode: Prefix Sum v2 (inclusive scan, float32, 256M elements)

## Result
Best: 482us mean on GPU MODE leaderboard (B200 on Modal)
Local: 478us mean, 475us best (B200 local)
Speedup: 2.2x over torch.cumsum baseline (1080us -> 482us)
Approach: Triton 3-pass (reduce, cumsum block sums, scan+add)

## Approaches tried (in order)
1. torch.cumsum baseline -> 1080us on Modal, 625us local
2. CUB DeviceScan::InclusiveSum -> 553us on Modal, 540us local (single-pass decoupled lookback)
3. Thrust inclusive_scan -> 2970us (much worse, older algorithm)
4. Custom 3-pass CUDA (reduce + CUB scan block sums + CUB BlockScan) -> 522-640us (CUB BlockLoad/BlockStore transpose overhead)
5. Triton 3-pass (tl.sum reduce + torch.cumsum + tl.cumsum scan_add) -> 477-484us **BEST**
6. Triton+CUB hybrid (Triton passes + CUB for middle) -> 478us (same as pure Triton)
7. Custom decoupled lookback (multiple attempts) -> all deadlocked or >5000us
8. Persistent kernel with decoupled lookback -> deadlocked
9. tl.associative_scan vs tl.cumsum -> identical performance
10. Cooperative groups grid sync -> compilation issues with load_inline

## Key insights
1. **Triton 3-pass beats CUB single-pass** on B200 for this workload (477us vs 540us). CUB's decoupled lookback spin-waiting adds more overhead than the extra data pass.
2. **3-pass data flow: 3GB total** (1GB read for reduce, 1GB read + 1GB write for scan+add). Theoretical floor at 7.2 TB/s = 417us. Achieved 87% efficiency.
3. **Per-pass breakdown**: reduce = 166us (1GB read, 6.5 TB/s), torch.cumsum middle = 11us, scan+add = 309us (2GB r/w, 6.5 TB/s).
4. **Decoupled lookback deadlocks** on B200 because spin-waiting thread 0 blocks the SM, preventing other blocks from making progress. Even with atomic tile counter and 1 block/SM, deadlocks due to tile ordering.
5. **Triton num_warps=4 is optimal** for BS=1024. More warps (8+) significantly degrades performance.
6. **Block size 1024-4096 are all equivalent** (~477-479us). BS=512 is worse (596us) due to too many blocks.
7. **L2 cache irrelevant**: 1GB input doesn't fit in 126MB L2, so this is purely HBM-bound.

## What I would do differently
1. Start with Triton 3-pass immediately (reduce + cumsum + scan_add). Don't waste time on CUB or custom CUDA.
2. Don't attempt decoupled lookback -- it's extremely difficult to implement correctly and doesn't beat the simpler 3-pass on B200.
3. Focus on per-pass bandwidth utilization rather than reducing passes. The 3-pass approach at 87% bandwidth efficiency is hard to beat.
4. The theoretical improvement from 3-pass (3GB) to 1-pass (2GB) is only 33%, bringing floor from 417us to 278us. Not worth the complexity.
