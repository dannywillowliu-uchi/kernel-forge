# Episode: Histogram v2 (uint8 -> 256 bins, atomic contention test)

## Result
Leaderboard: 14.9us mean, 14.3us best (B200 on Modal)
Target: 12.054us (#1 by agac.mp4)
Speedup: 13x over PyTorch bincount reference (198us)
Kernel CUDA time: 7.5us (profiler), memset: 1.3us, total CUDA: 8.8us

## Approach
Per-warp sub-histograms in shared memory with 256 threads (8 warps).
uint4 vectorized loads (16 bytes per load), __ldg for read-only data.
cudaMemsetAsync for output zeroing (overlaps with kernel launch).
atomicAdd to per-warp shared memory histograms, then reduce across warps.

## Approaches tried (in order)
1. Per-warp smem histograms + uint4 loads (256 threads, 160 blocks) -> 17.6us leaderboard
2. Warp-aggregated atomics (__match_any_sync) -> 99us (per-byte vote too expensive)
3. 1024 threads (32 warps) -> same 17us (more warps doesn't help)
4. cudaMemsetAsync instead of zero_() -> 16.2us locally (saves 0.3us memset)
5. Register caching (4-slot LRU before smem atomic) -> 20.4us (branching overhead)
6. Per-thread histograms (128KB smem, uint16 counters) -> 47us (smem too large)
7. Direct global memory atomics (L2 cached) -> 1972us (terrible contention)
8. Double uint4 loads per iteration -> same 8.4us kernel
9. Block count sweep: T=256 B=320 best -> 14.1us locally
10. Padded smem layout (stride 257) -> same performance
11. CUB DeviceHistogram (int32) -> 6.85us kernel but 3 launches = 15.2us total
12. CUB with int64 output -> 130us (64-bit smem atomics much slower)
13. CUB BlockHistogram -> 84us (not designed for large data)
14. Interleaved smem layout (bin * STRIDE + warp_id) -> 7.65us (marginal)
15. Inline PTX atom.shared.add.u32 -> same (compiler already optimal)
16. Cooperative groups grid sync -> linker error with load_inline
17. Atomic flag grid sync (in-kernel zero) -> 9.2us (spin-wait overhead)
18. 128 threads (4 warps) -> 9.4us (more contention per warp)
19. Software pipelining (prefetch next uint4) -> same (compiler already does it)
20. Triton tl.histogram -> 62us
21. L2 persistence policy -> no effect (output naturally fits in L2)

## Key insights
1. Shared memory atomicAdd contention is the bottleneck, not global memory bandwidth (10MB = 1.4us at 7.2TB/s, but kernel takes 7.5us)
2. CUB's DeviceHistogram (int32) has a faster kernel (6.85 vs 7.5us) but 3 kernel launches vs 2 makes it slower end-to-end
3. CUDA event timing overhead is ~5-6us on B200 (dominates for fast kernels)
4. __match_any_sync is too expensive per-byte (use it sparingly, not 16x per load)
5. Blackwell handles smem bank conflicts well in hardware; manual bank conflict avoidance doesn't help much
6. 64-bit shared memory atomics are ~20x slower than 32-bit on B200
7. More sub-histograms (more warps) helps up to 8 warps, then diminishing returns
8. Block count matters: 320 blocks (2 per SM) optimal for 10M elements

## Gap to #1 (12.054us)
The gap is ~2us. Possible explanations:
- They may use CUB with a custom fused init+widen kernel (2 launches instead of 3)
- They may have found a way to reduce CUDA event overhead
- They may use CUDA graphs or other launch-reducing techniques
- Their kernel may be fundamentally faster via a different algorithm

## What I'd do differently
1. Focus earlier on measuring CUDA event overhead to understand the ceiling
2. Try CUB's AgentHistogram internals directly (custom kernel with CUB's optimized inner loop)
3. Consider writing a single-kernel CUB-style implementation that handles int64 output natively
