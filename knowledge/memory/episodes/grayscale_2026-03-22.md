# Episode: Grayscale v2 (RGB to Y conversion)

## Result
Best: 595.7us (16384x16384, cold cache), 599us on GPU MODE eval
Speedup: 9.0x over PyTorch reference
Hardware utilization: 99.3% of achievable HBM bandwidth (7.21 TB/s of 7.26 TB/s measured peak)
Leaderboard: #3 on B200 (599us, #1 is 597us)

## Approaches tried (in order)
1. CUDA float4 vectorized loads + __ldg + __fmaf_rn -> 826us total, 596us for 16384 -> baseline
2. Various thread counts (128-1024) -> all within 1us -> 1024 marginally best for cold cache
3. maxrregcount sweep (24-40) -> 28 marginally best
4. CuPy RawKernel -> same launch overhead as load_inline (myth busted)
5. ctypes pre-compiled -> same overhead
6. Triton -> 623us cold cache (worse than CUDA due to strided loads)
7. 8 pixels per thread -> worse (616us, reduced parallelism)
8. Grid-stride persistent kernel -> worse (626us)
9. Write-through stores (st.global.wt) -> same
10. Streaming stores (st.global.cs) -> marginally tighter tail latency
11. __launch_bounds__(1024, 1) with maxreg=28 -> 598us cold -> best config
12. sm_100 explicit target -> same

## Key insight
The kernel is at the hardware bandwidth limit. The remaining gap to #1 (1.7us) is noise/eval variance, not an optimization opportunity. The AoS (RGBRGB...) memory layout causes stride-48-byte access between adjacent threads -- this 10% structural overhead is unavoidable without changing the input format. CuPy does NOT have lower launch overhead than load_inline -- this is a myth.

## What would I do differently
1. Start with the CUDA float4 kernel immediately (it's the obvious right approach)
2. Don't waste time on CuPy/ctypes hoping for lower launch overhead
3. Test with cold-cache L2 flush from the start (warm cache times are misleading)
4. Accept 598us as the ceiling and submit early
