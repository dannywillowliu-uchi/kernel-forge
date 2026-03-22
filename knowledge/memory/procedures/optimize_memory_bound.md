# Procedure: Optimize a Memory-Bound Kernel

## When to use
Arithmetic intensity < 120 FLOPs/byte. Examples: elementwise ops, reductions, normalization, simple transforms (grayscale, transpose).

## Steps
1. **Compute the bandwidth floor** -- total bytes read + written / HBM bandwidth (use 7.2 TB/s measured, not 8 TB/s spec)
2. **Write a CUDA kernel with float4 vectorized loads** -- 128-bit loads are mandatory for peak bandwidth. Use __ldg for read-only data.
3. **Use __fmaf_rn for FMA chains** -- reduces instruction count
4. **Choose 256-1024 threads/block** -- sweep to find best for your access pattern. 1024 is often best for cold-cache workloads.
5. **Use -maxrregcount=28** -- forces higher occupancy, helps hide memory latency
6. **Measure with cold L2 cache** -- flush L2 with a 128MB zero buffer before each timed iteration
7. **If within 10% of bandwidth floor, stop** -- you're at the hardware limit

## Common mistakes
- Expecting CuPy or ctypes to have lower launch overhead than load_inline (they don't)
- Measuring with warm cache (misleading for bandwidth-bound kernels)
- Trying Triton for simple elementwise (strided loads are slower than float4)
- Over-investing in launch overhead optimization when the kernel itself dominates

## Evidence
- grayscale_v2: achieved 99.3% measured bandwidth using this procedure
- RMSNorm (SOL-ExecBench): achieved 93.6% SOL Score using similar approach
