// Pattern: Warp-level reduction using __shfl_down_sync
// Use for: sum, max, min within a warp (32 threads) without shared memory
// Speedup: 2-5x vs shared memory reduction for small reductions
// When: reduction dimension <= 1024 (fits in a few warps)

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// Block-level reduce: warp reduce + shared memory for cross-warp
template<int BLOCK_SIZE>
__device__ float block_reduce_sum(float val) {
    __shared__ float smem[BLOCK_SIZE / 32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();

    val = (threadIdx.x < BLOCK_SIZE / 32) ? smem[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    return val;
}

// Usage in a kernel:
// float thread_sum = 0.0f;
// for (int i = threadIdx.x; i < N; i += blockDim.x)
//     thread_sum += data[i];
// float block_sum = block_reduce_sum<256>(thread_sum);
