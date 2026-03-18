// Pattern: Shared memory tiling with bank conflict avoidance
// Use for: matmul, convolution, any op with data reuse
// Speedup: 3-10x for compute-bound kernels
// When: same data read by multiple threads (high arithmetic intensity)

// Bank-conflict-free shared memory: pad by 1 element
// Banks are 4 bytes wide, 32 banks. Stride of 32 causes conflicts.
// Adding +1 shifts bank alignment.
__shared__ float As[TILE_M][TILE_K + 1];  // +1 avoids conflicts
__shared__ float Bs[TILE_K][TILE_N + 1];

// Basic tiled matmul pattern:
template<int BM, int BN, int BK>
__global__ void matmul_tiled(
    const float* __restrict__ A,  // [M, K]
    const float* __restrict__ B,  // [K, N]
    float* __restrict__ C,        // [M, N]
    int M, int K, int N
) {
    __shared__ float As[BM][BK + 1];
    __shared__ float Bs[BK][BN + 1];

    int row = blockIdx.y * BM + threadIdx.y;
    int col = blockIdx.x * BN + threadIdx.x;
    float acc = 0.0f;

    for (int tile = 0; tile < (K + BK - 1) / BK; ++tile) {
        // Coalesced load from global to shared
        int a_col = tile * BK + threadIdx.x;
        int b_row = tile * BK + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        // Compute from shared memory
        #pragma unroll
        for (int k = 0; k < BK; ++k)
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// Launch: dim3 grid((N+BN-1)/BN, (M+BM-1)/BM), dim3 block(BN, BM)
// Optimal BM=BN=128, BK=16 on B200
