// Pattern: Vectorized 128-bit loads using float4
// Use for: memory-bound kernels where load throughput is the bottleneck
// Speedup: 1.3-2x by reducing instruction count 4x
// When: data is 16-byte aligned and dimension divisible by 4

// Basic pattern: load 4 floats at once
__device__ __forceinline__ float4 load_float4(const float* ptr) {
    return *reinterpret_cast<const float4*>(ptr);
}

__device__ __forceinline__ void store_float4(float* ptr, float4 val) {
    *reinterpret_cast<float4*>(ptr) = val;
}

// Elementwise kernel with vectorized loads:
__global__ void elementwise_vec4(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N  // must be divisible by 4
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx >= N) return;

    float4 in = load_float4(input + idx);
    float4 out;
    // Apply operation to each component
    out.x = fmaxf(in.x, 0.0f);  // ReLU example
    out.y = fmaxf(in.y, 0.0f);
    out.z = fmaxf(in.z, 0.0f);
    out.w = fmaxf(in.w, 0.0f);
    store_float4(output + idx, out);
}

// For BF16: use __nv_bfloat162 for 2-element vectorized loads
// For FP16: use half2 for 2-element vectorized loads
// For int8/FP8: use int4 (uint4) for 16-element loads

// Alignment: cudaMalloc guarantees 256-byte alignment
// Requirement: N % 4 == 0 for float4, pad if needed
