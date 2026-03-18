// Pattern: NVFP4 (E2M1) encoding and decoding
// Use for: WaferBench NVFP4 problems, inference quantization on B200
// Context: FP4 is Blackwell-exclusive, 4-bit float with 2-level scaling
//   Format: 1 sign + 2 exponent + 1 mantissa bits
//   Range: [-6, 6], values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
//   Scaling: E4M3 FP8 scale per 16-value block + FP32 global scale
//   Packed: 2 FP4 values per uint8_t byte

// From GPT-5.4's winning WaferBench submission:
__device__ __forceinline__ uint8_t float_to_fp4_e2m1(float val) {
    uint8_t sign = 0;
    if (val < 0.0f) { sign = 1; val = -val; }
    if (val > 6.0f) val = 6.0f;  // clamp to FP4 max
    uint8_t encoded;
    if      (val <= 0.25f) encoded = 0b000;  // 0
    else if (val < 0.75f)  encoded = 0b001;  // 0.5
    else if (val <= 1.25f) encoded = 0b010;  // 1.0
    else if (val < 1.75f)  encoded = 0b011;  // 1.5
    else if (val <= 2.5f)  encoded = 0b100;  // 2.0
    else if (val < 3.5f)   encoded = 0b101;  // 3.0
    else if (val <= 5.0f)  encoded = 0b110;  // 4.0
    else                   encoded = 0b111;  // 6.0
    return (sign << 3) | encoded;
}

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t encoded) {
    static __device__ const float LUT[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,   // positive
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // negative
    };
    return LUT[encoded & 0xF];
}

// E4M3 FP8 scale encoding (for block-level scales)
__device__ __forceinline__ uint8_t float_to_e4m3(float val) {
    if (val <= 0.0f) return 0;
    if (val > 448.0f) val = 448.0f;  // E4M3 max
    // Use CUDA's built-in conversion
    return __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float e4m3_to_float(uint8_t val) {
    return __nv_cvt_fp8_to_float(val, __NV_E4M3);
}

// Pack two FP4 values into one byte
__device__ __forceinline__ uint8_t pack_fp4(uint8_t low, uint8_t high) {
    return (high << 4) | (low & 0xF);
}

// Fused quantization: float -> FP4 with block scaling
// Each block of 16 values gets one E4M3 scale factor
// Overall tensor gets one FP32 global scale
// Effective bits per value: 4 + 8/16 = 4.5 bits
template<int BLOCK_SIZE_Q>  // typically 16
__global__ void quantize_to_fp4(
    const float* __restrict__ input,     // [rows, cols]
    uint8_t* __restrict__ fp4_output,    // [rows, cols/2] packed
    uint8_t* __restrict__ block_scales,  // [rows, cols/BLOCK_SIZE_Q] E4M3
    float global_scale,
    int rows, int cols
) {
    int row = blockIdx.y;
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_blocks = cols / BLOCK_SIZE_Q;
    if (block_idx >= num_blocks) return;

    int col_start = block_idx * BLOCK_SIZE_Q;

    // Find max absolute value in this block
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_Q; ++i) {
        float v = input[row * cols + col_start + i];
        amax = fmaxf(amax, fabsf(v));
    }

    // Compute block scale: scale * global_scale should map amax to FP4 max (6.0)
    float scale = amax / (6.0f * global_scale);
    float inv_scale = (scale > 0.0f) ? 1.0f / scale : 0.0f;

    // Store block scale as E4M3
    block_scales[row * num_blocks + block_idx] = float_to_e4m3(scale);

    // Quantize and pack
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_Q; i += 2) {
        float v0 = input[row * cols + col_start + i] * inv_scale * (1.0f / global_scale);
        float v1 = input[row * cols + col_start + i + 1] * inv_scale * (1.0f / global_scale);
        uint8_t q0 = float_to_fp4_e2m1(v0);
        uint8_t q1 = float_to_fp4_e2m1(v1);
        fp4_output[row * (cols / 2) + (col_start + i) / 2] = pack_fp4(q0, q1);
    }
}
