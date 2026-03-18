// Pattern: Online softmax (single-pass numerically stable)
// Use for: softmax, attention score normalization
// Speedup: 1.1-1.5x over 2-pass, eliminates one HBM read
// Key: Milakov & Gimelshein algorithm - update running max+sum in one pass

// Standard 2-pass: pass1=find max, pass2=exp(x-max)/sum
// Online 1-pass: maintain running max and compensated sum simultaneously

__device__ void online_softmax_row(
    const float* __restrict__ input,
    float* __restrict__ output,
    int N  // row length
) {
    float max_val = -INFINITY;
    float sum_exp = 0.0f;

    // Single pass: track running max and compensated sum
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        float x = input[i];
        if (x > max_val) {
            // Rescale previous sum when max changes
            sum_exp = sum_exp * expf(max_val - x);
            max_val = x;
        }
        sum_exp += expf(x - max_val);
    }

    // Warp reduce the max and sum (need special handling)
    // ... (use warp_reduce_max + compensated sum merge)

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) * inv_sum;
    }
}

// FlashAttention uses this pattern:
// For each KV block:
//   new_max = max(old_max, row_max(QK^T))
//   rescale = exp(old_max - new_max)
//   O = O * rescale + exp(QK^T - new_max) @ V
//   sum = sum * rescale + row_sum(exp(QK^T - new_max))
