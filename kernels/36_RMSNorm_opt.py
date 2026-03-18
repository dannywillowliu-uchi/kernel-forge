import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

// v24: Best config from sweep: 256 threads, __launch_bounds__(256, 2)
// Two-pass with 8-wide ILP loads.
// Also try: interleave reads and accumulates differently.

__global__ __launch_bounds__(256, 2) void rmsnorm_kernel(
    const float* __restrict__ X,
    float* __restrict__ Y,
    const int HW,
    const float eps
) {
    const int batch_idx = blockIdx.y;
    const int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (spatial_idx >= HW) return;
    
    const float* x_base = X + (long)batch_idx * 64 * HW + spatial_idx;
    float* y_base = Y + (long)batch_idx * 64 * HW + spatial_idx;
    const long stride = HW;
    
    float sum_sq = 0.0f;
    float r0, r1, r2, r3, r4, r5, r6, r7;
    
    // Pass 1: 8-wide ILP accumulation
    r0 = x_base[0*stride]; r1 = x_base[1*stride]; r2 = x_base[2*stride]; r3 = x_base[3*stride];
    r4 = x_base[4*stride]; r5 = x_base[5*stride]; r6 = x_base[6*stride]; r7 = x_base[7*stride];
    sum_sq += r0*r0 + r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6 + r7*r7;
    
    r0 = x_base[8*stride]; r1 = x_base[9*stride]; r2 = x_base[10*stride]; r3 = x_base[11*stride];
    r4 = x_base[12*stride]; r5 = x_base[13*stride]; r6 = x_base[14*stride]; r7 = x_base[15*stride];
    sum_sq += r0*r0 + r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6 + r7*r7;
    
    r0 = x_base[16*stride]; r1 = x_base[17*stride]; r2 = x_base[18*stride]; r3 = x_base[19*stride];
    r4 = x_base[20*stride]; r5 = x_base[21*stride]; r6 = x_base[22*stride]; r7 = x_base[23*stride];
    sum_sq += r0*r0 + r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6 + r7*r7;
    
    r0 = x_base[24*stride]; r1 = x_base[25*stride]; r2 = x_base[26*stride]; r3 = x_base[27*stride];
    r4 = x_base[28*stride]; r5 = x_base[29*stride]; r6 = x_base[30*stride]; r7 = x_base[31*stride];
    sum_sq += r0*r0 + r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6 + r7*r7;
    
    r0 = x_base[32*stride]; r1 = x_base[33*stride]; r2 = x_base[34*stride]; r3 = x_base[35*stride];
    r4 = x_base[36*stride]; r5 = x_base[37*stride]; r6 = x_base[38*stride]; r7 = x_base[39*stride];
    sum_sq += r0*r0 + r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6 + r7*r7;
    
    r0 = x_base[40*stride]; r1 = x_base[41*stride]; r2 = x_base[42*stride]; r3 = x_base[43*stride];
    r4 = x_base[44*stride]; r5 = x_base[45*stride]; r6 = x_base[46*stride]; r7 = x_base[47*stride];
    sum_sq += r0*r0 + r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6 + r7*r7;
    
    r0 = x_base[48*stride]; r1 = x_base[49*stride]; r2 = x_base[50*stride]; r3 = x_base[51*stride];
    r4 = x_base[52*stride]; r5 = x_base[53*stride]; r6 = x_base[54*stride]; r7 = x_base[55*stride];
    sum_sq += r0*r0 + r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6 + r7*r7;
    
    r0 = x_base[56*stride]; r1 = x_base[57*stride]; r2 = x_base[58*stride]; r3 = x_base[59*stride];
    r4 = x_base[60*stride]; r5 = x_base[61*stride]; r6 = x_base[62*stride]; r7 = x_base[63*stride];
    sum_sq += r0*r0 + r1*r1 + r2*r2 + r3*r3 + r4*r4 + r5*r5 + r6*r6 + r7*r7;
    
    float inv_rms = rsqrtf(sum_sq * 0.015625f + eps);
    
    // Pass 2: Re-read + normalize with same ILP pattern
    r0 = x_base[0*stride]; r1 = x_base[1*stride]; r2 = x_base[2*stride]; r3 = x_base[3*stride];
    r4 = x_base[4*stride]; r5 = x_base[5*stride]; r6 = x_base[6*stride]; r7 = x_base[7*stride];
    y_base[0*stride] = r0*inv_rms; y_base[1*stride] = r1*inv_rms;
    y_base[2*stride] = r2*inv_rms; y_base[3*stride] = r3*inv_rms;
    y_base[4*stride] = r4*inv_rms; y_base[5*stride] = r5*inv_rms;
    y_base[6*stride] = r6*inv_rms; y_base[7*stride] = r7*inv_rms;
    
    r0 = x_base[8*stride]; r1 = x_base[9*stride]; r2 = x_base[10*stride]; r3 = x_base[11*stride];
    r4 = x_base[12*stride]; r5 = x_base[13*stride]; r6 = x_base[14*stride]; r7 = x_base[15*stride];
    y_base[8*stride] = r0*inv_rms; y_base[9*stride] = r1*inv_rms;
    y_base[10*stride] = r2*inv_rms; y_base[11*stride] = r3*inv_rms;
    y_base[12*stride] = r4*inv_rms; y_base[13*stride] = r5*inv_rms;
    y_base[14*stride] = r6*inv_rms; y_base[15*stride] = r7*inv_rms;
    
    r0 = x_base[16*stride]; r1 = x_base[17*stride]; r2 = x_base[18*stride]; r3 = x_base[19*stride];
    r4 = x_base[20*stride]; r5 = x_base[21*stride]; r6 = x_base[22*stride]; r7 = x_base[23*stride];
    y_base[16*stride] = r0*inv_rms; y_base[17*stride] = r1*inv_rms;
    y_base[18*stride] = r2*inv_rms; y_base[19*stride] = r3*inv_rms;
    y_base[20*stride] = r4*inv_rms; y_base[21*stride] = r5*inv_rms;
    y_base[22*stride] = r6*inv_rms; y_base[23*stride] = r7*inv_rms;
    
    r0 = x_base[24*stride]; r1 = x_base[25*stride]; r2 = x_base[26*stride]; r3 = x_base[27*stride];
    r4 = x_base[28*stride]; r5 = x_base[29*stride]; r6 = x_base[30*stride]; r7 = x_base[31*stride];
    y_base[24*stride] = r0*inv_rms; y_base[25*stride] = r1*inv_rms;
    y_base[26*stride] = r2*inv_rms; y_base[27*stride] = r3*inv_rms;
    y_base[28*stride] = r4*inv_rms; y_base[29*stride] = r5*inv_rms;
    y_base[30*stride] = r6*inv_rms; y_base[31*stride] = r7*inv_rms;
    
    r0 = x_base[32*stride]; r1 = x_base[33*stride]; r2 = x_base[34*stride]; r3 = x_base[35*stride];
    r4 = x_base[36*stride]; r5 = x_base[37*stride]; r6 = x_base[38*stride]; r7 = x_base[39*stride];
    y_base[32*stride] = r0*inv_rms; y_base[33*stride] = r1*inv_rms;
    y_base[34*stride] = r2*inv_rms; y_base[35*stride] = r3*inv_rms;
    y_base[36*stride] = r4*inv_rms; y_base[37*stride] = r5*inv_rms;
    y_base[38*stride] = r6*inv_rms; y_base[39*stride] = r7*inv_rms;
    
    r0 = x_base[40*stride]; r1 = x_base[41*stride]; r2 = x_base[42*stride]; r3 = x_base[43*stride];
    r4 = x_base[44*stride]; r5 = x_base[45*stride]; r6 = x_base[46*stride]; r7 = x_base[47*stride];
    y_base[40*stride] = r0*inv_rms; y_base[41*stride] = r1*inv_rms;
    y_base[42*stride] = r2*inv_rms; y_base[43*stride] = r3*inv_rms;
    y_base[44*stride] = r4*inv_rms; y_base[45*stride] = r5*inv_rms;
    y_base[46*stride] = r6*inv_rms; y_base[47*stride] = r7*inv_rms;
    
    r0 = x_base[48*stride]; r1 = x_base[49*stride]; r2 = x_base[50*stride]; r3 = x_base[51*stride];
    r4 = x_base[52*stride]; r5 = x_base[53*stride]; r6 = x_base[54*stride]; r7 = x_base[55*stride];
    y_base[48*stride] = r0*inv_rms; y_base[49*stride] = r1*inv_rms;
    y_base[50*stride] = r2*inv_rms; y_base[51*stride] = r3*inv_rms;
    y_base[52*stride] = r4*inv_rms; y_base[53*stride] = r5*inv_rms;
    y_base[54*stride] = r6*inv_rms; y_base[55*stride] = r7*inv_rms;
    
    r0 = x_base[56*stride]; r1 = x_base[57*stride]; r2 = x_base[58*stride]; r3 = x_base[59*stride];
    r4 = x_base[60*stride]; r5 = x_base[61*stride]; r6 = x_base[62*stride]; r7 = x_base[63*stride];
    y_base[56*stride] = r0*inv_rms; y_base[57*stride] = r1*inv_rms;
    y_base[58*stride] = r2*inv_rms; y_base[59*stride] = r3*inv_rms;
    y_base[60*stride] = r4*inv_rms; y_base[61*stride] = r5*inv_rms;
    y_base[62*stride] = r6*inv_rms; y_base[63*stride] = r7*inv_rms;
}

torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps) {
    const int B = x.size(0);
    const int HW = x.size(2) * x.size(3);
    
    auto y = torch::empty_like(x);
    
    const int THREADS = 256;
    dim3 grid((HW + THREADS - 1) / THREADS, B);
    dim3 block(THREADS);
    
    rmsnorm_kernel<<<grid, block>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        HW, eps
    );
    
    return y;
}
""" 

cpp_source = """
torch::Tensor rmsnorm_cuda(torch::Tensor x, float eps);
"""

rmsnorm_module = load_inline(
    name="rmsnorm_cuda_v24",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["rmsnorm_cuda"],
    verbose=False,
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
)


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm_module.rmsnorm_cuda(x, self.eps)
