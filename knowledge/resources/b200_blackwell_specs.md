# NVIDIA B200 / Blackwell Architecture Reference

## Hardware Specifications

| Parameter | Value | Notes |
|---|---|---|
| SM count | 148 | Dual-die: 2 x 74 enabled SMs |
| CUDA cores per SM | 128 | FP32/INT32 |
| Tensor Cores per SM | 4 | 5th-generation (tcgen05) |
| Shared memory per SM | Up to 228 KB | Configurable carveout |
| Tensor Memory (TMEM) per SM | 256 KB | Dedicated accumulator register file for Tensor Cores |
| Register file per SM | 64 KB (256K 32-bit registers) | 255 max per thread |
| L2 cache | 126 MB | Partitioned across dual-die; 2.5x H100's 50 MB |
| L2 bandwidth | 21 TB/s local, 16.8 TB/s cross-partition | |
| HBM3e capacity | 192 GB | |
| HBM3e bandwidth | 8 TB/s | 8 stacks |
| TMEM read BW | 16 TB/s per SM | |
| TMEM write BW | 8 TB/s per SM | |
| Peak FP4 | 7,702 TFLOPS | E2M1 (NVFP4) format |
| Peak FP8 | 3,851 TFLOPS | E4M3 / E5M2 |
| Peak BF16 | 1,929 TFLOPS | |
| Peak FP32 | 481 TFLOPS | |
| Peak FP64 | 44.8 TFLOPS | |
| Max warps per SM | 64 (sm100) | |
| Max thread blocks per SM | 32 | |
| Max cluster size (portable) | 8 | Non-portable: 16 on B200 |
| Compute capability | sm_100 | |
| CUDA version | 13.0 | |

## Blackwell-Specific Features

### tcgen05 (5th-gen Tensor Cores)
- Outputs directly to TMEM (not registers), enabling true 3-stage pipeline: TMA-load / MMA-compute / TMEM-store
- Single-thread dispatch (vs Hopper's warpgroup-collective wgmma)
- Max single-SM tile: 128x256x16 BF16; with 2-SM CTA pairs: 256x256x16
- Supports FP4, FP6, FP8, BF16, TF32, FP64
- TMEM: 420-cycle miss latency (58% reduction vs Hopper ~1000 cycles)
- Must explicitly alloc/dealloc TMEM; all CTA warps must be active during MMA

### TMA (Tensor Memory Accelerator)
- Single-thread issues bulk async copy of entire tiles from GMEM to SMEM
- Hardware address calculation eliminates register pressure
- Supports 1D-5D tensor copies with automatic boundary predication
- Multicast: one TMA op feeds all CTAs in a cluster (up to 16 on B200)

### Thread Block Clusters
- Up to 16 CTAs co-scheduled on adjacent SMs
- Distributed shared memory: CTAs access each other's SMEM
- TMA multicast reduces redundant global loads by cluster_size factor

### Key Differences from Hopper
- FP16 on CUDA cores runs at SAME rate as FP32 (not 2x); must use tensor cores for FP16 throughput
- SMEM up to 228 KB (vs 228 KB on Hopper -- similar, but TMEM is new)
- 4-5 pipeline stages often needed (vs 2-3 on Hopper) due to 420-cycle TMEM miss
- FlashAttention-4 on B200: 1,613 TFLOPS/s (71% utilization), 1.3x faster than cuDNN, 2.7x faster than Triton

## Key Optimization Guidelines for B200

1. **Tile sizes:** M and N must be multiples of 128 for full tensor core throughput (64x64 runs at 1/4 speed)
2. **SMEM carveout:** Request up to 228 KB explicitly via cudaFuncSetAttribute
3. **Use TMA** for all loads (not cp.async); single thread issues entire tile
4. **Warp specialization:** Producer warps (TMA) + consumer warps (MMA) for >70% utilization
5. **L2 cache:** 126 MB enables persistence of weight matrices for repeated inference
6. **NVFP4:** 7,702 TFLOPS available for inference; 2x throughput vs FP8
