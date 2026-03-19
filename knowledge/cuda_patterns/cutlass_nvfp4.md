# CUTLASS NVFP4 GEMM on sm100 (Blackwell)

Requires: CUDA 12.8+, CUTLASS 3.x, sm100

## Element types
- `cutlass::nv_float4_t<cutlass::float_e2m1_t>` -- NVFP4
- `cutlass::mx_float4_t<cutlass::float_e2m1_t>` -- MXFP4
- `cutlass::float_ue8m0_t` -- output scale factors

## Key classes
- `OpClassBlockScaledTensorOp` -- operator class for block-scaled
- `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB()` -- computes scale layouts
- `LinCombBlockScaleFactor` -- epilogue that generates output scales

## Tile shape
- MmaTileShape: 128x128x256 typical for NVFP4
- AlignmentA/B: 32 (32 fp4 = 16 bytes)
- Block scale vector size: 16

## Example: CUTLASS example 72b
github.com/NVIDIA/cutlass/blob/main/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu
