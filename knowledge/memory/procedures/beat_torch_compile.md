# Procedure: When and How to Beat torch.compile

## When to use
When torch.compile gives good results but you suspect there's headroom from cross-boundary fusion.

## When NOT to use
When torch.compile is already within 10% of hardware peak. In that case, accept the result.

## Steps
1. **Profile the compiled kernel** -- identify the separate CUDA kernels torch.compile generates
2. **Identify fusion boundaries** -- torch.compile CANNOT fuse:
   - LayerNorm + Linear projection
   - Any operation + matmul (aten::mm, aten::bmm)
   - Operations across .contiguous() calls
3. **Estimate savings from cross-boundary fusion** -- sum the time of kernels that COULD be fused. If < 15% of total, it's not worth the effort.
4. **Check if a library already has the fused kernel** -- Liger, cuEquivariance, FlashAttention, Apex. Verify architecture matches exactly.
5. **If no library match, write custom Triton** -- fuse the specific operations that torch.compile keeps separate. Start with the largest time contributor.
6. **Compare fairly** -- torch.compile's auto-fusion is surprisingly good. Custom kernels often end up slower because of worse instruction scheduling.

## Common mistakes
- Assuming torch.compile is slow without profiling (it often generates near-optimal code)
- Trying to replace cuBLAS matmul with custom Triton (cuBLAS wins for standard shapes)
- Using library kernels without checking architecture match (cuEquivariance has different gate ordering than some models)
- Splitting the torch.compile graph (breaks fusion, adds overhead)

## Evidence
- TriMul: torch.compile at 0.78ms, all custom approaches were slower or matched
- TriMul: cuEquivariance components individually fast but total slower due to kernel launch overhead between them
- Attention backward: torch.compile with TF32 gave 3.5-5x over reference
