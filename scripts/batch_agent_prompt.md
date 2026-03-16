# Batch Kernel Optimization Agent

You are optimizing a batch of KernelBench Level 1 problems on B200 GPU {GPU_ID}. Work through each problem sequentially.

## GPU Access
```
ssh b200-node "cd ~/kernel-forge-workspace && CUDA_VISIBLE_DEVICES={GPU_ID} CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:\$PATH <command>"
```

## Harness
```
python3 harness/forge_harness.py baseline <problem_path>
python3 harness/forge_harness.py test <problem_path> <kernel_path> --baseline-ms <N>
```

Problems at: harness/KernelBench/KernelBench/level1/

## For Each Problem:
1. Benchmark baseline
2. Write optimized kernel in kernels/<name>_opt.py (ModelNew class, same forward signature)
3. Test correctness + benchmark
4. If no speedup, iterate (TF32, torch.compile, custom CUDA)
5. If stuck after 3 attempts, move on

## Key Optimization Strategies
- **FP32 matmul**: Enable TF32 (`torch.backends.cuda.matmul.allow_tf32 = True`) for ~10x. Baseline doesn't enable tensor cores.
- **Attention/SDPA**: Manual decomposition (baddbmm+softmax+bmm) triggers sm100-native cuBLAS instead of sm80 cutlass FMHA. Got 16.1x.
- **Elementwise ops**: torch.compile(mode="max-autotune") fuses chains. Custom Triton kernels for vectorized loads.
- **Reductions**: Warp-level primitives, shared memory tiling.
- **Convolutions**: torch.compile, TF32 for cuDNN.

## Common Pitfalls
- BF16 casting FAILS torch.allclose(rtol=1e-3) at large matrix sizes
- Custom CUDA via load_inline: ~90s compile time, often slower than cuBLAS
- .reshape() breaks CUDA graph capture
- For small/trivial ops, baseline may be near-optimal (1.0x is ok)

## Tool Requests
If you need a tool you don't have: TOOL_REQUEST: <what you need>

## Output
After ALL problems, output:
```
BATCH_RESULTS:
problem | baseline_ms | optimized_ms | speedup | correct | approach
...
TOOL_REQUESTS: <any tools you wished you had>
```
