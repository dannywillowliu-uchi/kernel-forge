# Gotchas

## [2026-03-15 01:00] transformer_decode
torch.compile recompilation triggered when function signature changes from (Tensor, None) to (Tensor, Tensor) -- caused 64% regression on decode kernel. The compiled graph is specialized to the argument types at first call. Changing argument types forces a full recompile.

## [2026-03-15 01:00] cuda_graph_capture
.reshape() breaks CUDA graph capture because it may conditionally return a view OR a copy depending on memory layout. This non-determinism breaks CUDA graph replay which requires every operation to behave identically. Always use .contiguous().view() instead -- it fails loudly if the tensor isn't contiguous rather than silently copying.

## [2026-03-15 01:00] precision_casting
torch.autocast context manager adds per-operation type checking overhead. Each matmul/linear goes through autocast dispatch logic. Casting model weights to bf16 explicitly at init (model.bfloat16()) and removing all autocast context managers gave +47% throughput. Trade-off: slightly lower accuracy since bf16 is used everywhere including LayerNorm accumulation.

## [2026-03-15 01:00] quantization_overhead
FP8 and INT8 weight-only quantization via torchao was 15x and 42% slower respectively at batch=1. Dequantization overhead dominates at small batch sizes where the bandwidth savings are minimal. Quantization only helps when the model is memory-bandwidth-bound at large batch sizes.

## [2026-03-15 01:00] torch_compile_modes
torch.compile mode="reduce-overhead" (CUDA graph trees) fails when KV cache uses .copy_() operations -- detected as "mutated inputs". mode="max-autotune" (Triton kernels) was 50x slower for tiny batch=1 decode shapes because Triton kernels have high launch overhead for small workloads. mode="default" is the reliable choice for kernel fusion without overhead.

## [2026-03-15 01:00] cuda_graph_kv_cache
Manual CUDA graph capture requires replacing .copy_() with .scatter_() for KV cache writes (scatter uses static destination tensor address). Also requires passing full-length cache + boolean mask instead of dynamically sliced cache[:, :, :kv_len, :]. This combination of compile + manual CUDA graph gave +39% over compile alone.

## [2026-03-16 08:43:08 UTC] (ref: 1_Square_matrix_multiplication_)

Problem 1_Square_matrix_multiplication_: strategy 'shared_memory_tiling' failed with correctness_failure. Error: Traceback (most recent call last):
  File "/home/danny/kernel-forge-workspace/harness/forge_harness.py", line 120, in <module>
    cmd_test(args)
  Fi

## [2026-03-16 08:54:01 UTC] (ref: 23_Softmax)

Problem 23_Softmax: strategy 'online_softmax' failed with correctness_failure. Error: Traceback (most recent call last):
  File "/home/danny/kernel-forge-workspace/harness/forge_harness.py", line 120, in <module>
    cmd_test(args)
  Fi

## [2026-03-16 08:54:01 UTC] (ref: 23_Softmax)

Problem 23_Softmax: strategy 'Shared Memory Tiling with Fused Reductions' failed with correctness_failure. Error: Traceback (most recent call last):
  File "/home/danny/kernel-forge-workspace/harness/forge_harness.py", line 120, in <module>
    cmd_test(args)
  Fi

## [2026-03-16 08:59:23 UTC] (ref: 26_GELU_)

Problem 26_GELU_: strategy 'vectorized_loads_stores' failed with compilation_error. Error: Traceback (most recent call last):
  File "/home/danny/.local/lib/python3.12/site-packages/torch/utils/cpp_extension.py", line 2693, in _run_ninja_bui

## [2026-03-16 09:08:39 UTC] (ref: 37_FrobeniusNorm_)

Problem 37_FrobeniusNorm_: strategy 'Kernel Fusion with Streaming' failed with timeout. Error: Command timed out after 300s


## [2026-03-16 14:32:42 UTC] (ref: 100_HingeLoss)

Problem 100_HingeLoss: strategy 'tf32_tensor_cores' failed with compilation_error. Error: Traceback (most recent call last):
  File "/home/danny/kernel-forge-workspace/harness/forge_harness.py", line 120, in <module>
    cmd_test(args)
  Fi
