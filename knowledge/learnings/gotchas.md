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
