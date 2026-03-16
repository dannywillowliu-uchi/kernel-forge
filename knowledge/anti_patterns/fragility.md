# Fragility Patterns

Patterns where seemingly correct code changes cause catastrophic regressions.

### reshape_breaks_cuda_graph
- **Category:** fragility / cuda_graph
- **Applicability:** any kernel using CUDA graph capture
- **Description:** .reshape() returns view if contiguous, silently copies if not. CUDA graph capture records specific memory pointers. If .reshape() returned view during capture but copy during replay, graph writes to wrong pointer -- silent memory corruption.
- **Detection:** Graph capture succeeds but results are incorrect on replay. Test with non-contiguous inputs.
- **Prevention:** Use .contiguous().view() instead. .view() fails loudly if non-contiguous.

### torch_compile_recompilation
- **Category:** fragility / torch_compile
- **Applicability:** any compiled function with optional arguments
- **Description:** torch.compile specializes on argument types at first call. Changing from f(tensor, None) to f(tensor, tensor) triggers full recompile. Caused 64% regression on decode kernel.
- **Detection:** Use TORCH_LOGS=recompiles to log guard failures.
- **Prevention:** Normalize signatures -- use empty tensor sentinel instead of None. Use torch._dynamo.explain() to audit guards.

### autocast_dispatch_overhead
- **Category:** fragility / precision
- **Applicability:** inference kernels at high throughput
- **Description:** torch.autocast adds per-operation type-checking overhead. Explicit model.bfloat16() at init removes dispatch overhead entirely: +47% throughput.
- **Trade-off:** bf16 everywhere including LayerNorm accumulation = slightly lower accuracy.
- **Prevention:** For inference, prefer explicit dtype casting. Reserve autocast for training.

### cuda_graph_static_addresses
- **Category:** fragility / cuda_graph
- **Applicability:** any CUDA graph with mutable state
- **Description:** CUDA graphs bake in GPU memory pointers during capture. If input tensor occupies different address on replay, graph writes to wrong pointer. No .item(), .cpu(), data-dependent shapes, or dynamic allocation inside captured region.
- **Prevention for KV cache:** Replace .copy_() with .scatter_() on full static tensor. Pass full-length cache + boolean mask instead of dynamic slice.

### cuda_graph_dynamic_allocation
- **Category:** fragility / cuda_graph
- **Applicability:** any CUDA graph creating intermediate tensors
- **Description:** torch.empty(), torch.zeros() inside captured region allocates different address each time. Graph's kernel arguments still point to capture-time address.
- **Prevention:** Pre-allocate ALL intermediate tensors before graph capture. Only write to pre-allocated buffers inside capture region.

### compile_reduce_overhead_rerecording
- **Category:** fragility / torch_compile
- **Applicability:** training or any backward-pass context
- **Description:** mode="reduce-overhead" and mode="max-autotune" re-record CUDA graphs every iteration when backward pass involved. Produces 12x slowdown (27 data/s -> 2.3 data/s).
- **Prevention:** Use mode="default" for training. Reserve reduce-overhead for inference-only. Use max-autotune-no-cudagraphs as fallback.

### compile_kv_cache_mutations
- **Category:** fragility / torch_compile
- **Applicability:** inference with KV cache
- **Description:** mode="reduce-overhead" detects .copy_() on KV cache as "mutated inputs" and falls back from graph capture. mode="max-autotune" Triton kernels have 50x overhead at batch=1 tiny shapes.
- **Mode guide:**
  - default: kernel fusion, reliable, works everywhere
  - reduce-overhead: fails on mutated inputs, re-records on backward
  - max-autotune: 50x overhead at batch=1, re-records on backward
  - max-autotune-no-cudagraphs: Triton autotuning without graph fragility

### torch_compile_graph_breaks
- **Category:** fragility / torch_compile
- **Applicability:** any compiled function
- **Description:** TorchDynamo inserts graph break on untraceable code. Each break requires GPU-CPU sync, fragmenting computation.
- **Common causes:** data-dependent control flow, .item(), .data_ptr(), print(), C extensions, NumPy on CUDA tensors
- **Detection:** torch._dynamo.explain(fn)(*args) for break report. torch.compile(fn, fullgraph=True) to error on first break.
- **Prevention:** torch.cond() for conditionals. Move logging outside compiled functions. Avoid .item() in hot paths.

### load_inline_latency
- **Category:** fragility / compilation
- **Applicability:** KernelBench format kernels
- **Description:** load_inline() imports all 18K Libtorch headers regardless of kernel content. 90-second compile times for toy kernels. In autonomous loop with dozens of candidates, adds hours.
- **Mitigation:** Pre-warm compilation cache at startup. Batch candidate generation then compile sequentially with 300s timeout. Log compiler stderr fully.

### quantization_small_batch
- **Category:** fragility / precision
- **Applicability:** batch=1 or small batch inference
- **Description:** FP8 was 15x slower, INT8 42% slower at batch=1. Dequantization overhead dominates when not memory-bandwidth-bound.
- **Prevention:** Profile before applying quantization. At batch=1, prefer explicit bf16 over quantization.
