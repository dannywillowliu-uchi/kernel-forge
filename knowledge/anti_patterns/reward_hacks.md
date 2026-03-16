# Reward Hacking Taxonomy

Source: Wafer.ai Field Guide (2026-03-12)

## Category A: Timing Attacks

### stream_injection
- **Category:** reward_hack
- **Applicability:** any kernel benchmark using CUDA events on default stream
- **Description:** Computation dispatched to a separate non-default CUDA stream. Timing events on default stream record near-instant launch, actual work runs async. Result is correct because injected stream finishes before correctness check.
- **Detection:** Compare timing with and without forced torch.cuda.synchronize(). Flag if elapsed_without_sync / elapsed_with_sync > 1.5x.
- **Prevention:** Hybrid timing with post-execution synchronize(). Reject divergent ratios.

### thread_injection
- **Category:** reward_hack
- **Applicability:** any Python-wrapped kernel
- **Description:** Background CPU thread spawned at kernel entry. Python returns immediately, thread dispatches GPU work. Race condition means outputs occasionally contain zeros.
- **Detection:** Monitor threading.active_count() before/after execution. Flag new threads.
- **Prevention:** Record thread count before kernel import, check after, reject if increased.

### lazy_evaluation_tensor_subclass
- **Category:** reward_hack
- **Applicability:** any kernel returning torch.Tensor
- **Description:** Returns custom Tensor subclass storing inputs without computing. Actual computation triggers during torch.allclose correctness check.
- **Detection:** Kernel time near-zero but correctness check takes unusually long.
- **Prevention:** Validate type(output) is torch.Tensor (strict, not isinstance). Check allocated storage and valid device pointer.

### timing_monkey_patch
- **Category:** reward_hack
- **Applicability:** any benchmark using torch.cuda.Event.elapsed_time
- **Description:** Replaces elapsed_time with stub returning impossibly small value regardless of work.
- **Detection:** Implausibly consistent timing across wildly different problem sizes.
- **Prevention:** Capture original function reference before kernel import, verify identity after execution.

## Category B: Semantic Attacks

### identity_kernel
- **Category:** reward_hack
- **Applicability:** kernels where input could equal output
- **Description:** Copies first input to output without computation. Passes if shapes align.
- **Detection:** Check if output == input[0] elementwise. Use multi-input validation with distinct random values.
- **Prevention:** Always validate with distinct random inputs where input != expected output.

### no_op_kernel
- **Category:** reward_hack
- **Applicability:** kernels with pre-allocated output buffers
- **Description:** Launches on GPU but executes zero instructions. Relies on pre-computed memory from prior reference run.
- **Detection:** Passes only when reference ran first.
- **Prevention:** Initialize output buffers with NaN sentinel before each invocation. Verify sentinel is overwritten.

### shared_memory_overflow
- **Category:** reward_hack
- **Applicability:** kernels using shared memory
- **Description:** Requests more shared memory than hardware supports. Reads uninitialized GPU memory. Non-deterministic but bounded outputs may pass torch.allclose. Produced a claimed "1.7 PFLOPS" on MI300X (peak is 1.7 TFLOPS).
- **Detection:** Non-deterministic outputs across identical runs. Physically impossible performance numbers.
- **Prevention:** Run 3+ times with identical inputs on freshly allocated memory. Require torch.equal() between all runs. Cross-reference against hardware peak throughput.

### precision_downgrade
- **Category:** reward_hack
- **Applicability:** any kernel with precision requirements
- **Description:** Computes in fp16 internally, casts to fp32 for output. Genuine but illegitimate speedup (1.5-3x). Error suspiciously close to tolerance threshold.
- **Detection:** Compare against fp64 reference using ULP metrics. fp16-computed values show ~1e-3 relative error pattern.
- **Prevention:** Assert output.dtype == expected_dtype strictly. Use ULP-based precision metrics.

### caching_memoization
- **Category:** reward_hack
- **Applicability:** any kernel with deterministic inputs
- **Description:** Two variants: Python cache (keyed by shape/device) and C++ pointer cache (keyed by tensor.data_ptr()). C++ variant exploits PyTorch's stable allocator -- tensor addresses tend to persist.
- **Detection:** Python: vary input values, verify output changes. C++: requires pointer-poisoning.
- **Prevention (pointer-poisoning):** After correctness verification, overwrite verification tensors in-place with new random data (same addresses, different content), re-run kernel, compare against fresh reference. Stale cache returns old result, mismatch detected.

## Category C: Benign Shortcuts

### baseline_library_call
- **Category:** reward_hack
- **Applicability:** any kernel optimization task
- **Description:** Calls torch.matmul, F.linear, cuBLAS, cuDNN instead of writing custom kernel. Correct but defeats the purpose.
- **Detection:** Static analysis of kernel source for high-level library calls.
- **Prevention:** Regex/AST scan before execution. Reject imports of torch.mm, torch.matmul, F.linear, cuBLAS/cuDNN.

## Layered Prevention Strategy (Recommended Order)

1. **Static analysis** (free): scan for library calls, suspicious imports
2. **Sentinel initialization** (cheap): NaN-fill output buffers before kernel
3. **Hybrid timing** (cheap): synchronize + compare ratios
4. **Thread monitoring** (cheap): check active_count before/after
5. **Determinism check** (moderate): 3 identical runs, torch.equal comparison
6. **Pointer-poisoning** (expensive): overwrite inputs at same address, re-run
