# Known Failure Modes of LLM-Generated Kernels

## Error Distribution (KernelBench, 100 kernels one-shot)

| Error type | Count | Pct |
|---|---|---|
| Correct | 50 | 50% |
| Output value mismatch | 19 | 19% |
| Compilation error | 17 | 17% |
| Runtime error | 10 | 10% |
| Output shape mismatch | 4 | 4% |

## Patterns

### operation_class_blindspot
- **Category:** llm_failure
- **Description:** LLMs have systematic blind spots on specific operation classes. DeepSeek-V1 failed all 34 convolution variants in KernelBench L1 even with 100 samples. Strided convolution with non-standard padding is unrepresented in training data.
- **Implication:** When 5 consecutive attempts fail on same kernel class, likely a blind spot. Flag for human review rather than exhausting budget.

### execution_vs_correctness_asymmetry
- **Category:** llm_failure
- **Description:** Reasoning models (o1, R1) produce fewer compilation/runtime failures through self-correction but achieve no better functional correctness. Compiler error feedback is more actionable than correctness feedback. Iterative refinement improves execution from 12% to 43% but correct kernels "almost always fail due to functional incorrectness."
- **Implication:** Compilation errors are recoverable; correctness errors require re-diagnosis, not retry.

### optimization_correctness_tradeoff
- **Category:** llm_failure
- **Description:** Attempting advanced optimizations (tensor core wmma, complex tiling) correlates with higher error rates. Models add complex code that introduces bugs.
- **Implication:** Prefer incremental optimization over large rewrites. Start from known-correct baseline, apply one optimization at a time.

### complexity_cliff
- **Category:** llm_failure
- **Description:** L1 single ops: reasonable success. L2 operator fusion: significantly harder (fusion boundary reasoning). L3 full architectures: <12% success, cross-layer interactions overwhelm model reasoning.
- **Implication:** Start with L1, build knowledge, progress to L2. L3 requires fundamentally different approach.

### one_shot_hallucination
- **Category:** llm_failure
- **Description:** Single-pass generation produces hallucinated API calls, incorrect thread indexing, wrong data layout assumptions. Without execution feedback, undetectable.
- **Prevention:** Always compile and execute before acceptance. No "plausible-looking" kernel is ever done.

### semantic_drift_in_refinement
- **Category:** llm_failure
- **Description:** Iterative refinement can drift from original semantics. Control flow may be "frozen" to single traced path, breaking data-dependent behavior.
- **Prevention:** Whole-model parity checks comparing end-to-end output, not just per-kernel.

### llm_driven_control_flow
- **Category:** llm_failure
- **Description:** Letting LLM decide whether to stop/retry/escalate leads to inconsistent behavior. LLMs hallucinate success and underestimate failure modes.
- **Prevention:** Deterministic Python orchestration. LLM generates code; Python decides control flow based on actual results.

### debugging_trap
- **Category:** llm_failure
- **Description:** LLM cycles through same reflection steps without escaping a bug class. Generates "different" fix that is semantically equivalent, loops indefinitely.
- **Detection:** Track error message set. Same error class repeating 3+ times = trapped.
- **Prevention:** Hard iteration cap on debugging loop. After N unsuccessful attempts on same error, abandon and try new strategy. (GEAK finding: direct prompting ~8-15% accuracy vs agent with caps: 54-63%)

### cuda_training_data_scarcity
- **Category:** llm_failure
- **Description:** CUDA code is ~0.073% of The Stack v1.2. LLMs have severely limited exposure to GPU patterns, especially hardware intrinsics (wmma, warp shuffle, async memcpy). Models under-generate these and over-generate naive implementations.
- **Prevention:** Knowledge base and system prompt must inject hardware-specific patterns.

### hardware_instruction_blindspot
- **Category:** llm_failure
- **Description:** Models cannot reliably leverage TensorCore wmma, vectorized load/store (float4), warp-shuffle, or Blackwell-specific features. Attempts often syntactically plausible but semantically incorrect.
- **Prevention:** Extra validation scrutiny for kernels attempting wmma/tensor core ops. ULP comparison against bf16 reference.

### indexing_boundary_errors
- **Category:** llm_failure
- **Description:** Off-by-one errors in thread-to-data index mappings. Common for non-power-of-2 dimensions, strided layouts, shared memory padding. Partially correct output (correct for aligned cases, wrong for edge tiles).
- **Prevention:** Always test with non-power-of-2 input dimensions (127, 511, 1023) in addition to power-of-2.

### warp_divergence_introduction
- **Category:** llm_failure
- **Description:** LLMs frequently introduce data-dependent branches inside warps causing divergence and performance regression. Kernel is correct but slower.
- **Detection:** ncu sm__warps_eligible_active.avg metric.
- **Prevention:** Flag conditional branches inside thread blocks. Prefer predicated execution over branching.
