# Agent-Based GPU Kernel Optimization Methodology

## Core Principle

LLM-generated kernels fail primarily through lack of execution feedback, not lack of knowledge. The optimization loop is effective when every hypothesis is grounded in measurement, plateau detection drives strategy switching, and every run writes structured learnings that compound across problems.

## Phase 1: Intake and Classification

Before writing any code, classify the kernel to load the right prior knowledge:

1. **Kernel type:** matmul, reduction, attention, elementwise, fused op, convolution
2. **Bottleneck hypothesis from shape analysis:**
   - Small batch, large model = likely launch-overhead-bound or memory-bound
   - Large batch, small kernel = likely occupancy-limited
   - Operations with reuse opportunity (attention, matmul) = candidate for tiling + shared memory
3. Query knowledge base: "What strategies worked on similar kernel types?"
4. Load anti-patterns for this kernel type into context before generating code

## Phase 2: Establish Baseline

Never skip. Run the reference implementation first:
- Compile and benchmark on target hardware (warmup + timed reps via CUDA events)
- Record: wall-clock time, input shapes, batch size, optimization goal

The baseline is the anchor. A strategy that produces correct output but no speedup is a failed strategy.

## Phase 3: Diagnose Bottleneck

### Profiling Tier Progression

```
Tier 1: CUDA events timing (fast, always-on)
  -> if plateau detected: escalate to
Tier 2: ncu with key metrics
  - sm__throughput (compute utilization)
  - dram__throughput (memory throughput)
  - sm__warps_active (occupancy)
  - l2__throughput (L2 cache pressure)
```

### Bottleneck Classification

| Signal | Bottleneck | Primary Lever |
|--------|-----------|---------------|
| dram throughput near 100%, compute low | Memory-bound | Tiling, shared memory, vectorized loads, fusion |
| Compute near 100%, memory low | Compute-bound | Tensor cores, warp-level primitives, algorithm change |
| Low occupancy, neither saturated | Occupancy-limited | Reduce register pressure, adjust block size, pipeline loads |
| Fast ops, measured gap from profiling | Launch-overhead | Kernel fusion, CUDA graphs, op batching |

Do not skip diagnosis. Applying the wrong technique wastes attempts and produces plateaus.

## Phase 4: Strategy Selection

Ordered by typical impact, but always driven by diagnosis:

| Category | Strategies | When to Apply |
|----------|-----------|---------------|
| Algorithmic | Sparsity, simplification, KV cache reuse | When math is redundant; highest ceiling (13x) |
| Precision | Explicit bf16, fp8 at large batch | Memory-bandwidth-bound AND accuracy permits |
| Fusion | Elementwise, matmul+activation, attention fusion | Multiple kernels write/read intermediates |
| Tiling + Shared Memory | Blocked matmul, shared mem reduction | Memory-bound with data reuse |
| Compiler | torch.compile(mode="default"), Triton | Level 2 multi-op sequences |
| Execution | CUDA graphs, static allocation | After other optimizations; ~1.1-1.4x, fragile |

**Hierarchy heuristic:** algorithmic > precision > compiler > memory > pipeline > execution. Override whenever profiling contradicts this.

## Phase 5: Parallel Candidate Generation

When multiple strategies are viable after diagnosis:
- Each subagent gets: problem + diagnosis + assigned strategy + prior attempts
- All candidates validated for correctness before any benchmarking
- All correct candidates benchmarked sequentially (no GPU contention)
- Best result becomes new baseline; all results recorded

### Compilation Retry Loop

```
generate kernel source
  -> compile
    -> if error: append compiler stderr, retry (max 3x)
    -> if still failing: classify as compilation_error, move on
  -> validate correctness (randomized inputs)
    -> if incorrect: classify as correctness_failure, DO NOT BENCHMARK, re-diagnose
  -> benchmark
    -> if slower: classify as performance_regression, re-diagnose
```

## Phase 6: Plateau Detection and Strategy Switching

The SA+peephole lesson applied to kernel optimization.

**Plateau:** N consecutive attempts with < threshold% relative improvement.

| State | Action |
|-------|--------|
| Plateau at CUDA events tier | Escalate to ncu; re-diagnose with richer data |
| Plateau at ncu, strategies not exhausted | Switch strategy category |
| Plateau at ncu, all obvious strategies tried | Spawn exploration subagents |
| Plateau at ncu, no improvement after exploration | Accept best; write learnings; terminate |

**Key insight:** When one search method plateaus, switch to a complementary technique at different granularity:
- Fine-grained parameter tuning plateaus -> structural change (fusion, algorithm)
- Structural change plateaus -> fine-grained parameter tuning
- Both plateau -> bottleneck may be irreducible; accept best

## Phase 7: Failure Classification

Every failure is diagnostic signal:

| Failure Type | Signal | Next Action |
|---|---|---|
| Compilation error | stderr with line/column | Fix syntax; 3 retries max |
| Link error | Undefined symbol | Fix includes, load_inline function list |
| Runtime segfault | SIGSEGV | Fix indexing, shared memory bounds, launch config |
| Runtime OOM | CUDA out of memory | Smaller tile sizes, reduce register pressure |
| Correctness failure | torch.allclose False | Re-examine the algorithm, not just the code |
| Numerical instability | Fails on edge cases | fp32 intermediates for accumulation |
| Timeout | >300s | Check sync primitives and loop bounds |
| Regression | Correct but slower | Wrong strategy for this bottleneck |

**Discipline:** Correctness failures are re-diagnosed from scratch, never retried with minor tweaks.

## Phase 8: Validation Protocol

1. Correctness BEFORE benchmarking (reject incorrect kernels immediately)
2. Test on randomized inputs, not just harness test cases
3. Test edge cases: zeros, large values, denormals
4. For CUDA graphs: verify .contiguous().view() not .reshape(); no conditional control flow
5. For torch.compile: verify function signature stability
6. Timing validation: reject suspiciously low variance

## KernelBench Level-Specific Guidance

| Level | Problems | Key Insight | Strategy Bias |
|-------|----------|-------------|---------------|
| L1 (single ops) | 100 | PyTorch's closed-source kernels are hard to beat; tensor cores largely unsuccessful | Shared memory tiling, vectorized loads |
| L2 (multi-op) | 100 | Iterative refinement with feedback: 36% -> 72% fast_1 | Operator fusion; torch.compile is strong competition |
| L3 (full arch) | 50 | <4% fast_1; whole-model composition is the bottleneck | Don't over-fuse; model-level algorithmic changes |

## Research Findings by System

### KernelFalcon (Meta)
- 100% correctness on all 250 KernelBench tasks via parallel search + early-exit
- Key: execution-based verification, constraint-based context, persistent memory
- Failure: over-aggressive fusion (register spilling, occupancy collapse)

### GEAK (Triton Agent)
- Knowledge injection is the largest single lever (+52% call accuracy)
- Parallel scaling shows log-linear accuracy improvement
- Sequential refinement has diminishing returns beyond ~10 iterations

### CUDA Agent (Agentic RL)
- Milestone-based reward schedule outperforms raw speedup signal
- Learned policies outperform static compiler heuristics on complex fusion
- CUDA represents <0.01% of pretraining data -- knowledge injection critical

### Karpathy AutoResearch
- Fixed time budget per experiment enables high iteration velocity (~12/hour)
- Two-phase: broad exploration first, narrow exploitation second
- Core file carries instructions, constraints, and stopping criteria simultaneously

## Sources

- KernelBench paper (arxiv 2502.10517)
- KernelFalcon (PyTorch blog, Meta)
- GEAK (arxiv 2507.23194)
- CUDA Agent (arxiv 2602.24286)
- Karpathy AutoResearch (github.com/karpathy/autoresearch)
