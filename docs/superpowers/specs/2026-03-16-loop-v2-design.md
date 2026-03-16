# Optimization Loop v2: Closing the Gaps

## Current Loop (v1) -- What's Wrong

```
BASELINE (timing only)
  -> GENERATE (fresh kernel, no profiling context)
    -> TEST (pass/fail)
      -> EVALUATE (speedup number)
        -> STRATEGIZE (agent vibes)
          -> loop
```

Problems:
1. Agent never sees real profiling data (only wall-clock ms)
2. Each attempt generates from scratch (can't iterate on what's working)
3. No parameter search (one kernel per strategy, no exploration)
4. Strategy selection is uninformed (agent doesn't see roofline/profiling)
5. No tracking of whether experience suggestions helped

## Redesigned Loop (v2)

```
PHASE 1: UNDERSTAND
  1a. BASELINE benchmark (wall-clock timing)
  1b. BASELINE PROFILE (ncu: compute util, memory throughput, occupancy)
  1c. ROOFLINE ANALYSIS (utilization %, headroom, bound type)
  1d. LOAD EXPERIENCE (advisory context from similar problems)

PHASE 2: OPTIMIZE (iterative)
  ┌─────────────────────────────────────────────────┐
  │ 2a. DECIDE what to try:                         │
  │     - First attempt: strategy from experience   │
  │       + roofline analysis                       │
  │     - After success: refine the best kernel     │
  │       OR search parameters on current approach  │
  │     - After failure: try different strategy      │
  │       based on failure type + profiling data     │
  │                                                  │
  │ 2b. GENERATE kernel:                             │
  │     Agent receives:                              │
  │     - Problem + reference source                 │
  │     - Profiling data (ncu metrics, not just ms)  │
  │     - Roofline position (% of peak, bound type)  │
  │     - Best kernel so far (source code)           │
  │     - All prior attempts with outcomes + WHY      │
  │     - Experience from similar problems            │
  │     - Decision from 2a (refine vs new strategy)   │
  │                                                  │
  │ 2c. TEST (correctness + benchmark)               │
  │                                                  │
  │ 2d. PROFILE (if correct: ncu on new kernel)       │
  │                                                  │
  │ 2e. ANALYZE DELTA:                                │
  │     - What changed vs best-so-far?               │
  │     - Did utilization improve?                    │
  │     - Did bound type change?                      │
  │     - Was the experience suggestion helpful?      │
  │                                                  │
  │ 2f. EVALUATE: should we continue?                 │
  │     - Near peak -> STOP                          │
  │     - Headroom + improving -> CONTINUE            │
  │     - Plateau + no headroom -> STOP              │
  │     - Plateau + headroom -> ESCALATE (ncu deeper) │
  └──────────────── loop ───────────────────────────┘

PHASE 3: RECORD
  3a. Structured experience per attempt (with profiling delta)
  3b. Record whether experience suggestion was followed + helpful
  3c. Write generalized insights for the kernel trait class
```

## Key Changes from v1

### 1. Profiling Depth

Agent prompt BEFORE (v1):
```
Runtime: 0.17ms
Speedup: 12.6x
```

Agent prompt AFTER (v2):
```
Runtime: 0.17ms (12.6x speedup)
Roofline: 83.5% of TF32 peak, 16.5% headroom, COMPUTE-BOUND
ncu metrics:
  compute_utilization: 83.5%
  memory_throughput: 22.1% of peak
  occupancy: 67.3%
  warp_stall_reason: not_selected (42%), wait (31%)
  l2_hit_rate: 78.2%
Conclusion: compute-limited at 83.5%. Main stall is warp scheduling.
  Memory is NOT the bottleneck. Strategies targeting memory access
  will NOT help. Focus on: tensor core throughput, occupancy,
  warp-level parallelism.
```

### 2. Iterative Refinement

v1: "Generate an optimized kernel for this problem"
v2: "Here is the current best kernel (12.6x, 83.5% peak).
     Improve it by addressing the warp scheduling stall."

The agent prompt includes the FULL source of the best kernel found
so far. This enables incremental improvement rather than starting
from scratch each time.

Two generation modes:
- `generate_new`: Start from reference (first attempt or strategy switch)
- `refine_existing`: Take best kernel + profiling delta, improve it

### 3. Parameter Search

After a correct kernel with a working strategy, explore variations:
- Different tile sizes (BM=64,128,256 x BN=64,128,256)
- Different block dimensions (128,256,512 threads)
- Different unroll factors (1,2,4,8)
- With/without vectorized loads

This is the "fine-grained parameter tuning" from the SA+peephole pattern.
Implemented via parallel subagent generation: spawn 3 subagents, each
trying a different parameterization of the same strategy.

### 4. Data-Driven Strategy Selection

Strategy selection prompt includes:
```
## Current State
- Runtime: 0.17ms, Speedup: 12.6x
- Roofline: 83.5% TF32 peak, COMPUTE-BOUND
- Bottleneck: warp scheduling (42% not_selected stalls)
- Memory is NOT limiting (22.1% of bandwidth peak)

## Experience from similar problems
- tf32_tensor_cores: 12.8x avg (100% success)
- register_blocking: not tried yet (typical 2-4x for compute-bound)

## Available strategies (sorted by relevance to bottleneck)
- occupancy_tuning: may help with warp scheduling stalls
- register_blocking: increases ILP, reduces stalls
- warp_specialization: advanced, for >90% utilization target

## What NOT to try
- shared_memory_tiling: memory is not the bottleneck
- vectorized_loads: memory is not the bottleneck
- bf16: FAILED correctness on similar problems (3 times)
```

### 5. Experience Suggestion Tracking

Each ExperienceRecord gains:
```
suggestion_from_experience: str | None  # what was suggested
suggestion_followed: bool               # did agent follow it
suggestion_helpful: bool                # did it lead to improvement
```

This enables learning whether experience suggestions are actually
useful, and downweighting suggestions that frequently don't help.

## Implementation Plan

1. Add ncu profiling to baseline + post-correct-attempt
2. Include best_kernel_source and profiling data in generate prompt
3. Add refine_existing mode to agent prompt builder
4. Add parameter search subagent spawning
5. Include roofline + ncu data in strategy selection prompt
6. Add suggestion tracking fields to ExperienceRecord
7. Update orchestrator run() to use new loop structure

Each can be implemented incrementally. Priority order:
1 + 4 (profiling in prompts) -- biggest impact on strategy quality
2 + 3 (iterative refinement) -- biggest impact on kernel quality
5 (suggestion tracking) -- enables self-improvement
6 (parameter search) -- enables fine-grained optimization
