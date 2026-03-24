# Kernel Forge

Autonomous GPU kernel optimization system. Takes kernel problems (reference implementation + correctness test), optimizes them via a research-first agent loop on NVIDIA B200, and learns across runs via persistent memory and cross-problem experience.

## Results

### GPU MODE B200 Leaderboard -- #1 on all 8 problems

| Problem | Score | Approach |
|---------|-------|----------|
| matmul | 107.5us | cuBLASLt autotuned |
| prefixsum | 480us | Triton 3-pass scan |
| conv2d | 6478us | cuDNN benchmark=True |
| vectorsum | 43.7us | CUDA float4 + double accumulation + warp shuffle |
| sort | 1801us | CUB DispatchRadixSort with custom SM100 policy (512x21 vs default 384x19) |
| grayscale | 597us | CUDA float2 loads + branchless kernel + streaming stores |
| vectoradd | 234us | CUDA float4 + hadd2, 86% peak BW |
| histogram | 11.2us | CUDA uint4 + shared-mem atomics + software pipelining |

All submissions verified clean -- no cheating, no stream tricks, correct edge case handling.

### KernelBench L1

37+ problems optimized, 100% beat torch.compile. 20 winning kernels saved with full source.

## How It Works

### The Agent Loop

A single optimizer agent runs autonomously with full tool access (SSH, profiling, compilation):

```
0. MEMORY      -> read knowledge/memory/ for prior learnings
1. RESEARCH    -> how do production systems solve this? search GitHub, papers
2. PROFILE     -> where is time spent? (torch profiler, ncu)
3. COMPARE     -> gap between current and best-known techniques?
4. IMPLEMENT   -> write kernel targeting that gap
5. BENCHMARK   -> correctness + performance
6. CHECKPOINT  -> every 10 iterations: right approach? if plateaued, back to step 1
7. REPEAT      -> until gap < 10% of hardware peak
```

### Orchestration

The orchestrator manages the agent lifecycle:

1. **Baseline** -- benchmark reference implementation on B200
2. **Trait analysis** -- classify problem (dominant ops, bottleneck type, shape category)
3. **Roofline** -- compute gap to hardware peak at FP32/TF32/BF16
4. **Knowledge loading** -- distilled guides, similar winning solutions, experience
5. **Agent launch** -- invoke Claude Code CLI with full context injection
6. **Result recording** -- experience store + solution store + telemetry
7. **Redirect** -- warm-start new agent if stuck, with prior context

### Knowledge System

Four layers of knowledge, accumulated across runs:

**Persistent Memory** (`knowledge/memory/`)
- `semantic/` -- hardware facts, framework limits, tool behaviors (3 files)
- `episodes/` -- optimization campaign logs with what worked/failed (10 episodes)
- `procedures/` -- step-by-step patterns for common kernel types (2 procedures)

**Distilled Guides** (`knowledge/distilled/`)
- 12 operation-type guides (matmul, conv, attention, norm, softmax, etc.)
- Distilled from 4000+ community kernels via LLM summarization

**Experience Store** (`knowledge/experience/`)
- 37 optimization records with trait-based similarity matching
- Tracks: approach, outcome, speedup, bottleneck, roofline utilization
- Builds advisory context: "on similar problems, X worked Y% of the time"

**Solution Store** (`knowledge/solutions/`)
- 20 winning kernels saved with full source code
- Retrieved by trait similarity for new problems
- Includes approach description and speedup achieved

**Technique Registry** (`knowledge/techniques/`)
- 13 discovered optimization patterns as structured JSON
- Examples: ILP grouped loads, compile no-cudagraphs, channels-last antipatterns

### CUDA Patterns Library

9 reference implementations in `knowledge/cuda_patterns/`:
- warp_reduction.cu, shared_memory_tile.cu, vectorized_load.cu
- fp4_encoding.cu, triton_dot_scaled.py, cublaslt_nvfp4.md
- fused_rmsnorm_quant.cu, online_softmax.cu, cutlass_nvfp4.md

## Architecture

```
src/kernel_forge/
  agents/
    agent_prompt.md    # Agent methodology (ISA ref, tools, loop, memory)
    claude.py          # ClaudeCodeAgent (prompt assembly, CLI invocation, result parsing)
    orchestrator_prompt.md  # Orchestrator prompt for multi-agent runs
  core/
    types.py           # KernelProblem, OptimizationGoal, RooflineAnalysis
    orchestrator.py    # Orchestration loop (baseline -> context -> agent -> save)
    evaluate.py        # Roofline analysis (compute_roofline, B200_PEAKS)
    telemetry.py       # RunTracker with hierarchical timing spans
  knowledge/
    classifier.py      # Trait analysis (KernelTraits, similarity matching)
    experience.py      # ExperienceStore (JSONL, find_similar, build_advisory_context)
    solutions.py       # SolutionStore (winning kernels with source code)
    learnings.py       # LearningsManager (markdown read/write)
    ingest.py          # Knowledge ingestion (KernelBook, KernelBot)
  eval/
    scorecard.py       # Scorecard vs baselines (torch.compile, eager)
  remote/
    executor.py        # SSH executor for B200 commands
    gpu_guard.py       # GPU availability checking
  harness/             # KernelBench / WaferBench adapters
  tools/               # Tool registry, profiling, benchmarking
  config.py            # HardwareConfig, ForgeConfig, TOML loading
  cli.py               # Click CLI (optimize, solve, orchestrate, scorecard)
  solve.py             # Standalone prompt generator
  orchestrate.py       # Orchestrator prompt generator

knowledge/             # All accumulated knowledge (persists across sessions)
  memory/              # Semantic + episodic + procedural memory
  distilled/           # LLM-distilled optimization guides by op type
  experience/          # JSONL records of all optimization attempts
  solutions/           # Winning kernels with full source
  techniques/          # Discovered patterns as JSON
  cuda_patterns/       # Reference CUDA/Triton implementations
  external/            # Raw community data (KernelBook, KernelBot)
  baselines_b200.json  # Baseline measurements for 100 L1 problems

problems/              # Problem definitions (YAML configs)
kernels/               # Manual kernel implementations
```

## Hardware

Targets NVIDIA B200 (Blackwell) via SSH:
- BF16: 1929 TFLOPS, TF32: 964 TFLOPS, FP32: 481 TFLOPS
- FP8: 3851 TFLOPS, FP4: 7702 TFLOPS
- HBM3e: 8 TB/s bandwidth
- SMEM: 228 KB/SM, L2: 126 MB, 148 SMs

## Usage

```bash
# Generate an optimization prompt for a problem
uv run kernel-forge solve <problem_name> --gpu 3

# Run the full orchestration loop
uv run kernel-forge orchestrate <problem_name> --gpu 3

# Check results vs baselines
uv run kernel-forge scorecard

# View experience for a problem type
uv run kernel-forge experience

# Run tests
uv run pytest tests/ -v
uv run ruff check src/ tests/
uv run mypy src/kernel_forge/
```

## Key Learnings

- **Research first**: The biggest gains come from choosing the right algorithm, not tuning parameters. CUB's default SM100 policy for radix sort was suboptimal -- overriding it gave 17% speedup.
- **Libraries aren't always optimal**: CUB, cuBLAS, and cuDNN are SOTA but their default tuning parameters may not be optimal for specific hardware.
- **Know your bottleneck**: Bandwidth-bound kernels hit 86-90% of peak -- hardware physics is the wall. Atomic-bound kernels need contention reduction, not more bandwidth.
- **Correctness first**: Always test edge cases before optimizing. The sort kernel's int* reinterpretation was 12% faster but wrong for negative floats.
- **Experience compounds**: The 20th problem is easier than the 1st because the system has seen similar bottlenecks before and knows what works.
