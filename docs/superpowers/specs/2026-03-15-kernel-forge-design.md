# Kernel Forge: Autonomous GPU Kernel Optimization System

**Date:** 2026-03-15
**Status:** Design approved
**Author:** Danny Liu + Claude

## 1. Problem Statement

Create a purpose-built autonomous system that optimizes GPU kernels for maximum performance on NVIDIA B200 hardware. The system should:

- Take a kernel problem (reference implementation + correctness test) and produce the fastest correct kernel it can
- Learn from each optimization run to improve over time across kernel types
- Apply known optimization strategies dynamically based on profiling, not from a rigid checklist
- Support human supervision during the exploratory phase, with increasing autonomy as trust builds
- Be built using the autodev swarm framework

### Goals

- Close the loop on autonomous kernel optimization: profile -> diagnose -> strategize -> implement -> validate -> benchmark -> evaluate -> repeat
- Build a multi-kernel generalization layer that informs starting strategies for new kernels
- Achieve competitive scores on KernelBench (250 CUDA problems), then specialize to WaferBench NVFP4 (6 fused FP4 kernels on B200)
- Prevent reward hacking through robust validation and anti-pattern knowledge

### Non-Goals

- Real-time serving / production deployment (this is a research/exploration tool)
- Multi-GPU distributed kernels (single GPU 2 on shared B200 node)
- AMD MI300X / ROCm support (designed for but not implemented yet)
- Fine-tuning LLMs on kernel data (using LLMs as-is via API/CLI)

### Constraints

- Shared B200 node: exclusive lock on GPU 2 via `CUDA_VISIBLE_DEVICES=2`, periodic `nvidia-smi` checks
- SSH access: `b200-node` (45.76.244.62, user: danny)
- Agent backbone: Claude Code only for now, but model abstraction layer must support others
- Build tool: autodev swarm mode
- Human-in-the-loop: supervised runs initially (operator watches, intervenes on bad decisions)

## 2. Architecture Overview

### System Diagram

```
                    ┌──────────────────────────┐
                    │      CLI / Operator       │
                    │  (Danny, supervised mode)  │
                    └────────────┬───────────────┘
                                 │
                                 v
                    ┌──────────────────────────┐
                    │     PRIMARY AGENT         │
                    │  (Core Optimization Loop)  │
                    │                            │
                    │  profile -> diagnose ->    │
                    │  strategize -> implement ->│
                    │  validate -> benchmark ->  │
                    │  evaluate -> loop/report   │
                    └──┬──────────┬──────────┬──┘
                       │          │          │
              ┌────────┘    ┌─────┘    ┌─────┘
              v             v          v
     ┌──────────────┐ ┌─────────┐ ┌─────────────┐
     │  Tool System  │ │Knowledge│ │  Subagents   │
     │               │ │  Base   │ │ (short-lived)│
     │ - profiling   │ │         │ │              │
     │ - benchmarking│ │ - SQLite│ │ - candidate  │
     │ - compilation │ │ - .md   │ │   generation │
     │ - gpu_status  │ │ - seed  │ │ - param sweep│
     │ - extensible  │ │   data  │ │ - research   │
     └──────┬────────┘ └────┬────┘ └──────────────┘
            │               │
            v               v
     ┌─────────────┐ ┌──────────────┐
     │  B200 GPU 2  │ │  SQLite DB    │
     │  (via SSH)   │ │  + Markdown   │
     └─────────────┘ └──────────────┘
```

### Design Principles

1. **Diagnosis-driven, not checklist-driven.** The agent re-profiles and re-diagnoses after every attempt. Strategy selection is informed by what the profiler reveals, not a fixed ordering.
2. **Convergence-driven termination.** The loop stops when it can't find further improvement after N attempts, not after a fixed step count.
3. **Tiered profiling.** CUDA events by default (fast), `ncu` on plateau detection (3 attempts with <2% improvement).
4. **Extensible tools.** The agent can recognize "I need a tool that doesn't exist" and either build it or flag for human input.
5. **Knowledge compounds.** Every run writes back structured results and qualitative learnings. The system gets better over time.

## 3. Core Loop

The primary agent executes this loop for each kernel problem:

### Step 1: INTAKE
- Accept kernel problem: reference implementation, correctness test, input shapes
- Accept optimization goal (latency/throughput/memory/balanced)
- Load KernelBench harness adapter for problem format

### Step 2: ANALYZE
- Classify kernel type (matmul, reduction, attention, fused op, etc.)
- Query knowledge base: "What strategies worked on similar kernels?"
- Load relevant learnings into context (gotchas, tool recipes)
- Load anti-pattern knowledge (reward hacking prevention)

### Step 3: BASELINE
- Compile and benchmark reference implementation on B200 GPU 2
- Profile with CUDA events (wall-clock timing)
- Record baseline metrics in DB

### Step 4: DIAGNOSE
- Determine bottleneck type: memory-bound, compute-bound, launch-overhead, occupancy-limited
- At default tier: infer from CUDA events timing + kernel characteristics
- At escalation tier: use `ncu` for detailed metrics (memory throughput, compute utilization, warp stall reasons)
- Escalation trigger: 3 consecutive attempts with <2% improvement

### Step 5: STRATEGIZE
- Select optimization approach based on diagnosis + knowledge DB + optimization goal
- Weight strategies by goal context:
  - Latency goal: prioritize per-call time reduction at target batch size
  - Throughput goal: prioritize ops/second, batching, occupancy
  - Memory goal: prioritize peak VRAM reduction
- Optionally spawn subagents for parallel candidate generation (multiple strategies explored simultaneously)
- The agent draws from a knowledge base of strategies, not a hardcoded list. Strategies are weighted by prior success on similar kernels.

### Step 6: IMPLEMENT
- Write optimized kernel (CUDA or Triton, matching problem format)
- Primary agent or subagent produces kernel code

### Step 6.5: FAILURE HANDLING

When implementation or execution fails, classify the failure and feed it back into diagnosis:

| Failure Type | Signal | Agent Action |
|---|---|---|
| **Compilation error** | nvcc/hipcc stderr with error line/column | Fix syntax, check architecture flags, verify CUDA API usage |
| **Link error** | Undefined symbol, missing library | Fix includes, check load_inline function list |
| **Runtime segfault** | SIGSEGV, illegal memory access | Fix indexing, check shared memory bounds, verify launch config |
| **Runtime OOM** | CUDA out of memory | Reduce shared memory, reduce register pressure, smaller tile sizes |
| **Correctness failure** | torch.allclose returns False | Algorithm is wrong -- re-examine the mathematical transformation |
| **Numerical instability** | Correctness fails on some inputs, passes others | Precision issue -- check accumulation order, use higher precision intermediates |
| **Timeout** | Command exceeds 300s | Likely infinite loop or deadlock -- check synchronization, loop bounds |
| **Performance regression** | Correct but slower than baseline | Strategy was wrong for this bottleneck -- re-diagnose |

Each failure type produces a `FailureReport` that includes:
- The failure classification
- Raw error output (compiler stderr, crash log, etc.)
- The kernel source that caused it
- The strategy that was being attempted

This feeds back into DIAGNOSE -- the agent uses the failure type to adjust its approach, not just retry blindly.

### Step 7: VALIDATE
- Correctness check runs BEFORE benchmark timing (not after)
- `torch.allclose(rtol=1e-3, atol=1e-3)` against reference implementation
- Validation on randomized inputs, not just harness test inputs
- Reject incorrect kernels immediately - no benchmarking of wrong code

### Step 8: BENCHMARK
- CUDA events timing (warmup + timed reps, matching KernelBench harness methodology)
- Compare to best-so-far
- Record all metrics in DB

### Step 9: EVALUATE
- Score against optimization goal
- Update knowledge base: strategy used, speedup achieved, kernel type, hardware
- Write qualitative learnings if discovery was made
- Track cumulative cost (LLM API calls + compute time)

**Convergence & Termination Parameters:**

| Parameter | Default | Configurable | Description |
|---|---|---|---|
| `plateau_threshold` | 2% | per-problem | Minimum improvement to not count as plateau |
| `plateau_window_cuda_events` | 3 | per-difficulty | Consecutive sub-threshold attempts before escalating to ncu |
| `plateau_window_ncu` | 3 | per-difficulty | Consecutive sub-threshold attempts at ncu tier before stopping |
| `max_attempts_per_problem` | 25 | per-problem | Hard cap on total attempts |
| `max_cost_per_problem_usd` | 5.00 | global | Cost budget per problem (LLM API cost) |
| `max_wall_time_per_problem` | 30min | per-problem | Wall-time budget |
| `max_consecutive_failures` | 5 | global | Stop if 5 consecutive attempts fail (compile/crash/incorrect) |

Termination triggers (any one stops the loop):
1. Convergence: `plateau_window_ncu` consecutive attempts with <`plateau_threshold` improvement at ncu tier
2. Budget: cumulative cost exceeds `max_cost_per_problem_usd`
3. Time: wall time exceeds `max_wall_time_per_problem`
4. Attempts: total attempts exceed `max_attempts_per_problem`
5. Failures: `max_consecutive_failures` consecutive non-improvement attempts (compile errors, crashes, incorrect)

### Step 10: REPORT
- Emit final results: best kernel, speedup vs baseline, profiling summary
- Emit learnings: what worked, what didn't, gotchas discovered
- Update knowledge base for future runs on similar kernels

## 4. Knowledge & Learning System

### Layer 1: Structured Database (SQLite)

**strategies table:**
- `id`, `name`, `category` (memory_opt, compute_opt, precision_opt, pipeline_opt, execution_opt, algorithmic)
- `description` (LLM-readable explanation)
- `applicability` (what kernel types / bottleneck types this strategy addresses)
- `expected_impact` (typical speedup range from literature/experience)

**attempts table:**
- `id`, `kernel_problem`, `strategy_id`, `speedup`, `correct`, `hardware`
- `optimization_goal`, `batch_size`, `profiling_tier`
- `kernel_source_hash` (for dedup)
- `timestamp`

**kernel_profiles table:**
- `id`, `kernel_problem`, `bottleneck_type`
- `memory_throughput_pct`, `compute_utilization_pct`, `occupancy_pct`
- `warp_stall_reasons` (JSON), `l2_hit_rate`, `shared_mem_usage`
- `profiling_tier` (cuda_events / ncu)

**kernel_classifications table:**
- `id`, `kernel_problem`, `kernel_type`, `input_shapes` (JSON)
- `difficulty_level`, `benchmark_suite` (kernelbench / waferbench)

Enables queries like:
- "What strategies had >1.2x speedup on memory-bound matmul kernels?"
- "What's the best approach for fused attention with FP4 quantization?"
- "Which strategies haven't been tried on this kernel type?"

### Layer 2: Qualitative Learnings (Markdown)

```
knowledge/
  learnings/
    gotchas.md           # Fragility patterns, things that break
    insights.md          # Why things work, non-obvious discoveries
    tool_recipes.md      # How to use profiling tools effectively
  strategies/            # Detailed strategy descriptions
  resources/             # Distilled GPU optimization fundamentals
  anti_patterns/         # Reward hacks, validation bypasses
  process/               # Agent optimization methodology
```

Quality gate: learnings must include specific kernel/file references and explain why, not just what. Scored similarly to autodev's learnings quality scorer, extended with kernel-specific terms (cache line, TLB, warp, occupancy, bank conflict).

### Knowledge Base Seeding (Pre-Implementation Research Phase)

Before any system code is written, a dedicated research phase curates foundational knowledge:

**Category 1: GPU Optimization Fundamentals**
- NVIDIA CUDA best practices guide
- B200/Blackwell architecture specifics (SM count, shared memory, cache hierarchy, tensor cores)
- gpu-mode/resource-stream curated materials
- Kernel optimization patterns: tiling, coalescing, bank conflicts, occupancy, warp-level primitives, vectorized loads

**Category 2: Agent-Based Optimization Process**
- Wafer.ai blog posts (reward hacks field guide, methodology, benchmarking approach)
- KernelBench paper (Stanford Scaling Intelligence) - problem format, evaluation methodology
- KernelFalcon (PyTorch blog) - autonomous GPU kernel generation architecture
- GEAK (Triton kernel AI agent) - evaluation benchmarks
- Danny's prior work patterns (fp4-multiplier SA+peephole alternation, takehome optimization hierarchy)

**Category 3: Anti-Patterns & Guardrails**
- Wafer reward hacking taxonomy: timing attacks, semantic attacks, benign circumvention
- Fragility patterns: `.reshape()` vs `.view()` in CUDA graphs, torch.compile recompilation triggers, autocast vs explicit dtype
- Known failure modes of LLM-generated kernels

Each resource is distilled into:
1. Structured entries in the strategies DB (techniques, expected impact, applicability)
2. Markdown documents with qualitative context for agent reasoning

## 5. Optimization Goals

Goals are first-class configuration, set per problem or per run:

```python
@dataclass
class OptimizationGoal:
    primary: Literal["latency", "throughput", "memory", "balanced"]
    constraints: GoalConstraints

@dataclass
class GoalConstraints:
    correctness_rtol: float = 1e-3
    correctness_atol: float = 1e-3
    max_memory_mb: int | None = None
    batch_sizes: list[int] = field(default_factory=lambda: [1])
    precision: str = "bf16"
```

### How goals drive the loop

- **DIAGNOSE:** A memory-bound kernel optimized for latency gets different treatment than one optimized for throughput.
- **STRATEGIZE:** Batching optimizations are irrelevant for latency-at-batch-1 but critical for throughput. Memory optimizations are always relevant for memory goal.
- **EVALUATE:** A kernel 1.5x faster at batch=1 but 0.8x at batch=16 is a win for latency, a loss for throughput.
- **DB recording:** Each attempt is tagged with the goal, so learnings are goal-contextualized.

### Reward Hacking Prevention

Part of the evaluation layer:
- Correctness validation on randomized inputs (not just harness test cases)
- Timing validation: reject suspiciously low variance or results that only beat reference on cached inputs
- Agent context includes Wafer reward hacks taxonomy as explicit "don't do this" knowledge
- Semantic attack detection: compare outputs on edge cases (zeros, NaN, inf, denormals)

## 6. Model Abstraction Layer

Protocol-based interface, implement only Claude Code now:

```python
class KernelAgent(Protocol):
    """Interface for any LLM that can generate/optimize kernels."""

    async def generate_kernel(
        self,
        problem: KernelProblem,
        context: OptimizationContext,
        goal: OptimizationGoal,
    ) -> KernelCandidate: ...

    async def diagnose_bottleneck(
        self,
        profile: ProfileData,
        kernel_source: str,
    ) -> Diagnosis: ...

    async def suggest_strategies(
        self,
        diagnosis: Diagnosis,
        available_strategies: list[Strategy],
        prior_attempts: list[Attempt],
    ) -> list[StrategySelection]: ...
```

### ClaudeCodeAgent Implementation

Wraps Claude Code CLI subprocess. Same subprocess pattern as autodev workers (`claude --permission-mode auto --max-turns 100 --output-format text`).

**Prompt structure per method:**

`generate_kernel`:
```
Context injected (in priority order, trimmed from bottom if context budget exceeded):
1. System prompt: role, output format, anti-patterns to avoid
2. Problem definition: reference implementation, input shapes, correctness test
3. Optimization goal: latency/throughput/memory + constraints
4. Current diagnosis: bottleneck type, profiling data
5. Selected strategy: what approach to take and why
6. Prior attempts on this problem: what was tried, speedup achieved, why it failed/succeeded (last 5)
7. Knowledge base context: relevant strategies + learnings for this kernel type (budget: 8K tokens)

Expected output: kernel source code wrapped in markers:
KERNEL_SOURCE_START
<cuda/triton code>
KERNEL_SOURCE_END
APPROACH_NOTES: <1-2 sentence explanation of what this kernel does differently>
```

`diagnose_bottleneck`:
```
Context: profiling data (CUDA events or ncu output) + kernel source + problem definition
Expected output:
DIAGNOSIS_START
bottleneck_type: memory_bound | compute_bound | launch_overhead | occupancy_limited | mixed
explanation: <why this is the bottleneck>
evidence: <specific metrics that indicate this>
DIAGNOSIS_END
```

`suggest_strategies`:
```
Context: diagnosis + available strategies from DB + prior attempts
Expected output:
STRATEGIES_START
1. strategy_name: <name> | rationale: <why this addresses the diagnosed bottleneck>
2. strategy_name: <name> | rationale: <why>
3. strategy_name: <name> | rationale: <why>
STRATEGIES_END
```

**Context budget:** 32K tokens total per invocation. If context exceeds budget, trim in reverse priority order (knowledge base context first, then prior attempts, then diagnosis details).

**Output parsing:** Marker-based (like autodev's `AD_RESULT`). If markers are missing or malformed, retry once with explicit "please format your response with the required markers" prompt. If second attempt fails, log the raw output and flag for human review.

**Retry on invalid CUDA:** If compilation fails, the error is appended to context and the agent is re-invoked with "Your previous kernel had a compilation error: <error>. Fix it." Max 3 compilation retries per attempt.

### Subagent Model

Subagents are short-lived Claude Code subprocesses for parallel candidate generation:

| Parameter | Value | Notes |
|---|---|---|
| Max concurrent subagents | 3 | Cost guard - each is a Claude API call |
| Subagent context | Problem + diagnosis + assigned strategy | Lighter than primary agent context |
| Subagent output | Kernel source + approach notes | Same marker format as primary agent |
| GPU access | None | Subagents generate code only; primary agent benchmarks |
| Timeout | 120s | Kernel generation shouldn't take longer |

**Candidate selection flow:**
1. Primary agent decides to explore N strategies in parallel
2. Spawns N subagents, one per strategy, each with the same problem + diagnosis but different strategy assignment
3. Collects all candidate kernels
4. Validates all candidates for correctness (sequentially, on GPU)
5. Benchmarks all correct candidates (sequentially, on GPU -- no GPU contention)
6. Best candidate becomes the new baseline; all results recorded in DB

**When to spawn subagents vs. generate directly:**
- Default: primary agent generates one kernel at a time (simpler, cheaper)
- Subagents triggered when: (a) diagnosis suggests multiple viable strategies, (b) plateau detected and exploring breadth is more valuable than depth, (c) explicitly requested by operator

### Cost Tracking

Every LLM invocation records:
- `input_tokens`, `output_tokens`, `estimated_cost_usd`
- Stored in `attempts` table alongside speedup/correctness results
- Cumulative cost per problem checked against `max_cost_per_problem_usd` before each new attempt

**Future:** `OpenAIAgent`, `GeminiAgent`, etc. Same interface, different backend.

**ComparisonRunner:** For Kernel Arena benchmarking - runs the same problem through multiple agents, records results side-by-side in the DB.

## 7. Tool System

### Remote Execution Model

All GPU-dependent tools execute on the B200 via SSH. The execution model:

**Connection:** Persistent SSH connection via `ControlMaster` multiplexing. One control socket (`~/.ssh/ctrl-b200-node`) shared across all tool invocations. Eliminates per-command SSH handshake overhead.

```
# ~/.ssh/config addition for kernel-forge
Host b200-node
    ControlMaster auto
    ControlPath ~/.ssh/ctrl-%r@%h:%p
    ControlPersist 600
    ServerAliveInterval 30
```

**Remote working directory:** `~/kernel-forge-workspace/` on the B200. Structure:
```
~/kernel-forge-workspace/
├── kernels/          # Kernel source files transferred for compilation/benchmarking
├── results/          # Benchmark results, profiling output
├── harness/          # KernelBench harness (cloned once, updated as needed)
└── env/              # Python venv with PyTorch, CUDA toolkit
```

**File transfer:** `rsync` for kernel source files (local -> remote) and results (remote -> local). Small files only (kernel source is typically <500 lines). No large dataset transfer.

**Environment activation:** Every remote command is wrapped:
```bash
ssh b200-node "cd ~/kernel-forge-workspace && source env/bin/activate && CUDA_VISIBLE_DEVICES=2 $COMMAND"
```

**Error handling:**
- SSH connection failure: retry 3x with 5s backoff, then report as blocked
- GPU OOM: capture error, feed back to agent as "reduce memory usage" signal
- Compilation failure: capture full compiler stderr, feed to agent for diagnosis
- Runtime segfault: capture signal info, feed to agent as "memory access error" signal
- Timeout (default 300s per command): kill remote process, report timeout with partial output

**GPU safety on shared node:**
- Before every benchmark run: `nvidia-smi --query-gpu=memory.used --format=csv -i 2` to verify GPU 2 is free
- If memory >100MiB used: abort and report "GPU 2 in use by another process"
- Periodic check (every 5 min during long runs) to detect if a coworker started using GPU 2

### Tool Registry

```python
class Tool(Protocol):
    name: str
    description: str  # LLM-readable, used for tool selection

    async def run(self, **kwargs) -> ToolResult: ...

class ToolRegistry:
    def register(self, tool: Tool) -> None: ...
    def get_available(self) -> list[Tool]: ...
    def request_new(self, spec: str) -> None: ...  # logs need + flags for human review
```

### Built-in Tools

| Tool | Purpose | Tier |
|------|---------|------|
| `cuda_events_bench` | Fast wall-clock timing (warmup + timed reps) | Default |
| `ncu_profile` | Deep kernel profiling via Nsight Compute | Escalation |
| `correctness_check` | torch.allclose against reference on randomized inputs | Every attempt |
| `knowledge_query` | Query strategies DB by kernel type, bottleneck, goal | Strategy selection |
| `learnings_read` | Load relevant markdown learnings into agent context | Context building |
| `learnings_write` | Record new qualitative insight | Post-attempt |
| `kernel_compile` | Compile CUDA/Triton kernel via load_inline or nvcc | Pre-benchmark |
| `gpu_status` | Check nvidia-smi, verify GPU 2 is free | Pre-run |

### Tool Extensibility

When the agent identifies a need for a tool that doesn't exist:
1. If simple (e.g., an `ncu` wrapper with specific metric filters): build it, register it, use it
2. If complex or safety-sensitive: flag for human review before registration

## 8. Benchmark Harness Integration

### KernelBench (Primary - 250 problems)

Adapter for ScalingIntelligence/KernelBench format:
- Problems: Python `.py` files with `Model` class (reference) + `get_inputs()` + `get_init_inputs()`
- Solutions: `ModelNew` class using `torch.utils.cpp_extension.load_inline()` for custom CUDA/HIP kernels
- Scoring: `score = speedup if correct else 0` (where `speedup = reference_time / optimized_time`)
- Correctness: `torch.allclose(rtol=1e-3, atol=1e-3)`
- Difficulty levels 1-4 provide natural curriculum progression

### WaferBench NVFP4 (Later - 6 problems)

Adapter for Wafer's B200-specific format:
- Problems: Fused FP4 inference kernels (add_rmsnorm_fp4quant, etc.)
- Solutions: Raw `.cu` files with PYBIND11 bindings
- Benchmarking: 500 warmup, 100 timed reps, CUDA events, L2 cache eviction
- Hardware: NVIDIA B200 only

## 9. Project Structure

```
kernel-forge/
├── pyproject.toml
├── CLAUDE.md                      # Project-specific Claude Code config
├── autodev.toml                   # autodev build configuration
├── src/
│   └── kernel_forge/
│       ├── __init__.py
│       ├── core/
│       │   ├── loop.py            # Primary agent optimization loop
│       │   ├── problem.py         # KernelProblem, OptimizationGoal types
│       │   └── evaluate.py        # Scoring, reward hack detection
│       ├── agents/
│       │   ├── base.py            # KernelAgent protocol
│       │   ├── claude.py          # Claude Code CLI implementation
│       │   └── subagents.py       # Candidate generation spawner
│       ├── knowledge/
│       │   ├── db.py              # SQLite: strategies, attempts, profiles
│       │   ├── learnings.py       # Markdown learnings manager
│       │   ├── seed/              # Pre-populated knowledge documents
│       │   └── query.py           # Knowledge retrieval for context injection
│       ├── remote/
│       │   ├── executor.py        # SSH command execution, connection management
│       │   ├── transfer.py        # rsync file transfer (kernels up, results down)
│       │   ├── gpu_guard.py       # GPU availability checking, contention detection
│       │   └── dry_run.py         # Mock executor for local development
│       ├── tools/
│       │   ├── registry.py        # Tool registration + discovery
│       │   ├── profiling.py       # cuda_events_bench, ncu_profile wrappers
│       │   ├── benchmark.py       # KernelBench harness integration
│       │   └── compiler.py        # Kernel compilation (load_inline, nvcc)
│       ├── harness/
│       │   ├── kernelbench.py     # ScalingIntelligence/KernelBench adapter
│       │   └── waferbench.py      # WaferBench NVFP4 adapter (stub)
│       └── config.py              # Hardware targets, GPU selection, goals
├── knowledge/
│   ├── strategies/                # Seeded optimization technique descriptions
│   ├── resources/                 # Distilled GPU optimization docs
│   ├── anti_patterns/             # Reward hacks, fragility patterns
│   └── process/                   # Agent optimization methodology
├── tests/
│   ├── test_knowledge_db.py
│   ├── test_tools.py
│   ├── test_evaluate.py
│   └── test_harness.py
└── scripts/
    └── run.py                     # CLI entry point
```

## 10. Build Plan (autodev)

The system will be built using autodev's swarm mode. The implementation plan will be structured as autodev-compatible phases with verification criteria per phase:

**Phase 0: Knowledge Base Research & Seeding** (pre-code)
- Research and distill GPU optimization resources
- Research agent optimization methodology
- Research anti-patterns and guardrails
- Populate `knowledge/` directory

**Phase 1: Foundation** (types, DB, config, remote execution)
- Data types: KernelProblem, OptimizationGoal, Strategy, Attempt, Diagnosis, FailureReport
- SQLite schema and DB layer (with schema_version migration support)
- Configuration system (hardware targets, GPU selection)
- Remote execution layer: SSH executor, rsync transfer, GPU guard
- Dry run executor for local development
- B200 environment setup: create workspace, install venv, clone KernelBench harness

**Phase 2: Tool System**
- Tool protocol and registry
- Built-in tools: cuda_events_bench, correctness_check, kernel_compile, gpu_status
- ncu_profile wrapper
- All tools use remote executor for GPU operations

**Phase 3: Knowledge System**
- Learnings manager (read/write/quality-gate)
- Knowledge query layer (DB queries for strategy selection)
- Seed data loading from `knowledge/` into DB

**Phase 4: Agent Layer**
- KernelAgent protocol
- ClaudeCodeAgent implementation (Claude Code CLI subprocess)
- Subagent spawner for parallel candidate generation

**Phase 5: Core Loop**
- Primary agent optimization loop (steps 1-10)
- Failure taxonomy and error-informed retry logic
- Plateau detection and profiling escalation
- Convergence detection with configurable parameters
- Cost tracking and budget enforcement

**Phase 6: Harness Integration**
- KernelBench adapter (problem loading, solution format, scoring)
- End-to-end test: run one KernelBench problem through the full loop

**Phase 7: Evaluation & Reporting**
- Scoring against optimization goals
- Reward hack detection
- Run reporting (results, learnings, profile summary)

**Phase 8: CLI & Operator Interface**
- CLI entry point for running problems
- Supervised mode: operator approval gates, Rich-based live display
- Telegram notifications for milestones/blockers
- Structured run logging (runs/ directory with JSON + JSONL output)

Each phase has verification: `pytest + ruff + mypy` must pass before commit.

**Note on Phase 0:** Knowledge base seeding is research-heavy and open-ended. This phase should be done with heavy human supervision (Danny directing the research, reviewing output quality) rather than fully autonomous autodev execution. The output quality here sets the ceiling for everything else.

## 11. Hardware Configuration

```toml
[hardware.b200]
ssh_host = "b200-node"
ssh_user = "danny"
gpu_id = 2
cuda_visible_devices = "2"
gpu_check_interval_seconds = 300
cuda_version = "13.0"
driver_version = "580.95.05"
gpu_memory_mb = 183359
architecture = "blackwell"
```

## 12. Observability & Logging

Every optimization run produces a structured log:

```
runs/
  {problem_name}_{timestamp}/
    run.json           # Run metadata: problem, goal, hardware, total attempts, best speedup, cost
    attempts.jsonl     # One JSON line per attempt: strategy, speedup, correct, cost, duration
    best_kernel.cu     # Best kernel source
    baseline_profile/  # Baseline profiling output
    best_profile/      # Best kernel profiling output
    learnings.md       # Qualitative learnings from this run
```

**Structured logging:** Python `logging` with JSON formatter. Levels:
- `INFO`: loop transitions (DIAGNOSE -> STRATEGIZE -> IMPLEMENT), benchmark results, strategy selections
- `DEBUG`: full profiling output, kernel source diffs, knowledge base queries
- `WARNING`: plateau detected, GPU contention detected, cost threshold approaching
- `ERROR`: compilation failures, runtime crashes, SSH connection issues

**Console output (supervised mode):** Rich-based live display showing:
- Current step in the loop
- Best speedup so far vs baseline
- Attempt count / budget remaining
- Last attempt result (speedup, correct/incorrect)

## 13. Schema Migration

SQLite schema includes a `schema_version` table:

```sql
CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
INSERT INTO schema_version VALUES (1);
```

Migration functions in `db.py`:
```python
MIGRATIONS = {
    1: "initial schema",
    2: "ALTER TABLE attempts ADD COLUMN cost_usd REAL",
    # ...
}
```

On startup, check current version and apply any pending migrations sequentially. Simple and sufficient for a research tool.

## 14. Dry Run Mode

For development without GPU access:

```python
class DryRunExecutor:
    """Mock GPU backend using cached/synthetic profiling data."""
    # Returns pre-recorded benchmark results for known kernels
    # Generates synthetic profiling data for unknown kernels
    # Validates correctness locally (CPU-only torch)
```

Enables testing the full optimization loop (knowledge queries, strategy selection, convergence detection, cost tracking) without SSH/GPU access. Invaluable for autodev workers building the system on a local machine.

## 15. Success Criteria

### Exploratory Phase (current)
- [ ] System can take a KernelBench Level 1 problem and produce a correct, faster kernel without human intervention
- [ ] Knowledge base accumulates useful learnings across multiple problems
- [ ] Profiling correctly identifies bottleneck types and drives strategy selection
- [ ] No reward hacking in produced kernels

### Maturity Phase (later)
- [ ] Competitive KernelBench scores across all difficulty levels
- [ ] Cross-kernel generalization: strategies learned on matmul improve attention optimization
- [ ] WaferBench NVFP4 specialization producing competitive FP4 kernels
- [ ] Model comparison infrastructure working (Claude vs GPT vs Gemini)
