# Kernel Forge

Autonomous GPU kernel optimization using agent-to-agent collaboration. An orchestrator agent drives strategy while optimizer agents write and benchmark kernels on NVIDIA B200.

## How It Works

Kernel Forge uses two collaborating agents to optimize GPU kernels:

**Orchestrator** -- A senior GPU performance engineer who understands roofline analysis, hardware bottlenecks, and optimization strategy. Reads code, profiles kernels, computes the speed-of-light, and directs the optimizer. Watches the optimizer's live reasoning and uses SendMessage to course-correct mid-run.

**Optimizer** -- A kernel implementation specialist who writes PyTorch, Triton, or CUDA C++ kernels. Profiles, diagnoses bottlenecks, writes code, benchmarks. Receives strategic guidance from the orchestrator and adjusts approach accordingly.

The agents communicate bidirectionally:
- Orchestrator reads the optimizer's output trace in real time
- Orchestrator sends messages to redirect the optimizer mid-run
- Optimizer writes benchmark checkpoints the orchestrator monitors
- Both share the kernel code on disk (submission.py on the GPU node)

### The Loop

```
Orchestrator                          Optimizer
    |                                     |
    |-- reads problem, profiles,          |
    |   computes roofline                 |
    |                                     |
    |-- spawns optimizer with strategy -->|
    |                                     |-- profiles baseline
    |-- reads optimizer's reasoning       |-- writes kernel
    |-- monitors checkpoints              |-- benchmarks (checkpoint.jsonl)
    |                                     |-- iterates...
    |-- [SendMessage: "bottleneck is X"]->|
    |                                     |-- adjusts approach
    |-- [sees plateau in checkpoints]     |
    |-- [SendMessage: "try approach Y"] ->|
    |                                     |-- tries Y
    |-- [sees convergence]                |
    |-- stops optimizer                   |
    |-- reports results                   |
```

## Project Structure

```
src/kernel_forge/
  agents/           # Orchestrator + optimizer prompt templates
  core/             # Data types, optimization loop, evaluation
  knowledge/        # SQLite DB, markdown learnings, query layer
  remote/           # SSH executor, GPU guard
  tools/            # Tool registry, profiling, benchmarking
  config.py         # Hardware config, forge config
  cli.py            # Click CLI entry point
  orchestrate.py    # Prompt generator for orchestration

problems/           # Problem definitions (YAML)
knowledge/          # Distilled GPU optimization guides
runs/               # Orchestration run logs
```

## Quick Start

```bash
# Run the orchestration loop on a problem
uv run kernel-forge orchestrate trimul --gpu 3

# Or manually: generate the prompt and paste into Claude Code
uv run python -c "
from kernel_forge.orchestrate import build_orchestrate_prompt
import yaml
with open('problems/trimul.yaml') as f:
    problem = yaml.safe_load(f)
prompt = build_orchestrate_prompt(
    problem_name='trimul',
    problem_dir='trimul',
    problem_context=yaml.dump(problem),
    gpu_id=3,
)
print(prompt)
"
```

## Hardware

Targets NVIDIA B200 (Blackwell) via SSH:
- BF16: 1929 TFLOPS
- TF32: 964 TFLOPS
- HBM: 8 TB/s
- SMEM: 228 KB/SM

## Problem Format

Problems are defined in YAML:

```yaml
name: "problem_name"
description: "What the kernel does"
type: "kernel"

measure:
  command: "python bench.py --benchmark"
  metric: "geomean_ms"
  direction: "minimize"

edit:
  target_files: ["submission.py"]
  read_only_files: ["reference.py", "bench.py"]
  working_directory: "~/kernel-forge-workspace/problem"

execution:
  mode: "ssh"
  ssh_host: b200-node
  gpu_id: 3
```

## Results

### SOL-ExecBench (NVIDIA B200 benchmark)
- **069_rms_norm**: 8.5x speedup locally (15x on largest workload), SOL Score 0.936
- **001_attention_backward**: 3.5-5.3x speedup locally, SOL Score 0.460 on leaderboard

### GPU MODE TriMul Competition
- 9.25x speedup over reference (0.786ms geomean)
