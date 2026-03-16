# Kernel Forge Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an autonomous GPU kernel optimization system that takes kernel problems, optimizes them via a profile-diagnose-strategize-implement-benchmark loop, and learns across runs.

**Architecture:** Single primary agent runs the optimization loop on B200 GPU 2 via SSH, spawning short-lived subagents for parallel candidate generation. SQLite + markdown knowledge base compounds learnings. Built with autodev.

**Tech Stack:** Python 3.12, asyncio, SQLite, SSH (asyncssh), Rich, Click, PyTorch (remote), CUDA (remote)

**Spec:** docs/superpowers/specs/2026-03-15-kernel-forge-design.md

---

## Chunk 1: Project Scaffolding + Data Types

### Task 1: Project scaffolding

**Files:**
- Create: pyproject.toml
- Create: src/kernel_forge/__init__.py
- Create: CLAUDE.md
- Create: .gitignore
- Create: tests/__init__.py

- [ ] **Step 1: Create pyproject.toml** with project metadata, dependencies (click, rich, asyncssh, aiosqlite), dev deps (pytest, pytest-asyncio, ruff, mypy), entry point kernel-forge=kernel_forge.cli:main, ruff config (py312, line-length=100), mypy strict, pytest asyncio_mode=auto

- [ ] **Step 2: Create src/kernel_forge/__init__.py** with version 0.1.0

- [ ] **Step 3: Create .gitignore** (pycache, mypy_cache, egg-info, dist, .venv, *.db, runs/)

- [ ] **Step 4: Create CLAUDE.md** with project description, tech stack, code style (tabs, double quotes, type hints), test commands, architecture overview

- [ ] **Step 5: Create tests/__init__.py** (empty)

- [ ] **Step 6: Install and verify** -- uv venv --python 3.12 and uv pip install -e ".[dev]"

- [ ] **Step 7: Verify toolchain** -- ruff check, mypy, pytest all pass with no files

- [ ] **Step 8: Commit** -- "feat: project scaffolding with pyproject.toml and CLAUDE.md"

---

### Task 2: Core data types

**Files:**
- Create: src/kernel_forge/core/__init__.py
- Create: src/kernel_forge/core/types.py
- Create: tests/test_types.py

All data types from spec Section 5 and Section 3:
- KernelProblem (name, reference_source, input_shapes, benchmark_suite, difficulty_level)
- OptimizationGoal (primary: latency|throughput|memory|balanced, constraints: GoalConstraints)
- GoalConstraints (correctness_rtol, correctness_atol, max_memory_mb, batch_sizes, precision)
- Strategy (id, name, category: StrategyCategory, description, applicability, expected_impact)
- StrategyCategory enum (memory_opt, compute_opt, precision_opt, pipeline_opt, execution_opt, algorithmic)
- Attempt (kernel_problem, strategy_name, speedup, correct, hardware, optimization_goal, kernel_source_hash, input_tokens, output_tokens, cost_usd, profiling_tier, failure_report)
- KernelCandidate (source, approach_notes, strategy_name)
- ProfileData (runtime_us, profiling_tier, metrics dict, raw_output)
- Diagnosis (bottleneck_type: BottleneckType, explanation, evidence dict, profiling_tier)
- BottleneckType enum (memory_bound, compute_bound, launch_overhead, occupancy_limited, mixed)
- FailureReport (failure_type: FailureType, error_output, kernel_source, strategy_name)
- FailureType enum (compilation_error, link_error, runtime_segfault, runtime_oom, correctness_failure, numerical_instability, timeout, performance_regression)
- TerminationConfig (plateau_threshold=0.02, plateau_window_cuda_events=3, plateau_window_ncu=3, max_attempts=25, max_cost_usd=5.0, max_wall_time_seconds=1800, max_consecutive_failures=5)

- [ ] **Step 1: Write tests** for creation and default values of each type
- [ ] **Step 2: Run test, verify fail**
- [ ] **Step 3: Implement all types as dataclasses with enums**
- [ ] **Step 4: Run tests + lint + mypy**
- [ ] **Step 5: Commit** -- "feat: core data types for kernel problems, strategies, attempts, and diagnostics"

---

### Task 3: Configuration system

**Files:**
- Create: src/kernel_forge/config.py
- Create: tests/test_config.py

- HardwareConfig (ssh_host, ssh_user, gpu_id, cuda_visible_devices, remote_workspace, gpu_check_interval_seconds, command_timeout_seconds, gpu_memory_threshold_mib)
- HardwareConfig.wrap_remote_command(cmd) -- wraps with cd, source activate, CUDA_VISIBLE_DEVICES
- ForgeConfig (hardware, termination, knowledge_dir, runs_dir, db_path, dry_run, max_concurrent_subagents)
- default_config() returns sensible defaults matching spec Section 11
- load_config(path) loads from TOML file, merges with defaults

- [ ] **Step 1: Write tests** for default_config, wrap_remote_command, load_config from TOML
- [ ] **Step 2: Run test, verify fail**
- [ ] **Step 3: Implement config module**
- [ ] **Step 4: Run tests + lint + mypy**
- [ ] **Step 5: Commit** -- "feat: configuration system with hardware, termination, and TOML loading"

---

### Task 4: SQLite knowledge database

**Files:**
- Create: src/kernel_forge/knowledge/__init__.py
- Create: src/kernel_forge/knowledge/db.py
- Create: tests/test_db.py

Schema from spec Section 4:
- schema_version table (version INTEGER PRIMARY KEY)
- strategies table (id, name UNIQUE, category, description, applicability, expected_impact)
- attempts table (id, kernel_problem, strategy_name, speedup, correct, hardware, optimization_goal, profiling_tier, kernel_source_hash, input_tokens, output_tokens, cost_usd, timestamp)
- kernel_profiles table (id, kernel_problem, bottleneck_type, memory_throughput_pct, compute_utilization_pct, occupancy_pct, warp_stall_reasons, l2_hit_rate, shared_mem_usage, profiling_tier, timestamp)
- kernel_classifications table (id, kernel_problem UNIQUE, kernel_type, input_shapes, difficulty_level, benchmark_suite)
- Indexes on attempts(kernel_problem), attempts(strategy_name), strategies(category)

KnowledgeDB class with methods:
- initialize() -- create tables, set schema version
- close()
- get_schema_version()
- insert_strategy(Strategy) -> int
- get_strategies_for_bottleneck(bottleneck_type) -> list[Strategy]
- insert_attempt(Attempt) -> int
- get_attempts_for_problem(kernel_problem) -> list[Attempt]
- get_best_strategies_for_kernel_type(kernel_type, limit) -> list[dict] with avg_speedup
- get_total_cost_for_problem(kernel_problem) -> float

- [ ] **Step 1: Write tests** for initialize, insert/query strategies, insert/query attempts, best strategies, total cost
- [ ] **Step 2: Run test, verify fail**
- [ ] **Step 3: Implement KnowledgeDB with aiosqlite**
- [ ] **Step 4: Run tests + lint + mypy**
- [ ] **Step 5: Commit** -- "feat: SQLite knowledge database with strategies, attempts, and query layer"

---

### Task 5: Remote execution layer

**Files:**
- Create: src/kernel_forge/remote/__init__.py
- Create: src/kernel_forge/remote/executor.py
- Create: src/kernel_forge/remote/gpu_guard.py
- Create: src/kernel_forge/remote/dry_run.py
- Create: tests/test_remote.py

From spec Section 7 (Remote Execution Model):

CommandResult dataclass (stdout, stderr, exit_code, timed_out, success property)

Executor protocol with:
- run(command, timeout=300) -> CommandResult
- upload(local_path, remote_path)
- download(remote_path, local_path)

RemoteExecutor implements Executor:
- Uses asyncio.create_subprocess_exec with ssh
- Wraps commands with cd workspace, source activate, CUDA_VISIBLE_DEVICES
- Handles timeout via asyncio.wait_for, kills process on timeout
- upload/download via rsync subprocess

GpuGuard:
- check() -> GpuStatus (available bool, memory_used_mib, message)
- Runs nvidia-smi query on target GPU, parses memory usage
- Returns unavailable if memory > threshold (default 100 MiB)

DryRunExecutor implements Executor:
- Returns synthetic nvidia-smi output (0 MiB)
- Returns synthetic benchmark results for bench commands
- No-op upload/download

- [ ] **Step 1: Write tests** using DryRunExecutor for all paths
- [ ] **Step 2: Run test, verify fail**
- [ ] **Step 3: Implement all three modules**
- [ ] **Step 4: Run tests + lint + mypy**
- [ ] **Step 5: Commit** -- "feat: remote execution layer with SSH executor, GPU guard, and dry-run mock"

---

## Chunk 2: Tool System + Knowledge System

### Task 6: Tool registry and protocol

**Files:**
- Create: src/kernel_forge/tools/__init__.py
- Create: src/kernel_forge/tools/registry.py
- Create: tests/test_tools.py

ToolResult dataclass (success, data dict, output str, error str)

Tool protocol (name str, description str, run(**kwargs) -> ToolResult)

ToolRegistry:
- register(tool) -- adds to internal dict by name
- get_available() -> list[Tool]
- run(tool_name, **kwargs) -> ToolResult -- raises KeyError if unknown
- request_new(spec) -- logs the request to pending_requests list

- [ ] **Step 1: Write tests** for register, run, unknown tool, request_new
- [ ] **Step 2: Implement registry**
- [ ] **Step 3: Run tests + lint**
- [ ] **Step 4: Commit** -- "feat: tool registry with protocol, registration, and extensibility"

---

### Task 7: Built-in tools

**Files:**
- Create: src/kernel_forge/tools/profiling.py (GpuStatusTool, CudaEventsBench, NcuProfile)
- Create: src/kernel_forge/tools/compiler.py (KernelCompiler)
- Create: src/kernel_forge/tools/benchmark.py (CorrectnessTool)
- Create: tests/test_builtin_tools.py

Each tool takes an Executor in constructor. All tests use DryRunExecutor.

GpuStatusTool -- wraps GpuGuard, returns available/memory_used_mib
CudaEventsBench -- runs benchmark harness script via executor, parses JSON output
NcuProfile -- runs ncu via executor with configurable metrics, returns raw output
KernelCompiler -- writes kernel source to remote file, attempts compilation
CorrectnessTool -- runs correctness check script via executor, parses JSON result

- [ ] **Step 1: Write tests** with DryRunExecutor
- [ ] **Step 2: Implement all tools**
- [ ] **Step 3: Run tests + lint**
- [ ] **Step 4: Commit** -- "feat: built-in tools for profiling, compilation, benchmarking, and correctness"

---

### Task 8: Learnings manager

**Files:**
- Create: src/kernel_forge/knowledge/learnings.py
- Create: tests/test_learnings.py

LearningsManager(knowledge_dir):
- write(category, content, kernel_ref) -> bool -- quality-gated, appends timestamped entry to category.md
- read_relevant(kernel_type, max_tokens=8000) -> list[str] -- searches all .md files for kernel_type mentions
- read_all(max_tokens=8000) -> str -- all learnings concatenated

Quality gate: score based on content length (>=50 chars: +1), kernel_ref present (+1), contains bug/fix/gotcha/regression (+1), contains kernel terms (+0.5), too short (<30 chars: -1). Threshold >= 0.5.

- [ ] **Step 1: Write tests** for write, read_relevant, quality gate accept/reject
- [ ] **Step 2: Implement learnings manager**
- [ ] **Step 3: Run tests + lint**
- [ ] **Step 4: Commit** -- "feat: markdown learnings manager with quality gate and relevance filtering"

---

### Task 9: Knowledge query layer

**Files:**
- Create: src/kernel_forge/knowledge/query.py
- Create: tests/test_knowledge_query.py

KnowledgeQuery(db, learnings):
- build_context(kernel_problem, kernel_type, bottleneck_type, max_tokens=8000) -> str
  Builds agent context string from:
  1. Best strategies from DB for this kernel type
  2. Strategies matching bottleneck type
  3. Prior attempts on this problem (last 5)
  4. Relevant learnings from markdown
  Respects token budget, trims from bottom.

- [ ] **Step 1: Write tests** with seeded DB + learnings
- [ ] **Step 2: Implement query layer**
- [ ] **Step 3: Run tests + lint**
- [ ] **Step 4: Commit** -- "feat: knowledge query layer for context injection from DB + learnings"

---

## Chunk 3: Agent Layer + Core Loop

### Task 10: KernelAgent protocol + ClaudeCodeAgent

**Files:**
- Create: src/kernel_forge/agents/__init__.py
- Create: src/kernel_forge/agents/base.py (KernelAgent protocol)
- Create: src/kernel_forge/agents/prompts.py (prompt builders + output parsers)
- Create: src/kernel_forge/agents/claude.py (ClaudeCodeAgent)
- Create: tests/test_agents.py

KernelAgent protocol:
- generate_kernel(problem, goal, diagnosis, strategy_name, prior_attempts, knowledge_context) -> KernelCandidate | None
- diagnose_bottleneck(profile, kernel_source, problem) -> Diagnosis | None
- suggest_strategies(diagnosis, available_strategies, prior_attempts) -> list[str]

Prompt builders (from spec Section 6):
- build_generate_prompt() -- includes problem, goal, diagnosis, strategy, prior attempts, knowledge context, output format with KERNEL_SOURCE_START/END markers
- build_diagnose_prompt() -- includes profile data, kernel source, problem, expects DIAGNOSIS_START/END
- build_suggest_strategies_prompt() -- includes diagnosis, available strategies, prior attempts, expects STRATEGIES_START/END

Output parsers:
- parse_kernel_output(raw) -> KernelCandidate | None -- regex for KERNEL_SOURCE_START/END and APPROACH_NOTES
- parse_diagnosis_output(raw) -> Diagnosis | None -- regex for DIAGNOSIS_START/END, bottleneck_type, explanation, evidence

ClaudeCodeAgent:
- _invoke_claude(prompt) -> str -- runs claude CLI subprocess with --permission-mode auto --max-turns 50 --output-format text --model sonnet -p prompt
- generate_kernel: builds prompt, invokes, parses. On parse failure, retries once with format reminder
- diagnose_bottleneck: builds prompt, invokes, parses
- suggest_strategies: builds prompt, invokes, parses strategy names. Fallback to ["shared_mem_tiling"]

- [ ] **Step 1: Write tests** for prompt builders and output parsers (no actual Claude invocation)
- [ ] **Step 2: Implement base.py, prompts.py, claude.py**
- [ ] **Step 3: Run tests + lint + mypy**
- [ ] **Step 4: Commit** -- "feat: agent layer with KernelAgent protocol, Claude Code impl, prompt templates, and output parsers"

---

### Task 11: Core optimization loop + evaluation

**Files:**
- Create: src/kernel_forge/core/loop.py
- Create: src/kernel_forge/core/evaluate.py
- Create: tests/test_loop.py
- Create: tests/test_evaluate.py

evaluate.py:
- compute_score(speedup, correct) -> float -- speedup if correct else 0.0
- should_escalate_profiling(recent_attempts, config) -> bool -- True if last N correct attempts have <threshold relative improvement
- should_terminate(attempt_count, total_cost, elapsed_seconds, consecutive_failures, config) -> bool -- True if any termination trigger hit
- classify_failure(exit_code, stderr, stdout) -> FailureType -- pattern matching on error signals

loop.py:
- LoopState dataclass: problem, goal, config, attempt_count, best_speedup, best_kernel_source, total_cost, consecutive_failures, profiling_tier, start_time, attempts list
- record_attempt(speedup, correct, cost) -- updates state
- should_stop property -- delegates to should_terminate
- should_escalate property -- delegates to should_escalate_profiling
- kernel_hash(source) -> str -- sha256 truncated to 16 chars

- [ ] **Step 1: Write evaluate tests** for plateau detection, termination conditions, failure classification, score computation
- [ ] **Step 2: Implement evaluate.py**
- [ ] **Step 3: Write loop tests** for LoopState creation, record_attempt, should_stop, should_escalate
- [ ] **Step 4: Implement loop.py**
- [ ] **Step 5: Run all tests + lint + mypy**
- [ ] **Step 6: Commit** -- "feat: core optimization loop with state tracking, evaluation, and termination logic"

---

## Chunk 4: Harness + CLI + Integration

### Task 12: KernelBench harness adapter

**Files:**
- Create: src/kernel_forge/harness/__init__.py
- Create: src/kernel_forge/harness/kernelbench.py
- Create: tests/test_harness.py

KernelBenchAdapter(problems_dir):
- list_problems(difficulty=None) -> list[KernelProblem] -- scans levelN/ dirs for .py files, parses difficulty from dir name
- get_problem(name) -> KernelProblem | None

- [ ] **Step 1: Write tests** with sample problem .py file in tmp_path
- [ ] **Step 2: Implement adapter**
- [ ] **Step 3: Run tests + lint**
- [ ] **Step 4: Commit** -- "feat: KernelBench harness adapter for problem loading and listing"

---

### Task 13: CLI entry point

**Files:**
- Create: src/kernel_forge/cli.py
- Create: tests/test_cli.py

Click CLI with commands:
- main group with --version
- optimize PROBLEM_NAME --goal --config --dry-run --difficulty
- list-problems --difficulty --problems-dir
- report PROBLEM_NAME

- [ ] **Step 1: Write tests** for --help and --version
- [ ] **Step 2: Implement CLI**
- [ ] **Step 3: Run tests + lint**
- [ ] **Step 4: Commit** -- "feat: CLI entry point with optimize, list-problems, and report commands"

---

### Task 14: ForgeRunner integration wiring

**Files:**
- Create: src/kernel_forge/core/runner.py
- Create: tests/test_runner.py

ForgeRunner(config):
- initialize() -- creates DB, executor (dry_run or remote), tool registry, learnings manager, knowledge query
- shutdown() -- closes DB
- prepare_run(problem) -> Path -- creates timestamped run directory with run.json metadata

- [ ] **Step 1: Write tests** for initialize, prepare_run
- [ ] **Step 2: Implement ForgeRunner**
- [ ] **Step 3: Run tests + lint**
- [ ] **Step 4: Commit** -- "feat: ForgeRunner wiring all components together"

---

### Task 15: Full verification pass

- [ ] **Step 1: Run full test suite** -- uv run pytest tests/ -v --tb=short
- [ ] **Step 2: Run linter** -- uv run ruff check src/ tests/
- [ ] **Step 3: Run type checker** -- uv run mypy src/kernel_forge/
- [ ] **Step 4: Commit any fixes** -- "fix: resolve any lint/type issues from full verification"

---

## Chunk 5: Knowledge Seeding (Human-Supervised)

### Task 16: Knowledge base directory structure and seed data

**Human-supervised.** Autodev worker sets up directory structure and initial seed files, Danny reviews quality.

- [ ] **Step 1: Create knowledge directory structure** with strategies/, anti_patterns/, resources/, process/, learnings/ subdirs
- [ ] **Step 2: Create seed_db.py script** that parses strategy metadata from markdown and inserts into SQLite
- [ ] **Step 3: Research and populate** -- use research subagents to fetch/distill NVIDIA CUDA best practices, gpu-mode resources, Wafer.ai blog, KernelBench paper, KernelFalcon, GEAK, Danny's prior work patterns
- [ ] **Step 4: Review checkpoint with Danny**
- [ ] **Step 5: Run seed script and commit**

---

## Chunk 6: B200 Environment Setup + End-to-End

### Task 17: Remote workspace setup on B200

- [ ] **Step 1: Create scripts/setup_remote.sh** -- SSH to b200-node, create workspace, venv, install PyTorch, clone KernelBench, verify GPU access
- [ ] **Step 2: Run setup script**
- [ ] **Step 3: Update SSH config** with ControlMaster multiplexing
- [ ] **Step 4: Commit** -- "feat: B200 remote workspace setup script"

### Task 18: End-to-end dry run test

- [ ] **Step 1: Run one KernelBench Level 1 problem through full loop in dry-run mode**
- [ ] **Step 2: Verify run output** (run.json, attempts.jsonl exist with valid content)
- [ ] **Step 3: Commit any fixes**

### Task 19: autodev.toml configuration

- [ ] **Step 1: Create autodev.toml** with target name, verification command, swarm config
- [ ] **Step 2: Commit** -- "feat: autodev.toml build configuration"
