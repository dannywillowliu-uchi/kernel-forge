# Kernel Forge

Autonomous GPU kernel optimization system. Takes kernel problems (reference implementation + correctness test), optimizes them via a profile-diagnose-strategize-implement-benchmark loop on NVIDIA B200, and learns across runs.

## Tech Stack

- Python 3.12, asyncio, SQLite, SSH (asyncssh), Rich, Click
- Remote execution on B200 GPU 2 via SSH
- Claude Code as the agent backbone

## Code Style

- Indentation: Tabs
- Quotes: Double quotes
- Type hints on all public APIs
- Minimal comments, only when logic is complex

## Commands

```bash
uv run pytest tests/ -v          # Run tests
uv run ruff check src/ tests/    # Lint
uv run mypy src/kernel_forge/    # Type check
```

## Architecture

```
src/kernel_forge/
  core/        # Data types, optimization loop, evaluation
  agents/      # KernelAgent protocol, Claude Code impl
  knowledge/   # SQLite DB, markdown learnings, query layer
  remote/      # SSH executor, GPU guard, dry-run mock
  tools/       # Tool registry, profiling, benchmarking, compilation
  harness/     # KernelBench / WaferBench adapters
  config.py    # Hardware config, forge config, TOML loading
  cli.py       # Click CLI entry point
```
