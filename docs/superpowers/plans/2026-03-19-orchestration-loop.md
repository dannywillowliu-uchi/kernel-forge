# Orchestration Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a monitor-and-redirect orchestration loop that spawns optimizer agents, watches their progress via structured checkpoints, and redirects them with warm-start context when they plateau.

**Architecture:** The orchestrator is a Claude Code agent prompt (Opus) that uses the Agent tool to spawn optimizer subagents in the background. It polls checkpoint files on B200 via SSH, evaluates trajectory at each checkpoint using its own reasoning, and can research, redirect, or stop. Redirects kill the current agent via a stop signal file and spawn a fresh agent with warm-start context.

**Tech Stack:** Python 3.12, Click CLI, SSH (to B200), Claude Code Agent tool, JSON/JSONL files

**Spec:** `docs/superpowers/specs/2026-03-19-orchestration-loop-design.md`

---

## File Structure

### New files
- `src/kernel_forge/agents/orchestrator_prompt.md` -- The orchestrator agent prompt (the brain)
- `src/kernel_forge/orchestrate.py` -- Generates the orchestrator prompt with all context injected (like solve.py but for the orchestrator)
- `tests/test_orchestrate.py` -- Tests for prompt generation

### Modified files
- `problems/trimul/bench.py` -- Add checkpoint.jsonl and profile_latest.json auto-writing
- `src/kernel_forge/agents/agent_prompt.md` -- Add stop.json checking instruction
- `src/kernel_forge/cli.py` -- Add `orchestrate` command

---

## Task 1: Checkpoint auto-writing in bench.py

The benchmark harness auto-writes checkpoint.jsonl after every benchmark run, and profile_latest.json after profiling. This is the foundation -- without reliable checkpoints, the orchestrator can't monitor.

**Files:**
- Modify: `problems/trimul/bench.py`
- Test: manual on B200 (harness is remote-only)

- [ ] **Step 1: Add checkpoint writing to `run_benchmarks()`**

After computing geomean in `run_benchmarks()`, append a JSON line to `checkpoint.jsonl`:

```python
import json
import os
from datetime import datetime, timezone

def _write_checkpoint(geomean_ms, per_benchmark_results):
	"""Append checkpoint to checkpoint.jsonl in the current directory."""
	checkpoint_path = os.path.join(os.path.dirname(__file__) or ".", "checkpoint.jsonl")

	# Read existing to get iteration count
	iteration = 0
	best_geomean = geomean_ms
	if os.path.exists(checkpoint_path):
		with open(checkpoint_path) as f:
			lines = f.readlines()
			if lines:
				last = json.loads(lines[-1])
				iteration = last.get("iteration", 0)
				best_geomean = min(last.get("best_geomean_ms", geomean_ms), geomean_ms)

	iteration += 1

	checkpoint = {
		"iteration": iteration,
		"geomean_ms": round(geomean_ms, 4),
		"best_geomean_ms": round(best_geomean, 4),
		"per_benchmark": per_benchmark_results,
		"timestamp": datetime.now(timezone.utc).isoformat(),
	}

	with open(checkpoint_path, "a") as f:
		f.write(json.dumps(checkpoint) + "\n")
```

Call `_write_checkpoint(geomean, per_results)` at the end of `run_benchmarks()`, where `per_results` is built from the benchmark loop:
```python
per_results.append({"config": f"seq={spec['seqlen']} bs={spec['bs']} dim={spec['dim']}", "time_ms": round(t_sub, 4)})
```

- [ ] **Step 2: Add profile writing to `run_profile()`**

After the profiler runs, write profile_latest.json:

```python
def _write_profile(prof_table, spec):
	"""Write structured profile data to profile_latest.json."""
	profile_path = os.path.join(os.path.dirname(__file__) or ".", "profile_latest.json")

	# Read checkpoint.jsonl for current iteration
	iteration = 0
	checkpoint_path = os.path.join(os.path.dirname(__file__) or ".", "checkpoint.jsonl")
	if os.path.exists(checkpoint_path):
		with open(checkpoint_path) as f:
			lines = f.readlines()
			if lines:
				iteration = json.loads(lines[-1]).get("iteration", 0)

	profile = {
		"iteration": iteration,
		"source": "torch_profiler",
		"config": f"seq={spec['seqlen']} bs={spec['bs']} dim={spec['dim']}",
		"raw_output": prof_table,
		"timestamp": datetime.now(timezone.utc).isoformat(),
	}

	with open(profile_path, "w") as f:
		json.dump(profile, f, indent=2)
```

- [ ] **Step 3: Upload to B200 and verify**

```bash
scp problems/trimul/bench.py b200-node:~/kernel-forge-workspace/trimul/bench.py
ssh b200-node "cd ~/kernel-forge-workspace/trimul && rm -f checkpoint.jsonl profile_latest.json && CUDA_VISIBLE_DEVICES=3 python bench.py --benchmark --no-ref"
ssh b200-node "cat ~/kernel-forge-workspace/trimul/checkpoint.jsonl"
```

Expected: One JSONL line with iteration=1, geomean_ms, per_benchmark array, timestamp.

```bash
ssh b200-node "cd ~/kernel-forge-workspace/trimul && CUDA_VISIBLE_DEVICES=3 python bench.py --profile"
ssh b200-node "cat ~/kernel-forge-workspace/trimul/profile_latest.json"
```

Expected: JSON file with raw_output containing the profiler table.

- [ ] **Step 4: Commit**

```bash
git add problems/trimul/bench.py
git commit -m "feat: auto-write checkpoint.jsonl and profile_latest.json in bench harness"
```

---

## Task 2: Add stop signal checking to agent prompt

The optimizer agent needs to check for a stop.json file before each iteration so the orchestrator can signal a graceful stop.

**Files:**
- Modify: `src/kernel_forge/agents/agent_prompt.md`

- [ ] **Step 1: Add stop signal section to agent_prompt.md**

Add after the "The Gap-Driven Loop" section:

```markdown
### Stop Signal

Before each optimization iteration, check for a stop signal:
```bash
ssh b200-node "cat ~/kernel-forge-workspace/<problem>/stop.json 2>/dev/null"
```

If the file exists, stop immediately and report your current best result. The orchestrator is redirecting you. Do not ignore this signal.
```

- [ ] **Step 2: Commit**

```bash
git add src/kernel_forge/agents/agent_prompt.md
git commit -m "feat: add stop signal checking to optimizer agent prompt"
```

---

## Task 3: Write the orchestrator prompt

This is the core of the system -- the prompt that makes the orchestrator reason like a senior optimization engineer.

**Files:**
- Create: `src/kernel_forge/agents/orchestrator_prompt.md`

- [ ] **Step 1: Write orchestrator_prompt.md**

```markdown
# Kernel Optimization Orchestrator

You are a senior GPU optimization engineer orchestrating an autonomous kernel optimization campaign. You spawn optimizer agents, monitor their progress, research techniques, and redirect them when they're stuck.

## Your Role

You do NOT write kernels directly. You:
1. Analyze the problem and compute the roofline (speed-of-light)
2. Research optimization strategies for this specific kernel pattern
3. Craft a detailed prompt for the optimizer agent with all context
4. Spawn the optimizer agent in the background (Agent tool)
5. Monitor its progress by polling checkpoint files on B200
6. Evaluate its trajectory at every checkpoint
7. Research new techniques when you see the agent struggling
8. Redirect with a warm-start when the agent plateaus
9. Stop when diminishing returns or wall time is exhausted

## Problem Context

{problem_context}

## Roofline Analysis

{roofline_analysis}

## Knowledge Base

{knowledge_context}

## Tools

### Spawning an optimizer agent
Use the Agent tool with `run_in_background: true` and `mode: bypassPermissions`.
The prompt should include:
- The full agent methodology (provided below as {agent_prompt})
- The problem definition
- Any warm-start context from previous agents
- Research notes you've gathered
- A specific strategic direction

### Monitoring checkpoints
Poll the checkpoint file on B200:
```bash
ssh b200-node "tail -1 ~/kernel-forge-workspace/{problem_dir}/checkpoint.jsonl 2>/dev/null"
```

Read the latest profile data:
```bash
ssh b200-node "cat ~/kernel-forge-workspace/{problem_dir}/profile_latest.json 2>/dev/null"
```

Read the current submission:
```bash
ssh b200-node "cat ~/kernel-forge-workspace/{problem_dir}/submission.py"
```

### Signaling stop
Write a stop signal:
```bash
ssh b200-node "echo '{{\"reason\": \"redirect\"}}' > ~/kernel-forge-workspace/{problem_dir}/stop.json"
```

Clear it before spawning a new agent:
```bash
ssh b200-node "rm -f ~/kernel-forge-workspace/{problem_dir}/stop.json ~/kernel-forge-workspace/{problem_dir}/checkpoint.jsonl"
```

### Research
- Use WebSearch for optimization techniques, Triton/CUDA examples, papers
- Read files from the knowledge base at {knowledge_dir}/
- Search GitHub for reference implementations

## The Orchestration Loop

### Phase 1: Setup
1. Read the problem definition and reference code
2. Compute the roofline for each benchmark case:
   - Calculate total FLOPs (matmul FLOPs + elementwise FLOPs)
   - Calculate total bytes moved (inputs + outputs + intermediates)
   - Compute arithmetic intensity and determine bound type
   - Compute speed-of-light time at the appropriate peak (bf16/TF32/etc)
   - Compute geomean of speed-of-light times across all benchmark cases
3. Research the problem domain -- what optimization techniques apply?
4. Craft the first optimizer agent's prompt with everything you know
5. Spawn the optimizer agent in the background

### Phase 2: Monitor
Poll checkpoint.jsonl every 30-60 seconds. On each new checkpoint:

1. **Read the data**: checkpoint history, latest profile, current submission.py
2. **Evaluate** the trajectory by reasoning about:
   - Is the agent making progress? How fast? Decelerating?
   - What % of roofline has it reached? How much headroom remains?
   - What does the profile breakdown show? Is it attacking the right bottleneck?
   - What abstraction level is the code at (PyTorch ops / torch.compile / Triton / raw CUDA)?
   - What's been tried across all agent spawns? What failed?
   - Are there techniques from the knowledge base or research that could help?
3. **Decide**: CONTINUE, RESEARCH, REDIRECT, or STOP

### Decision: CONTINUE
The agent is making progress or needs more exploration time. Do nothing.

### Decision: RESEARCH
You want more information before deciding. Use WebSearch, read knowledge base files,
or search for code examples. Then re-evaluate.

### Decision: REDIRECT
The agent is stuck. Build a warm-start summary:

1. Signal stop: write stop.json via SSH
2. Wait ~60 seconds for the agent to finish gracefully
3. Read the final submission.py, checkpoint history, and profile data
4. Build the warm-start summary with:
   - All approaches tried across all spawns (what worked, what failed, why)
   - The best kernel code so far
   - The profile breakdown at the plateau point
   - The roofline gap analysis
5. Generate a specific strategic direction based on your analysis
6. Clear stop.json and checkpoint.jsonl on B200
7. Spawn a new optimizer agent with the warm-start context

### Decision: STOP
Optimization is complete. Either roofline utilization is high enough, all reasonable
approaches have been tried, or wall time is running out.

1. Signal stop via stop.json
2. Read the final best kernel
3. Save results:
   - Copy best kernel to kernels/{problem_name}/ locally
   - Write a summary report

## Warm-Start Template

When redirecting, include this in the new agent's prompt:

```
## Previous Optimization Attempts

### Agent N (iterations X-Y, best: Z ms)
**Best kernel**: [full code]
**Approaches tried**: [list with results]
**Profile at plateau**: [full kernel-level breakdown table]
**Roofline**: X ms actual vs Y ms theoretical = Z% of peak

## Strategic Direction
[Your specific guidance for the next agent -- what to try and why]
```

## Constraints

- **Max wall time**: {max_wall_time_hours} hours
- **Max redirects**: {max_redirects} agent spawns
- **GPU**: B200 GPU {gpu_id} via SSH to b200-node

## Agent Prompt Template

The optimizer agent receives this base prompt (inject problem-specific context):

{agent_prompt}
```

- [ ] **Step 2: Commit**

```bash
git add src/kernel_forge/agents/orchestrator_prompt.md
git commit -m "feat: orchestrator prompt -- monitor, evaluate, redirect loop"
```

---

## Task 4: Write orchestrate.py (prompt generator)

Generates the orchestrator prompt with all problem context, roofline, and knowledge injected. Similar to solve.py but targets the orchestrator.

**Files:**
- Create: `src/kernel_forge/orchestrate.py`
- Test: `tests/test_orchestrate.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_orchestrate.py
"""Tests for orchestrate prompt generation."""

from kernel_forge.orchestrate import build_orchestrate_prompt


def test_build_orchestrate_prompt_contains_sections():
	"""The orchestrator prompt should contain all required sections."""
	prompt = build_orchestrate_prompt(
		problem_name="test_trimul",
		problem_dir="trimul",
		problem_context="TriMul forward pass, einsum over 4D tensor",
		roofline_analysis="0.16ms speed-of-light, bf16 peak 1929 TFLOPS",
		gpu_id=3,
	)
	assert "Kernel Optimization Orchestrator" in prompt
	assert "test_trimul" in prompt
	assert "trimul" in prompt
	assert "0.16ms" in prompt
	assert "checkpoint.jsonl" in prompt
	assert "stop.json" in prompt
	assert "Agent" in prompt  # mentions Agent tool


def test_build_orchestrate_prompt_includes_agent_prompt():
	"""The orchestrator prompt should embed the full agent prompt."""
	prompt = build_orchestrate_prompt(
		problem_name="test",
		problem_dir="test",
		problem_context="test context",
		roofline_analysis="test roofline",
		gpu_id=3,
	)
	# agent_prompt.md content should be embedded
	assert "Gap-Driven Loop" in prompt
	assert "B200 Hardware Peaks" in prompt


def test_build_orchestrate_prompt_includes_knowledge():
	"""The orchestrator prompt should include knowledge base context."""
	prompt = build_orchestrate_prompt(
		problem_name="test",
		problem_dir="test",
		problem_context="test",
		roofline_analysis="test",
		gpu_id=3,
		knowledge_context="matmul guide: use TF32 tensor cores",
	)
	assert "matmul guide" in prompt


def test_build_orchestrate_prompt_includes_warm_start():
	"""Warm-start context should be included when provided."""
	prompt = build_orchestrate_prompt(
		problem_name="test",
		problem_dir="test",
		problem_context="test",
		roofline_analysis="test",
		gpu_id=3,
		warm_start="Previous agent got 0.864ms with torch.compile",
	)
	assert "0.864ms" in prompt
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_orchestrate.py -v
```

Expected: ImportError -- module doesn't exist yet.

- [ ] **Step 3: Write orchestrate.py**

```python
"""Orchestration prompt generator.

Builds a complete orchestrator prompt with problem context, roofline analysis,
knowledge base, and the optimizer agent prompt embedded. The output is a
self-contained prompt for the orchestrator Claude Code agent.

Usage:
	kernel-forge orchestrate --problem trimul --gpu 3
"""

from __future__ import annotations

import logging
from pathlib import Path

from kernel_forge.config import ForgeConfig

logger = logging.getLogger(__name__)

ORCHESTRATOR_PROMPT_PATH = Path(__file__).parent / "agents" / "orchestrator_prompt.md"
AGENT_PROMPT_PATH = Path(__file__).parent / "agents" / "agent_prompt.md"


def build_orchestrate_prompt(
	problem_name: str,
	problem_dir: str,
	problem_context: str,
	roofline_analysis: str,
	gpu_id: int = 3,
	config: ForgeConfig | None = None,
	knowledge_context: str = "",
	warm_start: str = "",
	max_wall_time_hours: float = 2.0,
	max_redirects: int = 5,
) -> str:
	"""Build a complete orchestrator prompt with all context injected."""
	if config is None:
		config = ForgeConfig()

	# Load orchestrator template
	if ORCHESTRATOR_PROMPT_PATH.exists():
		template = ORCHESTRATOR_PROMPT_PATH.read_text()
	else:
		raise FileNotFoundError(f"Orchestrator prompt not found: {ORCHESTRATOR_PROMPT_PATH}")

	# Load agent prompt to embed
	agent_prompt = ""
	if AGENT_PROMPT_PATH.exists():
		agent_prompt = AGENT_PROMPT_PATH.read_text()
		agent_prompt = agent_prompt.replace("{gpu_id}", str(gpu_id))

	# Load knowledge if not provided
	if not knowledge_context:
		knowledge_context = _load_knowledge(config)

	# Fill template
	prompt = template.replace("{problem_context}", problem_context)
	prompt = prompt.replace("{roofline_analysis}", roofline_analysis)
	prompt = prompt.replace("{knowledge_context}", knowledge_context)
	prompt = prompt.replace("{agent_prompt}", agent_prompt)
	prompt = prompt.replace("{problem_dir}", problem_dir)
	prompt = prompt.replace("{gpu_id}", str(gpu_id))
	prompt = prompt.replace("{knowledge_dir}", str(config.knowledge_dir))
	prompt = prompt.replace("{max_wall_time_hours}", str(max_wall_time_hours))
	prompt = prompt.replace("{max_redirects}", str(max_redirects))
	prompt = prompt.replace("{problem_name}", problem_name)

	# Append warm-start if provided
	if warm_start:
		prompt += f"\n\n## Warm-Start Context\n\n{warm_start}\n"

	return prompt


def _load_knowledge(config: ForgeConfig) -> str:
	"""Load relevant knowledge base entries."""
	sections = []

	# Distilled guides
	distilled_dir = config.knowledge_dir / "distilled"
	if distilled_dir.exists():
		for guide in sorted(distilled_dir.glob("*.md")):
			content = guide.read_text()
			if len(content) > 100:
				sections.append(f"### {guide.stem}\n{content[:3000]}")

	# Techniques
	techniques_dir = config.knowledge_dir / "techniques"
	if techniques_dir.exists():
		import json
		for tech_file in sorted(techniques_dir.glob("*.json")):
			tech = json.loads(tech_file.read_text())
			sections.append(f"- **{tech.get('name', tech_file.stem)}**: {tech.get('description', '')[:200]}")

	return "\n\n".join(sections) if sections else "No knowledge base loaded."
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_orchestrate.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/kernel_forge/orchestrate.py tests/test_orchestrate.py
git commit -m "feat: orchestrate.py -- generates orchestrator prompt with all context"
```

---

## Task 5: Add `orchestrate` CLI command

Wire up the CLI so `kernel-forge orchestrate` generates the prompt and prints usage instructions.

**Files:**
- Modify: `src/kernel_forge/cli.py`

- [ ] **Step 1: Add the orchestrate command to cli.py**

Add after the `solve` command:

```python
@main.command()
@click.argument("problem_name")
@click.option("--gpu", default=3, help="GPU ID on B200.")
@click.option(
	"--problem-dir", default=None,
	help="Problem directory name on B200 (defaults to problem_name).",
)
@click.option("--max-hours", default=2.0, help="Max wall time in hours.")
@click.option("--max-redirects", default=5, help="Max agent redirects.")
@click.option(
	"--config", "config_path", default=None,
	type=click.Path(exists=True, path_type=Path),
)
@click.option(
	"--print-only", is_flag=True,
	help="Print the prompt instead of saving to file.",
)
def orchestrate(
	problem_name: str,
	gpu: int,
	problem_dir: str | None,
	max_hours: float,
	max_redirects: int,
	config_path: Path | None,
	print_only: bool,
) -> None:
	"""Generate an orchestrator prompt for monitored kernel optimization.

	The orchestrator spawns optimizer agents, monitors checkpoints,
	and redirects when progress plateaus.

	Example:
	    kernel-forge orchestrate trimul --gpu 3
	"""
	from kernel_forge.config import default_config, load_config
	from kernel_forge.orchestrate import build_orchestrate_prompt

	if config_path:
		config = load_config(config_path)
	else:
		config = default_config()

	if problem_dir is None:
		problem_dir = problem_name

	# Build problem context from YAML if available
	problem_context = ""
	yaml_path = Path("problems") / f"{problem_name}.yaml"
	if yaml_path.exists():
		problem_context = yaml_path.read_text()

	# TODO: compute roofline automatically from problem definition
	roofline_analysis = "Compute roofline as your first step (see orchestrator prompt instructions)."

	prompt = build_orchestrate_prompt(
		problem_name=problem_name,
		problem_dir=problem_dir,
		problem_context=problem_context,
		roofline_analysis=roofline_analysis,
		gpu_id=gpu,
		config=config,
		max_wall_time_hours=max_hours,
		max_redirects=max_redirects,
	)

	if print_only:
		click.echo(prompt)
	else:
		from kernel_forge.solve import save_prompt
		path = save_prompt(prompt, f"{problem_name}_orchestrator")
		click.echo(f"Orchestrator prompt saved to: {path}")
		click.echo(f"Prompt size: {len(prompt)} chars")
		click.echo("")
		click.echo("To run:")
		click.echo(
			f"  claude --permission-mode bypassPermissions "
			f"-p \"$(cat {path})\" --model opus "
			f"--max-turns 0 --output-format text"
		)
```

- [ ] **Step 2: Test the command**

```bash
uv run kernel-forge orchestrate trimul --gpu 3 --print-only 2>&1 | head -20
```

Expected: Prints the orchestrator prompt starting with "# Kernel Optimization Orchestrator".

- [ ] **Step 3: Commit**

```bash
git add src/kernel_forge/cli.py
git commit -m "feat: kernel-forge orchestrate CLI command"
```

---

## Task 6: Integration test -- run on TriMul

End-to-end validation: generate the orchestrator prompt, verify it contains everything needed, and do a dry run of the checkpoint polling.

**Files:**
- No new files -- this is a manual integration test

- [ ] **Step 1: Generate the orchestrator prompt**

```bash
uv run kernel-forge orchestrate trimul --gpu 3
```

Verify the saved prompt contains:
- Problem context from `problems/trimul.yaml`
- Roofline instructions
- Embedded agent_prompt.md with B200 hardware peaks
- Checkpoint polling commands with correct paths
- Stop signal commands
- Knowledge base entries (distilled guides, techniques)

- [ ] **Step 2: Verify checkpoint infrastructure on B200**

```bash
# Clear any old state
ssh b200-node "cd ~/kernel-forge-workspace/trimul && rm -f checkpoint.jsonl profile_latest.json stop.json"

# Run one benchmark cycle
ssh b200-node "cd ~/kernel-forge-workspace/trimul && CUDA_VISIBLE_DEVICES=3 python bench.py --benchmark --no-ref"

# Verify checkpoint was written
ssh b200-node "cat ~/kernel-forge-workspace/trimul/checkpoint.jsonl"

# Run profiling
ssh b200-node "cd ~/kernel-forge-workspace/trimul && CUDA_VISIBLE_DEVICES=3 python bench.py --profile"

# Verify profile was written
ssh b200-node "cat ~/kernel-forge-workspace/trimul/profile_latest.json | python3 -m json.tool | head -10"
```

- [ ] **Step 3: Test stop signal flow**

```bash
# Write stop signal
ssh b200-node "echo '{\"reason\": \"test\"}' > ~/kernel-forge-workspace/trimul/stop.json"

# Verify it exists
ssh b200-node "cat ~/kernel-forge-workspace/trimul/stop.json"

# Clean up
ssh b200-node "rm -f ~/kernel-forge-workspace/trimul/stop.json"
```

- [ ] **Step 4: Final commit with all files**

```bash
uv run pytest tests/ -v --timeout=30
uv run ruff check src/ tests/
git add -A
git commit -m "feat: orchestration loop -- monitor, evaluate, redirect for kernel optimization"
```

---

## Execution Order

Tasks 1-2 are independent (harness changes vs prompt changes).
Task 3 depends on Task 2 (orchestrator references stop signal).
Task 4 depends on Task 3 (loads orchestrator prompt template).
Task 5 depends on Task 4 (wires up the CLI).
Task 6 depends on all previous tasks.

```
Task 1 (bench.py checkpoints) ─────────────────┐
Task 2 (stop signal in agent prompt) ──┐        │
                                       ▼        ▼
                              Task 3 (orchestrator prompt)
                                       │
                                       ▼
                              Task 4 (orchestrate.py)
                                       │
                                       ▼
                              Task 5 (CLI command)
                                       │
                                       ▼
                              Task 6 (integration test)
```
