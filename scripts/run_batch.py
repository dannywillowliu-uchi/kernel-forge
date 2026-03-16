#!/usr/bin/env python3
"""Run a batch of KernelBench problems via subagent with automated experience.

Reads prior experience from the store, builds advisory context,
launches the agent, then records results back to the store.

Usage:
    uv run python scripts/run_batch.py --start 37 --end 46 --gpu 3
    uv run python scripts/run_batch.py --problems "19_ReLU,20_LeakyReLU" --gpu 3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernel_forge.harness.kernelbench import KernelBenchAdapter
from kernel_forge.knowledge.classifier import analyze_traits
from kernel_forge.knowledge.experience import ExperienceRecord, ExperienceStore


def build_batch_prompt(
	problems: list[str],
	gpu_id: int,
	experience: ExperienceStore,
	adapter: KernelBenchAdapter,
) -> str:
	"""Build agent prompt with automated experience context."""

	# Gather experience for each problem's traits
	experience_sections: list[str] = []
	seen_contexts: set[str] = set()

	for name in problems:
		problem = adapter.get_problem(name)
		if problem is None:
			continue
		traits = analyze_traits(name, problem.reference_source, problem.input_shapes)
		ctx = experience.build_advisory_context(traits, max_tokens=500)
		if ctx and ctx not in seen_contexts:
			experience_sections.append(ctx)
			seen_contexts.add(ctx)

	experience_text = "\n\n".join(experience_sections) if experience_sections else ""

	# Build problem list
	problem_list = "\n".join(
		f"{i+1}. {name}.py" for i, name in enumerate(problems)
	)

	prompt = f"""You are a GPU kernel optimization agent running KernelBench Level 1 problems on B200 GPU {gpu_id}.

## GPU Access
```
ssh b200-node "cd ~/kernel-forge-workspace && CUDA_VISIBLE_DEVICES={gpu_id} CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:\\$PATH <command>"
```

## Harness
```
python3 harness/forge_harness.py baseline <problem_path>
python3 harness/forge_harness.py test <problem_path> <kernel_path> --baseline-ms <N>
```
Problems at: harness/KernelBench/KernelBench/level1/

## Problems (do all sequentially)
{problem_list}

{experience_text}

## Optimization Strategy Guide

### For matmul/bmm/mm operations (compute-bound):
- TF32: `torch.backends.cuda.matmul.allow_tf32 = True` gives 3-15x
- Pad irregular shapes to multiples of 128
- Remove .contiguous() on transposed inputs (cuBLAS handles natively)
- Reshape higher-dim to 2D matmul when possible

### For elementwise operations (memory-bound):
- Write Triton kernels -- they achieve 1.5x over PyTorch eager on B200
- PyTorch's native kernels are NOT at HBM bandwidth ceiling on B200
- Look for multi-kernel baselines: fusing N kernels into 1 saves N-1 memory passes
- Softsign (3 kernels->1): 5.24x. Softplus (multi->1): 2.21x. Swish (2->1): 2.50x

### For norms/reductions:
- Custom Triton with online algorithms (single-pass stats)
- BatchNorm eval: precompute scale+shift, single kernel: 2.52x
- RMSNorm: Triton 2D tile single-pass: 5.98x
- GroupNorm/InstanceNorm: Triton two-pass per-group: 3-4x

### For convolutions:
- TF32 for cuDNN acceleration
- torch.compile may help with surrounding ops

### General:
- Read each problem file first to understand the operation
- Always pass --baseline-ms for accurate measurement
- If op is single native PyTorch kernel at bandwidth ceiling after Triton attempt, accept ~1.0x
- Max 3 attempts per problem, then move on

## Tool Requests
If you need a tool you don't have: TOOL_REQUEST: <what you need>

## Output
```
BATCH_RESULTS:
problem | baseline_ms | optimized_ms | speedup | correct | approach
...
TOOL_REQUESTS: <any>
```"""
	return prompt


def parse_batch_results(output: str) -> list[dict]:
	"""Parse BATCH_RESULTS table from agent output."""
	results = []
	in_table = False
	for line in output.split("\n"):
		if "BATCH_RESULTS" in line:
			in_table = True
			continue
		if in_table and "|" in line:
			parts = [p.strip() for p in line.split("|")]
			if len(parts) >= 6 and parts[0] and parts[0] != "problem":
				try:
					results.append({
						"problem": parts[0],
						"baseline_ms": float(parts[1]) if parts[1] else 0,
						"optimized_ms": float(parts[2]) if parts[2] else 0,
						"speedup": float(parts[3].replace("x", "")) if parts[3] else 0,
						"correct": "yes" in parts[4].lower() or "true" in parts[4].lower(),
						"approach": parts[5] if len(parts) > 5 else "",
					})
				except (ValueError, IndexError):
					continue
		if in_table and "```" in line and results:
			break
	return results


def record_results(
	results: list[dict],
	experience: ExperienceStore,
	adapter: KernelBenchAdapter,
) -> None:
	"""Write batch results to experience store."""
	for r in results:
		problem = adapter.get_problem(r["problem"])
		traits = analyze_traits(
			r["problem"],
			problem.reference_source if problem else "",
			problem.input_shapes if problem else {},
		)
		experience.record(ExperienceRecord(
			problem_name=r["problem"],
			dominant_ops=traits.dominant_ops,
			strategy_name=r.get("approach", "")[:50],
			approach_notes=r.get("approach", ""),
			outcome="success" if r["speedup"] > 1.0 else "no_improvement",
			speedup=r["speedup"],
			baseline_ms=r["baseline_ms"],
			optimized_ms=r["optimized_ms"],
			bottleneck_type=traits.estimated_bottleneck,
			roofline_utilization_pct=0.0,
			root_cause=r.get("approach", ""),
			has_data_reuse=traits.has_data_reuse,
			shape_category=traits.shape_category,
			estimated_bottleneck=traits.estimated_bottleneck,
			input_shapes=problem.input_shapes if problem else {},
		))
	print(f"Recorded {len(results)} results to experience store")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--start", type=int, default=None)
	parser.add_argument("--end", type=int, default=None)
	parser.add_argument("--problems", type=str, default=None)
	parser.add_argument("--gpu", type=int, default=3)
	parser.add_argument("--output-prompt", action="store_true",
		help="Just print the prompt, don't run")
	args = parser.parse_args()

	adapter = KernelBenchAdapter(Path("knowledge/kernelbench"))
	experience = ExperienceStore(Path("knowledge/experience"))
	all_problems = adapter.list_problems(difficulty=1)

	if args.problems:
		names = [n.strip() for n in args.problems.split(",")]
	elif args.start is not None and args.end is not None:
		all_names = [p.name for p in all_problems]
		names = all_names[args.start - 1:args.end]
	else:
		print("Specify --start/--end or --problems")
		sys.exit(1)

	print(f"Batch: {len(names)} problems on GPU {args.gpu}")
	for n in names:
		print(f"  - {n}")

	prompt = build_batch_prompt(names, args.gpu, experience, adapter)

	if args.output_prompt:
		print("\n" + prompt)
		sys.exit(0)

	# Save prompt for reference
	ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
	prompt_file = Path(f"runs/batch_{ts}_prompt.md")
	prompt_file.parent.mkdir(parents=True, exist_ok=True)
	prompt_file.write_text(prompt)
	print(f"Prompt saved to {prompt_file}")
	print(f"\nTo run: copy prompt and launch as subagent")
	print(f"After results, run:")
	print(f"  uv run python scripts/run_batch.py --record <results.json>")
