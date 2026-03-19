"""Standalone kernel optimization solver.

Takes any kernel optimization problem and produces the full agent prompt
with all knowledge layers injected. Can be run from any Claude Code
session or automation.

Usage:
    # Generate prompt for a KernelBench problem
    kernel-forge solve 36_RMSNorm_ --gpu 3

    # Generate prompt for a custom problem
    kernel-forge solve --problem-file my_kernel.py --gpu 3

    # Generate prompt for a GPU MODE competition
    kernel-forge solve --problem-dir path/to/gpumode_problem --gpu 3

The output is a self-contained prompt file that can be:
1. Used as a subagent prompt (Agent tool in Claude Code)
2. Run via: claude --permission-mode bypassPermissions -p "$(cat prompt.md)"
3. Pasted into any Claude interface
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from kernel_forge.config import ForgeConfig
from kernel_forge.eval.scorecard import get_gap_context_for_problem, load_baselines
from kernel_forge.knowledge.classifier import analyze_traits
from kernel_forge.knowledge.experience import ExperienceStore
from kernel_forge.knowledge.solutions import SolutionStore

logger = logging.getLogger(__name__)

AGENT_PROMPT_PATH = Path(__file__).parent / "agents" / "agent_prompt.md"


def build_solve_prompt(
	problem_name: str,
	problem_source: str = "",
	problem_file: str = "",
	gpu_id: int = 3,
	config: ForgeConfig | None = None,
	custom_instructions: str = "",
) -> str:
	"""Build a complete, self-contained optimization prompt.

	Injects all knowledge layers automatically:
	1. Agent methodology + tools
	2. Problem definition
	3. Roofline analysis
	4. Gap context (baselines to beat)
	5. Distilled guide for this op type
	6. Triton/CUDA examples
	7. Similar winning solutions
	8. Experience from similar problems
	9. Technique registry
	"""
	if config is None:
		config = ForgeConfig()

	sections: list[str] = []

	# 1. Agent methodology
	if AGENT_PROMPT_PATH.exists():
		prompt_text = AGENT_PROMPT_PATH.read_text()
		prompt_text = prompt_text.replace("{gpu_id}", str(gpu_id))
		sections.append(prompt_text)

	# 2. Problem definition
	if problem_source:
		sections.append(
			f"## Problem: {problem_name}\n\n"
			f"```python\n{problem_source}\n```\n"
		)
	elif problem_file:
		sections.append(
			f"## Problem: {problem_name}\n\n"
			f"Problem file on B200: `{problem_file}`\n"
			f"Read it first to understand the operation.\n"
		)

	# 3. Trait analysis
	traits = analyze_traits(problem_name, problem_source)
	sections.append(
		f"## Trait Analysis\n{traits.summary()}\n"
	)

	# 4. Roofline + baselines (gap context)
	baselines_path = config.knowledge_dir / "baselines_b200.json"
	if baselines_path.exists():
		baselines = load_baselines(baselines_path)
		gap_ctx = get_gap_context_for_problem(
			problem_name, baselines
		)
		if gap_ctx:
			sections.append(gap_ctx)

	# 5. Distilled guide for this op type
	distilled_dir = config.knowledge_dir / "distilled"
	if distilled_dir.exists() and traits.dominant_ops:
		op_to_file = {
			"matmul": "matmul", "conv": "conv",
			"attention": "attention", "softmax": "softmax",
			"norm": "norm", "elementwise": "elementwise",
			"reduction": "reduce", "loss": "other",
			"pooling": "pooling",
		}
		for op in traits.dominant_ops:
			fname = op_to_file.get(op, op)
			guide_path = distilled_dir / f"{fname}.md"
			if guide_path.exists():
				content = guide_path.read_text()
				if len(content) > 100:
					sections.append(
						f"## Optimization Guide ({op})\n\n"
						f"{content[:6000]}\n"
					)
				break

	# 6. Similar winning solutions (actual code)
	solutions = SolutionStore(config.knowledge_dir / "solutions")
	similar = solutions.get_winning_kernel_for_similar(
		traits, top_k=2
	)
	if similar:
		parts = [
			"## Winning kernels from similar problems"
		]
		for sol in similar:
			code = sol.winning_kernel[:2000]
			parts.append(
				f"\n### {sol.problem} ({sol.speedup:.1f}x)\n"
				f"Approach: {sol.approach}\n"
				f"```python\n{code}\n```"
			)
		sections.append("\n".join(parts))

	# 7. Experience from similar problems
	experience = ExperienceStore(
		config.experience.store_path
	)
	exp_ctx = experience.build_advisory_context(
		traits, max_tokens=3000
	)
	if exp_ctx:
		sections.append(exp_ctx)

	# 8. Relevant techniques
	techniques_dir = config.knowledge_dir / "techniques"
	if techniques_dir.exists():
		relevant_techniques = []
		for tech_file in techniques_dir.glob("*.json"):
			tech = json.loads(tech_file.read_text())
			# Check if technique applies to this problem
			when = tech.get("when_to_apply", "").lower()
			for op in traits.dominant_ops:
				if op in when or "any" in when:
					relevant_techniques.append(tech)
					break
		if relevant_techniques:
			parts = ["## Relevant Techniques"]
			for t in relevant_techniques[:5]:
				parts.append(
					f"- **{t['name']}**: {t['description'][:200]}"
				)
			sections.append("\n".join(parts))

	# 9. Custom instructions
	if custom_instructions:
		sections.append(
			f"## Additional Instructions\n\n"
			f"{custom_instructions}\n"
		)

	# 10. Task
	sections.append(
		f"\n## Your Task\n\n"
		f"Optimize this kernel on B200 GPU {gpu_id}. "
		f"Follow the gap-driven loop. Profile with ncu. "
		f"Iterate until near-roofline or budget exhausted.\n\n"
		f"Report: BEST_KERNEL_PATH, BEST_SPEEDUP, "
		f"FINAL_UTILIZATION, GAP_REMAINING, APPROACH, "
		f"WHY_IT_WORKED, WHAT_FAILED, "
		f"NOVEL_TECHNIQUES, TOOL_REQUESTS\n"
	)

	return "\n\n".join(sections)


def save_prompt(
	prompt: str,
	problem_name: str,
	output_dir: Path | None = None,
) -> Path:
	"""Save the generated prompt to a file."""
	if output_dir is None:
		output_dir = Path("runs")
	output_dir.mkdir(parents=True, exist_ok=True)

	ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
	prompt_path = output_dir / f"{problem_name}_{ts}_prompt.md"
	prompt_path.write_text(prompt)
	return prompt_path
