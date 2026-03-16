"""Prompt builders and output parsers for kernel agents."""

from __future__ import annotations

import re
from typing import Any

from kernel_forge.core.types import (
	Attempt,
	BottleneckType,
	Diagnosis,
	KernelCandidate,
	KernelProblem,
	OptimizationGoal,
	ProfileData,
	Strategy,
)


def build_generate_prompt(
	problem: KernelProblem,
	goal: OptimizationGoal,
	diagnosis: Diagnosis | None,
	strategy_name: str,
	prior_attempts: list[Attempt],
	knowledge_context: str,
) -> str:
	"""Build a prompt for kernel generation.

	Includes problem, goal, diagnosis, strategy, prior attempts, and
	knowledge context. Output format uses KERNEL_SOURCE_START/END markers.
	"""
	sections: list[str] = []

	sections.append(
		"You are an expert GPU kernel optimization engineer. "
		"Generate an optimized CUDA or Triton kernel for the given problem.\n\n"
		"IMPORTANT: Wrap your kernel source code in markers:\n"
		"KERNEL_SOURCE_START\n<your kernel code>\nKERNEL_SOURCE_END\n"
		"APPROACH_NOTES: <1-2 sentence explanation of what this kernel does differently>"
	)

	sections.append(
		f"## Problem: {problem.name}\n\n"
		f"Reference implementation:\n```\n{problem.reference_source}\n```\n\n"
		f"Input shapes: {problem.input_shapes}"
	)

	sections.append(
		f"## Optimization Goal\n\n"
		f"Primary: {goal.primary}\n"
		f"Correctness tolerances: rtol={goal.constraints.correctness_rtol}, "
		f"atol={goal.constraints.correctness_atol}\n"
		f"Precision: {goal.constraints.precision}\n"
		f"Batch sizes: {goal.constraints.batch_sizes}"
	)

	if diagnosis:
		sections.append(
			f"## Current Diagnosis\n\n"
			f"Bottleneck: {diagnosis.bottleneck_type.value}\n"
			f"Explanation: {diagnosis.explanation}\n"
			f"Evidence: {diagnosis.evidence}"
		)

	sections.append(f"## Strategy\n\nApply: {strategy_name}")

	if prior_attempts:
		lines = ["## Prior Attempts"]
		for a in prior_attempts[-5:]:
			status = "correct" if a.correct else "INCORRECT"
			lines.append(
				f"- {a.strategy_name}: {a.speedup:.2f}x ({status})"
			)
		sections.append("\n".join(lines))

	if knowledge_context:
		sections.append(f"## Knowledge Context\n\n{knowledge_context}")

	return "\n\n".join(sections)


def build_diagnose_prompt(
	profile: ProfileData,
	kernel_source: str,
	problem: KernelProblem,
) -> str:
	"""Build a prompt for bottleneck diagnosis.

	Includes profile data, kernel source, and problem definition.
	Expects DIAGNOSIS_START/END markers in the response.
	"""
	sections: list[str] = []

	sections.append(
		"You are an expert GPU performance analyst. "
		"Diagnose the performance bottleneck of the given kernel.\n\n"
		"IMPORTANT: Wrap your diagnosis in markers:\n"
		"DIAGNOSIS_START\n"
		"bottleneck_type: memory_bound | compute_bound | launch_overhead | "
		"occupancy_limited | mixed\n"
		"explanation: <why this is the bottleneck>\n"
		"evidence: <specific metrics that indicate this>\n"
		"DIAGNOSIS_END"
	)

	sections.append(
		f"## Problem: {problem.name}\n\n"
		f"Input shapes: {problem.input_shapes}"
	)

	sections.append(
		f"## Profiling Data (tier: {profile.profiling_tier})\n\n"
		f"Runtime: {profile.runtime_us} us\n"
		f"Metrics: {profile.metrics}"
	)

	if profile.raw_output:
		sections.append(f"## Raw Profiler Output\n\n{profile.raw_output}")

	sections.append(f"## Kernel Source\n\n```\n{kernel_source}\n```")

	return "\n\n".join(sections)


def build_suggest_strategies_prompt(
	diagnosis: Diagnosis,
	available_strategies: list[Strategy],
	prior_attempts: list[Attempt],
) -> str:
	"""Build a prompt for strategy suggestion.

	Includes diagnosis, available strategies, and prior attempts.
	Expects STRATEGIES_START/END markers with numbered strategy list.
	"""
	sections: list[str] = []

	sections.append(
		"You are an expert GPU optimization strategist. "
		"Suggest the best optimization strategies for the diagnosed bottleneck.\n\n"
		"IMPORTANT: Wrap your suggestions in markers:\n"
		"STRATEGIES_START\n"
		"1. strategy_name: <name> | rationale: <why this addresses the bottleneck>\n"
		"2. strategy_name: <name> | rationale: <why>\n"
		"3. strategy_name: <name> | rationale: <why>\n"
		"STRATEGIES_END"
	)

	sections.append(
		f"## Diagnosis\n\n"
		f"Bottleneck: {diagnosis.bottleneck_type.value}\n"
		f"Explanation: {diagnosis.explanation}\n"
		f"Evidence: {diagnosis.evidence}"
	)

	if available_strategies:
		lines = ["## Available Strategies"]
		for s in available_strategies:
			lines.append(
				f"- **{s.name}** ({s.category.value}): {s.description}\n"
				f"  Applicability: {s.applicability}\n"
				f"  Expected impact: {s.expected_impact}"
			)
		sections.append("\n".join(lines))

	if prior_attempts:
		lines = ["## Prior Attempts"]
		for a in prior_attempts[-5:]:
			status = "correct" if a.correct else "INCORRECT"
			lines.append(
				f"- {a.strategy_name}: {a.speedup:.2f}x ({status})"
			)
		sections.append("\n".join(lines))

	return "\n\n".join(sections)


def parse_kernel_output(raw: str) -> KernelCandidate | None:
	"""Parse kernel source and approach notes from agent output.

	Looks for KERNEL_SOURCE_START/END markers and APPROACH_NOTES line.
	Returns None if markers are not found.
	"""
	source_match = re.search(
		r"KERNEL_SOURCE_START\s*\n(.*?)KERNEL_SOURCE_END",
		raw,
		re.DOTALL,
	)
	if not source_match:
		return None

	source = source_match.group(1).strip()
	if not source:
		return None

	notes_match = re.search(
		r"APPROACH_NOTES:\s*(.+?)(?:\n|$)",
		raw,
	)
	approach_notes = notes_match.group(1).strip() if notes_match else ""

	return KernelCandidate(
		source=source,
		approach_notes=approach_notes,
		strategy_name="",  # Caller fills in
	)


def parse_diagnosis_output(raw: str) -> Diagnosis | None:
	"""Parse diagnosis from agent output.

	Looks for DIAGNOSIS_START/END markers with bottleneck_type, explanation,
	and evidence fields. Returns None if markers or required fields missing.
	"""
	diag_match = re.search(
		r"DIAGNOSIS_START\s*\n(.*?)DIAGNOSIS_END",
		raw,
		re.DOTALL,
	)
	if not diag_match:
		return None

	block = diag_match.group(1)

	bt_match = re.search(r"bottleneck_type:\s*(\S+)", block)
	if not bt_match:
		return None

	bt_str = bt_match.group(1).strip()
	try:
		bottleneck_type = BottleneckType(bt_str)
	except ValueError:
		return None

	expl_match = re.search(r"explanation:\s*(.+?)(?:\n|$)", block)
	explanation = expl_match.group(1).strip() if expl_match else ""

	evidence_match = re.search(r"evidence:\s*(.+?)(?:\n|$)", block)
	evidence_str = evidence_match.group(1).strip() if evidence_match else ""
	evidence: dict[str, Any] = {"raw": evidence_str} if evidence_str else {}

	return Diagnosis(
		bottleneck_type=bottleneck_type,
		explanation=explanation,
		evidence=evidence,
		profiling_tier="unknown",  # Caller should override
	)


def parse_strategies_output(raw: str) -> list[str]:
	"""Parse strategy names from agent output.

	Looks for STRATEGIES_START/END markers with numbered strategy lines.
	Returns list of strategy names, or empty list if parsing fails.
	"""
	strat_match = re.search(
		r"STRATEGIES_START\s*\n(.*?)STRATEGIES_END",
		raw,
		re.DOTALL,
	)
	if not strat_match:
		return []

	block = strat_match.group(1)
	names: list[str] = []

	for line in block.strip().split("\n"):
		name_match = re.search(r"strategy_name:\s*([^|]+)", line)
		if name_match:
			name = name_match.group(1).strip()
			if name:
				names.append(name)

	return names
