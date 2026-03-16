"""Tests for agent layer: prompt builders and output parsers."""

from __future__ import annotations

from kernel_forge.agents.base import KernelAgent
from kernel_forge.agents.claude import ClaudeCodeAgent
from kernel_forge.agents.prompts import (
	build_diagnose_prompt,
	build_generate_prompt,
	build_suggest_strategies_prompt,
	parse_diagnosis_output,
	parse_kernel_output,
	parse_strategies_output,
)
from kernel_forge.core.types import (
	Attempt,
	BottleneckType,
	Diagnosis,
	KernelProblem,
	OptimizationGoal,
	ProfileData,
	Strategy,
	StrategyCategory,
)


# -- Fixtures --


def _make_problem() -> KernelProblem:
	return KernelProblem(
		name="matmul_basic",
		reference_source="def model(a, b): return a @ b",
		input_shapes={"a": [512, 512], "b": [512, 512]},
	)


def _make_goal() -> OptimizationGoal:
	return OptimizationGoal(primary="latency")


def _make_diagnosis() -> Diagnosis:
	return Diagnosis(
		bottleneck_type=BottleneckType.MEMORY_BOUND,
		explanation="High memory throughput, low compute utilization",
		evidence={"memory_throughput_pct": 85.0},
		profiling_tier="ncu",
	)


def _make_profile() -> ProfileData:
	return ProfileData(
		runtime_us=150.0,
		profiling_tier="cuda_events",
		metrics={"memory_throughput_pct": 85.0},
	)


def _make_strategies() -> list[Strategy]:
	return [
		Strategy(
			id="1",
			name="shared_mem_tiling",
			category=StrategyCategory.MEMORY_OPT,
			description="Use shared memory to tile matrix blocks",
			applicability="memory_bound matmul",
			expected_impact="1.5-3x",
		),
		Strategy(
			id="2",
			name="vectorized_loads",
			category=StrategyCategory.MEMORY_OPT,
			description="Use vectorized memory loads",
			applicability="memory_bound elementwise",
			expected_impact="1.3-1.8x",
		),
	]


def _make_attempts() -> list[Attempt]:
	return [
		Attempt(
			kernel_problem="matmul_basic",
			strategy_name="shared_mem_tiling",
			speedup=1.8,
			correct=True,
			hardware="b200",
			optimization_goal="latency",
		),
		Attempt(
			kernel_problem="matmul_basic",
			strategy_name="vectorized_loads",
			speedup=0.9,
			correct=False,
			hardware="b200",
			optimization_goal="latency",
		),
	]


# -- Prompt builder tests --


class TestBuildGeneratePrompt:
	def test_includes_problem_info(self) -> None:
		prompt = build_generate_prompt(
			problem=_make_problem(),
			goal=_make_goal(),
			diagnosis=None,
			strategy_name="shared_mem_tiling",
			prior_attempts=[],
			knowledge_context="",
		)
		assert "matmul_basic" in prompt
		assert "def model(a, b)" in prompt
		assert "512" in prompt

	def test_includes_goal(self) -> None:
		prompt = build_generate_prompt(
			problem=_make_problem(),
			goal=_make_goal(),
			diagnosis=None,
			strategy_name="shared_mem_tiling",
			prior_attempts=[],
			knowledge_context="",
		)
		assert "latency" in prompt

	def test_includes_diagnosis_when_provided(self) -> None:
		prompt = build_generate_prompt(
			problem=_make_problem(),
			goal=_make_goal(),
			diagnosis=_make_diagnosis(),
			strategy_name="shared_mem_tiling",
			prior_attempts=[],
			knowledge_context="",
		)
		assert "memory_bound" in prompt
		assert "High memory throughput" in prompt

	def test_no_diagnosis_section_when_none(self) -> None:
		prompt = build_generate_prompt(
			problem=_make_problem(),
			goal=_make_goal(),
			diagnosis=None,
			strategy_name="shared_mem_tiling",
			prior_attempts=[],
			knowledge_context="",
		)
		assert "Current Diagnosis" not in prompt

	def test_includes_strategy(self) -> None:
		prompt = build_generate_prompt(
			problem=_make_problem(),
			goal=_make_goal(),
			diagnosis=None,
			strategy_name="shared_mem_tiling",
			prior_attempts=[],
			knowledge_context="",
		)
		assert "shared_mem_tiling" in prompt

	def test_includes_prior_attempts(self) -> None:
		prompt = build_generate_prompt(
			problem=_make_problem(),
			goal=_make_goal(),
			diagnosis=None,
			strategy_name="shared_mem_tiling",
			prior_attempts=_make_attempts(),
			knowledge_context="",
		)
		assert "Prior Attempts" in prompt
		assert "1.80x" in prompt
		assert "INCORRECT" in prompt

	def test_includes_knowledge_context(self) -> None:
		prompt = build_generate_prompt(
			problem=_make_problem(),
			goal=_make_goal(),
			diagnosis=None,
			strategy_name="shared_mem_tiling",
			prior_attempts=[],
			knowledge_context="Tiling works best with 32x32 blocks.",
		)
		assert "Tiling works best with 32x32 blocks" in prompt

	def test_includes_output_format_instructions(self) -> None:
		prompt = build_generate_prompt(
			problem=_make_problem(),
			goal=_make_goal(),
			diagnosis=None,
			strategy_name="shared_mem_tiling",
			prior_attempts=[],
			knowledge_context="",
		)
		assert "KERNEL_SOURCE_START" in prompt
		assert "KERNEL_SOURCE_END" in prompt
		assert "APPROACH_NOTES" in prompt


class TestBuildDiagnosePrompt:
	def test_includes_profile_data(self) -> None:
		prompt = build_diagnose_prompt(
			profile=_make_profile(),
			kernel_source="__global__ void kernel() {}",
			problem=_make_problem(),
		)
		assert "150.0" in prompt
		assert "cuda_events" in prompt

	def test_includes_kernel_source(self) -> None:
		prompt = build_diagnose_prompt(
			profile=_make_profile(),
			kernel_source="__global__ void kernel() {}",
			problem=_make_problem(),
		)
		assert "__global__ void kernel()" in prompt

	def test_includes_problem(self) -> None:
		prompt = build_diagnose_prompt(
			profile=_make_profile(),
			kernel_source="__global__ void kernel() {}",
			problem=_make_problem(),
		)
		assert "matmul_basic" in prompt

	def test_includes_output_format(self) -> None:
		prompt = build_diagnose_prompt(
			profile=_make_profile(),
			kernel_source="test",
			problem=_make_problem(),
		)
		assert "DIAGNOSIS_START" in prompt
		assert "DIAGNOSIS_END" in prompt

	def test_includes_raw_output_when_present(self) -> None:
		profile = ProfileData(
			runtime_us=150.0,
			profiling_tier="ncu",
			raw_output="ncu detailed output here",
		)
		prompt = build_diagnose_prompt(
			profile=profile,
			kernel_source="test",
			problem=_make_problem(),
		)
		assert "ncu detailed output here" in prompt

	def test_no_raw_output_section_when_none(self) -> None:
		prompt = build_diagnose_prompt(
			profile=_make_profile(),
			kernel_source="test",
			problem=_make_problem(),
		)
		assert "Raw Profiler Output" not in prompt


class TestBuildSuggestStrategiesPrompt:
	def test_includes_diagnosis(self) -> None:
		prompt = build_suggest_strategies_prompt(
			diagnosis=_make_diagnosis(),
			available_strategies=[],
			prior_attempts=[],
		)
		assert "memory_bound" in prompt
		assert "High memory throughput" in prompt

	def test_includes_available_strategies(self) -> None:
		prompt = build_suggest_strategies_prompt(
			diagnosis=_make_diagnosis(),
			available_strategies=_make_strategies(),
			prior_attempts=[],
		)
		assert "shared_mem_tiling" in prompt
		assert "vectorized_loads" in prompt

	def test_includes_prior_attempts(self) -> None:
		prompt = build_suggest_strategies_prompt(
			diagnosis=_make_diagnosis(),
			available_strategies=[],
			prior_attempts=_make_attempts(),
		)
		assert "1.80x" in prompt

	def test_includes_output_format(self) -> None:
		prompt = build_suggest_strategies_prompt(
			diagnosis=_make_diagnosis(),
			available_strategies=[],
			prior_attempts=[],
		)
		assert "STRATEGIES_START" in prompt
		assert "STRATEGIES_END" in prompt


# -- Output parser tests --


class TestParseKernelOutput:
	def test_parses_valid_output(self) -> None:
		raw = (
			"Some preamble text.\n\n"
			"KERNEL_SOURCE_START\n"
			"__global__ void optimized_kernel() {\n"
			"    // tiled implementation\n"
			"}\n"
			"KERNEL_SOURCE_END\n"
			"APPROACH_NOTES: Used shared memory tiling with 32x32 blocks.\n"
		)
		candidate = parse_kernel_output(raw)
		assert candidate is not None
		assert "__global__ void optimized_kernel()" in candidate.source
		assert "shared memory tiling" in candidate.approach_notes

	def test_returns_none_on_missing_markers(self) -> None:
		raw = "Here is some kernel code but no markers."
		assert parse_kernel_output(raw) is None

	def test_returns_none_on_empty_source(self) -> None:
		raw = "KERNEL_SOURCE_START\n\nKERNEL_SOURCE_END\n"
		assert parse_kernel_output(raw) is None

	def test_missing_approach_notes_gives_empty_string(self) -> None:
		raw = (
			"KERNEL_SOURCE_START\n"
			"void kernel() {}\n"
			"KERNEL_SOURCE_END\n"
		)
		candidate = parse_kernel_output(raw)
		assert candidate is not None
		assert candidate.approach_notes == ""

	def test_strategy_name_is_empty(self) -> None:
		"""Caller should fill in strategy_name after parsing."""
		raw = (
			"KERNEL_SOURCE_START\n"
			"void kernel() {}\n"
			"KERNEL_SOURCE_END\n"
		)
		candidate = parse_kernel_output(raw)
		assert candidate is not None
		assert candidate.strategy_name == ""

	def test_multiline_source(self) -> None:
		raw = (
			"KERNEL_SOURCE_START\n"
			"#include <cuda.h>\n"
			"\n"
			"__global__ void kernel(float* a, float* b, float* c) {\n"
			"    int idx = threadIdx.x + blockIdx.x * blockDim.x;\n"
			"    c[idx] = a[idx] + b[idx];\n"
			"}\n"
			"KERNEL_SOURCE_END\n"
			"APPROACH_NOTES: Simple element-wise addition.\n"
		)
		candidate = parse_kernel_output(raw)
		assert candidate is not None
		assert "threadIdx" in candidate.source
		assert "#include" in candidate.source


class TestParseDiagnosisOutput:
	def test_parses_valid_output(self) -> None:
		raw = (
			"Some analysis preamble.\n\n"
			"DIAGNOSIS_START\n"
			"bottleneck_type: memory_bound\n"
			"explanation: High memory throughput, low compute utilization\n"
			"evidence: memory_throughput_pct=85%, compute_utilization=30%\n"
			"DIAGNOSIS_END\n"
		)
		diag = parse_diagnosis_output(raw)
		assert diag is not None
		assert diag.bottleneck_type == BottleneckType.MEMORY_BOUND
		assert "High memory throughput" in diag.explanation
		assert "raw" in diag.evidence

	def test_returns_none_on_missing_markers(self) -> None:
		raw = "The bottleneck is memory_bound because..."
		assert parse_diagnosis_output(raw) is None

	def test_returns_none_on_missing_bottleneck_type(self) -> None:
		raw = (
			"DIAGNOSIS_START\n"
			"explanation: something\n"
			"evidence: something\n"
			"DIAGNOSIS_END\n"
		)
		assert parse_diagnosis_output(raw) is None

	def test_returns_none_on_invalid_bottleneck_type(self) -> None:
		raw = (
			"DIAGNOSIS_START\n"
			"bottleneck_type: nonexistent_type\n"
			"explanation: something\n"
			"DIAGNOSIS_END\n"
		)
		assert parse_diagnosis_output(raw) is None

	def test_all_bottleneck_types(self) -> None:
		for bt in BottleneckType:
			raw = (
				f"DIAGNOSIS_START\n"
				f"bottleneck_type: {bt.value}\n"
				f"explanation: test\n"
				f"DIAGNOSIS_END\n"
			)
			diag = parse_diagnosis_output(raw)
			assert diag is not None
			assert diag.bottleneck_type == bt

	def test_profiling_tier_defaults_to_unknown(self) -> None:
		raw = (
			"DIAGNOSIS_START\n"
			"bottleneck_type: compute_bound\n"
			"explanation: test\n"
			"DIAGNOSIS_END\n"
		)
		diag = parse_diagnosis_output(raw)
		assert diag is not None
		assert diag.profiling_tier == "unknown"


class TestParseStrategiesOutput:
	def test_parses_valid_output(self) -> None:
		raw = (
			"Here are my suggestions.\n\n"
			"STRATEGIES_START\n"
			"1. strategy_name: shared_mem_tiling | rationale: addresses memory bottleneck\n"
			"2. strategy_name: vectorized_loads | rationale: improves memory throughput\n"
			"3. strategy_name: loop_unrolling | rationale: reduces overhead\n"
			"STRATEGIES_END\n"
		)
		names = parse_strategies_output(raw)
		assert names == ["shared_mem_tiling", "vectorized_loads", "loop_unrolling"]

	def test_returns_empty_on_missing_markers(self) -> None:
		raw = "I suggest shared_mem_tiling and vectorized_loads."
		assert parse_strategies_output(raw) == []

	def test_returns_empty_on_no_valid_lines(self) -> None:
		raw = (
			"STRATEGIES_START\n"
			"some random text without strategy_name\n"
			"STRATEGIES_END\n"
		)
		assert parse_strategies_output(raw) == []

	def test_single_strategy(self) -> None:
		raw = (
			"STRATEGIES_START\n"
			"1. strategy_name: shared_mem_tiling | rationale: best for this case\n"
			"STRATEGIES_END\n"
		)
		names = parse_strategies_output(raw)
		assert names == ["shared_mem_tiling"]


# -- Protocol conformance --


class TestKernelAgentProtocol:
	def test_claude_agent_is_kernel_agent(self) -> None:
		agent = ClaudeCodeAgent()
		assert isinstance(agent, KernelAgent)
