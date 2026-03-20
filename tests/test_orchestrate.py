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
	assert "TriMul forward pass" in prompt
	assert "trimul" in prompt
	assert "0.16ms" in prompt
	assert "checkpoint.jsonl" in prompt
	assert "stop.json" in prompt
	assert "Agent" in prompt


def test_build_orchestrate_prompt_includes_agent_prompt():
	"""The orchestrator prompt should embed the full agent prompt."""
	prompt = build_orchestrate_prompt(
		problem_name="test",
		problem_dir="test",
		problem_context="test context",
		roofline_analysis="test roofline",
		gpu_id=3,
	)
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
