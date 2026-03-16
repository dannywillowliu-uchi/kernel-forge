"""Tests for structured experience store."""

from __future__ import annotations

from pathlib import Path

from kernel_forge.knowledge.experience import ExperienceRecord, ExperienceStore


def _make_record(
	problem: str = "matmul_4096",
	kernel_class: str = "matmul",
	strategy: str = "tf32_tensor_cores",
	outcome: str = "success",
	speedup: float = 12.6,
	root_cause: str = "TF32 enables tensor cores on FP32 baseline",
) -> ExperienceRecord:
	return ExperienceRecord(
		problem_name=problem,
		kernel_class=kernel_class,
		strategy_name=strategy,
		approach_notes="Enable TF32 tensor cores",
		outcome=outcome,
		speedup=speedup,
		baseline_ms=2.16,
		optimized_ms=0.17,
		bottleneck_type="compute_bound",
		roofline_utilization_pct=83.5,
		root_cause=root_cause,
		input_shapes={"input_0": [4096, 4096]},
		precision_constraint="fp32_strict",
	)


class TestExperienceStore:
	def test_record_and_read(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		store.record(_make_record())
		records = store.get_records()
		assert len(records) == 1
		assert records[0].kernel_class == "matmul"
		assert records[0].speedup == 12.6

	def test_filter_by_class(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		store.record(_make_record(kernel_class="matmul"))
		store.record(_make_record(
			problem="relu", kernel_class="elementwise",
			strategy="torch_compile", speedup=1.05,
		))
		matmul_recs = store.get_records("matmul")
		assert len(matmul_recs) == 1
		all_recs = store.get_records()
		assert len(all_recs) == 2

	def test_derive_pattern(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		# Multiple matmul experiences
		store.record(_make_record(problem="matmul_1", speedup=12.6))
		store.record(_make_record(problem="matmul_2", speedup=13.1))
		store.record(_make_record(
			problem="matmul_1", strategy="bf16",
			outcome="correctness_failure", speedup=0.0,
			root_cause="BF16 precision loss at large matrix sizes",
		))

		pattern = store.get_pattern("matmul")
		assert pattern is not None
		assert pattern.total_problems == 2
		assert pattern.total_attempts == 3
		assert len(pattern.best_strategies) >= 1
		assert pattern.best_strategies[0]["name"] == "tf32_tensor_cores"
		assert pattern.best_strategies[0]["avg_speedup"] > 12.0
		assert len(pattern.common_failures) >= 1

	def test_build_context(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		store.record(_make_record(problem="m1", speedup=12.6))
		store.record(_make_record(problem="m2", speedup=13.1))
		store.record(_make_record(
			problem="m1", strategy="bf16",
			outcome="correctness_failure", speedup=0.0,
			root_cause="BF16 precision insufficient",
		))

		ctx = store.build_context_for_class("matmul")
		assert "matmul" in ctx
		assert "tf32_tensor_cores" in ctx
		assert "What works" in ctx
		assert "What fails" in ctx

	def test_insights_bottleneck_consistency(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		for i in range(5):
			store.record(_make_record(problem=f"m{i}"))

		pattern = store.get_pattern("matmul")
		assert pattern is not None
		# All compute_bound -> should note consistency
		assert any("compute_bound" in i for i in pattern.insights)

	def test_empty_store(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		assert store.get_records() == []
		assert store.get_pattern("matmul") is None
		assert store.build_context_for_class("matmul") == ""
