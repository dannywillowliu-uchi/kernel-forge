"""Tests for structured experience store."""

from __future__ import annotations

from pathlib import Path

from kernel_forge.knowledge.classifier import KernelTraits
from kernel_forge.knowledge.experience import ExperienceRecord, ExperienceStore


def _make_record(
	problem: str = "matmul_4096",
	ops: list[str] | None = None,
	strategy: str = "tf32_tensor_cores",
	outcome: str = "success",
	speedup: float = 12.6,
	root_cause: str = "TF32 enables tensor cores on FP32 baseline",
	bottleneck: str = "compute_bound",
	reuse: bool = True,
	shape: str = "large",
) -> ExperienceRecord:
	return ExperienceRecord(
		problem_name=problem,
		dominant_ops=ops or ["matmul"],
		strategy_name=strategy,
		approach_notes="Enable TF32 tensor cores",
		outcome=outcome,
		speedup=speedup,
		baseline_ms=2.16,
		optimized_ms=0.17,
		bottleneck_type=bottleneck,
		roofline_utilization_pct=83.5,
		root_cause=root_cause,
		has_data_reuse=reuse,
		shape_category=shape,
		estimated_bottleneck=bottleneck,
		input_shapes={"input_0": [4096, 4096]},
	)


class TestExperienceStore:
	def test_record_and_read(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		store.record(_make_record())
		records = store.get_all_records()
		assert len(records) == 1
		assert records[0].dominant_ops == ["matmul"]

	def test_find_similar_by_ops(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		store.record(_make_record(ops=["matmul"]))
		store.record(_make_record(
			problem="relu", ops=["elementwise"],
			strategy="torch_compile", speedup=1.05,
			bottleneck="memory_bound", reuse=False,
		))

		# Query with matmul traits
		traits = KernelTraits(
			dominant_ops=["matmul"],
			estimated_bottleneck="compute_bound",
			has_data_reuse=True,
		)
		similar = store.find_similar(traits)
		# Matmul record should be more similar
		assert len(similar) >= 1
		assert similar[0].record.dominant_ops == ["matmul"]
		assert similar[0].similarity > 0.5

	def test_find_similar_threshold(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		store.record(_make_record(
			ops=["elementwise"],
			bottleneck="memory_bound",
			reuse=False,
		))

		# Query with totally different traits
		traits = KernelTraits(
			dominant_ops=["matmul"],
			estimated_bottleneck="compute_bound",
			has_data_reuse=True,
		)
		similar = store.find_similar(traits, min_similarity=0.8)
		assert len(similar) == 0  # too different

	def test_advisory_context(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		store.record(_make_record(problem="m1", speedup=12.6))
		store.record(_make_record(problem="m2", speedup=13.1))
		store.record(_make_record(
			problem="m1", strategy="bf16",
			outcome="correctness_failure", speedup=0.0,
			root_cause="BF16 precision insufficient",
		))

		traits = KernelTraits(
			dominant_ops=["matmul"],
			estimated_bottleneck="compute_bound",
			has_data_reuse=True,
		)
		ctx = store.build_advisory_context(traits)
		assert "advisory" in ctx.lower()
		assert "tf32_tensor_cores" in ctx

	def test_empty_store(self, tmp_path: Path) -> None:
		store = ExperienceStore(tmp_path / "exp")
		traits = KernelTraits(dominant_ops=["matmul"])
		assert store.find_similar(traits) == []
		assert store.build_advisory_context(traits) == ""
