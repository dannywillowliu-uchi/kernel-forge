"""Tests for run telemetry."""

from __future__ import annotations

from pathlib import Path

from kernel_forge.core.telemetry import RunTracker


class TestRunTracker:
	def test_basic_span(self) -> None:
		tracker = RunTracker("test_problem")
		with tracker.span("baseline") as s:
			s.set("baseline_ms", 2.16)
		tracker.finish()
		summary = tracker.summary()
		assert summary["problem"] == "test_problem"
		assert summary["total_ms"] >= 0  # may be ~0 for fast tests

	def test_nested_spans(self) -> None:
		tracker = RunTracker("test")
		with tracker.span("outer"):
			pass
		with tracker.span("inner"):
			pass
		tracker.finish()
		assert len(tracker.root.children) == 2

	def test_agent_call_tracking(self) -> None:
		tracker = RunTracker("test")
		tracker.record_agent_call(500.0, tokens_in=3000, tokens_out=1500)
		tracker.record_agent_call(300.0, tokens_in=2000, tokens_out=1000)
		tracker.finish()
		summary = tracker.summary()
		assert summary["agent_calls"] == 2
		assert summary["tokens_in"] == 5000
		assert summary["tokens_out"] == 2500
		assert summary["agent_time_ms"] == 800.0

	def test_gpu_time_tracking(self) -> None:
		tracker = RunTracker("test")
		tracker.record_gpu_time(100.0)
		tracker.record_gpu_time(50.0)
		tracker.finish()
		assert tracker.summary()["gpu_time_ms"] == 150.0

	def test_report_string(self) -> None:
		tracker = RunTracker("matmul")
		tracker.record_agent_call(1000.0)
		tracker.record_gpu_time(500.0)
		tracker.finish()
		report = tracker.report()
		assert "matmul" in report
		assert "Agent" in report
		assert "GPU" in report

	def test_save_to_json(self, tmp_path: Path) -> None:
		tracker = RunTracker("test")
		with tracker.span("step1") as s:
			s.set("key", "value")
		tracker.finish()
		out = tmp_path / "telemetry.json"
		tracker.save(out)
		assert out.exists()
		import json
		data = json.loads(out.read_text())
		assert "summary" in data
		assert "trace" in data
		assert data["trace"]["children"][0]["name"] == "step1"
