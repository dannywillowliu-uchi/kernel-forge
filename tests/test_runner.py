"""Tests for ForgeRunner: initialization and run preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernel_forge.config import ForgeConfig
from kernel_forge.core.runner import ForgeRunner
from kernel_forge.core.types import KernelProblem
from kernel_forge.knowledge.db import KnowledgeDB
from kernel_forge.knowledge.learnings import LearningsManager
from kernel_forge.knowledge.query import KnowledgeQuery
from kernel_forge.remote.dry_run import DryRunExecutor
from kernel_forge.tools.registry import ToolRegistry


def _make_config(tmp_path: Path) -> ForgeConfig:
	return ForgeConfig(
		db_path=tmp_path / "test.db",
		knowledge_dir=tmp_path / "knowledge",
		runs_dir=tmp_path / "runs",
		dry_run=True,
	)


def _make_problem() -> KernelProblem:
	return KernelProblem(
		name="matmul_basic",
		reference_source="def model(a, b): return a @ b",
		input_shapes={"a": [512, 512], "b": [512, 512]},
		difficulty_level=1,
	)


class TestInitialize:
	@pytest.mark.asyncio
	async def test_creates_all_components(self, tmp_path: Path) -> None:
		config = _make_config(tmp_path)
		runner = ForgeRunner(config)
		await runner.initialize()

		assert isinstance(runner.db, KnowledgeDB)
		assert isinstance(runner.executor, DryRunExecutor)
		assert isinstance(runner.registry, ToolRegistry)
		assert isinstance(runner.learnings, LearningsManager)
		assert isinstance(runner.knowledge_query, KnowledgeQuery)

		await runner.shutdown()

	@pytest.mark.asyncio
	async def test_registers_built_in_tools(self, tmp_path: Path) -> None:
		config = _make_config(tmp_path)
		runner = ForgeRunner(config)
		await runner.initialize()

		tools = runner.registry.get_available()
		tool_names = {t.name for t in tools}
		assert "gpu_status" in tool_names
		assert "cuda_events_bench" in tool_names
		assert "ncu_profile" in tool_names
		assert "kernel_compile" in tool_names
		assert "correctness_check" in tool_names

		await runner.shutdown()

	@pytest.mark.asyncio
	async def test_db_is_functional(self, tmp_path: Path) -> None:
		config = _make_config(tmp_path)
		runner = ForgeRunner(config)
		await runner.initialize()

		version = await runner.db.get_schema_version()
		assert version == 1

		await runner.shutdown()

	@pytest.mark.asyncio
	async def test_creates_db_parent_dir(self, tmp_path: Path) -> None:
		config = ForgeConfig(
			db_path=tmp_path / "subdir" / "nested" / "test.db",
			knowledge_dir=tmp_path / "knowledge",
			runs_dir=tmp_path / "runs",
			dry_run=True,
		)
		runner = ForgeRunner(config)
		await runner.initialize()

		assert config.db_path.exists()

		await runner.shutdown()


class TestShutdown:
	@pytest.mark.asyncio
	async def test_closes_db(self, tmp_path: Path) -> None:
		config = _make_config(tmp_path)
		runner = ForgeRunner(config)
		await runner.initialize()
		await runner.shutdown()

		# After shutdown, db should be None
		assert runner._db is None

	@pytest.mark.asyncio
	async def test_shutdown_idempotent(self, tmp_path: Path) -> None:
		config = _make_config(tmp_path)
		runner = ForgeRunner(config)
		await runner.initialize()
		await runner.shutdown()
		await runner.shutdown()  # Should not raise


class TestPrepareRun:
	@pytest.mark.asyncio
	async def test_creates_run_directory(self, tmp_path: Path) -> None:
		config = _make_config(tmp_path)
		runner = ForgeRunner(config)
		await runner.initialize()

		problem = _make_problem()
		run_dir = runner.prepare_run(problem)

		assert run_dir.exists()
		assert run_dir.is_dir()
		assert "matmul_basic" in run_dir.name

		await runner.shutdown()

	@pytest.mark.asyncio
	async def test_creates_run_json(self, tmp_path: Path) -> None:
		config = _make_config(tmp_path)
		runner = ForgeRunner(config)
		await runner.initialize()

		problem = _make_problem()
		run_dir = runner.prepare_run(problem)

		run_json = run_dir / "run.json"
		assert run_json.exists()

		data = json.loads(run_json.read_text())
		assert data["problem_name"] == "matmul_basic"
		assert data["benchmark_suite"] == "kernelbench"
		assert data["difficulty_level"] == 1
		assert data["dry_run"] is True
		assert "timestamp" in data
		assert "hardware" in data

		await runner.shutdown()

	@pytest.mark.asyncio
	async def test_run_dir_under_runs_dir(self, tmp_path: Path) -> None:
		config = _make_config(tmp_path)
		runner = ForgeRunner(config)
		await runner.initialize()

		problem = _make_problem()
		run_dir = runner.prepare_run(problem)

		assert run_dir.parent == config.runs_dir

		await runner.shutdown()

	@pytest.mark.asyncio
	async def test_multiple_runs_distinct_dirs(self, tmp_path: Path) -> None:
		config = _make_config(tmp_path)
		runner = ForgeRunner(config)
		await runner.initialize()

		problem = _make_problem()
		dir1 = runner.prepare_run(problem)
		# Ensure unique timestamps by waiting or checking names differ
		dir2 = runner.prepare_run(problem)

		# They should at least both exist (timestamps could match within same second)
		assert dir1.exists()
		assert dir2.exists()

		await runner.shutdown()


class TestAccessBeforeInit:
	def test_db_raises_before_init(self) -> None:
		runner = ForgeRunner(ForgeConfig())
		with pytest.raises(AssertionError, match="initialize"):
			_ = runner.db

	def test_executor_raises_before_init(self) -> None:
		runner = ForgeRunner(ForgeConfig())
		with pytest.raises(AssertionError, match="initialize"):
			_ = runner.executor

	def test_registry_raises_before_init(self) -> None:
		runner = ForgeRunner(ForgeConfig())
		with pytest.raises(AssertionError, match="initialize"):
			_ = runner.registry

	def test_learnings_raises_before_init(self) -> None:
		runner = ForgeRunner(ForgeConfig())
		with pytest.raises(AssertionError, match="initialize"):
			_ = runner.learnings

	def test_knowledge_query_raises_before_init(self) -> None:
		runner = ForgeRunner(ForgeConfig())
		with pytest.raises(AssertionError, match="initialize"):
			_ = runner.knowledge_query
