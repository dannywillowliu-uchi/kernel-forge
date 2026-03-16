"""ForgeRunner: wires all components together for a complete optimization run."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from kernel_forge.config import ForgeConfig
from kernel_forge.core.types import KernelProblem
from kernel_forge.knowledge.db import KnowledgeDB
from kernel_forge.knowledge.learnings import LearningsManager
from kernel_forge.knowledge.query import KnowledgeQuery
from kernel_forge.remote.dry_run import DryRunExecutor
from kernel_forge.remote.executor import Executor
from kernel_forge.tools.benchmark import CorrectnessTool
from kernel_forge.tools.compiler import KernelCompiler
from kernel_forge.tools.profiling import CudaEventsBench, GpuStatusTool, NcuProfile
from kernel_forge.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class ForgeRunner:
	"""Orchestrates all kernel-forge components for an optimization run."""

	def __init__(self, config: ForgeConfig) -> None:
		self._config = config
		self._db: KnowledgeDB | None = None
		self._executor: Executor | None = None
		self._registry: ToolRegistry | None = None
		self._learnings: LearningsManager | None = None
		self._knowledge_query: KnowledgeQuery | None = None

	@property
	def db(self) -> KnowledgeDB:
		assert self._db is not None, "Call initialize() first"
		return self._db

	@property
	def executor(self) -> Executor:
		assert self._executor is not None, "Call initialize() first"
		return self._executor

	@property
	def registry(self) -> ToolRegistry:
		assert self._registry is not None, "Call initialize() first"
		return self._registry

	@property
	def learnings(self) -> LearningsManager:
		assert self._learnings is not None, "Call initialize() first"
		return self._learnings

	@property
	def knowledge_query(self) -> KnowledgeQuery:
		assert self._knowledge_query is not None, "Call initialize() first"
		return self._knowledge_query

	async def initialize(self) -> None:
		"""Create and wire up all components.

		Creates the DB, executor (dry-run or remote), tool registry with
		built-in tools, learnings manager, and knowledge query layer.
		"""
		# Database
		self._config.db_path.parent.mkdir(parents=True, exist_ok=True)
		self._db = KnowledgeDB(self._config.db_path)
		await self._db.initialize()

		# Executor
		if self._config.dry_run:
			self._executor = DryRunExecutor()
		else:
			# Deferred: RemoteExecutor for real SSH connections
			self._executor = DryRunExecutor()
			logger.warning(
				"RemoteExecutor not yet implemented, falling back to DryRunExecutor"
			)

		# Tool registry
		hw = self._config.hardware
		self._registry = ToolRegistry()
		self._registry.register(GpuStatusTool(
			self._executor,
			gpu_id=hw.gpu_id,
			memory_threshold_mib=hw.gpu_memory_threshold_mib,
		))
		self._registry.register(CudaEventsBench(self._executor))
		self._registry.register(NcuProfile(self._executor))
		self._registry.register(KernelCompiler(self._executor))
		self._registry.register(CorrectnessTool(self._executor))

		# Knowledge
		self._learnings = LearningsManager(self._config.knowledge_dir)
		self._knowledge_query = KnowledgeQuery(self._db, self._learnings)

		logger.info("ForgeRunner initialized (dry_run=%s)", self._config.dry_run)

	async def shutdown(self) -> None:
		"""Close database and release resources."""
		if self._db is not None:
			await self._db.close()
			self._db = None
		logger.info("ForgeRunner shut down")

	def prepare_run(self, problem: KernelProblem) -> Path:
		"""Create a timestamped run directory with run.json metadata.

		Returns the path to the run directory.
		"""
		timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
		run_dir = self._config.runs_dir / f"{problem.name}_{timestamp}"
		run_dir.mkdir(parents=True, exist_ok=True)

		metadata = {
			"problem_name": problem.name,
			"benchmark_suite": problem.benchmark_suite,
			"difficulty_level": problem.difficulty_level,
			"timestamp": timestamp,
			"dry_run": self._config.dry_run,
			"hardware": {
				"ssh_host": self._config.hardware.ssh_host,
				"gpu_id": self._config.hardware.gpu_id,
			},
		}

		run_json = run_dir / "run.json"
		run_json.write_text(json.dumps(metadata, indent=2))

		logger.info("Prepared run directory: %s", run_dir)
		return run_dir
