"""SQLite knowledge database for strategies, attempts, and kernel profiles."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

from kernel_forge.core.types import (
	Attempt,
	BottleneckType,
	Strategy,
	StrategyCategory,
)

SCHEMA_VERSION = 1

CREATE_TABLES = """
CREATE TABLE IF NOT EXISTS schema_version (
	version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS strategies (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	name TEXT UNIQUE NOT NULL,
	category TEXT NOT NULL,
	description TEXT NOT NULL,
	applicability TEXT NOT NULL,
	expected_impact TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attempts (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	kernel_problem TEXT NOT NULL,
	strategy_name TEXT NOT NULL,
	speedup REAL NOT NULL,
	correct INTEGER NOT NULL,
	hardware TEXT NOT NULL,
	optimization_goal TEXT NOT NULL,
	profiling_tier TEXT NOT NULL DEFAULT 'cuda_events',
	kernel_source_hash TEXT,
	input_tokens INTEGER,
	output_tokens INTEGER,
	cost_usd REAL,
	timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kernel_profiles (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	kernel_problem TEXT NOT NULL,
	bottleneck_type TEXT NOT NULL,
	memory_throughput_pct REAL,
	compute_utilization_pct REAL,
	occupancy_pct REAL,
	warp_stall_reasons TEXT,
	l2_hit_rate REAL,
	shared_mem_usage REAL,
	profiling_tier TEXT NOT NULL DEFAULT 'cuda_events',
	timestamp TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS kernel_classifications (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	kernel_problem TEXT UNIQUE NOT NULL,
	kernel_type TEXT NOT NULL,
	input_shapes TEXT,
	difficulty_level INTEGER,
	benchmark_suite TEXT NOT NULL DEFAULT 'kernelbench'
);

CREATE INDEX IF NOT EXISTS idx_attempts_kernel ON attempts(kernel_problem);
CREATE INDEX IF NOT EXISTS idx_attempts_strategy ON attempts(strategy_name);
CREATE INDEX IF NOT EXISTS idx_strategies_category ON strategies(category);
"""


class KnowledgeDB:
	"""Async SQLite database for kernel optimization knowledge."""

	def __init__(self, db_path: Path | str) -> None:
		self._db_path = str(db_path)
		self._db: aiosqlite.Connection | None = None

	async def initialize(self) -> None:
		"""Create tables and set schema version."""
		self._db = await aiosqlite.connect(self._db_path)
		await self._db.executescript(CREATE_TABLES)
		# Set schema version if not already set
		cursor = await self._db.execute("SELECT COUNT(*) FROM schema_version")
		row = await cursor.fetchone()
		if row and row[0] == 0:
			await self._db.execute(
				"INSERT INTO schema_version (version) VALUES (?)",
				(SCHEMA_VERSION,),
			)
		await self._db.commit()

	async def close(self) -> None:
		"""Close the database connection."""
		if self._db:
			await self._db.close()
			self._db = None

	async def get_schema_version(self) -> int:
		"""Return the current schema version."""
		assert self._db is not None
		cursor = await self._db.execute("SELECT MAX(version) FROM schema_version")
		row = await cursor.fetchone()
		return row[0] if row and row[0] is not None else 0

	async def insert_strategy(self, strategy: Strategy) -> int:
		"""Insert a strategy and return its row ID."""
		assert self._db is not None
		cursor = await self._db.execute(
			"INSERT INTO strategies (name, category, description, applicability, expected_impact) "
			"VALUES (?, ?, ?, ?, ?)",
			(
				strategy.name,
				strategy.category.value,
				strategy.description,
				strategy.applicability,
				strategy.expected_impact,
			),
		)
		await self._db.commit()
		assert cursor.lastrowid is not None
		return cursor.lastrowid

	async def get_strategies_for_bottleneck(
		self, bottleneck_type: BottleneckType
	) -> list[Strategy]:
		"""Return strategies whose applicability mentions the bottleneck type."""
		assert self._db is not None
		cursor = await self._db.execute(
			"SELECT id, name, category, description, applicability, expected_impact "
			"FROM strategies WHERE applicability LIKE ?",
			(f"%{bottleneck_type.value}%",),
		)
		rows = await cursor.fetchall()
		return [
			Strategy(
				id=str(row[0]),
				name=row[1],
				category=StrategyCategory(row[2]),
				description=row[3],
				applicability=row[4],
				expected_impact=row[5],
			)
			for row in rows
		]

	async def insert_attempt(self, attempt: Attempt) -> int:
		"""Insert an attempt record and return its row ID."""
		assert self._db is not None
		cursor = await self._db.execute(
			"INSERT INTO attempts "
			"(kernel_problem, strategy_name, speedup, correct, hardware, "
			"optimization_goal, profiling_tier, kernel_source_hash, "
			"input_tokens, output_tokens, cost_usd) "
			"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
			(
				attempt.kernel_problem,
				attempt.strategy_name,
				attempt.speedup,
				int(attempt.correct),
				attempt.hardware,
				attempt.optimization_goal,
				attempt.profiling_tier,
				attempt.kernel_source_hash,
				attempt.input_tokens,
				attempt.output_tokens,
				attempt.cost_usd,
			),
		)
		await self._db.commit()
		assert cursor.lastrowid is not None
		return cursor.lastrowid

	async def get_attempts_for_problem(self, kernel_problem: str) -> list[Attempt]:
		"""Return all attempts for a given kernel problem."""
		assert self._db is not None
		cursor = await self._db.execute(
			"SELECT kernel_problem, strategy_name, speedup, correct, hardware, "
			"optimization_goal, kernel_source_hash, input_tokens, output_tokens, "
			"cost_usd, profiling_tier "
			"FROM attempts WHERE kernel_problem = ? ORDER BY id",
			(kernel_problem,),
		)
		rows = await cursor.fetchall()
		return [
			Attempt(
				kernel_problem=row[0],
				strategy_name=row[1],
				speedup=row[2],
				correct=bool(row[3]),
				hardware=row[4],
				optimization_goal=row[5],
				kernel_source_hash=row[6],
				input_tokens=row[7],
				output_tokens=row[8],
				cost_usd=row[9],
				profiling_tier=row[10],
			)
			for row in rows
		]

	async def get_best_strategies_for_kernel_type(
		self, kernel_type: str, limit: int = 5
	) -> list[dict[str, object]]:
		"""Return the best strategies by average speedup for a kernel type.

		Joins attempts with kernel_classifications to find strategies that work
		well on the given kernel type. Returns dicts with strategy_name and avg_speedup.
		"""
		assert self._db is not None
		cursor = await self._db.execute(
			"SELECT a.strategy_name, AVG(a.speedup) as avg_speedup, COUNT(*) as attempt_count "
			"FROM attempts a "
			"JOIN kernel_classifications kc ON a.kernel_problem = kc.kernel_problem "
			"WHERE kc.kernel_type = ? AND a.correct = 1 "
			"GROUP BY a.strategy_name "
			"ORDER BY avg_speedup DESC "
			"LIMIT ?",
			(kernel_type, limit),
		)
		rows = await cursor.fetchall()
		return [
			{
				"strategy_name": row[0],
				"avg_speedup": row[1],
				"attempt_count": row[2],
			}
			for row in rows
		]

	async def get_total_cost_for_problem(self, kernel_problem: str) -> float:
		"""Return the total LLM API cost for a given problem."""
		assert self._db is not None
		cursor = await self._db.execute(
			"SELECT COALESCE(SUM(cost_usd), 0.0) FROM attempts WHERE kernel_problem = ?",
			(kernel_problem,),
		)
		row = await cursor.fetchone()
		return float(row[0]) if row else 0.0

	async def insert_kernel_classification(
		self,
		kernel_problem: str,
		kernel_type: str,
		input_shapes: str | None = None,
		difficulty_level: int | None = None,
		benchmark_suite: str = "kernelbench",
	) -> int:
		"""Insert a kernel classification and return its row ID."""
		assert self._db is not None
		cursor = await self._db.execute(
			"INSERT OR REPLACE INTO kernel_classifications "
			"(kernel_problem, kernel_type, input_shapes, difficulty_level, benchmark_suite) "
			"VALUES (?, ?, ?, ?, ?)",
			(kernel_problem, kernel_type, input_shapes, difficulty_level, benchmark_suite),
		)
		await self._db.commit()
		assert cursor.lastrowid is not None
		return cursor.lastrowid
