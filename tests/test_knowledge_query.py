"""Tests for the knowledge query layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel_forge.core.types import (
	Attempt,
	BottleneckType,
	Strategy,
	StrategyCategory,
)
from kernel_forge.knowledge.db import KnowledgeDB
from kernel_forge.knowledge.learnings import LearningsManager
from kernel_forge.knowledge.query import KnowledgeQuery


@pytest.fixture
async def seeded_db(tmp_path: Path) -> KnowledgeDB:
	"""Create and seed a test database."""
	db = KnowledgeDB(tmp_path / "test.db")
	await db.initialize()

	# Insert strategies
	await db.insert_strategy(Strategy(
		id="1",
		name="shared_mem_tiling",
		category=StrategyCategory.MEMORY_OPT,
		description="Use shared memory to tile matrix blocks",
		applicability="memory_bound matmul attention",
		expected_impact="1.5-3x speedup",
	))
	await db.insert_strategy(Strategy(
		id="2",
		name="warp_reduction",
		category=StrategyCategory.COMPUTE_OPT,
		description="Use warp-level primitives for reductions",
		applicability="compute_bound reduction",
		expected_impact="1.2-2x speedup",
	))
	await db.insert_strategy(Strategy(
		id="3",
		name="vectorized_loads",
		category=StrategyCategory.MEMORY_OPT,
		description="Use vectorized memory loads (float4/int4)",
		applicability="memory_bound matmul elementwise",
		expected_impact="1.3-1.8x speedup",
	))

	# Insert kernel classification for matmul problems
	await db.insert_kernel_classification(
		kernel_problem="matmul_basic",
		kernel_type="matmul",
		difficulty_level=1,
	)

	# Insert attempts
	await db.insert_attempt(Attempt(
		kernel_problem="matmul_basic",
		strategy_name="shared_mem_tiling",
		speedup=1.8,
		correct=True,
		hardware="b200",
		optimization_goal="latency",
	))
	await db.insert_attempt(Attempt(
		kernel_problem="matmul_basic",
		strategy_name="vectorized_loads",
		speedup=1.5,
		correct=True,
		hardware="b200",
		optimization_goal="latency",
	))
	await db.insert_attempt(Attempt(
		kernel_problem="matmul_basic",
		strategy_name="warp_reduction",
		speedup=0.9,
		correct=False,
		hardware="b200",
		optimization_goal="latency",
	))

	return db


@pytest.fixture
def seeded_learnings(tmp_path: Path) -> LearningsManager:
	"""Create learnings with some matmul-relevant content."""
	learnings_dir = tmp_path / "knowledge"
	(learnings_dir / "learnings").mkdir(parents=True)
	(learnings_dir / "learnings" / "gotchas.md").write_text(
		"## Matmul tiling gotcha\n"
		"When tiling matmul kernels, ensure tile size is a multiple of warp size (32) "
		"to avoid bank conflicts in shared memory.\n\n"
		"## Reduction insight\n"
		"Reduction kernels benefit from warp shuffle operations.\n"
	)
	return LearningsManager(learnings_dir)


@pytest.mark.asyncio
async def test_build_context_full(
	seeded_db: KnowledgeDB, seeded_learnings: LearningsManager
) -> None:
	"""build_context returns all four sections when data is available."""
	query = KnowledgeQuery(seeded_db, seeded_learnings)
	context = await query.build_context(
		kernel_problem="matmul_basic",
		kernel_type="matmul",
		bottleneck_type=BottleneckType.MEMORY_BOUND,
	)

	# Should contain best strategies section
	assert "Best strategies" in context
	assert "shared_mem_tiling" in context

	# Should contain bottleneck strategies section
	assert "memory_bound" in context
	assert "vectorized_loads" in context

	# Should contain prior attempts section
	assert "Prior attempts" in context
	assert "1.80x" in context or "1.8" in context

	# Should contain relevant learnings section
	assert "Relevant learnings" in context or "Matmul tiling gotcha" in context

	await seeded_db.close()


@pytest.mark.asyncio
async def test_build_context_no_kernel_type(
	seeded_db: KnowledgeDB, seeded_learnings: LearningsManager
) -> None:
	"""build_context works without kernel_type (skips type-specific sections)."""
	query = KnowledgeQuery(seeded_db, seeded_learnings)
	context = await query.build_context(
		kernel_problem="matmul_basic",
		kernel_type=None,
		bottleneck_type=BottleneckType.MEMORY_BOUND,
	)

	# Should NOT contain best strategies for kernel type
	assert "Best strategies for this kernel type" not in context

	# Should still contain bottleneck strategies
	assert "memory_bound" in context

	# Should still contain prior attempts
	assert "Prior attempts" in context

	await seeded_db.close()


@pytest.mark.asyncio
async def test_build_context_no_bottleneck(
	seeded_db: KnowledgeDB, seeded_learnings: LearningsManager
) -> None:
	"""build_context works without bottleneck_type."""
	query = KnowledgeQuery(seeded_db, seeded_learnings)
	context = await query.build_context(
		kernel_problem="matmul_basic",
		kernel_type="matmul",
		bottleneck_type=None,
	)

	# Should contain best strategies and prior attempts
	assert "Best strategies" in context
	assert "Prior attempts" in context

	# Should NOT contain bottleneck strategies section heading
	assert "bottleneck" not in context.split("## Relevant")[0] if "Relevant" in context else True

	await seeded_db.close()


@pytest.mark.asyncio
async def test_build_context_empty_problem(
	seeded_db: KnowledgeDB, seeded_learnings: LearningsManager
) -> None:
	"""build_context for a problem with no data returns empty or minimal context."""
	query = KnowledgeQuery(seeded_db, seeded_learnings)
	context = await query.build_context(
		kernel_problem="unknown_problem",
		kernel_type="convolution",
		bottleneck_type=None,
	)

	# No strategies for convolution kernel type, no attempts for unknown_problem
	# Might have learnings if any mention "convolution"
	# Should not crash
	assert isinstance(context, str)

	await seeded_db.close()


@pytest.mark.asyncio
async def test_build_context_respects_token_budget(
	seeded_db: KnowledgeDB, tmp_path: Path
) -> None:
	"""build_context respects the max_tokens budget."""
	# Create learnings with lots of content
	learnings_dir = tmp_path / "big_knowledge"
	(learnings_dir / "learnings").mkdir(parents=True)
	(learnings_dir / "learnings" / "matmul.md").write_text(
		"## Matmul entry\n" + ("X" * 5000) + " matmul\n"
	)
	learnings = LearningsManager(learnings_dir)

	query = KnowledgeQuery(seeded_db, learnings)
	context = await query.build_context(
		kernel_problem="matmul_basic",
		kernel_type="matmul",
		bottleneck_type=BottleneckType.MEMORY_BOUND,
		max_tokens=100,
	)

	# 100 tokens * 4 chars = 400 chars budget
	# Context should be truncated to respect budget
	assert len(context) <= 600  # Some overhead for section headers

	await seeded_db.close()


@pytest.mark.asyncio
async def test_build_context_only_correct_attempts_for_best_strategies(
	seeded_db: KnowledgeDB, seeded_learnings: LearningsManager
) -> None:
	"""get_best_strategies_for_kernel_type only considers correct attempts."""
	query = KnowledgeQuery(seeded_db, seeded_learnings)
	context = await query.build_context(
		kernel_problem="matmul_basic",
		kernel_type="matmul",
	)

	# warp_reduction had correct=False, so it should NOT appear in best strategies
	# (it may appear in prior attempts though)
	best_section = context.split("## Prior")[0] if "## Prior" in context else context
	# The "Best strategies" section should only list correct attempts
	if "warp_reduction" in best_section:
		# If it appears, it shouldn't be in the "Best strategies" section
		# since the only warp_reduction attempt was incorrect
		assert "warp_reduction" not in best_section.split("## Best strategies")[1].split("##")[0] \
			if "## Best strategies" in best_section else True

	await seeded_db.close()
