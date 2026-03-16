"""Tests for the learnings manager."""

from __future__ import annotations

from pathlib import Path

from kernel_forge.knowledge.learnings import LearningsManager, _score_quality


def test_score_quality_high_quality() -> None:
	"""Long content with kernel ref and action terms scores high."""
	score = _score_quality(
		"Bug fix: discovered that shared memory bank conflicts cause 2x slowdown "
		"when tile size is not a multiple of 32. Workaround: pad tiles to 32.",
		kernel_ref="matmul_tiled_v3",
	)
	# length >= 50: +1, kernel_ref: +1, action terms (bug, fix, discovered, workaround): +1,
	# kernel terms (shared memory, bank conflict, tile): +0.5
	assert score >= 3.0


def test_score_quality_minimal_accept() -> None:
	"""Content that just barely passes the quality gate."""
	score = _score_quality(
		"This is a longer content string that has more than fifty characters in it.",
		kernel_ref=None,
	)
	# length >= 50: +1, no kernel_ref, no action terms, no kernel terms
	assert score >= 0.5


def test_score_quality_too_short_reject() -> None:
	"""Very short content without other signals is rejected."""
	score = _score_quality("short", kernel_ref=None)
	# length < 30: -1, nothing else
	assert score < 0.5


def test_score_quality_short_but_ref_accepts() -> None:
	"""Short content with a kernel ref can be accepted."""
	score = _score_quality(
		"This kernel bug causes shared memory bank conflicts to slow things down.",
		kernel_ref="matmul_v2",
	)
	# length >= 50: +1, kernel_ref: +1, action terms (bug): +1, kernel terms: +0.5
	assert score >= 0.5


def test_write_accepted(tmp_path: Path) -> None:
	"""Quality content is written to the correct category file."""
	mgr = LearningsManager(tmp_path)
	accepted = mgr.write(
		"gotchas",
		"Bug: shared memory bank conflicts cause 2x slowdown when tile size "
		"is not a multiple of 32. Fix: pad tiles to next multiple of 32.",
		kernel_ref="matmul_tiled",
	)
	assert accepted is True
	gotchas_file = tmp_path / "learnings" / "gotchas.md"
	assert gotchas_file.exists()
	content = gotchas_file.read_text()
	assert "bank conflicts" in content
	assert "matmul_tiled" in content


def test_write_rejected(tmp_path: Path) -> None:
	"""Low-quality content is rejected."""
	mgr = LearningsManager(tmp_path)
	accepted = mgr.write("gotchas", "bad", kernel_ref=None)
	assert accepted is False
	gotchas_file = tmp_path / "learnings" / "gotchas.md"
	assert not gotchas_file.exists()


def test_write_multiple_entries(tmp_path: Path) -> None:
	"""Multiple writes append to the same file."""
	mgr = LearningsManager(tmp_path)
	mgr.write(
		"insights",
		"Discovered that vectorized loads improve throughput by 1.5x on "
		"memory-bound kernels when data is aligned to 128 bytes.",
		kernel_ref="reduce_v2",
	)
	mgr.write(
		"insights",
		"Learned that occupancy above 50% doesn't improve performance for "
		"compute-bound kernels on B200 architecture.",
		kernel_ref="matmul_v4",
	)
	file_path = tmp_path / "learnings" / "insights.md"
	content = file_path.read_text()
	assert "vectorized loads" in content
	assert "occupancy" in content
	assert content.count("## [") == 2


def test_read_relevant_finds_matches(tmp_path: Path) -> None:
	"""read_relevant returns sections matching the kernel type."""
	learnings_dir = tmp_path / "learnings"
	learnings_dir.mkdir(parents=True)
	(learnings_dir / "gotchas.md").write_text(
		"## Entry about matmul\n"
		"Matmul kernels benefit from shared memory tiling.\n\n"
		"## Entry about reduction\n"
		"Reduction kernels need warp-level primitives.\n\n"
		"## Another matmul note\n"
		"Matmul with FP4 quantization has precision issues.\n"
	)
	mgr = LearningsManager(tmp_path)
	results = mgr.read_relevant("matmul")
	assert len(results) == 2
	assert all("matmul" in r.lower() for r in results)


def test_read_relevant_no_matches(tmp_path: Path) -> None:
	"""read_relevant returns empty list when no matches found."""
	learnings_dir = tmp_path / "learnings"
	learnings_dir.mkdir(parents=True)
	(learnings_dir / "gotchas.md").write_text(
		"## Entry about reduction\n"
		"Reduction kernels need warp-level primitives.\n"
	)
	mgr = LearningsManager(tmp_path)
	results = mgr.read_relevant("convolution")
	assert results == []


def test_read_relevant_respects_token_budget(tmp_path: Path) -> None:
	"""read_relevant stops when token budget is reached."""
	learnings_dir = tmp_path / "learnings"
	learnings_dir.mkdir(parents=True)
	# Write many matmul entries
	entries = []
	for i in range(100):
		entries.append(f"## Matmul entry {i}\n" + "A" * 500 + " matmul " + "B" * 500 + "\n")
	(learnings_dir / "gotchas.md").write_text("\n".join(entries))

	mgr = LearningsManager(tmp_path)
	# Very small token budget
	results = mgr.read_relevant("matmul", max_tokens=100)
	# Should have fewer than 100 entries due to budget
	assert len(results) < 100


def test_read_all(tmp_path: Path) -> None:
	"""read_all returns all learnings concatenated."""
	learnings_dir = tmp_path / "learnings"
	learnings_dir.mkdir(parents=True)
	(learnings_dir / "gotchas.md").write_text("## Gotcha 1\nSome gotcha content.\n")
	(learnings_dir / "insights.md").write_text("## Insight 1\nSome insight content.\n")

	mgr = LearningsManager(tmp_path)
	all_text = mgr.read_all()
	assert "gotchas" in all_text
	assert "insights" in all_text
	assert "Gotcha 1" in all_text
	assert "Insight 1" in all_text


def test_read_all_empty(tmp_path: Path) -> None:
	"""read_all returns empty string when no learnings exist."""
	mgr = LearningsManager(tmp_path)
	assert mgr.read_all() == ""


def test_read_all_respects_token_budget(tmp_path: Path) -> None:
	"""read_all truncates when token budget is reached."""
	learnings_dir = tmp_path / "learnings"
	learnings_dir.mkdir(parents=True)
	# Write a large file
	(learnings_dir / "big.md").write_text("## Big entry\n" + "X" * 10000 + "\n")

	mgr = LearningsManager(tmp_path)
	result = mgr.read_all(max_tokens=50)
	# 50 tokens * 4 chars = 200 chars max
	assert len(result) <= 300  # Some overhead for headers


def test_read_relevant_searches_all_subdirs(tmp_path: Path) -> None:
	"""read_relevant searches the entire knowledge directory tree."""
	(tmp_path / "strategies").mkdir(parents=True)
	(tmp_path / "strategies" / "memory_opt.md").write_text(
		"## Tiling for matmul\nUse shared memory tiling for matmul kernels.\n"
	)
	(tmp_path / "learnings").mkdir(parents=True)
	(tmp_path / "learnings" / "gotchas.md").write_text(
		"## Matmul gotcha\nMatmul with small tiles causes bank conflicts.\n"
	)

	mgr = LearningsManager(tmp_path)
	results = mgr.read_relevant("matmul")
	assert len(results) >= 2
