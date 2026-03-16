"""Markdown learnings manager with quality gate and relevance filtering."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

# Kernel-specific terms that indicate quality content
KERNEL_TERMS = frozenset({
	"cache line", "tlb", "warp", "occupancy", "bank conflict",
	"coalescing", "shared memory", "register", "tile", "tiling",
	"vectorized", "fused", "kernel", "cuda", "triton", "gpu",
	"throughput", "latency", "memory bound", "compute bound",
	"sm", "tensor core", "fp4", "fp8", "bf16", "fp16",
	"nvcc", "ncu", "nsight", "profiling", "benchmark",
	"threadblock", "grid", "block", "thread", "sync",
	"l1", "l2", "dram", "hbm", "bandwidth",
})

# Terms indicating actionable learning
ACTION_TERMS = frozenset({
	"bug", "fix", "gotcha", "regression", "workaround",
	"avoid", "instead", "because", "discovered", "learned",
	"insight", "trick", "pattern", "anti-pattern",
})


def _score_quality(content: str, kernel_ref: str | None) -> float:
	"""Score learning content quality. Threshold >= 0.5 to accept."""
	score = 0.0
	content_lower = content.lower()

	# Length check
	if len(content) >= 50:
		score += 1.0
	if len(content) < 30:
		score -= 1.0

	# Kernel reference present
	if kernel_ref:
		score += 1.0

	# Contains actionable terms
	if any(term in content_lower for term in ACTION_TERMS):
		score += 1.0

	# Contains kernel-specific terms
	if any(term in content_lower for term in KERNEL_TERMS):
		score += 0.5

	return score


class LearningsManager:
	"""Manages markdown-based qualitative learnings with quality gating."""

	QUALITY_THRESHOLD = 0.5

	def __init__(self, knowledge_dir: Path) -> None:
		self._knowledge_dir = knowledge_dir
		self._learnings_dir = knowledge_dir / "learnings"
		self._learnings_dir.mkdir(parents=True, exist_ok=True)

	def write(self, category: str, content: str, kernel_ref: str | None = None) -> bool:
		"""Write a learning entry if it passes the quality gate.

		Returns True if the learning was accepted and written, False if rejected.
		"""
		score = _score_quality(content, kernel_ref)
		if score < self.QUALITY_THRESHOLD:
			return False

		file_path = self._learnings_dir / f"{category}.md"
		timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

		entry_parts = [f"\n## [{timestamp}]"]
		if kernel_ref:
			entry_parts.append(f" (ref: {kernel_ref})")
		entry_parts.append(f"\n\n{content}\n")
		entry = "".join(entry_parts)

		with open(file_path, "a") as f:
			f.write(entry)

		return True

	def read_relevant(self, kernel_type: str, max_tokens: int = 8000) -> list[str]:
		"""Search all .md files in the knowledge dir for kernel_type mentions.

		Returns a list of matching entries, respecting the approximate token budget.
		"""
		results: list[str] = []
		total_chars = 0
		# Approximate: 1 token ~= 4 chars
		max_chars = max_tokens * 4

		pattern = re.compile(re.escape(kernel_type), re.IGNORECASE)

		# Search all markdown files in the entire knowledge directory tree
		for md_file in sorted(self._knowledge_dir.rglob("*.md")):
			try:
				text = md_file.read_text()
			except OSError:
				continue

			# Split into sections (## headings)
			sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)

			for section in sections:
				if pattern.search(section):
					section_stripped = section.strip()
					if not section_stripped:
						continue
					section_len = len(section_stripped)
					if total_chars + section_len > max_chars:
						break
					results.append(section_stripped)
					total_chars += section_len

			if total_chars >= max_chars:
				break

		return results

	def read_all(self, max_tokens: int = 8000) -> str:
		"""Return all learnings concatenated, respecting approximate token budget."""
		parts: list[str] = []
		total_chars = 0
		max_chars = max_tokens * 4

		for md_file in sorted(self._learnings_dir.rglob("*.md")):
			try:
				text = md_file.read_text()
			except OSError:
				continue

			text_stripped = text.strip()
			if not text_stripped:
				continue

			header = f"--- {md_file.stem} ---\n"
			section = header + text_stripped
			section_len = len(section)

			if total_chars + section_len > max_chars:
				# Add what fits
				remaining = max_chars - total_chars
				if remaining > 100:  # Only add if meaningful
					parts.append(section[:remaining] + "\n[...truncated]")
				break

			parts.append(section)
			total_chars += section_len

		return "\n\n".join(parts)
