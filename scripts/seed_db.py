#!/usr/bin/env python3
"""Seed the knowledge database from strategy markdown files.

Parses strategy metadata blocks from knowledge/strategies/*.md and inserts
into the SQLite database. Run once to populate, idempotent (uses INSERT OR IGNORE).

Usage:
	uv run python scripts/seed_db.py [--db-path kernel_forge.db]
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernel_forge.knowledge.db import KnowledgeDB
from kernel_forge.core.types import Strategy, StrategyCategory


def parse_strategies_from_markdown(file_path: Path) -> list[Strategy]:
	"""Parse strategy metadata blocks from a markdown file.

	Expected format in markdown:
	### strategy_name
	- **Category:** memory_opt
	- **Applicability:** memory_bound matmul kernels
	- **Expected Impact:** 1.5x-3.0x
	- **Description:** Use shared memory to reduce global memory accesses...
	"""
	text = file_path.read_text()
	strategies: list[Strategy] = []

	# Split by ### headers
	sections = re.split(r"\n### ", text)
	for section in sections[1:]:  # skip preamble before first ###
		lines = section.strip().split("\n")
		name = lines[0].strip().lower().replace(" ", "_")

		category_match = re.search(r"\*\*Category:\*\*\s*(\S+)", section)
		applicability_match = re.search(r"\*\*Applicability:\*\*\s*(.+)", section)
		impact_match = re.search(r"\*\*Expected Impact:\*\*\s*(.+)", section)
		desc_match = re.search(r"\*\*Description:\*\*\s*(.+)", section, re.DOTALL)

		category_str = category_match.group(1).strip() if category_match else "memory_opt"
		try:
			category = StrategyCategory(category_str)
		except ValueError:
			category = StrategyCategory.MEMORY_OPT

		description = ""
		if desc_match:
			# Take everything after Description: until the next **field** or end
			desc_text = desc_match.group(1)
			# Stop at the next metadata field or section
			desc_text = re.split(r"\n\*\*\w+", desc_text)[0]
			description = desc_text.strip()

		strategies.append(Strategy(
			name=name,
			category=category,
			description=description,
			applicability=applicability_match.group(1).strip() if applicability_match else "",
			expected_impact=impact_match.group(1).strip() if impact_match else "",
		))

	return strategies


async def main(db_path: str = "kernel_forge.db") -> None:
	knowledge_dir = Path(__file__).parent.parent / "knowledge" / "strategies"
	if not knowledge_dir.exists():
		print(f"Knowledge directory not found: {knowledge_dir}")
		return

	db = KnowledgeDB(db_path)
	await db.initialize()

	total = 0
	for md_file in sorted(knowledge_dir.glob("*.md")):
		if md_file.name == "README.md":
			continue
		strategies = parse_strategies_from_markdown(md_file)
		for strategy in strategies:
			sid = await db.insert_strategy(strategy)
			if sid:
				print(f"  + {strategy.name} ({strategy.category.value})")
				total += 1

	print(f"\nSeeded {total} strategies into {db_path}")
	await db.close()


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description="Seed knowledge database")
	parser.add_argument("--db-path", default="kernel_forge.db")
	args = parser.parse_args()
	asyncio.run(main(args.db_path))
