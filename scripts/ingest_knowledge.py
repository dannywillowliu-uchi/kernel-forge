#!/usr/bin/env python3
"""Ingest external knowledge sources into the Kernel Forge knowledge base.

Usage:
    uv run python scripts/ingest_knowledge.py --all
    uv run python scripts/ingest_knowledge.py --kernelbook
    uv run python scripts/ingest_knowledge.py --kernelbot
    uv run python scripts/ingest_knowledge.py --index
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernel_forge.knowledge.ingest import (
	build_triton_examples_index,
	extract_patterns,
	ingest_kernelbook,
	ingest_kernelbot,
)

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s: %(message)s",
	datefmt="%H:%M:%S",
)

OUTPUT_DIR = Path("knowledge/external")


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Ingest external knowledge sources"
	)
	parser.add_argument(
		"--kernelbook", action="store_true",
		help="Ingest GPU MODE KernelBook (18K PyTorch->Triton pairs)",
	)
	parser.add_argument(
		"--kernelbot", action="store_true",
		help="Ingest GPU MODE kernelbot competition submissions",
	)
	parser.add_argument(
		"--index", action="store_true",
		help="Build Triton examples index from KernelBook",
	)
	parser.add_argument(
		"--all", action="store_true",
		help="Run all ingestion steps",
	)
	parser.add_argument(
		"--max-entries", type=int, default=5000,
		help="Max entries per source (default 5000)",
	)
	args = parser.parse_args()

	if not any([args.kernelbook, args.kernelbot, args.index, args.all]):
		parser.print_help()
		return

	all_kernels = []

	if args.kernelbook or args.all:
		print("\n=== Ingesting KernelBook ===")
		kernels = ingest_kernelbook(
			OUTPUT_DIR, max_entries=args.max_entries
		)
		all_kernels.extend(kernels)
		print(f"  {len(kernels)} entries ingested")

	if args.kernelbot or args.all:
		print("\n=== Ingesting KernelBot ===")
		kernels = ingest_kernelbot(
			OUTPUT_DIR, max_entries=args.max_entries
		)
		all_kernels.extend(kernels)
		print(f"  {len(kernels)} entries ingested")

	if all_kernels:
		print("\n=== Extracting patterns ===")
		patterns = extract_patterns(all_kernels)
		for p in patterns:
			print(f"  {p['type']}: {p['insight']}")

		# Save patterns
		import json
		patterns_path = OUTPUT_DIR / "patterns.json"
		patterns_path.write_text(json.dumps(patterns, indent=2))
		print(f"  Saved to {patterns_path}")

	if args.index or args.all:
		kb_path = OUTPUT_DIR / "kernelbook.jsonl"
		if kb_path.exists():
			print("\n=== Building Triton examples index ===")
			index = build_triton_examples_index(
				kb_path,
				OUTPUT_DIR / "triton_examples_index.json",
			)
			for op_type, examples in index.items():
				print(f"  {op_type}: {len(examples)} examples")
		else:
			print("  KernelBook not yet ingested, run --kernelbook first")

	print("\n=== Summary ===")
	if OUTPUT_DIR.exists():
		for f in sorted(OUTPUT_DIR.iterdir()):
			size = f.stat().st_size
			unit = "KB" if size > 1024 else "B"
			val = size / 1024 if size > 1024 else size
			print(f"  {f.name}: {val:.1f} {unit}")


if __name__ == "__main__":
	main()
