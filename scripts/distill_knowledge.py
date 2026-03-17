#!/usr/bin/env python3
"""Distill external kernel data into actionable operation-type summaries.

Uses Opus to analyze groups of kernels and extract:
- What makes winning solutions fast
- Common patterns and techniques
- Key insights per operation type

Output: knowledge/distilled/ directory with per-op summaries
that get injected into agent prompts.

Usage:
    uv run python scripts/distill_knowledge.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s %(levelname)s: %(message)s",
	datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

EXTERNAL_DIR = Path("knowledge/external")
OUTPUT_DIR = Path("knowledge/distilled")

OP_KEYWORDS = {
	"matmul": ["matmul", "mm", "bmm", "gemm", "linear", "addmm"],
	"softmax": ["softmax", "log_softmax"],
	"attention": ["attention", "sdpa", "scaled_dot", "flash"],
	"conv": ["conv2d", "conv1d", "conv3d", "convolution", "depthwise"],
	"norm": [
		"layernorm", "batchnorm", "groupnorm", "rmsnorm",
		"instancenorm", "layer_norm", "batch_norm",
	],
	"elementwise": [
		"relu", "gelu", "silu", "sigmoid", "tanh", "elu",
		"swish", "softplus", "mish", "leaky_relu",
	],
	"reduce": ["sum", "mean", "max", "min", "prod", "norm", "argmax"],
	"pooling": ["maxpool", "avgpool", "adaptivepool", "pool"],
	"moe": ["moe", "mixture_of_experts", "expert"],
}


def classify_kernel(name: str, code: str) -> str:
	text = (name + " " + code[:500]).lower()
	for op_type, keywords in OP_KEYWORDS.items():
		if any(kw in text for kw in keywords):
			return op_type
	return "other"


def load_and_group_kernels() -> dict[str, list[dict]]:
	groups: dict[str, list[dict]] = defaultdict(list)

	for jsonl_file in EXTERNAL_DIR.glob("*.jsonl"):
		with open(jsonl_file) as f:
			for line in f:
				entry = json.loads(line)
				op = classify_kernel(
					entry.get("problem_name", ""),
					entry.get("optimized_code", ""),
				)
				groups[op].append({
					"source": jsonl_file.stem,
					"name": entry.get("problem_name", "unknown"),
					"code": entry.get("optimized_code", "")[:3000],
					"hardware": entry.get("hardware", ""),
				})

	return dict(groups)


def build_distillation_prompt(
	op_type: str, kernels: list[dict]
) -> str:
	samples = kernels[:5]
	code_sections = []
	for i, k in enumerate(samples):
		code_sections.append(
			f"### Example {i+1}: {k['name']} "
			f"(source: {k['source']})\n"
			f"```\n{k['code'][:2000]}\n```"
		)

	examples_text = "\n\n".join(code_sections)

	return (
		f"Analyze these GPU kernel solutions for "
		f"**{op_type}** operations and extract actionable "
		f"optimization knowledge.\n\n"
		f"I have {len(kernels)} kernels for {op_type} ops. "
		f"Here are {len(samples)} representative examples:\n\n"
		f"{examples_text}\n\n"
		f"Produce a structured summary:\n\n"
		f"## {op_type.upper()} Optimization Guide\n\n"
		f"### What makes fast {op_type} kernels fast\n"
		f"(2-3 key techniques with WHY they help)\n\n"
		f"### Common code patterns\n"
		f"(Block sizes, tiling, memory access patterns)\n\n"
		f"### B200/Blackwell considerations\n"
		f"(Tensor cores, 228KB SMEM, 126MB L2)\n\n"
		f"### Typical speedup range\n"
		f"(Over PyTorch baseline)\n\n"
		f"### Common mistakes\n"
		f"(Approaches that look promising but don't work)\n\n"
		f"### Memory-bound vs compute-bound\n"
		f"(At what shapes does the bottleneck shift?)\n\n"
		f"Be specific and actionable. This guide will be "
		f"injected into an optimization agent's prompt."
	)


async def distill_op_type(
	op_type: str, kernels: list[dict]
) -> str:
	"""Use Claude to distill knowledge for one op type."""
	prompt = build_distillation_prompt(op_type, kernels)

	cmd = [
		"claude",
		"--permission-mode", "auto",
		"--max-turns", "1",
		"--output-format", "text",
		"--model", "sonnet",
		"-p", prompt,
	]

	logger.info(
		"Distilling %s (%d kernels)...", op_type, len(kernels)
	)

	proc = await asyncio.create_subprocess_exec(
		*cmd,
		stdout=asyncio.subprocess.PIPE,
		stderr=asyncio.subprocess.PIPE,
	)
	try:
		stdout, stderr = await asyncio.wait_for(
			proc.communicate(), timeout=120
		)
	except asyncio.TimeoutError:
		logger.warning("Timeout distilling %s", op_type)
		try:
			proc.kill()
		except ProcessLookupError:
			pass
		return ""

	return stdout.decode("utf-8", errors="replace")


async def main() -> None:
	groups = load_and_group_kernels()

	logger.info("Loaded kernel groups:")
	for op, kernels in sorted(
		groups.items(), key=lambda x: -len(x[1])
	):
		logger.info("  %s: %d kernels", op, len(kernels))

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	for op_type, kernels in groups.items():
		if len(kernels) < 3:
			logger.info(
				"Skipping %s (%d kernels)", op_type, len(kernels)
			)
			continue

		summary = await distill_op_type(op_type, kernels)
		if not summary:
			continue

		out_path = OUTPUT_DIR / f"{op_type}.md"
		out_path.write_text(summary)
		logger.info("  Saved %s (%d chars)", out_path, len(summary))

	# Build combined summary
	combined = [
		"# Distilled GPU Optimization Knowledge\n",
		"Generated from 4000+ community kernels "
		"(GPU MODE KernelBook + kernelbot).\n",
	]
	for md_file in sorted(OUTPUT_DIR.glob("*.md")):
		if md_file.name == "combined.md":
			continue
		content = md_file.read_text()
		combined.append(f"\n---\n\n{content}")

	(OUTPUT_DIR / "combined.md").write_text("\n".join(combined))
	logger.info("Combined summary saved")


if __name__ == "__main__":
	asyncio.run(main())
