"""KernelBench harness adapter for problem loading and listing."""

from __future__ import annotations

import re
from pathlib import Path

from kernel_forge.core.types import KernelProblem


class KernelBenchAdapter:
	"""Adapter for ScalingIntelligence/KernelBench problem format.

	Problems are Python .py files organized in levelN/ directories where N
	indicates difficulty (1-4). Each file contains a Model class with the
	reference implementation, get_inputs(), and get_init_inputs().
	"""

	def __init__(self, problems_dir: Path | str) -> None:
		self._problems_dir = Path(problems_dir)

	def list_problems(self, difficulty: int | None = None) -> list[KernelProblem]:
		"""List all problems, optionally filtered by difficulty level.

		Scans levelN/ directories for .py files. Parses difficulty from
		directory name (e.g., level1/ -> difficulty 1).
		"""
		problems: list[KernelProblem] = []

		if not self._problems_dir.exists():
			return problems

		for level_dir in sorted(self._problems_dir.iterdir()):
			if not level_dir.is_dir():
				continue

			match = re.match(r"level(\d+)", level_dir.name)
			if not match:
				continue

			level = int(match.group(1))
			if difficulty is not None and level != difficulty:
				continue

			for py_file in sorted(level_dir.glob("*.py")):
				problem = self._load_problem(py_file, level)
				if problem is not None:
					problems.append(problem)

		return problems

	def get_problem(self, name: str) -> KernelProblem | None:
		"""Look up a single problem by name across all difficulty levels."""
		if not self._problems_dir.exists():
			return None

		for level_dir in sorted(self._problems_dir.iterdir()):
			if not level_dir.is_dir():
				continue

			match = re.match(r"level(\d+)", level_dir.name)
			if not match:
				continue

			level = int(match.group(1))

			# Try exact filename match (with or without .py)
			py_file = level_dir / f"{name}.py"
			if py_file.exists():
				return self._load_problem(py_file, level)

			# Also try matching the stem of existing files
			for candidate in level_dir.glob("*.py"):
				if candidate.stem == name:
					return self._load_problem(candidate, level)

		return None

	def _load_problem(self, py_file: Path, difficulty_level: int) -> KernelProblem | None:
		"""Load a KernelProblem from a Python file."""
		try:
			source = py_file.read_text()
		except OSError:
			return None

		name = py_file.stem
		input_shapes = self._extract_input_shapes(source)

		return KernelProblem(
			name=name,
			reference_source=source,
			input_shapes=input_shapes,
			benchmark_suite="kernelbench",
			difficulty_level=difficulty_level,
		)

	def _extract_input_shapes(self, source: str) -> dict[str, list[int]]:
		"""Best-effort extraction of input shapes from get_inputs() function.

		Looks for torch.randn(...) calls and extracts shape tuples.
		Returns empty dict if parsing fails (non-critical).
		"""
		shapes: dict[str, list[int]] = {}

		# Look for get_inputs function body
		match = re.search(
			r"def\s+get_inputs\s*\(\s*\)\s*:(.+?)(?=\ndef\s|\Z)",
			source,
			re.DOTALL,
		)
		if not match:
			return shapes

		body = match.group(1)

		# Find torch.randn(N, M, ...) patterns
		idx = 0
		for randn_match in re.finditer(r"torch\.randn\s*\(([^)]+)\)", body):
			args = randn_match.group(1)
			# Extract integer literals
			dims = re.findall(r"\b(\d+)\b", args)
			if dims:
				shapes[f"input_{idx}"] = [int(d) for d in dims]
				idx += 1

		return shapes
