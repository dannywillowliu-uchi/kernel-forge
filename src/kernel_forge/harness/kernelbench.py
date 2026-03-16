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
		"""Best-effort extraction of input shapes from source.

		Resolves module-level integer variable assignments (e.g., N = 2048 * 2)
		via safe arithmetic evaluation, then substitutes into torch tensor
		creation calls in get_inputs(). Returns empty dict if parsing fails.
		"""
		shapes: dict[str, list[int]] = {}

		# Step 1: resolve module-level variable assignments
		variables: dict[str, int] = {}
		for var_match in re.finditer(
			r"^(\w+)\s*=\s*(.+)$", source, re.MULTILINE
		):
			name = var_match.group(1)
			expr = var_match.group(2).strip()
			val = self._safe_eval_int(expr, variables)
			if val is not None:
				variables[name] = val

		# Step 2: find get_inputs function body
		match = re.search(
			r"def\s+get_inputs\s*\(\s*\)\s*:(.+?)(?=\ndef\s|\Z)",
			source,
			re.DOTALL,
		)
		if not match:
			return shapes

		body = match.group(1)

		# Step 3: find torch tensor creation calls and resolve dims
		idx = 0
		for tensor_match in re.finditer(
			r"torch\.(?:randn?|zeros|ones|empty)\s*\(([^)]+)\)", body
		):
			args = tensor_match.group(1)
			dims: list[int] = []
			for part in args.split(","):
				part = part.strip()
				if not part or "device" in part or "dtype" in part:
					continue
				val = self._safe_eval_int(part, variables)
				if val is not None:
					dims.append(val)
			if dims:
				shapes[f"input_{idx}"] = dims
				idx += 1

		return shapes

	@staticmethod
	def _safe_eval_int(
		expr: str, variables: dict[str, int]
	) -> int | None:
		"""Safely evaluate simple integer arithmetic expressions.

		Supports: integer literals, *, +, -, //, **, variable references.
		Uses AST node visitor to reject anything that isn't arithmetic.
		"""
		import ast as _ast
		import operator

		expr = expr.strip()
		if not expr:
			return None

		# Substitute known variables
		resolved = expr
		for name, val in sorted(
			variables.items(), key=lambda x: -len(x[0])
		):
			resolved = re.sub(rf"\b{name}\b", str(val), resolved)

		ops = {
			_ast.Add: operator.add,
			_ast.Sub: operator.sub,
			_ast.Mult: operator.mul,
			_ast.FloorDiv: operator.floordiv,
			_ast.Pow: operator.pow,
			_ast.Div: operator.truediv,
		}

		def _eval_node(node: _ast.expr) -> int | float:
			if isinstance(node, _ast.Constant) and isinstance(
				node.value, (int, float)
			):
				return node.value
			if isinstance(node, _ast.BinOp):
				op_fn = ops.get(type(node.op))
				if op_fn is None:
					raise ValueError(f"Unsupported op: {node.op}")
				return op_fn(
					_eval_node(node.left), _eval_node(node.right)
				)
			if isinstance(node, _ast.UnaryOp) and isinstance(
				node.op, _ast.USub
			):
				return -_eval_node(node.operand)
			raise ValueError(f"Unsupported node: {type(node)}")

		try:
			tree = _ast.parse(resolved, mode="eval")
			result = _eval_node(tree.body)
			return int(result) if isinstance(result, (int, float)) else None
		except Exception:
			return None
