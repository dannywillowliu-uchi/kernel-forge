"""Solution store: saves winning kernels and optimization trajectories.

For each problem, stores:
1. The winning kernel source code
2. The full trajectory (version -> speedup -> what changed)
3. What worked, what didn't, and why

This is the most valuable data in the system. When the agent
encounters a similar problem later, it can see the actual code
that won, not just "Triton got 5.98x."
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class KernelVersion:
	"""A single kernel version in the optimization trajectory."""

	version: int
	name: str
	source: str
	speedup: float
	correct: bool
	approach: str  # what this version tried
	delta: str  # what changed from previous version


@dataclass
class OptimizationTrajectory:
	"""Full trajectory of an optimization run."""

	problem: str
	baseline_ms: float
	best_speedup: float
	best_version: int
	total_versions: int
	total_time_seconds: float
	versions: list[KernelVersion] = field(default_factory=list)

	def summary(self) -> str:
		"""One-line summary."""
		return (
			f"{self.problem}: {self.best_speedup:.2f}x in "
			f"{self.total_versions} versions "
			f"({self.total_time_seconds:.0f}s)"
		)


@dataclass
class Solution:
	"""A complete solution for a kernel problem."""

	problem: str
	winning_kernel: str  # source code
	speedup: float
	approach: str
	trajectory: OptimizationTrajectory | None = None
	hardware: str = "b200"
	traits: dict = field(default_factory=dict)


class SolutionStore:
	"""Persists winning kernels and trajectories.

	Directory structure:
	  solutions/
	    <problem>/
	      kernel.py          # winning kernel source
	      trajectory.json    # version history with speedups
	      solution.json      # metadata (speedup, approach, traits)
	"""

	def __init__(self, path: Path | str) -> None:
		self._path = Path(path)
		self._path.mkdir(parents=True, exist_ok=True)

	def save(self, solution: Solution) -> Path:
		"""Save a complete solution."""
		sol_dir = self._path / solution.problem
		sol_dir.mkdir(parents=True, exist_ok=True)

		# Save winning kernel
		kernel_path = sol_dir / "kernel.py"
		kernel_path.write_text(solution.winning_kernel)

		# Save metadata
		meta = {
			"problem": solution.problem,
			"speedup": solution.speedup,
			"approach": solution.approach,
			"hardware": solution.hardware,
			"traits": solution.traits,
		}
		(sol_dir / "solution.json").write_text(
			json.dumps(meta, indent=2)
		)

		# Save trajectory if available
		if solution.trajectory:
			traj_data = {
				"problem": solution.trajectory.problem,
				"baseline_ms": solution.trajectory.baseline_ms,
				"best_speedup": solution.trajectory.best_speedup,
				"best_version": solution.trajectory.best_version,
				"total_versions": solution.trajectory.total_versions,
				"total_time_seconds": (
					solution.trajectory.total_time_seconds
				),
				"versions": [
					asdict(v) for v in solution.trajectory.versions
				],
			}
			(sol_dir / "trajectory.json").write_text(
				json.dumps(traj_data, indent=2)
			)

		logger.info(
			"Saved solution: %s (%.2fx)",
			solution.problem, solution.speedup,
		)
		return sol_dir

	def get(self, problem: str) -> Solution | None:
		"""Load a saved solution."""
		sol_dir = self._path / problem
		if not sol_dir.exists():
			return None

		meta_path = sol_dir / "solution.json"
		if not meta_path.exists():
			return None

		meta = json.loads(meta_path.read_text())
		kernel_path = sol_dir / "kernel.py"
		kernel_source = (
			kernel_path.read_text() if kernel_path.exists() else ""
		)

		trajectory = None
		traj_path = sol_dir / "trajectory.json"
		if traj_path.exists():
			traj_data = json.loads(traj_path.read_text())
			trajectory = OptimizationTrajectory(
				problem=traj_data["problem"],
				baseline_ms=traj_data["baseline_ms"],
				best_speedup=traj_data["best_speedup"],
				best_version=traj_data["best_version"],
				total_versions=traj_data["total_versions"],
				total_time_seconds=traj_data["total_time_seconds"],
				versions=[
					KernelVersion(**v)
					for v in traj_data.get("versions", [])
				],
			)

		return Solution(
			problem=meta["problem"],
			winning_kernel=kernel_source,
			speedup=meta["speedup"],
			approach=meta["approach"],
			hardware=meta.get("hardware", "b200"),
			traits=meta.get("traits", {}),
			trajectory=trajectory,
		)

	def list_solutions(self) -> list[dict]:
		"""List all saved solutions with metadata."""
		solutions = []
		for sol_dir in sorted(self._path.iterdir()):
			if not sol_dir.is_dir():
				continue
			meta_path = sol_dir / "solution.json"
			if meta_path.exists():
				meta = json.loads(meta_path.read_text())
				solutions.append(meta)
		return solutions

	def get_winning_kernel_for_similar(
		self, traits: object, top_k: int = 3
	) -> list[Solution]:
		"""Find solutions for similar problems by trait matching.

		Returns actual winning kernel code the agent can reference.
		"""
		from kernel_forge.knowledge.classifier import KernelTraits

		solutions = []
		for sol_dir in self._path.iterdir():
			if not sol_dir.is_dir():
				continue
			sol = self.get(sol_dir.name)
			if sol and sol.winning_kernel:
				solutions.append(sol)

		if not solutions or not hasattr(traits, "similarity"):
			return solutions[:top_k]

		# Score by trait similarity
		scored = []
		for sol in solutions:
			sol_traits = KernelTraits(
				dominant_ops=sol.traits.get("dominant_ops", []),
				estimated_bottleneck=sol.traits.get(
					"estimated_bottleneck", "unknown"
				),
				has_data_reuse=sol.traits.get(
					"has_data_reuse", False
				),
				shape_category=sol.traits.get(
					"shape_category", "unknown"
				),
			)
			sim = traits.similarity(sol_traits)
			scored.append((sim, sol))

		scored.sort(key=lambda x: -x[0])
		return [sol for _, sol in scored[:top_k]]
