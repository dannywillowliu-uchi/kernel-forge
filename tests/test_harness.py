"""Tests for KernelBench harness adapter."""

from __future__ import annotations

from pathlib import Path

from kernel_forge.harness.kernelbench import KernelBenchAdapter


SAMPLE_PROBLEM = '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return a @ b

def get_inputs():
    a = torch.randn(512, 512)
    b = torch.randn(512, 256)
    return [a, b]

def get_init_inputs():
    return []
'''


SAMPLE_PROBLEM_2 = '''\
import torch

class Model(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)

def get_inputs():
    return [torch.randn(1024)]

def get_init_inputs():
    return []
'''


def _setup_problems(tmp_path: Path) -> Path:
	"""Create a sample KernelBench directory structure."""
	problems = tmp_path / "problems"
	(problems / "level1").mkdir(parents=True)
	(problems / "level2").mkdir(parents=True)
	(problems / "level1" / "matmul_basic.py").write_text(SAMPLE_PROBLEM)
	(problems / "level1" / "relu_simple.py").write_text(SAMPLE_PROBLEM_2)
	(problems / "level2" / "fused_attention.py").write_text(SAMPLE_PROBLEM)
	return problems


class TestListProblems:
	def test_lists_all(self, tmp_path: Path) -> None:
		problems_dir = _setup_problems(tmp_path)
		adapter = KernelBenchAdapter(problems_dir)
		problems = adapter.list_problems()
		assert len(problems) == 3
		names = {p.name for p in problems}
		assert names == {"matmul_basic", "relu_simple", "fused_attention"}

	def test_filter_by_difficulty(self, tmp_path: Path) -> None:
		problems_dir = _setup_problems(tmp_path)
		adapter = KernelBenchAdapter(problems_dir)
		level1 = adapter.list_problems(difficulty=1)
		assert len(level1) == 2
		assert all(p.difficulty_level == 1 for p in level1)

		level2 = adapter.list_problems(difficulty=2)
		assert len(level2) == 1
		assert level2[0].difficulty_level == 2

	def test_nonexistent_difficulty(self, tmp_path: Path) -> None:
		problems_dir = _setup_problems(tmp_path)
		adapter = KernelBenchAdapter(problems_dir)
		assert adapter.list_problems(difficulty=99) == []

	def test_empty_dir(self, tmp_path: Path) -> None:
		adapter = KernelBenchAdapter(tmp_path / "nonexistent")
		assert adapter.list_problems() == []

	def test_benchmark_suite(self, tmp_path: Path) -> None:
		problems_dir = _setup_problems(tmp_path)
		adapter = KernelBenchAdapter(problems_dir)
		problems = adapter.list_problems()
		assert all(p.benchmark_suite == "kernelbench" for p in problems)


class TestGetProblem:
	def test_finds_by_name(self, tmp_path: Path) -> None:
		problems_dir = _setup_problems(tmp_path)
		adapter = KernelBenchAdapter(problems_dir)
		problem = adapter.get_problem("matmul_basic")
		assert problem is not None
		assert problem.name == "matmul_basic"
		assert problem.difficulty_level == 1
		assert "def forward" in problem.reference_source

	def test_not_found(self, tmp_path: Path) -> None:
		problems_dir = _setup_problems(tmp_path)
		adapter = KernelBenchAdapter(problems_dir)
		assert adapter.get_problem("nonexistent") is None

	def test_finds_in_level2(self, tmp_path: Path) -> None:
		problems_dir = _setup_problems(tmp_path)
		adapter = KernelBenchAdapter(problems_dir)
		problem = adapter.get_problem("fused_attention")
		assert problem is not None
		assert problem.difficulty_level == 2

	def test_nonexistent_dir(self, tmp_path: Path) -> None:
		adapter = KernelBenchAdapter(tmp_path / "nonexistent")
		assert adapter.get_problem("anything") is None


class TestInputShapeExtraction:
	def test_extracts_randn_shapes(self, tmp_path: Path) -> None:
		problems_dir = _setup_problems(tmp_path)
		adapter = KernelBenchAdapter(problems_dir)
		problem = adapter.get_problem("matmul_basic")
		assert problem is not None
		# Should find two torch.randn calls: (512, 512) and (512, 256)
		assert len(problem.input_shapes) == 2
		assert problem.input_shapes["input_0"] == [512, 512]
		assert problem.input_shapes["input_1"] == [512, 256]

	def test_single_dim_input(self, tmp_path: Path) -> None:
		problems_dir = _setup_problems(tmp_path)
		adapter = KernelBenchAdapter(problems_dir)
		problem = adapter.get_problem("relu_simple")
		assert problem is not None
		assert len(problem.input_shapes) == 1
		assert problem.input_shapes["input_0"] == [1024]

	def test_no_get_inputs(self, tmp_path: Path) -> None:
		"""Problem without get_inputs returns empty shapes dict."""
		problems_dir = tmp_path / "problems" / "level1"
		problems_dir.mkdir(parents=True)
		(problems_dir / "minimal.py").write_text("class Model: pass\n")
		adapter = KernelBenchAdapter(tmp_path / "problems")
		problem = adapter.get_problem("minimal")
		assert problem is not None
		assert problem.input_shapes == {}
