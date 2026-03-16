"""Tests for built-in tools using DryRunExecutor."""

from __future__ import annotations

import pytest

from kernel_forge.remote.dry_run import DryRunExecutor
from kernel_forge.tools.benchmark import CorrectnessTool
from kernel_forge.tools.compiler import KernelCompiler
from kernel_forge.tools.profiling import CudaEventsBench, GpuStatusTool, NcuProfile
from kernel_forge.tools.registry import Tool

# -- GpuStatusTool --


@pytest.mark.asyncio
async def test_gpu_status_tool_available() -> None:
	executor = DryRunExecutor()
	tool = GpuStatusTool(executor)
	result = await tool.run()
	assert result.success is True
	assert result.data["available"] is True
	assert result.data["memory_used_mib"] == 0


@pytest.mark.asyncio
async def test_gpu_status_tool_name_and_description() -> None:
	executor = DryRunExecutor()
	tool = GpuStatusTool(executor)
	assert tool.name == "gpu_status"
	assert "GPU" in tool.description or "nvidia" in tool.description.lower()


@pytest.mark.asyncio
async def test_gpu_status_tool_satisfies_protocol() -> None:
	executor = DryRunExecutor()
	tool = GpuStatusTool(executor)
	assert isinstance(tool, Tool)


# -- CudaEventsBench --


@pytest.mark.asyncio
async def test_cuda_events_bench_success() -> None:
	executor = DryRunExecutor()
	tool = CudaEventsBench(executor)
	result = await tool.run(
		kernel_path="kernels/test.cu",
		problem_name="matmul",
	)
	assert result.success is True
	assert "runtime_us" in result.data
	assert result.data["speedup"] == 1.25


@pytest.mark.asyncio
async def test_cuda_events_bench_no_kernel_path() -> None:
	executor = DryRunExecutor()
	tool = CudaEventsBench(executor)
	result = await tool.run()
	assert result.success is False
	assert "kernel_path" in result.error


@pytest.mark.asyncio
async def test_cuda_events_bench_name() -> None:
	executor = DryRunExecutor()
	tool = CudaEventsBench(executor)
	assert tool.name == "cuda_events_bench"
	assert isinstance(tool, Tool)


# -- NcuProfile --


@pytest.mark.asyncio
async def test_ncu_profile_success() -> None:
	executor = DryRunExecutor()
	tool = NcuProfile(executor)
	result = await tool.run(
		kernel_path="kernels/test.cu",
		problem_name="matmul",
	)
	assert result.success is True
	assert "raw_output" in result.data
	assert "PROF" in result.output


@pytest.mark.asyncio
async def test_ncu_profile_no_kernel_path() -> None:
	executor = DryRunExecutor()
	tool = NcuProfile(executor)
	result = await tool.run()
	assert result.success is False
	assert "kernel_path" in result.error


@pytest.mark.asyncio
async def test_ncu_profile_name() -> None:
	executor = DryRunExecutor()
	tool = NcuProfile(executor)
	assert tool.name == "ncu_profile"
	assert isinstance(tool, Tool)


@pytest.mark.asyncio
async def test_ncu_profile_custom_metrics() -> None:
	executor = DryRunExecutor()
	tool = NcuProfile(executor)
	result = await tool.run(
		kernel_path="kernels/test.cu",
		problem_name="matmul",
		metrics=["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
	)
	assert result.success is True
	# Verify the command included the custom metric
	assert any("sm__throughput" in cmd for cmd in executor._command_log)


# -- KernelCompiler --


@pytest.mark.asyncio
async def test_kernel_compiler_success() -> None:
	executor = DryRunExecutor()
	tool = KernelCompiler(executor)
	result = await tool.run(
		kernel_source="__global__ void kernel() {}",
		kernel_path="kernels/test.cu",
	)
	assert result.success is True
	assert result.data["kernel_path"] == "kernels/test.cu"
	# Verify upload was called
	assert len(executor._upload_log) == 1


@pytest.mark.asyncio
async def test_kernel_compiler_no_source() -> None:
	executor = DryRunExecutor()
	tool = KernelCompiler(executor)
	result = await tool.run()
	assert result.success is False
	assert "kernel_source" in result.error


@pytest.mark.asyncio
async def test_kernel_compiler_name() -> None:
	executor = DryRunExecutor()
	tool = KernelCompiler(executor)
	assert tool.name == "kernel_compile"
	assert isinstance(tool, Tool)


# -- CorrectnessTool --


@pytest.mark.asyncio
async def test_correctness_tool_success() -> None:
	executor = DryRunExecutor()
	tool = CorrectnessTool(executor)
	result = await tool.run(
		kernel_path="kernels/test.cu",
		problem_name="matmul",
	)
	assert result.success is True
	assert result.data["correct"] is True
	assert result.data["all_close"] is True


@pytest.mark.asyncio
async def test_correctness_tool_no_kernel_path() -> None:
	executor = DryRunExecutor()
	tool = CorrectnessTool(executor)
	result = await tool.run()
	assert result.success is False
	assert "kernel_path" in result.error


@pytest.mark.asyncio
async def test_correctness_tool_name() -> None:
	executor = DryRunExecutor()
	tool = CorrectnessTool(executor)
	assert tool.name == "correctness_check"
	assert isinstance(tool, Tool)


@pytest.mark.asyncio
async def test_correctness_tool_custom_tolerances() -> None:
	executor = DryRunExecutor()
	tool = CorrectnessTool(executor)
	result = await tool.run(
		kernel_path="kernels/test.cu",
		problem_name="matmul",
		rtol=1e-5,
		atol=1e-5,
	)
	assert result.success is True
	# Verify the command used custom tolerances
	assert any("1e-05" in cmd for cmd in executor._command_log)
