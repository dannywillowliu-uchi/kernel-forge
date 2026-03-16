"""Tests for the tool registry and protocol."""

from __future__ import annotations

from typing import Any

import pytest

from kernel_forge.tools.registry import Tool, ToolRegistry, ToolResult


class FakeTool:
	"""A concrete tool for testing."""

	def __init__(self, name: str = "fake", description: str = "A fake tool") -> None:
		self._name = name
		self._description = description

	@property
	def name(self) -> str:
		return self._name

	@property
	def description(self) -> str:
		return self._description

	async def run(self, **kwargs: Any) -> ToolResult:
		return ToolResult(
			success=True,
			data={"echo": kwargs},
			output="ran fake tool",
		)


class FailTool:
	"""A tool that always fails."""

	@property
	def name(self) -> str:
		return "fail"

	@property
	def description(self) -> str:
		return "Always fails"

	async def run(self, **kwargs: Any) -> ToolResult:
		return ToolResult(success=False, error="intentional failure")


def test_tool_result_defaults() -> None:
	result = ToolResult(success=True)
	assert result.success is True
	assert result.data == {}
	assert result.output == ""
	assert result.error == ""


def test_tool_result_with_data() -> None:
	result = ToolResult(
		success=False,
		data={"key": "value"},
		output="some output",
		error="some error",
	)
	assert result.success is False
	assert result.data == {"key": "value"}
	assert result.output == "some output"
	assert result.error == "some error"


def test_fake_tool_satisfies_protocol() -> None:
	tool = FakeTool()
	assert isinstance(tool, Tool)


def test_registry_register_and_get_available() -> None:
	registry = ToolRegistry()
	assert registry.get_available() == []

	tool = FakeTool()
	registry.register(tool)
	available = registry.get_available()
	assert len(available) == 1
	assert available[0].name == "fake"


def test_registry_register_multiple() -> None:
	registry = ToolRegistry()
	registry.register(FakeTool("alpha", "First"))
	registry.register(FakeTool("beta", "Second"))
	available = registry.get_available()
	assert len(available) == 2
	names = {t.name for t in available}
	assert names == {"alpha", "beta"}


@pytest.mark.asyncio
async def test_registry_run_known_tool() -> None:
	registry = ToolRegistry()
	registry.register(FakeTool())
	result = await registry.run("fake", value=42)
	assert result.success is True
	assert result.data == {"echo": {"value": 42}}
	assert result.output == "ran fake tool"


@pytest.mark.asyncio
async def test_registry_run_unknown_tool() -> None:
	registry = ToolRegistry()
	with pytest.raises(KeyError, match="Unknown tool"):
		await registry.run("nonexistent")


@pytest.mark.asyncio
async def test_registry_run_failing_tool() -> None:
	registry = ToolRegistry()
	registry.register(FailTool())
	result = await registry.run("fail")
	assert result.success is False
	assert result.error == "intentional failure"


def test_registry_request_new() -> None:
	registry = ToolRegistry()
	assert registry.pending_requests == []

	registry.request_new("I need a tool that measures L2 cache hit rate")
	registry.request_new("Tool for warp divergence analysis")

	assert len(registry.pending_requests) == 2
	assert "L2 cache" in registry.pending_requests[0]
	assert "warp divergence" in registry.pending_requests[1]


def test_registry_overwrites_same_name() -> None:
	registry = ToolRegistry()
	registry.register(FakeTool("dupe", "First version"))
	registry.register(FakeTool("dupe", "Second version"))
	available = registry.get_available()
	assert len(available) == 1
	assert available[0].description == "Second version"
