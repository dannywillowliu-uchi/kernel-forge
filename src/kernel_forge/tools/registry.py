"""Tool registry with protocol, registration, and extensibility."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolResult:
	"""Result returned by a tool execution."""

	success: bool
	data: dict[str, Any] = field(default_factory=dict)
	output: str = ""
	error: str = ""


@runtime_checkable
class Tool(Protocol):
	"""Protocol for tools that can be registered and invoked."""

	@property
	def name(self) -> str: ...

	@property
	def description(self) -> str: ...

	async def run(self, **kwargs: Any) -> ToolResult: ...


class ToolRegistry:
	"""Registry for discovering and invoking tools by name."""

	def __init__(self) -> None:
		self._tools: dict[str, Tool] = {}
		self.pending_requests: list[str] = []

	def register(self, tool: Tool) -> None:
		"""Register a tool by its name."""
		self._tools[tool.name] = tool

	def get_available(self) -> list[Tool]:
		"""Return all registered tools."""
		return list(self._tools.values())

	async def run(self, tool_name: str, **kwargs: Any) -> ToolResult:
		"""Run a tool by name. Raises KeyError if tool not found."""
		if tool_name not in self._tools:
			raise KeyError(f"Unknown tool: {tool_name!r}")
		return await self._tools[tool_name].run(**kwargs)

	def request_new(self, spec: str) -> None:
		"""Log a request for a new tool that doesn't exist yet."""
		self.pending_requests.append(spec)
