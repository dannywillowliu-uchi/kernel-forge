"""Command execution protocol and result type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class CommandResult:
	"""Result of executing a command on a remote or local host."""

	stdout: str = ""
	stderr: str = ""
	exit_code: int = 0
	timed_out: bool = False

	@property
	def success(self) -> bool:
		return self.exit_code == 0 and not self.timed_out


@runtime_checkable
class Executor(Protocol):
	"""Protocol for executing commands on a target host."""

	async def run(self, command: str, timeout: int = 300) -> CommandResult: ...

	async def upload(self, local_path: str, remote_path: str) -> None: ...

	async def download(self, remote_path: str, local_path: str) -> None: ...
