"""SSH-based remote executor for B200 GPU operations."""

from __future__ import annotations

import asyncio
import logging

from kernel_forge.remote.executor import CommandResult

logger = logging.getLogger(__name__)


class SSHExecutor:
	"""Execute commands on the B200 via SSH subprocess."""

	def __init__(
		self,
		ssh_host: str = "b200-node",
		ssh_user: str = "danny",
		remote_workspace: str = "~/kernel-forge-workspace",
		cuda_visible_devices: str = "2",
		cuda_home: str = "/usr/local/cuda-12.8",
	) -> None:
		self._host = ssh_host
		self._user = ssh_user
		self._workspace = remote_workspace
		self._cuda_devices = cuda_visible_devices
		self._cuda_home = cuda_home

	def _wrap_command(self, command: str) -> str:
		return (
			f"cd {self._workspace} && "
			f"export CUDA_HOME={self._cuda_home} && "
			f"export PATH={self._cuda_home}/bin:$PATH && "
			f"CUDA_VISIBLE_DEVICES={self._cuda_devices} "
			f"{command}"
		)

	async def run(self, command: str, timeout: int = 300) -> CommandResult:
		wrapped = self._wrap_command(command)
		ssh_target = f"{self._user}@{self._host}"
		ssh_cmd = ["ssh", ssh_target, wrapped]

		logger.debug("SSH run: %s", command[:200])
		try:
			proc = await asyncio.create_subprocess_exec(
				*ssh_cmd,
				stdout=asyncio.subprocess.PIPE,
				stderr=asyncio.subprocess.PIPE,
			)
			stdout_bytes, stderr_bytes = await asyncio.wait_for(
				proc.communicate(), timeout=timeout
			)
			return CommandResult(
				stdout=stdout_bytes.decode("utf-8", errors="replace"),
				stderr=stderr_bytes.decode("utf-8", errors="replace"),
				exit_code=proc.returncode or 0,
			)
		except asyncio.TimeoutError:
			try:
				proc.kill()
			except ProcessLookupError:
				pass
			return CommandResult(
				stdout="",
				stderr=f"Command timed out after {timeout}s",
				exit_code=-1,
				timed_out=True,
			)
		except Exception as e:
			return CommandResult(
				stdout="",
				stderr=f"SSH error: {e}",
				exit_code=-1,
			)

	async def upload(self, local_path: str, remote_path: str) -> None:
		dest = f"{self._user}@{self._host}:{remote_path}"
		proc = await asyncio.create_subprocess_exec(
			"rsync", "-az", local_path, dest,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		_, stderr = await proc.communicate()
		if proc.returncode != 0:
			logger.error("Upload failed: %s", stderr.decode())

	async def download(self, remote_path: str, local_path: str) -> None:
		src = f"{self._user}@{self._host}:{remote_path}"
		proc = await asyncio.create_subprocess_exec(
			"rsync", "-az", src, local_path,
			stdout=asyncio.subprocess.PIPE,
			stderr=asyncio.subprocess.PIPE,
		)
		_, stderr = await proc.communicate()
		if proc.returncode != 0:
			logger.error("Download failed: %s", stderr.decode())
