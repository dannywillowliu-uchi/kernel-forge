"""Configuration system for kernel-forge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kernel_forge.core.types import TerminationConfig


@dataclass
class HardwareConfig:
	ssh_host: str = "b200-node"
	ssh_user: str = "danny"
	gpu_id: int = 2
	cuda_visible_devices: str = "2"
	remote_workspace: str = "~/kernel-forge-workspace"
	gpu_check_interval_seconds: int = 300
	command_timeout_seconds: int = 300
	gpu_memory_threshold_mib: int = 2048

	def wrap_remote_command(self, cmd: str) -> str:
		"""Wrap a command with cd, source activate, CUDA_VISIBLE_DEVICES."""
		return (
			f"cd {self.remote_workspace} && "
			f"source env/bin/activate && "
			f"CUDA_VISIBLE_DEVICES={self.cuda_visible_devices} "
			f"{cmd}"
		)


@dataclass
class ForgeConfig:
	hardware: HardwareConfig = field(default_factory=HardwareConfig)
	termination: TerminationConfig = field(default_factory=TerminationConfig)
	knowledge_dir: Path = field(default_factory=lambda: Path("knowledge"))
	runs_dir: Path = field(default_factory=lambda: Path("runs"))
	db_path: Path = field(default_factory=lambda: Path("kernel_forge.db"))
	dry_run: bool = False
	max_concurrent_subagents: int = 3


def default_config() -> ForgeConfig:
	"""Return sensible defaults matching spec Section 11."""
	return ForgeConfig()


def load_config(path: Path) -> ForgeConfig:
	"""Load config from TOML file, merge with defaults."""
	import tomllib

	with open(path, "rb") as f:
		data = tomllib.load(f)

	config = default_config()

	if "hardware" in data:
		hw = data["hardware"]
		for key, val in hw.items():
			if hasattr(config.hardware, key):
				setattr(config.hardware, key, val)

	if "termination" in data:
		term = data["termination"]
		for key, val in term.items():
			if hasattr(config.termination, key):
				setattr(config.termination, key, val)

	for key in ("knowledge_dir", "runs_dir", "db_path"):
		if key in data:
			setattr(config, key, Path(data[key]))

	if "dry_run" in data:
		config.dry_run = data["dry_run"]
	if "max_concurrent_subagents" in data:
		config.max_concurrent_subagents = data["max_concurrent_subagents"]

	return config
