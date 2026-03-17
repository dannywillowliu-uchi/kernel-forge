"""Declarative configuration for kernel-forge.

Inspired by NVIDIA AIQ Toolkit's YAML-first workflow definition.
All components (hardware, agent, evaluation, experience) are
configured declaratively and can be swapped without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from kernel_forge.core.types import TerminationConfig


@dataclass
class HardwareConfig:
	"""Target GPU hardware configuration."""

	ssh_host: str = "b200-node"
	ssh_user: str = "danny"
	gpu_id: int = 2
	cuda_visible_devices: str = "2"
	cuda_home: str = "/usr/local/cuda-12.8"
	remote_workspace: str = "~/kernel-forge-workspace"
	gpu_check_interval_seconds: int = 300
	command_timeout_seconds: int = 300
	gpu_memory_threshold_mib: int = 2048

	def ssh_prefix(self) -> str:
		"""SSH command prefix for this hardware target."""
		return (
			f"cd {self.remote_workspace} && "
			f"export CUDA_HOME={self.cuda_home} && "
			f"export PATH={self.cuda_home}/bin:$PATH && "
			f"CUDA_VISIBLE_DEVICES={self.cuda_visible_devices}"
		)

	def wrap_remote_command(self, cmd: str) -> str:
		return f"{self.ssh_prefix()} {cmd}"


@dataclass
class AgentConfig:
	"""Agent model and behavior configuration."""

	model: str = "opus"
	max_turns: int = 30
	timeout_seconds: int = 900
	permission_mode: str = "bypassPermissions"


@dataclass
class EvaluationConfig:
	"""Evaluation criteria for kernel correctness and performance."""

	correctness_rtol: float = 1e-3
	correctness_atol: float = 1e-3
	roofline_headroom_threshold: float = 10.0
	min_speedup_threshold: float = 1.0


@dataclass
class ExperienceConfig:
	"""Experience store configuration."""

	store_path: Path = field(
		default_factory=lambda: Path("knowledge/experience")
	)
	similarity_threshold: float = 0.3
	max_context_tokens: int = 4000


@dataclass
class ProblemsConfig:
	"""Problem set configuration."""

	suite: str = "kernelbench"
	level: int = 1
	problems_dir: Path = field(
		default_factory=lambda: Path("knowledge/kernelbench")
	)
	start: int | None = None
	end: int | None = None
	names: list[str] = field(default_factory=list)


@dataclass
class ForgeConfig:
	"""Top-level declarative configuration.

	All components are configured here. Load from YAML/TOML
	to swap hardware targets, agent models, evaluation criteria,
	etc. without code changes.
	"""

	hardware: HardwareConfig = field(default_factory=HardwareConfig)
	agent: AgentConfig = field(default_factory=AgentConfig)
	evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
	experience: ExperienceConfig = field(default_factory=ExperienceConfig)
	problems: ProblemsConfig = field(default_factory=ProblemsConfig)
	termination: TerminationConfig = field(default_factory=TerminationConfig)
	knowledge_dir: Path = field(default_factory=lambda: Path("knowledge"))
	runs_dir: Path = field(default_factory=lambda: Path("runs"))
	db_path: Path = field(default_factory=lambda: Path("kernel_forge.db"))
	dry_run: bool = False
	max_concurrent_subagents: int = 3


def default_config() -> ForgeConfig:
	return ForgeConfig()


def load_config(path: Path) -> ForgeConfig:
	"""Load config from TOML or YAML file."""
	suffix = path.suffix.lower()
	if suffix in (".yaml", ".yml"):
		return _load_yaml(path)
	return _load_toml(path)


def _load_toml(path: Path) -> ForgeConfig:
	import tomllib

	with open(path, "rb") as f:
		data = tomllib.load(f)
	return _merge_config(data)


def _load_yaml(path: Path) -> ForgeConfig:
	import yaml

	with open(path) as f:
		data = yaml.safe_load(f)
	return _merge_config(data or {})


def _merge_config(data: dict) -> ForgeConfig:
	"""Merge loaded data into default config."""
	config = default_config()

	# Map sections to config objects
	section_map = {
		"hardware": config.hardware,
		"agent": config.agent,
		"evaluation": config.evaluation,
		"experience": config.experience,
		"problems": config.problems,
		"termination": config.termination,
	}

	for section_name, section_obj in section_map.items():
		if section_name in data:
			for key, val in data[section_name].items():
				if hasattr(section_obj, key):
					# Convert path strings
					current = getattr(section_obj, key)
					if isinstance(current, Path):
						val = Path(val)
					setattr(section_obj, key, val)

	# Top-level fields
	for key in ("knowledge_dir", "runs_dir", "db_path"):
		if key in data:
			setattr(config, key, Path(data[key]))
	if "dry_run" in data:
		config.dry_run = data["dry_run"]
	if "max_concurrent_subagents" in data:
		config.max_concurrent_subagents = data["max_concurrent_subagents"]

	return config
