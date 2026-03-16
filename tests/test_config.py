"""Tests for the configuration system."""

from __future__ import annotations

from pathlib import Path

from kernel_forge.config import (
	ForgeConfig,
	HardwareConfig,
	default_config,
	load_config,
)
from kernel_forge.core.types import TerminationConfig


class TestHardwareConfig:
	def test_defaults(self) -> None:
		hw = HardwareConfig()
		assert hw.ssh_host == "b200-node"
		assert hw.ssh_user == "danny"
		assert hw.gpu_id == 2
		assert hw.cuda_visible_devices == "2"
		assert hw.remote_workspace == "~/kernel-forge-workspace"
		assert hw.gpu_check_interval_seconds == 300
		assert hw.command_timeout_seconds == 300
		assert hw.gpu_memory_threshold_mib == 2048

	def test_wrap_remote_command(self) -> None:
		hw = HardwareConfig()
		wrapped = hw.wrap_remote_command("python train.py")
		assert "cd ~/kernel-forge-workspace" in wrapped
		assert "source env/bin/activate" in wrapped
		assert "CUDA_VISIBLE_DEVICES=2" in wrapped
		assert "python train.py" in wrapped

	def test_wrap_remote_command_custom_workspace(self) -> None:
		hw = HardwareConfig(
			remote_workspace="/opt/workspace",
			cuda_visible_devices="3",
		)
		wrapped = hw.wrap_remote_command("nvidia-smi")
		assert "cd /opt/workspace" in wrapped
		assert "CUDA_VISIBLE_DEVICES=3" in wrapped
		assert "nvidia-smi" in wrapped


class TestForgeConfig:
	def test_defaults(self) -> None:
		config = ForgeConfig()
		assert isinstance(config.hardware, HardwareConfig)
		assert isinstance(config.termination, TerminationConfig)
		assert config.knowledge_dir == Path("knowledge")
		assert config.runs_dir == Path("runs")
		assert config.db_path == Path("kernel_forge.db")
		assert config.dry_run is False
		assert config.max_concurrent_subagents == 3


class TestDefaultConfig:
	def test_returns_forge_config(self) -> None:
		config = default_config()
		assert isinstance(config, ForgeConfig)

	def test_sensible_defaults(self) -> None:
		config = default_config()
		assert config.hardware.ssh_host == "b200-node"
		assert config.termination.max_attempts == 25
		assert config.termination.max_cost_usd == 5.0
		assert config.dry_run is False


class TestLoadConfig:
	def test_load_from_toml(self, tmp_path: Path) -> None:
		toml_file = tmp_path / "config.toml"
		toml_file.write_text(
			'[hardware]\n'
			'ssh_host = "custom-node"\n'
			'gpu_id = 3\n'
			'\n'
			'[termination]\n'
			'max_attempts = 50\n'
			'max_cost_usd = 10.0\n'
		)
		config = load_config(toml_file)
		assert config.hardware.ssh_host == "custom-node"
		assert config.hardware.gpu_id == 3
		# Unset fields retain defaults
		assert config.hardware.ssh_user == "danny"
		assert config.termination.max_attempts == 50
		assert config.termination.max_cost_usd == 10.0
		assert config.termination.plateau_threshold == 0.02

	def test_load_top_level_fields(self, tmp_path: Path) -> None:
		toml_file = tmp_path / "config.toml"
		toml_file.write_text(
			'dry_run = true\n'
			'max_concurrent_subagents = 5\n'
			'knowledge_dir = "/custom/knowledge"\n'
			'runs_dir = "/custom/runs"\n'
			'db_path = "/custom/db.sqlite"\n'
		)
		config = load_config(toml_file)
		assert config.dry_run is True
		assert config.max_concurrent_subagents == 5
		assert config.knowledge_dir == Path("/custom/knowledge")
		assert config.runs_dir == Path("/custom/runs")
		assert config.db_path == Path("/custom/db.sqlite")

	def test_load_empty_toml(self, tmp_path: Path) -> None:
		toml_file = tmp_path / "config.toml"
		toml_file.write_text("")
		config = load_config(toml_file)
		# Should return defaults
		assert config.hardware.ssh_host == "b200-node"
		assert config.termination.max_attempts == 25

	def test_load_merges_with_defaults(self, tmp_path: Path) -> None:
		toml_file = tmp_path / "config.toml"
		toml_file.write_text(
			'[hardware]\n'
			'ssh_host = "test-node"\n'
		)
		config = load_config(toml_file)
		# Overridden
		assert config.hardware.ssh_host == "test-node"
		# Defaults retained
		assert config.hardware.gpu_id == 2
		assert config.hardware.cuda_visible_devices == "2"
		assert config.termination.max_attempts == 25
