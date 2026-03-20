"""Dynamic problem configuration for any optimization target.

Instead of hardcoding "GPU kernel on B200", the system adapts to
any optimization problem: VLIW scheduling, GPU kernels, training
recipes, compiler optimization, etc.

Usage:
    config = ProblemConfig.from_yaml("problems/vliw_challenge.yaml")
    prompt = build_agent_prompt(config)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MeasureConfig:
	"""How to measure performance."""

	command: str  # e.g., "python3 tests/submission_tests.py"
	metric: str  # e.g., "cycles", "latency_ms", "val_bpb"
	direction: str = "minimize"  # "minimize" or "maximize"
	parse_regex: str = ""  # regex to extract metric from output
	unit: str = ""  # "cycles", "ms", "us", "bpb"


@dataclass
class ProfileConfig:
	"""How to diagnose bottlenecks."""

	commands: list[dict] = field(default_factory=list)
	# Each: {"name": "trace", "command": "python3 watch_trace.py", "description": "..."}


@dataclass
class TargetConfig:
	"""What does good look like."""

	leaderboard_best: float = 0.0
	our_best: float = 0.0
	theoretical_limit: float = 0.0
	hiring_threshold: float = 0.0  # for Anthropic-style challenges
	baselines: dict[str, float] = field(default_factory=dict)
	# e.g., {"starter": 147734, "claude_opus": 1363, "leaderboard_1": 954}


@dataclass
class EditConfig:
	"""What files the agent edits."""

	target_files: list[str] = field(default_factory=list)
	# e.g., ["perf_takehome.py"]
	read_only_files: list[str] = field(default_factory=list)
	# e.g., ["problem.py", "tests/submission_tests.py"]
	working_directory: str = ""


@dataclass
class ExecutionConfig:
	"""Where and how to run."""

	mode: str = "local"  # "local", "ssh", "modal"
	ssh_host: str = ""
	ssh_user: str = ""
	gpu_id: int | None = None
	env_vars: dict[str, str] = field(default_factory=dict)


@dataclass
class KnowledgeConfig:
	"""What context to inject."""

	strategy_hints: list[str] = field(default_factory=list)
	# e.g., ["VLIW packing", "SIMD vectorization", "loop unrolling"]
	reference_docs: list[str] = field(default_factory=list)
	# e.g., ["Read problem.py for ISA documentation"]
	anti_patterns: list[str] = field(default_factory=list)
	# e.g., ["Do not modify tests/", "CUDA graphs are disallowed"]
	web_search_hints: list[str] = field(default_factory=list)
	# e.g., ["VLIW instruction scheduling algorithms"]


@dataclass
class ProblemConfig:
	"""Complete configuration for any optimization problem."""

	name: str
	description: str
	type: str  # "kernel", "vliw", "training", "compiler", etc.

	measure: MeasureConfig = field(default_factory=MeasureConfig)
	profile: ProfileConfig = field(default_factory=ProfileConfig)
	targets: TargetConfig = field(default_factory=TargetConfig)
	edit: EditConfig = field(default_factory=EditConfig)
	execution: ExecutionConfig = field(default_factory=ExecutionConfig)
	knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)

	@classmethod
	def from_yaml(cls, path: str | Path) -> ProblemConfig:
		path = Path(path)
		with open(path) as f:
			data = yaml.safe_load(f)
		return cls._from_dict(data)

	@classmethod
	def _from_dict(cls, data: dict) -> ProblemConfig:
		return cls(
			name=data.get("name", "unknown"),
			description=data.get("description", ""),
			type=data.get("type", "general"),
			measure=MeasureConfig(**data.get("measure", {})),
			profile=ProfileConfig(
				commands=data.get("profile", {}).get("commands", [])
			),
			targets=TargetConfig(**data.get("targets", {})),
			edit=EditConfig(**data.get("edit", {})),
			execution=ExecutionConfig(**data.get("execution", {})),
			knowledge=KnowledgeConfig(**data.get("knowledge", {})),
		)


def build_agent_prompt(config: ProblemConfig) -> str:
	"""Build an agent prompt from any problem configuration."""
	sections = []

	# Header
	sections.append(
		f"# Optimization Agent: {config.name}\n\n"
		f"{config.description}\n"
	)

	# Execution environment
	if config.execution.mode == "local":
		sections.append(
			f"## Environment\n"
			f"Run locally in: `{config.edit.working_directory}`\n"
		)
	elif config.execution.mode == "ssh":
		host = config.execution.ssh_host
		env = " ".join(
			f"{k}={v}" for k, v in config.execution.env_vars.items()
		)
		sections.append(
			f"## Environment\n"
			f"Run via SSH:\n"
			f"```\nssh {host} \"cd {config.edit.working_directory}"
			f" && {env} <command>\"\n```\n"
		)

	# Measurement
	sections.append(
		f"## How to Measure\n"
		f"```\n{config.measure.command}\n```\n"
		f"Metric: **{config.measure.metric}** "
		f"({config.measure.direction}, "
		f"unit: {config.measure.unit})\n"
	)

	# Targets
	t = config.targets
	sections.append("## Targets\n")
	for label, value in t.baselines.items():
		sections.append(f"- {label}: **{value}** {config.measure.unit}")
	if t.leaderboard_best:
		sections.append(
			f"- Leaderboard #1: **{t.leaderboard_best}** "
			f"{config.measure.unit}"
		)
	if t.our_best:
		sections.append(
			f"- Our best: **{t.our_best}** "
			f"{config.measure.unit}"
		)
	sections.append("")

	# Profiling tools
	if config.profile.commands:
		sections.append("## Profiling Tools\n")
		for tool in config.profile.commands:
			sections.append(
				f"- **{tool['name']}**: "
				f"`{tool['command']}`\n"
				f"  {tool.get('description', '')}"
			)
		sections.append("")

	# Files
	sections.append("## Files\n")
	sections.append("**Edit these:**")
	for f in config.edit.target_files:
		sections.append(f"- `{f}`")
	sections.append("\n**Read these (DO NOT modify):**")
	for f in config.edit.read_only_files:
		sections.append(f"- `{f}`")
	sections.append("")

	# Knowledge
	k = config.knowledge
	if k.strategy_hints:
		sections.append("## Optimization Strategies\n")
		for hint in k.strategy_hints:
			sections.append(f"- {hint}")
		sections.append("")

	if k.reference_docs:
		sections.append("## Reference\n")
		for doc in k.reference_docs:
			sections.append(f"- {doc}")
		sections.append("")

	if k.anti_patterns:
		sections.append("## Rules / Anti-patterns\n")
		for ap in k.anti_patterns:
			sections.append(f"- {ap}")
		sections.append("")

	# The universal optimization loop
	sections.append(
		f"## The Optimization Loop\n\n"
		f"1. **MEASURE**: Run `{config.measure.command}` "
		f"to get current {config.measure.metric}\n"
		f"2. **POSITION**: Compare to targets above. "
		f"How far from the best?\n"
		f"3. **DIAGNOSE**: Use profiling tools to find "
		f"the bottleneck\n"
		f"4. **ACT**: Edit the target files to address "
		f"the bottleneck\n"
		f"5. **RE-MEASURE**: Run measurement again\n"
		f"6. **DECIDE**: Improved? Keep going. "
		f"Plateaued? Try different approach.\n\n"
		f"Iterate aggressively. Each measurement is fast.\n"
	)

	# Output format
	metric = config.measure.metric
	sections.append(
		f"## Output\n```\n"
		f"BEST_{metric.upper()}: <value>\n"
		f"VS_LEADERBOARD: <ratio>\n"
		f"APPROACH: <what worked>\n"
		f"WHAT_FAILED: <what didn't>\n"
		f"NOVEL_TECHNIQUES: <any new patterns>\n"
		f"TOOL_REQUESTS: <any tools that would help>\n"
		f"```\n"
	)

	return "\n".join(sections)
