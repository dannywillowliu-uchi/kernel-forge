"""Orchestration prompt generator.

Builds a lean orchestrator prompt with just problem context and constraints.
The orchestrator figures out roofline, strategy, and agent prompts itself.

Usage:
	kernel-forge orchestrate trimul --gpu 3
"""

from __future__ import annotations

import logging
from pathlib import Path

from kernel_forge.config import ForgeConfig

logger = logging.getLogger(__name__)

ORCHESTRATOR_PROMPT_PATH = Path(__file__).parent / "agents" / "orchestrator_prompt.md"


def build_orchestrate_prompt(
	problem_name: str,
	problem_dir: str,
	problem_context: str,
	roofline_analysis: str = "",
	gpu_id: int = 3,
	config: ForgeConfig | None = None,
	knowledge_context: str = "",
	warm_start: str = "",
	max_wall_time_hours: float = 2.0,
	max_redirects: int = 5,
) -> str:
	"""Build orchestrator prompt with problem context injected."""
	if config is None:
		config = ForgeConfig()

	if not ORCHESTRATOR_PROMPT_PATH.exists():
		raise FileNotFoundError(f"Orchestrator prompt not found: {ORCHESTRATOR_PROMPT_PATH}")

	template = ORCHESTRATOR_PROMPT_PATH.read_text()

	replacements = {
		"{problem_context}": problem_context,
		"{problem_dir}": problem_dir,
		"{gpu_id}": str(gpu_id),
		"{max_wall_time_hours}": str(max_wall_time_hours),
		"{max_redirects}": str(max_redirects),
	}
	for placeholder, value in replacements.items():
		template = template.replace(placeholder, value)

	if warm_start:
		template += f"\n\n## Previous Campaign Context\n\n{warm_start}\n"

	return template
