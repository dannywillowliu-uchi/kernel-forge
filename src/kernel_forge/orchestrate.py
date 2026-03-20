"""Orchestration prompt generator.

Builds a complete orchestrator prompt with problem context, roofline analysis,
knowledge base, and the optimizer agent prompt embedded.

Usage:
	kernel-forge orchestrate trimul --gpu 3
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from kernel_forge.config import ForgeConfig

logger = logging.getLogger(__name__)

ORCHESTRATOR_PROMPT_PATH = Path(__file__).parent / "agents" / "orchestrator_prompt.md"
AGENT_PROMPT_PATH = Path(__file__).parent / "agents" / "agent_prompt.md"


def build_orchestrate_prompt(
	problem_name: str,
	problem_dir: str,
	problem_context: str,
	roofline_analysis: str,
	gpu_id: int = 3,
	config: ForgeConfig | None = None,
	knowledge_context: str = "",
	warm_start: str = "",
	max_wall_time_hours: float = 2.0,
	max_redirects: int = 5,
) -> str:
	"""Build a complete orchestrator prompt with all context injected."""
	if config is None:
		config = ForgeConfig()

	if not ORCHESTRATOR_PROMPT_PATH.exists():
		raise FileNotFoundError(f"Orchestrator prompt not found: {ORCHESTRATOR_PROMPT_PATH}")

	template = ORCHESTRATOR_PROMPT_PATH.read_text()

	# Load agent prompt to embed
	agent_prompt = ""
	if AGENT_PROMPT_PATH.exists():
		agent_prompt = AGENT_PROMPT_PATH.read_text()
		agent_prompt = agent_prompt.replace("{gpu_id}", str(gpu_id))

	# Load knowledge if not provided
	if not knowledge_context:
		knowledge_context = _load_knowledge(config)

	# Fill template
	prompt = template
	replacements = {
		"{problem_context}": problem_context,
		"{roofline_analysis}": roofline_analysis,
		"{knowledge_context}": knowledge_context,
		"{agent_prompt}": agent_prompt,
		"{problem_dir}": problem_dir,
		"{gpu_id}": str(gpu_id),
		"{knowledge_dir}": str(config.knowledge_dir),
		"{max_wall_time_hours}": str(max_wall_time_hours),
		"{max_redirects}": str(max_redirects),
		"{problem_name}": problem_name,
	}
	for placeholder, value in replacements.items():
		prompt = prompt.replace(placeholder, value)

	if warm_start:
		prompt += f"\n\n## Warm-Start Context\n\n{warm_start}\n"

	return prompt


def _load_knowledge(config: ForgeConfig) -> str:
	"""Load relevant knowledge base entries."""
	sections = []

	# Distilled guides
	distilled_dir = config.knowledge_dir / "distilled"
	if distilled_dir.exists():
		for guide in sorted(distilled_dir.glob("*.md")):
			content = guide.read_text()
			if len(content) > 100:
				sections.append(f"### {guide.stem}\n{content[:3000]}")

	# Techniques
	techniques_dir = config.knowledge_dir / "techniques"
	if techniques_dir.exists():
		for tech_file in sorted(techniques_dir.glob("*.json")):
			tech = json.loads(tech_file.read_text())
			sections.append(
				f"- **{tech.get('name', tech_file.stem)}**: "
				f"{tech.get('description', '')[:200]}"
			)

	return "\n\n".join(sections) if sections else "No knowledge base loaded."
