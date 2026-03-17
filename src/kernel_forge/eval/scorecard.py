"""Scorecard: compare our results against baselines and identify gaps.

Loads B200 baselines (eager + torch.compile) and our experience store
results, computes per-problem and aggregate metrics, identifies where
we're underperforming and what to prioritize next.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ProblemScore:
	"""Score for a single problem against baselines."""

	problem: str
	eager_ms: float = 0.0
	compile_ms: float = 0.0
	our_ms: float = 0.0
	our_speedup_vs_eager: float = 0.0
	our_speedup_vs_compile: float = 0.0
	compile_speedup_vs_eager: float = 0.0
	# not_attempted, beats_both, beats_eager, no_speedup
	status: str = "not_attempted"
	gap_to_compile: float = 0.0
	approach: str = ""


@dataclass
class Scorecard:
	"""Aggregate evaluation scorecard."""

	total_problems: int = 0
	attempted: int = 0
	beats_eager: int = 0
	beats_compile: int = 0
	beats_both: int = 0
	no_speedup: int = 0
	not_attempted: int = 0

	avg_speedup_vs_eager: float = 0.0
	avg_speedup_vs_compile: float = 0.0

	# Problems sorted by gap (biggest opportunity first)
	gaps: list[ProblemScore] = field(default_factory=list)
	scores: list[ProblemScore] = field(default_factory=list)


def load_baselines(baselines_path: Path) -> dict[str, dict]:
	"""Load B200 baselines from JSON."""
	if not baselines_path.exists():
		logger.warning("No baselines file at %s", baselines_path)
		return {}
	data = json.loads(baselines_path.read_text())
	baselines = {}
	for entry in data:
		name = entry.get("problem", "")
		if name:
			baselines[name] = entry
	return baselines


def load_our_results(experience_path: Path) -> dict[str, dict]:
	"""Load our best result per problem from experience store."""
	if not experience_path.exists():
		return {}
	results: dict[str, dict] = {}
	with open(experience_path) as f:
		for line in f:
			if not line.strip():
				continue
			entry = json.loads(line)
			name = entry.get("problem_name", "")
			speedup = entry.get("speedup", 0)
			if name and (
				name not in results
				or speedup > results[name].get("speedup", 0)
			):
				results[name] = entry
	return results


def compute_scorecard(
	baselines: dict[str, dict],
	our_results: dict[str, dict],
) -> Scorecard:
	"""Compute full scorecard comparing our results to baselines."""
	card = Scorecard()
	card.total_problems = len(baselines)

	speedups_eager: list[float] = []
	speedups_compile: list[float] = []

	for problem, baseline in baselines.items():
		eager_ms = baseline.get("eager_ms", 0)
		compile_ms = baseline.get("compile_ms") or eager_ms
		compile_speedup = baseline.get("compile_speedup", 1.0)

		score = ProblemScore(
			problem=problem,
			eager_ms=eager_ms,
			compile_ms=compile_ms,
			compile_speedup_vs_eager=compile_speedup or 1.0,
		)

		if problem in our_results:
			our = our_results[problem]
			our_speedup = our.get("speedup", 0)
			our_ms = (
				eager_ms / our_speedup if our_speedup > 0 else 0
			)
			score.our_ms = our_ms
			score.our_speedup_vs_eager = our_speedup
			score.approach = our.get("approach_notes", "")[:80]

			if compile_ms and compile_ms > 0 and our_ms > 0:
				score.our_speedup_vs_compile = compile_ms / our_ms
				score.gap_to_compile = (
					(compile_ms - our_ms) / compile_ms
				)
			else:
				score.our_speedup_vs_compile = our_speedup
				score.gap_to_compile = 0

			card.attempted += 1
			speedups_eager.append(our_speedup)

			if our_speedup > 1.0:
				card.beats_eager += 1
			if score.our_speedup_vs_compile > 1.0:
				card.beats_compile += 1
				speedups_compile.append(
					score.our_speedup_vs_compile
				)
			if our_speedup > 1.0 and score.our_speedup_vs_compile > 1.0:
				card.beats_both += 1

			if our_speedup <= 1.0:
				card.no_speedup += 1
				score.status = "no_speedup"
			elif score.our_speedup_vs_compile > 1.0:
				score.status = "beats_both"
			else:
				score.status = "beats_eager"
		else:
			card.not_attempted += 1
			score.status = "not_attempted"

		card.scores.append(score)

	# Averages
	if speedups_eager:
		card.avg_speedup_vs_eager = (
			sum(speedups_eager) / len(speedups_eager)
		)
	if speedups_compile:
		card.avg_speedup_vs_compile = (
			sum(speedups_compile) / len(speedups_compile)
		)

	# Gaps: problems where we lose to compile or haven't attempted
	card.gaps = sorted(
		[s for s in card.scores if s.status != "beats_both"],
		key=lambda s: s.gap_to_compile,
	)

	return card


def format_scorecard(card: Scorecard) -> str:
	"""Human-readable scorecard."""
	lines = [
		"=" * 60,
		"KERNEL FORGE SCORECARD (B200)",
		"=" * 60,
		f"Problems: {card.total_problems}",
		f"Attempted: {card.attempted}/{card.total_problems}",
		"",
		f"Beats PyTorch eager:  {card.beats_eager}/{card.attempted}",
		f"Beats torch.compile:  {card.beats_compile}/{card.attempted}",
		f"Beats both:           {card.beats_both}/{card.attempted}",
		f"No speedup:           {card.no_speedup}/{card.attempted}",
		f"Not attempted:        {card.not_attempted}",
		"",
		f"Avg speedup vs eager:   {card.avg_speedup_vs_eager:.2f}x",
		f"Avg speedup vs compile: {card.avg_speedup_vs_compile:.2f}x",
		"",
		"--- TOP GAPS (prioritize these) ---",
	]

	for s in card.gaps[:15]:
		if s.status == "not_attempted":
			lines.append(
				f"  {s.problem:45s} NOT ATTEMPTED "
				f"(compile={s.compile_speedup_vs_eager:.2f}x)"
			)
		elif s.status == "beats_eager":
			lines.append(
				f"  {s.problem:45s} "
				f"us={s.our_speedup_vs_eager:.2f}x "
				f"compile={s.compile_speedup_vs_eager:.2f}x "
				f"GAP={s.gap_to_compile:+.1%}"
			)
		elif s.status == "no_speedup":
			lines.append(
				f"  {s.problem:45s} "
				f"us=1.0x "
				f"compile={s.compile_speedup_vs_eager:.2f}x "
				f"LOSING"
			)

	lines.append("")
	lines.append("--- WINS ---")
	wins = [
		s for s in card.scores if s.status == "beats_both"
	]
	for s in sorted(wins, key=lambda x: -x.our_speedup_vs_eager)[:10]:
		lines.append(
			f"  {s.problem:45s} "
			f"{s.our_speedup_vs_eager:.2f}x "
			f"(vs compile: {s.our_speedup_vs_compile:.2f}x)"
		)

	return "\n".join(lines)


def get_gap_context_for_problem(
	problem: str,
	baselines: dict[str, dict],
) -> str:
	"""Build context string telling the agent what to beat."""
	if problem not in baselines:
		return ""

	b = baselines[problem]
	eager_ms = b.get("eager_ms", 0)
	compile_ms = b.get("compile_ms")
	compile_speedup = b.get("compile_speedup", 1.0)

	lines = [
		"## Baselines to beat (B200 GPU)",
		f"- PyTorch eager: {eager_ms:.4f} ms",
	]

	if compile_ms and compile_ms > 0:
		lines.append(
			f"- torch.compile (max-autotune): "
			f"{compile_ms:.4f} ms "
			f"({compile_speedup:.2f}x over eager)"
		)
		lines.append(
			f"\n**Your target: beat {compile_ms:.4f} ms "
			f"(torch.compile). Anything slower is not "
			f"competitive.**"
		)
	else:
		lines.append(
			f"\n**Your target: beat {eager_ms:.4f} ms "
			f"(PyTorch eager).**"
		)

	return "\n".join(lines)
