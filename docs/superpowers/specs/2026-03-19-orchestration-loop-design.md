# Kernel Forge Orchestration Loop

## Problem

The current orchestrator is single-shot: it sets up context, invokes one autonomous agent, and collects results. There is no monitoring, no redirection, and no escalation. Every ambitious result in the project's history (RMSNorm 84.6% HBM, LayerNorm 64x, NVFP4 11.1us) was achieved with a human acting as the outer loop -- watching progress, computing roofline gaps, and redirecting the agent when it plateaued.

Today's TriMul benchmark demonstrated this gap: the agent achieved 9.25x speedup (0.786ms) using torch.compile tuning, but plateaued at 20% of roofline. It never escalated to custom Triton or CUDA despite the remaining 80% headroom being attributable to memory traffic between unfused kernels. A human watching would have redirected at turn 15.

## Design

### Architecture: Orchestrator-as-Agent

The orchestrator is a Claude Code agent (Opus) that acts as a senior optimization engineer. It runs within a Claude Code session with unlimited turns (`--max-turns 0`), constrained only by wall time. It uses the Agent tool to spawn optimizer subagents and SSH/WebSearch/Read for monitoring and research.

```
┌──────────────────────────────────────────────────┐
│ Orchestrator Agent (Opus, main session)          │
│ --max-turns 0 (unlimited, wall-time constrained) │
│                                                  │
│  1. Analyze problem + compute roofline           │
│  2. Research domain (WebSearch, knowledge base)   │
│  3. Craft optimizer prompt with all context       │
│  4. Spawn optimizer agent (background)            │
│  5. Monitor loop:                                 │
│     ├─ Poll checkpoint.jsonl via SSH              │
│     ├─ On new checkpoint: evaluate trajectory     │
│     ├─ Decision: CONTINUE | RESEARCH | REDIRECT | STOP │
│     ├─ RESEARCH: gather info, re-evaluate         │
│     ├─ REDIRECT: signal stop, warm-start new agent│
│     └─ STOP: signal stop, save results            │
│  6. Save final results + techniques discovered    │
│                                                  │
│  Tools: Agent, SSH, WebSearch, Read, Glob, Grep   │
└──────────┬───────────────────────────────────────┘
           │ spawns (Agent tool, background)
           ▼
┌──────────────────────────────────────────────────┐
│ Optimizer Agent (Opus, subagent)                 │
│                                                  │
│  - Writes/edits submission.py                     │
│  - Runs correctness tests on B200                 │
│  - Runs benchmarks on B200 (harness auto-writes   │
│    checkpoint.jsonl)                              │
│  - Runs profiling (ncu, torch profiler)           │
│  - Checks stop.json before each iteration         │
│  - Iterates via gap-driven loop                   │
│                                                  │
│  Tools: SSH (to B200), file editing               │
└──────────────────────────────────────────────────┘
```

### Checkpoint Protocol

Checkpoints are written **by the benchmark harness** (bench.py), not the optimizer agent. This makes them reliable -- they're a side effect of measurement, not a behavioral instruction to an LLM.

**Path**: `~/kernel-forge-workspace/<problem>/checkpoint.jsonl` (append-only)

Each line is a JSON object appended after every benchmark run:

```json
{
  "iteration": 7,
  "geomean_ms": 0.864,
  "best_geomean_ms": 0.864,
  "per_benchmark": [
    {"config": "seq=256 bs=2 dim=128", "time_ms": 0.244},
    {"config": "seq=1024 bs=1 dim=384", "time_ms": 2.591}
  ],
  "timestamp": "2026-03-19T23:45:00Z"
}
```

The harness writes this automatically. No agent cooperation required.

**Profile data** is written separately by the harness when profiling is run:

**Path**: `~/kernel-forge-workspace/<problem>/profile_latest.json`

```json
{
  "iteration": 7,
  "source": "torch_profiler",
  "breakdown": [
    {"kernel": "fused_projection_mm_bf16", "time_us": 460, "pct": 22.7},
    {"kernel": "layernorm_fused", "time_us": 505, "pct": 24.9},
    {"kernel": "elementwise_mask_sigmoid_gate", "time_us": 516, "pct": 25.5},
    {"kernel": "bmm_einsum", "time_us": 188, "pct": 9.3},
    {"kernel": "final_projection_tf32", "time_us": 344, "pct": 17.0}
  ],
  "total_us": 2025,
  "raw_output": "..."
}
```

This persists across agent spawns. The next agent can read it directly without re-profiling.

The orchestrator reads both via SSH:
- `ssh b200-node "tail -1 ~/kernel-forge-workspace/<problem>/checkpoint.jsonl"` (latest checkpoint)
- `ssh b200-node "cat ~/kernel-forge-workspace/<problem>/profile_latest.json"` (latest profile)

### Stop Signal

The orchestrator signals the optimizer agent to stop via a file on B200:

**Path**: `~/kernel-forge-workspace/<problem>/stop.json`

```json
{"reason": "redirect", "message": "Plateaued at 20% roofline. Redirecting to custom Triton approach."}
```

The optimizer agent is instructed to check for this file before each benchmark iteration. If it exists, the agent stops gracefully and reports its final state.

The orchestrator writes it via: `ssh b200-node "echo '{...}' > ~/kernel-forge-workspace/<problem>/stop.json"`

Before spawning a new agent, the orchestrator removes it: `ssh b200-node "rm -f ~/kernel-forge-workspace/<problem>/stop.json"`

### Monitor Loop

After spawning the optimizer agent in the background:

```
checkpoint_history = []
research_notes = []
warm_start_summaries = []
last_seen_iteration = 0

while wall_time < max_wall_time:
    poll checkpoint.jsonl via SSH (tail -1)

    if new checkpoint (iteration > last_seen_iteration):
        last_seen_iteration = iteration
        append to checkpoint_history
        read current submission.py from B200
        read profile_latest.json from B200 (if updated)

        evaluate:
            inputs: checkpoint_history, submission_code, profile_data,
                    problem_context, roofline_analysis, research_notes,
                    warm_start_summaries

            reasoning: Is the agent making progress? Is it at the right
                       abstraction level for the remaining gap? Is there a
                       better strategy based on the profile breakdown?
                       Would research help inform the decision?

            output: one of CONTINUE, RESEARCH, REDIRECT, STOP

        if CONTINUE: do nothing
        if RESEARCH: use WebSearch/Read to gather targeted info,
                     add to research_notes, re-evaluate
        if REDIRECT:
            write stop.json via SSH
            wait for agent to finish (or timeout after 60s)
            read final checkpoint + submission.py
            build warm-start summary
            clear stop.json
            clear checkpoint.jsonl (fresh for new agent)
            spawn new optimizer agent with warm-start context
        if STOP:
            write stop.json via SSH
            wait for agent to finish
            save final results
            break
```

### Evaluation

The orchestrator LLM evaluates dynamically at every checkpoint. There are no hardcoded rules for when to redirect -- the LLM reasons about the full picture:

- **Trajectory**: Are results improving? How fast? Decelerating?
- **Roofline position**: What % of peak? Is there meaningful headroom?
- **Profile breakdown**: Where is time going? Is the agent attacking the right bottleneck?
- **Code inspection**: What abstraction level is the agent at? Could a different level help?
- **History across spawns**: What's been tried before? What failed and why?
- **Domain knowledge**: Are there known techniques for this bottleneck pattern?

The evaluation is the orchestrator's own reasoning, not a separate API call. It sees the checkpoint data in its context and decides.

### Warm-Start Protocol

When the orchestrator decides to redirect:

1. **Signal stop** via stop.json on B200
2. **Wait** for the optimizer agent to finish gracefully
3. **Read** the current best submission.py and all checkpoint/profile data from B200
4. **Build** a warm-start summary:

```markdown
## Previous Optimization Attempts

### Agent 1 (iterations 1-16, best: 0.864ms)
**Best kernel**: [full submission.py attached]
**Approaches tried**:
- Fused 5 projections into one bf16 matmul (7.2ms -> 2.4ms)
- torch.compile max-autotune-no-cudagraphs (2.4ms -> 0.89ms)
- coordinate_descent_tuning + dynamic=False (0.89ms -> 0.86ms)
- Triton fused kernel (failed: 5x slower, single attempt, bad tiling)
- bf16 final projection (failed: correctness on cauchy distribution)

**Profile at plateau**:
| Kernel | Time | % |
|--------|------|---|
| fused_projection_mm_bf16 | 460us | 22.7% |
| layernorm_fused | 505us | 24.9% |
| elementwise_mask_sigmoid_gate | 516us | 25.5% |
| bmm_einsum | 188us | 9.3% |
| final_projection_tf32 | 344us | 17.0% |

**Roofline**: 0.864ms actual vs 0.16ms theoretical = 20% of peak.
Gap is memory-bound: 50% of time in elementwise ops writing intermediates to HBM.

## Strategic Direction
[Generated by orchestrator LLM -- specific to why the redirect is happening]
```

5. **Craft** the new optimizer agent's prompt with:
   - Full agent methodology (agent_prompt.md)
   - Problem definition
   - Warm-start summary (above) with full profile data
   - The best kernel so far (as starting point, saved on B200)
   - Any research notes accumulated by orchestrator
   - Specific strategic direction for this spawn

6. **Clear** checkpoint.jsonl on B200 (fresh tracking for new agent)
7. **Spawn** new optimizer agent in background

### Research Capability

At any evaluation point, the orchestrator can decide to research before acting. Research actions include:

- **WebSearch**: "Triton fused batched matmul with elementwise gating", "FlashAttention-style kernel fusion"
- **Knowledge base**: Read distilled guides, technique registry, similar solutions
- **Code search**: Look for reference implementations on GitHub
- **Documentation**: Check Triton/CUDA API docs for relevant primitives

Research findings accumulate in `research_notes` and get passed to both future evaluations and to agent warm-start prompts.

### Termination

The orchestrator stops when it evaluates and decides further progress is unlikely, or when hard limits are hit:

- **Max wall time**: configurable (default 2 hours)
- **Max agent spawns**: configurable (default 5 redirects)
- **Orchestrator judgment**: Roofline utilization is high enough, or all reasonable approaches exhausted

On stop, the orchestrator:
1. Saves the best kernel to `kernels/<problem>/`
2. Records the full trajectory (all checkpoints across all spawns)
3. Logs techniques discovered to the knowledge base
4. Writes a summary report

### Entry Point

The orchestrator is invoked via the existing `kernel-forge` CLI or as a Claude Code prompt:

```bash
# Via CLI (generates orchestrator prompt, user runs in Claude Code)
kernel-forge orchestrate --problem trimul --gpu 3

# Or directly as a prompt in Claude Code
# (the orchestrator prompt is self-contained)
```

The `orchestrate` command generates the orchestrator prompt with all problem context, roofline analysis, and knowledge pre-loaded. The user then runs it in a Claude Code session with `--max-turns 0`.

### What Changes vs Current System

| Aspect | Current | New |
|--------|---------|-----|
| Agent invocations per problem | 1 | 1-5+ (redirects) |
| Monitoring | None | Every checkpoint |
| Escalation | Agent self-manages (doesn't work) | Orchestrator LLM decides |
| Research | Upfront only | Continuous |
| Context between attempts | Lost | Warm-start summaries + profiling data |
| Roofline enforcement | In prompt (ignored) | Orchestrator evaluates actively |
| Checkpoint writes | Agent responsibility (unreliable) | Harness responsibility (automatic) |
| Agent termination | Process kill or timeout | Cooperative via stop.json |
| Turn budget | max_turns=30 | Unlimited (wall-time constrained) |

### What Doesn't Change

- The optimizer agent prompt (agent_prompt.md) -- still the same gap-driven methodology
- The benchmark/profiling tools on B200
- The knowledge base (distilled guides, techniques, solutions)

### Changes to bench.py

The benchmark harness gains two new behaviors:

1. **Auto-write checkpoint.jsonl**: After every benchmark run, append a JSON line with iteration, geomean, per-benchmark results, and timestamp. This is a ~10 line addition to bench.py.

2. **Auto-write profile_latest.json**: When `--profile` is run, write the structured breakdown to a JSON file alongside the human-readable output. Also ~10 lines.

3. **Check stop.json**: The optimizer agent is instructed to check for `stop.json` before each iteration. This is a behavioral instruction (not harness-enforced), but graceful -- if the agent ignores it, the orchestrator can wait for the agent to exhaust its turns naturally.

### Success Criteria

The orchestration loop is successful if:
1. On the TriMul problem, it autonomously achieves < 0.5ms geomean (currently 0.786ms with manual intervention, 0.16ms roofline) by escalating to custom Triton/CUDA
2. The orchestrator redirects at least once during a run, demonstrating it detected a plateau and escalated
3. No human intervention required during the run
