# Agent Memory System

Four types of persistent memory for kernel optimization agents. All stored as files in this directory -- portable via git clone.

## Memory Types

### 1. Semantic Memory (`semantic/`)
**What:** Facts about hardware, tools, and optimization techniques.
**When to read:** Always, at the start of every run.
**When to write:** When discovering a new hardware fact or tool behavior.

Files: one markdown file per topic.
```
semantic/
  b200_hardware.md       # HBM bandwidth, tensor core peaks, L2 size, etc.
  tools_behavior.md      # How tools actually behave (e.g., CuPy launch overhead = same as load_inline)
  framework_limits.md    # What torch.compile can/can't fuse, Triton limitations
```

### 2. Episodic Memory (`episodes/`)
**What:** Structured summaries of past optimization campaigns.
**When to read:** When starting a problem similar to a past episode.
**When to write:** After completing an optimization campaign.

Files: one markdown file per campaign.
```
episodes/
  trimul_2026-03-21.md
  grayscale_2026-03-22.md
  rmsnorm_sol_2026-03-21.md
```

Format:
```markdown
# Episode: [problem name]

## Result
Best: X us/ms, Y speedup, Z% of hardware peak

## Approaches tried (in order)
1. [approach] -> [result] -> [why it worked/failed]
2. ...

## Key insight
[The one thing that mattered most]

## What would I do differently
[If starting over, what approach would I take first?]
```

### 3. Procedural Memory (`procedures/`)
**What:** Step-by-step procedures for common optimization patterns.
**When to read:** When the current problem matches a known pattern.
**When to write:** When you discover a reliable procedure that works.

Files: one markdown file per procedure.
```
procedures/
  optimize_memory_bound.md    # How to optimize a memory-bound kernel
  optimize_compute_bound.md   # How to optimize a compute-bound kernel
  optimize_launch_bound.md    # How to reduce kernel launch overhead
  optimize_fused_pipeline.md  # How to fuse a multi-kernel pipeline
  beat_torch_compile.md       # When and how to beat torch.compile
```

Format:
```markdown
# Procedure: [name]

## When to use
[Pattern that triggers this procedure]

## Steps
1. [step] -- [why]
2. [step] -- [why]
...

## Common mistakes
- [mistake] -- [what to do instead]

## Evidence
- [problem X]: used this procedure, got Y speedup
- [problem Z]: used this procedure, got W speedup
```

### 4. Short-term Memory
Handled by the agent's context window during a run. Not persisted here.

## How agents use this

### Before starting a new problem:
1. Read `semantic/` for hardware and tool facts
2. Search `episodes/` for similar problems (match by op type, bound type)
3. Check `procedures/` for a matching optimization pattern
4. Use insights to choose initial approach

### After finishing:
1. Write an episode summary to `episodes/`
2. Update `semantic/` if new hardware/tool facts were discovered
3. Update or create `procedures/` if a reliable pattern was confirmed or discovered

## Cloud portability
Everything is plain files in git. Clone the repo and all memory is available. No database, vector store, or cloud service needed.

For cloud scaling (multiple agents in parallel), consider:
- Git-based merge for concurrent writes
- Or a shared JSONL append log with agent IDs
