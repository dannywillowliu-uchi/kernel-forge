# Novel Techniques Registry

Optimization techniques discovered by agents during runs.
Each technique is a reusable pattern that can be applied to future problems.

## Format

Each technique is a JSON file with:
- `name`: short identifier
- `description`: what the technique does
- `when_to_apply`: traits/bottleneck conditions
- `code_pattern`: the actual code (CUDA/Triton/Python)
- `evidence`: problems where it worked, speedups achieved
- `discovered_from`: which problem/run led to discovery
