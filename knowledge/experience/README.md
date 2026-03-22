# Experience Store

Persistent learnings from kernel optimization runs. Agents should read relevant records before starting and write new records when done.

## Format

`records.jsonl` -- one JSON object per line:

```json
{
  "problem": "matmul_fp16",
  "op_type": "matmul",
  "hardware": "b200",
  "approach": "cuBLAS via torch.matmul with TF32 enabled",
  "speedup": 12.95,
  "baseline_us": 2160,
  "best_us": 167,
  "what_worked": "Enable TF32, cuBLAS auto-selects CUTLASS sm100 tensorop kernel",
  "what_failed": "Custom Triton matmul was 30% slower than cuBLAS for all shapes",
  "key_insight": "For standard GEMM shapes, cuBLAS is near-optimal. Don't write custom kernels.",
  "timestamp": "2026-03-22"
}
```

## How agents use this

**Before optimizing:** Read records.jsonl, filter by `op_type` matching the current problem. Use `what_worked` and `key_insight` to inform strategy.

**After optimizing:** Append a new record with results. Be specific about what worked and what didn't -- this helps future agents avoid dead ends.

## Cloud portability

This is just a file in the git repo. Clone the repo on any machine and the experience travels with it. No database, no service, no credentials.
