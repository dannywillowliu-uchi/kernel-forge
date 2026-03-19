# Using Kernel Forge for Parameter Golf

## What Kernel Forge Is

Kernel Forge (`/Users/dannyliu/personal_projects/optimization/`) is an autonomous GPU kernel optimization system. It takes a kernel or model, profiles it against hardware peak, identifies bottlenecks, writes optimized kernels (PyTorch/Triton/CUDA), and iterates until near-roofline.

**Key results:** 65.9x on LayerNorm, 16.1x on attention, 84.6% HBM bandwidth on RMSNorm, within 10% of GPU MODE competition winners on NVFP4 GEMM.

## What Parameter Golf Needs

Train the best 16MB language model in 10 minutes on 8xH100. The optimization surface:
1. Model architecture (fit quality into 16MB)
2. Training recipe (optimizer, schedule, data)
3. **Kernel efficiency** (maximize training throughput on H100)
4. Multi-GPU communication (8xH100 NCCL/FSDP)
5. Compression (pack the most into 16MB)

## Where Kernel Forge Helps (items 3)

### Profile-Driven Kernel Optimization

Once you have a training script, Kernel Forge can:

1. **Profile the training loop** -- identify which kernels are bottlenecks
2. **Optimize attention** -- we achieved 16.1x on SDPA via manual decomposition
3. **Optimize norms** -- 65.9x on LayerNorm, 3.99x on RMSNorm via custom Triton/CUDA
4. **Optimize elementwise** -- 1.5-9x via kernel fusion
5. **Find TF32/precision wins** -- 10-15x on matmul by enabling tensor cores

### How to Use It

```bash
# 1. Generate an optimization prompt for any kernel
cd /Users/dannyliu/personal_projects/optimization
uv run kernel-forge solve <problem_name> --gpu <GPU_ID>

# 2. The prompt includes:
#    - Methodology (gap-driven: profile -> diagnose -> act -> re-measure)
#    - All B200 tools (ncu, nsys, benchmarking, PTX inspection)
#    - Distilled optimization guides from 4000 community kernels
#    - Winning kernel solutions from similar problems
#    - Triton/CUDA code patterns library
#    - 11 discovered optimization techniques

# 3. Run as a Claude Code subagent or CLI
claude --permission-mode bypassPermissions \
  -p "$(cat runs/<prompt_file>.md)" \
  --model opus --max-turns 30
```

### Key Files

| File | What it contains |
|------|-----------------|
| `src/kernel_forge/solve.py` | Auto-generates optimization prompts with full knowledge injection |
| `src/kernel_forge/agents/agent_prompt.md` | The agent methodology (gap-driven loop, tools, hardware specs) |
| `knowledge/distilled/` | 10 op-type optimization guides from community data |
| `knowledge/cuda_patterns/` | 9 CUDA/Triton code patterns (warp reduction, shared mem, FP4, etc.) |
| `knowledge/techniques/` | 11 discovered techniques (chunked stats, ILP loads, fusion, etc.) |
| `knowledge/solutions/` | 20 winning kernels with source code |
| `knowledge/external/` | 4000 community kernels from GPU MODE |
| `config.yaml` | Declarative config (swap hardware target here) |

## What Needs to Change for Parameter Golf

### 1. Hardware Target: H100 instead of B200

Edit `config.yaml`:
```yaml
hardware:
  ssh_host: <your-h100-host>  # or Modal/Runpod
  gpu_id: 0
  cuda_visible_devices: "0"
  cuda_home: /usr/local/cuda  # H100 typically 12.x
```

Update hardware peaks in `src/kernel_forge/core/types.py`:
```python
H100_PEAKS = HardwarePeaks(
    fp32_tflops=267.0,      # H100 SXM
    tf32_tflops=989.0,
    bf16_tflops=1979.0,
    fp8_tflops=3958.0,
    fp4_tflops=0.0,         # H100 doesn't have FP4
    hbm_bandwidth_tb_s=3.35,
    l2_bandwidth_tb_s=12.0,
)
```

### 2. Platform: Modal/Runpod instead of SSH

The system uses an `Executor` protocol (`src/kernel_forge/remote/executor.py`). Currently only `SSHExecutor` and `DryRunExecutor` exist. For Modal:

```python
class ModalExecutor:
    async def run(self, command: str, timeout: int = 300) -> CommandResult:
        # Use Modal sandbox API to execute commands
        ...
    async def upload(self, local_path: str, remote_path: str) -> None:
        # Modal volume mount
        ...
```

### 3. Multi-GPU (8xH100)

Kernel Forge is single-GPU only. For Parameter Golf's 8xH100:
- Profile NCCL communication overhead separately
- Optimize per-GPU kernels with Kernel Forge
- Use torch.distributed profiling for multi-GPU bottlenecks
- This is outside Kernel Forge's current scope

### 4. Training Loop Integration

Instead of optimizing isolated kernels, you'd:

1. Run the training script with `nsys` to get a timeline
2. Identify the top-5 most expensive kernels
3. For each kernel, create a KernelBench-style problem file:
   ```python
   class Model(nn.Module):
       def forward(self, x):
           return <the_expensive_operation>(x)

   def get_inputs():
       return [<tensors_matching_training_shapes>]
   ```
4. Run `kernel-forge solve <problem>` on each
5. Replace the optimized kernels in the training script

## What Kernel Forge Can NOT Do for Parameter Golf

- **Architecture search** -- choosing between MHA vs GQA vs MLA, depth vs width, etc.
- **Training recipe** -- learning rate schedules, optimizer choice, gradient accumulation
- **Data strategy** -- tokenizer choice, data mixing, curriculum
- **Multi-GPU optimization** -- FSDP sharding strategy, communication overlap
- **Compression** -- knowledge distillation, pruning, quantization-aware training

These require separate tools/agents. Kernel Forge is one piece of the pipeline.

## Recommended Approach

1. Start with an existing competitive training script (e.g., from NanoGPT speedrun community)
2. Profile with `nsys` to find bottlenecks
3. Use Kernel Forge to optimize the top kernel bottlenecks
4. Iterate: train -> profile -> optimize kernels -> train again
5. Use Kernel Forge's experience system to accumulate H100-specific knowledge

## Repo
Private: https://github.com/dannywillowliu-uchi/kernel-forge
