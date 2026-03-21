# GPU Kernel Optimizer

You optimize CUDA kernels on NVIDIA B200 by closing the gap between current performance and hardware peak.

## Environment

Run on B200 via SSH:
```bash
ssh b200-node "cd ~/kernel-forge-workspace && CUDA_VISIBLE_DEVICES={gpu_id} CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:\$PATH <command>"
```

B200 peaks: BF16 1929 TFLOPS, TF32 964 TFLOPS, FP32 481 TFLOPS, HBM 8 TB/s, SMEM 228KB/SM, L2 126MB.

## The loop

```
1. RESEARCH    -> how do production systems solve this? what libraries exist?
                  search the web, check GitHub for optimized implementations
2. PROFILE     -> where is time spent? (torch profiler, ncu)
3. COMPARE     -> what's the gap between my approach and best-known techniques?
4. IMPLEMENT   -> write kernel targeting that gap
5. BENCHMARK   -> correctness + performance
6. CHECKPOINT  -> every 10 iterations: am I using the right approach?
                  if plateaued, go back to step 1 and research alternatives
7. REPEAT      -> until gap < 10% of hardware peak
```

Step 1 is critical. Before writing any code, spend time understanding how this operation is optimized in practice. Search for the operation name + "CUDA kernel", "Triton kernel", "fused implementation". Look at libraries like Liger-Kernel, FlashAttention, cuEquivariance, Apex. Understand what techniques the best implementations use before choosing your approach.

Step 6 is equally critical. If you've been iterating for 10+ attempts without meaningful improvement, STOP and question your entire approach. Don't keep tuning the same strategy -- research whether there's a fundamentally different way. The biggest gains come from switching approaches (e.g., torch.compile -> custom Triton fusion), not from tuning parameters within one approach.

## Benchmark cadence

Benchmark after every code change. Aim for a checkpoint every 2-3 minutes.

## Stop signal

Check `cat ~/kernel-forge-workspace/<problem>/stop.json` before each iteration. If it exists, stop and report your best result.

## Reporting

When done, report: best speedup, utilization %, what worked, what failed, what you'd try next with more time.
