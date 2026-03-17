## CONV Optimization Guide

---

### What makes fast conv kernels fast

**1. Delegate conv to cuDNN/CUTLASS, fuse everything else in Triton**

All 5 examples use `extern_kernels.convolution` (cuDNN under the hood) for the actual weight × input multiply, then handle bias-add and activations in separate Triton pointwise kernels. The conv itself is near-optimal via cuDNN's autotuned GEMM paths (Winograd for 3×3, FFT for larger kernels, direct GEMM for 1×1). Your optimization leverage is in fusing the post-ops — every extra kernel launch reads/writes the full output tensor from HBM.

**2. Fuse bias + activation into a single in-place pass**

Example 1 fuses `bias_add + ReLU + threshold_backward` into one kernel (`in_out_ptr0` is read, modified, and stored in place). Example 4 fuses `cat + tanh * sigmoid` (GLU gate). Each un-fused elementwise op costs a full HBM round-trip on the output tensor. On B200 with 8 TB/s HBM3e, a 128MB activation tensor costs ~16µs per read — fusing 3 ops saves ~32µs.

**3. `eviction_policy='evict_last'` on broadcast operands**

Every example loads the bias vector with `eviction_policy='evict_last'`. This pins the bias in L2 across warps since it's tiny (C floats, e.g. 16 bytes for C=4) but accessed once per output element. Without this hint, L2 eviction under pressure would force repeated HBM reads.

---

### Common code patterns

**Block sizes:**
- `xnumel ≤ 32`: `XBLOCK=16-32`, `num_warps=1` — don't over-provision for tiny tensors
- `xnumel 64-512`: `XBLOCK=128`, `num_warps=4` — standard for medium bias-add fusions
- `xnumel > 4096`: `XBLOCK=256-1024`, `num_warps=8` — benchmark; larger blocks hide memory latency

**Bias broadcast pattern (all 5 examples):**
```python
x1 = xindex // spatial_size % C  # extract channel dim
bias = tl.load(in_ptr0 + x1, eviction_policy='evict_last')
out = conv_out + bias
```
This is the canonical pattern. `xindex // 16 % 4` means spatial=16, C=4. Adjust divisors for your shape.

**In-place fusion:**
```python
tmp = tl.load(in_out_ptr0 + x)  # read conv output
tmp = tmp + bias                  # bias add
tmp = triton_helpers.maximum(0, tmp)  # ReLU
tl.store(in_out_ptr0 + x, tmp)   # write back in-place
```
Use a separate `out_ptr` only when you need to preserve the pre-activation value (e.g., for backward mask in Example 1).

**Padding as a separate Triton kernel (Example 2):**
The `constant_pad_nd` kernel pads by materializing a larger buffer and zero-filling boundaries using conditional loads. This is correct but adds an extra kernel launch + HBM allocation. For fixed padding, prefer passing `padding=` directly to `extern_kernels.convolution` — cuDNN handles it in-register.

**CausalConv "over-pad then slice" (Example 3):**
```python
buf0 = extern_kernels.convolution(..., padding=(dilation*(kernel_size-1),))
return reinterpret_tensor(buf0, (B, C, T), strides, offset=0)  # slice off tail
```
`reinterpret_tensor` is zero-cost (metadata-only). Valid only when the slice is contiguous in the underlying buffer.

---

### B200/Blackwell considerations

**TMA (Tensor Memory Accelerator):** Use `tl.experimental_descriptor_load` (Triton 3.x) for async 2D tile loads into shared memory. Critical for conv kernels that tile over spatial dims — eliminates index calculation overhead and enables pipelining of memory and compute.

**228KB SMEM:** Blackwell doubles H100's SMEM. For a direct convolution tile of size `(Bx × Kx × BLOCK_N × BLOCK_K)` in fp16, you can hold ~56K elements per tile — enabling `BLOCK_M=128, BLOCK_N=256` tiles without spilling. Size your tiles to fill SMEM, not H100-era 164KB.

**FP8 tensor cores:** B200 has 4 PFLOPS FP8 vs 2 PFLOPS BF16. For inference, cast weights to `e4m3` and activations to `e4m3`/`e5m2`. None of the 5 examples use FP8 — all run in FP32 on toy shapes. Real production kernels should target BF16 minimum, FP8 where accuracy permits.

**126MB L2:** The full weight tensor of a ResNet conv layer (e.g., 512×512×3×3 FP16 = 4.7MB) fits in L2. For weight-stationary kernel loops, if weights fit in L2, subsequent output tiles read weights from L2 not HBM — bias the tiling to maximize weight reuse.

**Warp group size = 4 warps (128 threads):** Blackwell's MMA instruction operates on 4-warp groups. Set `num_warps` to a multiple of 4 for tensor core ops.

---

### Typical speedup range

| Scenario | Speedup over `torch.nn.Conv2d` |
|---|---|
| Bias-add fusion only (what these examples do) | 1.05–1.15× for small tensors, negligible for large |
| Bias + activation fusion | 1.1–1.3× |
| Custom tiled direct conv (small kernels, 1×1, 3×3) | 1.2–2.5× |
| FP8 vs FP32 | 3–4× |
| Full custom kernel (TMA + FP8 + tuned tiling) | 2–5× over cuDNN FP16 |

The examples are all tiny (batch=4, C=4, spatial=4×4) — speedup numbers at these sizes are noise-dominated. Real measurements require `N≥64, C≥64, HW≥32`.

---

### Common mistakes

**1. Writing a Triton conv from scratch when cuDNN is available**
For standard conv (stride=1, dilation=1, groups=1), cuDNN's autotuned paths (e.g., IMPLICIT_PRECOMP_GEMM, WINOGRAD_NONFUSED) beat hand-written Triton for most shapes. Write custom kernels only for: fused sequences cuDNN can't express, non-standard layouts, or shapes cuDNN heuristics mishandle.

**2. Padding in a separate kernel**
Example 2 materializes the padded input as a new buffer before calling convolution. This doubles HBM traffic for the input. Pass padding directly to `extern_kernels.convolution` and let cuDNN absorb it.

**3. XBLOCK too large for tiny tensors**
Setting `XBLOCK=256` with `num_warps=4` on `xnumel=16` wastes 240/256 threads. TorchInductor correctly uses `XBLOCK=16, num_warps=1` for Example 1. Always match block size to problem size.

**4. Ignoring `num_stages` for pipelining**
All examples use `num_stages=1`. For memory-bound kernels with predictable access patterns, `num_stages=3-4` enables software pipelining (prefetch next tile while computing current). At B200 memory bandwidth, this can recover 20–40% of latency for large tensors.

**5. FP32 accumulation when BF16 is sufficient**
These examples all operate in FP32. For inference, BF16 halves memory traffic and doubles tensor core throughput with negligible accuracy impact on most conv layers.

---

### Memory-bound vs compute-bound

The crossover depends on arithmetic intensity (FLOPs / bytes read):

| Shape | Intensity | Bottleneck |
|---|---|---|
| 1×1 conv, C≤64 | < 10 FLOPs/byte | Memory-bound (HBM) |
| 3×3 conv, C=64–256 | 10–50 FLOPs/byte | Balanced — tune tiling carefully |
| 3×3 conv, C≥512 | > 50 FLOPs/byte | Compute-bound (tensor cores) |
| Depthwise conv (groups=C) | 1–5 FLOPs/byte | Always memory-bound |
| 1×1 conv, large batch | > 100 FLOPs/byte | Compute-bound |

**Rule of thumb for B200:** With 8 TB/s HBM and 4 PFLOPS FP8, the ridge point is at ~500 FLOPs/byte. Most single-layer convolutions in practice are memory-bound unless batch size is very large (≥256) or channels are very wide (≥1024). Depthwise is always memory-bound — optimize for memory coalescing and vectorized loads (`tl.load` with aligned offsets) rather than tiling depth.

For memory-bound ops: maximize vectorization (load 4× fp32 per thread), minimize kernel launches (fuse aggressively), use `evict_last` on reused small tensors.

For compute-bound ops: maximize tensor core utilization (BF16/FP8, aligned `BLOCK_M/N/K`), use TMA for async loading, tune tile shapes to fill SMEM.
