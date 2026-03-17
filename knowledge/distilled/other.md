## OTHER Optimization Guide

### What makes fast other kernels fast

**1. Extern kernels for heavy compute, Triton for elementwise glue**
All examples delegate matmul (`mm`, `addmm`) and convolution to `extern_kernels` (cuBLAS/cuDNN), then use Triton *only* for fusing the lightweight ops (activations, bias add, residual add). This is correct because cuBLAS has hand-tuned Tensor Core paths that no Triton kernel at production shapes will beat for dense GEMM. The Triton layer exists to eliminate the extra memory roundtrips between ops.

**2. Elementwise fusion collapses memory traffic**
`fused_add_relu_threshold_backward` (SimpleBlock, BottleneckBlock) computes residual add + forward ReLU + backward mask in a single pass. Without fusion, this is 3 kernel launches and 3×N reads + 2×N writes. Fused: 1 launch, 2N reads + 2N writes. On B200 with 8 TB/s HBM, the bandwidth savings directly translate to speedup for these memory-bound ops.

**3. `evict_last` for broadcast loads**
When a small tensor (bias vector of size C) is broadcast across a large tensor (B×C×H×W), loading the same value N times would trash L1. `eviction_policy='evict_last'` pins those elements in cache. See MLP's `fused_gelu_0` — the bias (`in_ptr1`, size 4) is loaded with `evict_last` while the main buffer is streamed normally.

---

### Common code patterns

**Block sizes:**
- Tiny elementwise (≤64 elements): `XBLOCK=16`, 1 warp
- Small elementwise (64–512 elements): `XBLOCK=128`, `num_warps=4`
- Medium pointwise (512–4096): `XBLOCK=512`, `num_warps=8`
- Large pointwise: `XBLOCK=1024`, `num_warps=8`, `num_stages=1`

**In-place activation pattern** (zero extra buffer):
```python
# in_out_ptr0: reads and writes same buffer
tmp0 = tl.load(in_out_ptr0 + x0, xmask)
tmp1 = triton_helpers.maximum(tl.full([1], 0, tl.int32), tmp0)
tl.store(in_out_ptr0 + x0, tmp1, xmask)
```

**Dual output forward+backward in one pass:**
```python
# Compute relu output AND backward mask simultaneously
tmp4 = triton_helpers.maximum(tmp3, tmp2)   # forward: relu(x+residual)
tmp6 = tmp4 <= 0.0                           # backward: dead neuron mask
tl.store(out_ptr0 + x2, tmp4, xmask)        # save activation
tl.store(out_ptr1 + x2, tmp6, xmask)        # save grad mask
```
Eliminates a separate backward kernel launch during training.

**Conditional load for gather/cat ops (BehlerAngular pattern):**
```python
tmp4 = tmp0 < tmp3                    # branch on element index
tmp5 = tl.load(ptr + x1, tmp4 & mask, other=0.0)   # predicated load
```
Avoids warp divergence by using predicated loads instead of if/else branches.

---

### B200/Blackwell considerations

**Tensor Cores:** Only activated by `extern_kernels.mm`/`addmm`/`convolution`. For the Triton elementwise layer, Tensor Cores are irrelevant — these are pure memory-bandwidth ops. Don't try to use Tensor Cores for activation kernels.

**228KB SMEM:** Irrelevant for pure elementwise kernels (no tiling needed). Relevant only if you're reimplementing the matmul yourself — in that case tile to fit A and B tiles in SMEM. But default: use cuBLAS, not hand-rolled SMEM tiling.

**126MB L2:** "Other" ops at production batch sizes (B≥32, large hidden dims) will fit activation tensors in L2 across kernel launches. Design kernel launch sequences so consumers of a buffer run immediately after producers — L2 residency degrades with unrelated work between launches.

**NVLink/SM count:** B200 has 132 SMs. Kernels with fewer than ~132 blocks are underutilizing the GPU. At batch=4, hidden=4 (toy sizes in these examples), you're launching 1–4 blocks. At production: ensure `grid = ceil(N / XBLOCK)` yields ≥528 blocks (4× SM count) for full occupancy.

---

### Typical speedup range

| Pattern | Baseline (PyTorch eager) | Fused Triton | Notes |
|---|---|---|---|
| Bias + activation (ReLU/GELU/ELU) | 1.0× | 1.8–2.5× | Eliminates 1 read+write roundtrip |
| Add + ReLU + backward mask | 1.0× | 2.5–3.5× | 3-op → 1-pass |
| Cat + transform (BehlerAngular) | 1.0× | 1.5–2.0× | Avoids intermediate concat buffer |
| Full block (Conv + BN + ReLU + residual) | 1.0× | 1.2–1.6× | cuDNN handles conv; gains from elementwise tail |

At toy sizes (≤512 elements), measured speedup is unreliable due to launch overhead dominating. Real gains appear at B≥32.

---

### Common mistakes

**1. Reimplementing matmul in Triton** — `extern_kernels.mm` already dispatches to cuBLAS with Tensor Core paths tuned for B200. A custom Triton GEMM will be 0.5–0.8× the speed unless you're doing something exotic (e.g., fused attention with custom masking). Don't replace it.

**2. Separate kernels for chained elementwise ops** — `relu(add(conv(x), residual))` as three separate launches is 3× the memory traffic. The pattern in all examples is to fuse them into one Triton kernel.

**3. Forgetting `evict_last` for broadcast dimensions** — Loading a bias of size C into a (B×C×H×W) loop without pinning it causes repeated L1 evictions. Always annotate broadcast dimensions.

**4. Wrong XBLOCK for small tensors** — Using XBLOCK=1024 for a 16-element kernel launches 1/64th of a warp, wasting SM resources. Match XBLOCK to problem size: use the smallest power-of-2 ≥ problem size, capped at warp size (32) for tiny problems.

**5. Not fusing backward mask computation** — Computing the ReLU dead-neuron mask in a separate backward pass re-reads the activation buffer. The `threshold_backward` fusion pattern (BottleneckBlock, SimpleBlock) does this in the forward pass at zero extra cost.

---

### Memory-bound vs compute-bound

| Regime | Bottleneck | Crossover |
|---|---|---|
| Activation functions (ReLU, GELU, ELU) | Always memory-bound | N/A |
| Bias add, residual add, cat | Always memory-bound | N/A |
| Small matmul (M,N,K ≤ 256) | Memory-bound (GEMM overhead) | — |
| Medium matmul (M,N,K ~ 512–1024) | Transitioning | Depends on batch |
| Large matmul (M,N,K ≥ 2048) | Compute-bound (Tensor Cores) | — |
| Convolution (large spatial) | Compute-bound via cuDNN | Kernel size 3×3+ at C≥64 |

**Rule of thumb for "other" ops:** If the op is not a matmul or convolution, assume memory-bound and optimize for bandwidth: minimize reads/writes through fusion, use `evict_last` for broadcasts, keep kernel count low. Only investigate compute-boundedness if profiling shows >70% math throughput utilization.
