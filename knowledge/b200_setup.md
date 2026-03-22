# B200 Node Setup

Tools needed for kernel optimization on NVIDIA B200.

## System Requirements
- NVIDIA B200 GPU
- CUDA 12.8+
- Python 3.12+
- NVIDIA driver 580+

## Python Packages (user-local install)

```bash
# Core (likely already installed)
pip install --user torch triton

# Kernel writing
pip install --user cupy-cuda12x          # Low-overhead kernel dispatch
pip install --user liger-kernel           # Fused LN, cross-entropy, SwiGLU

# NVIDIA libraries
pip install --user cuequivariance-torch   # Equivariant ops API
pip install --user cuequivariance-ops-torch-cu12  # Fused CUDA kernels for equivariant ops

# Profiling (usually system-installed)
# ncu, nsys come with CUDA toolkit
```

## Verification

```python
import torch, triton, cupy, liger_kernel
import cuequivariance_torch, cuequivariance_ops_torch
x = torch.randn(100, 100, device="cuda")
print(x @ x.T)  # GPU works
```

## Usage Notes
- Use `--user` flag for pip installs on shared nodes
- Never use `--privileged` Docker or lock GPU clocks on shared nodes
- Use `CUDA_VISIBLE_DEVICES=N` to target your assigned GPU
