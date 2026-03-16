#!/bin/bash
# Setup kernel-forge workspace on B200 node
# Usage: bash scripts/setup_remote.sh

set -e

HOST="b200-node"
WORKSPACE="~/kernel-forge-workspace"
GPU_ID=2

echo "=== Setting up kernel-forge workspace on $HOST ==="

# Create directory structure
ssh $HOST "mkdir -p $WORKSPACE/{kernels,results,harness}"

# Clone KernelBench if not present
ssh $HOST "cd $WORKSPACE/harness && [ -d KernelBench ] || git clone https://github.com/ScalingIntelligence/KernelBench.git"

# Verify GPU access
echo "=== Verifying GPU $GPU_ID ==="
ssh $HOST "CUDA_VISIBLE_DEVICES=$GPU_ID python3 -c \"
import torch
print(f'Device: {torch.cuda.get_device_name(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
a = torch.randn(256, 256, device=\"cuda\")
b = torch.matmul(a, a.T)
print(f'Compute test passed: {b.shape}')
\""

# Verify KernelBench problems
echo "=== KernelBench problems ==="
ssh $HOST "echo 'Level 1:' && ls $WORKSPACE/harness/KernelBench/KernelBench/level1/ | wc -l && \
           echo 'Level 2:' && ls $WORKSPACE/harness/KernelBench/KernelBench/level2/ | wc -l && \
           echo 'Level 3:' && ls $WORKSPACE/harness/KernelBench/KernelBench/level3/ | wc -l"

echo "=== Setup complete ==="
