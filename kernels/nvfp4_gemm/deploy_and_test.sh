#!/bin/bash
# Deploy submission versions to B200 and test them
set -e

B200="b200-node"
REMOTE_DIR="~/kernel-forge-workspace/gpumode_nvfp4_gemm"
LOCAL_DIR="$(dirname "$0")"
GPU_ENV="CUDA_VISIBLE_DEVICES=3 CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:\$PATH"

echo "=== Deploying files ==="
scp "$LOCAL_DIR/benchmark.py" "$B200:$REMOTE_DIR/benchmark.py"

# Test each version
for v in v1_cached_scaledmm v2_triton v4_triton_cublas; do
    echo ""
    echo "=== Testing submission_${v} ==="
    scp "$LOCAL_DIR/submission_${v}.py" "$B200:$REMOTE_DIR/submission.py"
    ssh "$B200" "cd $REMOTE_DIR && $GPU_ENV python3 benchmark.py submission" 2>&1 || echo "FAILED: $v"
    echo ""
done
