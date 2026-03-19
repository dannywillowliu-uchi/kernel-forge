# Pattern: Triton tl.dot_scaled for FP4/FP8 block-scaled GEMM
# Available in Triton 3.6 on B200 (sm100)
# Use for: NVFP4 GEMM, FP8 GEMM with block scaling
#
# Signature:
#   tl.dot_scaled(
#       lhs,           # left matrix tile
#       lhs_scale,     # scale factors for lhs
#       lhs_format,    # "e2m1" for FP4, "e4m3" for FP8
#       rhs,           # right matrix tile
#       rhs_scale,     # scale factors for rhs
#       rhs_format,    # "e2m1" for FP4, "e4m3" for FP8
#       acc=None,      # accumulator (for chaining)
#       fast_math=False,
#       out_dtype=tl.float32,
#   )
#
# NOTE: TensorDescriptor NOT available in Triton 3.6
# Use tl.load/tl.store directly instead

import triton
import triton.language as tl


@triton.jit
def fp4_gemm_kernel(
	A_ptr, B_ptr,           # FP4 data pointers
	SA_ptr, SB_ptr,         # FP8 scale factor pointers
	C_ptr,                  # FP16 output pointer
	M, N, K,
	stride_am, stride_ak,  # A strides
	stride_bn, stride_bk,  # B strides (B is NxK)
	stride_cm, stride_cn,  # C strides
	BLOCK_M: tl.constexpr,
	BLOCK_N: tl.constexpr,
	BLOCK_K: tl.constexpr,
):
	pid_m = tl.program_id(0)
	pid_n = tl.program_id(1)

	# Tile offsets
	offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
	offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

	# Accumulator
	acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

	for k in range(0, K, BLOCK_K):
		offs_k = k + tl.arange(0, BLOCK_K)

		# Load FP4 tiles
		a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
		b = tl.load(B_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk)

		# Load scale factors (one per 16 elements along K)
		# Scale shape: [M, K//16] and [N, K//16]
		k_scale = k // 16
		sa = tl.load(SA_ptr + offs_m * (K // 16) + k_scale)
		sb = tl.load(SB_ptr + offs_n * (K // 16) + k_scale)

		# FP4 scaled dot product
		acc = tl.dot_scaled(
			a, sa, "e2m1",
			b, sb, "e2m1",
			acc=acc,
			out_dtype=tl.float32,
		)

	# Store result as FP16
	c = acc.to(tl.float16)
	tl.store(C_ptr + offs_m[:, None] * stride_cn + offs_n[None, :] * stride_cn, c)


# Recommended block sizes for B200:
# BLOCK_M=128, BLOCK_N=256, BLOCK_K=256
# These match the tensor core tile alignment
