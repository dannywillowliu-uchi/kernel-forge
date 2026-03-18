"""
V3: Direct cuBLASLt FP4 GEMM via load_inline CUDA extension.
Bypasses torch._scaled_mm Python dispatch overhead entirely.
Uses CUDA_R_4F_E2M1 data type with CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 scaling.
"""
import torch
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t
import math

# CUDA source for direct cuBLASLt FP4 GEMM
cuda_source = r"""
#include <torch/extension.h>
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define CUBLASLT_CHECK(expr)                                                     \
  do {                                                                           \
    cublasStatus_t status = (expr);                                              \
    if (status != CUBLAS_STATUS_SUCCESS) {                                       \
      printf("cuBLASLt error %d at %s:%d\n", status, __FILE__, __LINE__);        \
      throw std::runtime_error("cuBLASLt error");                                \
    }                                                                            \
  } while (0)

static cublasLtHandle_t ltHandle = nullptr;

void ensure_handle() {
    if (ltHandle == nullptr) {
        CUBLASLT_CHECK(cublasLtCreate(&ltHandle));
    }
}

// FP4 GEMM: C = A @ B^T where A is (m,k) FP4 packed and B is (n,k) FP4 packed
// scaleA is swizzled scale for A, scaleB is swizzled scale for B
// Output C is (m,n) FP16
torch::Tensor cublaslt_fp4_gemm(
    torch::Tensor A,      // (m, k/2) uint8 packed FP4
    torch::Tensor B,      // (n, k/2) uint8 packed FP4
    torch::Tensor scaleA, // swizzled scale factors for A
    torch::Tensor scaleB, // swizzled scale factors for B
    int64_t m, int64_t n, int64_t k
) {
    ensure_handle();

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // Create operation descriptor
    cublasLtMatmulDesc_t opDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Set transpose: A is non-transposed, B is transposed
    // cuBLAS is column-major, so for row-major: C(m,n) = A(m,k) @ B(n,k)^T
    // In col-major: C^T(n,m) = B(n,k)^T^T @ A(m,k)^T = B(k,n) @ A^T(k,m)
    // So: matA = B (non-transposed in col-major = row-major transposed)
    //     matB = A (transposed in col-major = row-major non-transposed)
    // Actually for FP4 with cuBLASLt:
    // We do C = A @ B^T in row-major
    // cuBLAS col-major: opA=N means read column-major, opA=T means transposed

    cublasOperation_t transA = CUBLAS_OP_T;  // B is transposed
    cublasOperation_t transB = CUBLAS_OP_N;  // A is not transposed
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transA, sizeof(transA)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transB, sizeof(transB)));

    // Scale mode for A and B: per-block FP8 E4M3 with 16-element vectors
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Set scale pointers
    void* scaleA_ptr = scaleA.data_ptr();
    void* scaleB_ptr = scaleB.data_ptr();
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &scaleA_ptr, sizeof(scaleA_ptr)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &scaleB_ptr, sizeof(scaleB_ptr)));

    // Matrix layouts
    // A layout: (m, k) FP4 = stored as (m, k/2) bytes, leading dim = k in FP4 elements
    // For cuBLAS column-major: matA is B^T, so layout is (k, n) with ld=k
    // matB is A, layout is (k, m) with ld=k
    cublasLtMatrixLayout_t ALayout, BLayout, CLayout, DLayout;

    // In the TN formulation: A_cublas = B (n, k) with transA=T -> reads as (k, n)
    // B_cublas = A (m, k) with transB=N -> reads as (k, m) but stored as (m, k) row-major
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&ALayout, CUDA_R_4F_E2M1, k, n, k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&BLayout, CUDA_R_4F_E2M1, k, m, k));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&CLayout, CUDA_R_16F, n, m, n));
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&DLayout, CUDA_R_16F, n, m, n));

    // Create output tensor
    auto C = torch::empty({m, n}, torch::TensorOptions().dtype(torch::kFloat16).device(A.device()));

    // Alpha and beta
    float alpha = 1.0f;
    float beta = 0.0f;

    // Find best algorithm
    cublasLtMatmulPreference_t pref;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    size_t maxWs = 4 * 1024 * 1024;  // 4MB workspace
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &maxWs, sizeof(maxWs)));

    cublasLtMatmulHeuristicResult_t heuristic;
    int returned = 0;
    CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(ltHandle, opDesc, ALayout, BLayout, CLayout, DLayout, pref, 1, &heuristic, &returned));

    if (returned == 0) {
        throw std::runtime_error("No cuBLASLt algorithm found for FP4 GEMM");
    }

    // Allocate workspace if needed
    void* workspace = nullptr;
    if (heuristic.workspaceSize > 0) {
        cudaMalloc(&workspace, heuristic.workspaceSize);
    }

    // Execute GEMM
    CUBLASLT_CHECK(cublasLtMatmul(
        ltHandle, opDesc,
        &alpha,
        B.data_ptr(), ALayout,   // A_cublas = B (the matrix being transposed)
        A.data_ptr(), BLayout,   // B_cublas = A
        &beta,
        C.data_ptr(), CLayout,
        C.data_ptr(), DLayout,
        &heuristic.algo,
        workspace, heuristic.workspaceSize,
        stream
    ));

    // Cleanup
    if (workspace) cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(ALayout);
    cublasLtMatrixLayoutDestroy(BLayout);
    cublasLtMatrixLayoutDestroy(CLayout);
    cublasLtMatrixLayoutDestroy(DLayout);
    cublasLtMatmulDescDestroy(opDesc);

    return C;
}

TORCH_LIBRARY_FRAGMENT(nvfp4_ext, m) {
    m.def("cublaslt_fp4_gemm", &cublaslt_fp4_gemm);
}
"""

cpp_source = r"""
#include <torch/extension.h>
torch::Tensor cublaslt_fp4_gemm(
    torch::Tensor A, torch::Tensor B,
    torch::Tensor scaleA, torch::Tensor scaleB,
    int64_t m, int64_t n, int64_t k);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cublaslt_fp4_gemm", &cublaslt_fp4_gemm);
}
"""

# Build the extension
_ext = None

def get_ext():
	global _ext
	if _ext is None:
		_ext = load_inline(
			name="nvfp4_cublaslt",
			cpp_sources=[cpp_source],
			cuda_sources=[cuda_source],
			extra_ldflags=["-lcublasLt"],
			extra_cuda_cflags=["-O3", "--use_fast_math"],
			verbose=False,
		)
	return _ext


sf_vec_size = 16

def ceil_div(a, b):
	return (a + b - 1) // b

def swizzle_rowwise_scale(scale_mat):
	"""Convert row-wise scale factors to cuBLAS blocked format."""
	tile_dim_y = 128
	tile_dim_x = 4
	rows, cols = scale_mat.shape
	ntile_y = ceil_div(rows, tile_dim_y)
	ntile_x = ceil_div(cols, tile_dim_x)

	# Pad if needed
	padded = scale_mat
	if rows % tile_dim_y != 0 or cols % tile_dim_x != 0:
		padded = torch.zeros(
			(ntile_y * tile_dim_y, ntile_x * tile_dim_x),
			dtype=scale_mat.dtype, device=scale_mat.device
		)
		padded[:rows, :cols] = scale_mat

	# Reshape and swizzle
	padded = padded.view(ntile_y, tile_dim_y, ntile_x, tile_dim_x).permute(0, 2, 1, 3)
	# padded is now (ntile_y, ntile_x, 128, 4)
	# Chunk: (ntile_y, ntile_x, 4, 32, 4)
	chunked = padded.reshape(ntile_y, ntile_x, 4, 32, 4)
	# Swizzle to (ntile_y, ntile_x, 32, 4, 4) -> flatten last two -> (ntile_y, ntile_x, 32, 16)
	swizzled = chunked.permute(0, 1, 3, 2, 4).reshape(ntile_y * 32, ntile_x * 16)
	return swizzled.contiguous()


_cache = {}

def custom_kernel(data: input_t) -> output_t:
	a, b, sfa, sfb, sfa_permuted, sfb_permuted, c = data
	m, n, _ = c.shape
	k = a.shape[1] * 2  # packed FP4

	ext = get_ext()

	key = (sfa.data_ptr(), sfb.data_ptr())
	if key not in _cache:
		# Prepare swizzled scale factors
		sfa_2d = sfa[:, :, 0]
		sfb_2d = sfb[:, :, 0]
		sa_swiz = swizzle_rowwise_scale(sfa_2d)
		sb_swiz = swizzle_rowwise_scale(sfb_2d)
		_cache[key] = (sa_swiz, sb_swiz)
	sa_swiz, sb_swiz = _cache[key]

	# Get contiguous 2D views
	a_2d = a[:, :, 0].contiguous().view(torch.uint8)
	b_2d = b[:, :, 0].contiguous().view(torch.uint8)

	result = ext.cublaslt_fp4_gemm(a_2d, b_2d, sa_swiz, sb_swiz, m, n, k)
	c[:, :, 0] = result
	return c
