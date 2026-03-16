#!/usr/bin/env python3
"""Seed the experience store with results from completed batches."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kernel_forge.knowledge.classifier import analyze_traits
from kernel_forge.knowledge.experience import ExperienceRecord, ExperienceStore
from kernel_forge.harness.kernelbench import KernelBenchAdapter

RESULTS = [
	# Batch 1 (problems 1-5)
	{"problem": "1_Square_matrix_multiplication_", "speedup": 12.95, "baseline_ms": 2.16, "optimized_ms": 0.17, "approach": "TF32 tensor cores"},
	{"problem": "2_Standard_matrix_multiplication_", "speedup": 13.43, "baseline_ms": 2.16, "optimized_ms": 0.16, "approach": "TF32 tensor cores"},
	{"problem": "3_Batched_matrix_multiplication", "speedup": 10.81, "baseline_ms": 4.29, "optimized_ms": 0.40, "approach": "TF32 tensor cores"},
	{"problem": "4_Matrix_vector_multiplication_", "speedup": 1.00, "baseline_ms": 1.56, "optimized_ms": 1.56, "approach": "Memory-bound baseline optimal"},
	{"problem": "5_Matrix_scalar_multiplication", "speedup": 1.00, "baseline_ms": 1.24, "optimized_ms": 1.24, "approach": "Memory-bound baseline optimal"},
	# Batch 2 (problems 6-10)
	{"problem": "6_Matmul_with_large_K_dimension_", "speedup": 3.18, "baseline_ms": 1.44, "optimized_ms": 0.45, "approach": "TF32"},
	{"problem": "7_Matmul_with_small_K_dimension_", "speedup": 4.42, "baseline_ms": 3.31, "optimized_ms": 0.75, "approach": "TF32"},
	{"problem": "8_Matmul_with_irregular_shapes_", "speedup": 8.86, "baseline_ms": 4.53, "optimized_ms": 0.51, "approach": "TF32 + pad to 128"},
	{"problem": "9_Tall_skinny_matrix_multiplication_", "speedup": 2.82, "baseline_ms": 1.97, "optimized_ms": 0.70, "approach": "TF32"},
	{"problem": "10_3D_tensor_matrix_multiplication", "speedup": 12.22, "baseline_ms": 0.86, "optimized_ms": 0.07, "approach": "TF32"},
	# Batch 3 (problems 11-18)
	{"problem": "11_4D_tensor_matrix_multiplication", "speedup": 8.13, "baseline_ms": 7.13, "optimized_ms": 0.88, "approach": "Reshape 4D to 2D + TF32"},
	{"problem": "12_Matmul_with_diagonal_matrices_", "speedup": 2.28, "baseline_ms": 0.05, "optimized_ms": 0.02, "approach": "Triton fused diag*B"},
	{"problem": "13_Matmul_for_symmetric_matrices", "speedup": 12.95, "baseline_ms": 2.16, "optimized_ms": 0.17, "approach": "TF32"},
	{"problem": "14_Matmul_for_upper_triangular_matrices", "speedup": 10.76, "baseline_ms": 2.21, "optimized_ms": 0.21, "approach": "TF32 + triu"},
	{"problem": "15_Matmul_for_lower_triangular_matrices", "speedup": 10.68, "baseline_ms": 2.21, "optimized_ms": 0.21, "approach": "TF32 + tril"},
	{"problem": "16_Matmul_with_transposed_A", "speedup": 15.30, "baseline_ms": 2.46, "optimized_ms": 0.16, "approach": "TF32, removed .contiguous()"},
	{"problem": "17_Matmul_with_transposed_B", "speedup": 13.83, "baseline_ms": 2.23, "optimized_ms": 0.16, "approach": "TF32, removed .contiguous()"},
	{"problem": "18_Matmul_with_transposed_both", "speedup": 15.18, "baseline_ms": 2.45, "optimized_ms": 0.16, "approach": "TF32, removed .contiguous()"},
	# Batch 4 (problems 19-26) - re-assessed with batch 5 findings
	{"problem": "19_ReLU", "speedup": 1.00, "baseline_ms": 4.03, "optimized_ms": 4.03, "approach": "Bandwidth ceiling single op"},
	{"problem": "20_LeakyReLU", "speedup": 1.00, "baseline_ms": 4.04, "optimized_ms": 4.04, "approach": "Bandwidth ceiling single op"},
	{"problem": "21_Sigmoid", "speedup": 1.00, "baseline_ms": 4.01, "optimized_ms": 4.01, "approach": "Bandwidth ceiling single op"},
	{"problem": "22_Tanh", "speedup": 1.01, "baseline_ms": 4.03, "optimized_ms": 3.99, "approach": "Bandwidth ceiling single op"},
	{"problem": "23_Softmax", "speedup": 1.00, "baseline_ms": 8.65, "optimized_ms": 8.65, "approach": "cunn_SoftMaxForward already optimal"},
	{"problem": "24_LogSoftmax", "speedup": 1.00, "baseline_ms": 8.45, "optimized_ms": 8.45, "approach": "cunn_SoftMaxForward already optimal"},
	{"problem": "25_Swish", "speedup": 2.50, "baseline_ms": 10.00, "optimized_ms": 4.01, "approach": "F.silu fuses 2 kernels into 1"},
	{"problem": "26_GELU_", "speedup": 1.00, "baseline_ms": 4.10, "optimized_ms": 4.10, "approach": "Single native kernel at bandwidth ceiling"},
	# Batch 5 (problems 27-36)
	{"problem": "27_SELU_", "speedup": 1.51, "baseline_ms": 6.04, "optimized_ms": 4.00, "approach": "Triton fused elementwise"},
	{"problem": "28_HardSigmoid", "speedup": 1.51, "baseline_ms": 6.04, "optimized_ms": 3.99, "approach": "Triton fused elementwise"},
	{"problem": "29_Softplus", "speedup": 2.21, "baseline_ms": 8.92, "optimized_ms": 4.03, "approach": "Triton fused with threshold"},
	{"problem": "30_Softsign", "speedup": 5.24, "baseline_ms": 20.93, "optimized_ms": 4.00, "approach": "Triton fused 3 kernels into 1"},
	{"problem": "31_ELU", "speedup": 1.51, "baseline_ms": 6.03, "optimized_ms": 3.99, "approach": "Triton fused elementwise"},
	{"problem": "32_HardTanh", "speedup": 1.51, "baseline_ms": 6.04, "optimized_ms": 3.99, "approach": "Triton fused elementwise"},
	{"problem": "33_BatchNorm", "speedup": 2.52, "baseline_ms": 7.25, "optimized_ms": 2.88, "approach": "Triton precomputed scale+shift"},
	{"problem": "34_InstanceNorm", "speedup": 3.29, "baseline_ms": 23.56, "optimized_ms": 7.15, "approach": "Triton two-pass online stats"},
	{"problem": "35_GroupNorm_", "speedup": 4.00, "baseline_ms": 34.26, "optimized_ms": 8.58, "approach": "Triton two-pass per-group"},
	{"problem": "36_RMSNorm_", "speedup": 5.98, "baseline_ms": 28.92, "optimized_ms": 4.84, "approach": "Triton single-pass 2D tile"},
	# Attention (separate run)
	{"problem": "97_ScaledDotProductAttention", "speedup": 16.10, "baseline_ms": 37.10, "optimized_ms": 2.30, "approach": "Manual decomposition + sm100 cuBLAS + TF32"},
]


def main() -> None:
	adapter = KernelBenchAdapter(Path("knowledge/kernelbench"))
	experience = ExperienceStore(Path("knowledge/experience"))

	for r in RESULTS:
		name = r["problem"]
		problem = adapter.get_problem(name)
		traits = analyze_traits(
			name,
			problem.reference_source if problem else "",
			problem.input_shapes if problem else {},
		)
		experience.record(ExperienceRecord(
			problem_name=name,
			dominant_ops=traits.dominant_ops,
			strategy_name=r["approach"][:50],
			approach_notes=r["approach"],
			outcome="success" if r["speedup"] > 1.0 else "no_improvement",
			speedup=r["speedup"],
			baseline_ms=r["baseline_ms"],
			optimized_ms=r["optimized_ms"],
			bottleneck_type=traits.estimated_bottleneck,
			roofline_utilization_pct=0.0,
			root_cause=r["approach"],
			has_data_reuse=traits.has_data_reuse,
			shape_category=traits.shape_category,
			estimated_bottleneck=traits.estimated_bottleneck,
			input_shapes=problem.input_shapes if problem else {},
		))
		status = "+" if r["speedup"] > 1.0 else "="
		print(f"  {status} {name}: {r['speedup']:.2f}x ({r['approach'][:40]})")

	print(f"\nSeeded {len(RESULTS)} results into experience store")


if __name__ == "__main__":
	main()
