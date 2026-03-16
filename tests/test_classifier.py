"""Tests for kernel problem classifier."""

from __future__ import annotations

from kernel_forge.knowledge.classifier import classify_kernel


class TestClassifyKernel:
	def test_square_matmul(self) -> None:
		r = classify_kernel("1_Square_matrix_multiplication_")
		assert r.kernel_class == "matmul"
		assert r.confidence > 0.5

	def test_standard_matmul(self) -> None:
		r = classify_kernel("2_Standard_matrix_multiplication_")
		assert r.kernel_class == "matmul"

	def test_batched_matmul(self) -> None:
		r = classify_kernel("3_Batched_matrix_multiplication")
		assert r.kernel_class == "matmul"

	def test_matmul_with_transposed(self) -> None:
		r = classify_kernel("16_Matmul_with_transposed_A")
		assert r.kernel_class == "matmul"

	def test_matrix_vector(self) -> None:
		r = classify_kernel("4_Matrix_vector_multiplication_")
		assert r.kernel_class == "matmul"

	def test_relu(self) -> None:
		r = classify_kernel("19_ReLU")
		assert r.kernel_class == "elementwise"

	def test_gelu(self) -> None:
		r = classify_kernel("26_GELU_")
		assert r.kernel_class == "elementwise"

	def test_sigmoid(self) -> None:
		r = classify_kernel("21_Sigmoid")
		assert r.kernel_class == "elementwise"

	def test_softmax(self) -> None:
		r = classify_kernel("23_Softmax")
		assert r.kernel_class == "reduction"

	def test_logsoftmax(self) -> None:
		r = classify_kernel("24_LogSoftmax")
		assert r.kernel_class == "reduction"

	def test_frobenius_norm(self) -> None:
		r = classify_kernel("37_FrobeniusNorm_")
		assert r.kernel_class == "reduction"

	def test_sum_reduction(self) -> None:
		r = classify_kernel("47_Sum_reduction_over_a_dimension")
		assert r.kernel_class == "reduction"

	def test_conv_1d(self) -> None:
		r = classify_kernel("67_conv_standard_1D")
		assert r.kernel_class == "conv"

	def test_conv_2d(self) -> None:
		r = classify_kernel(
			"70_conv_standard_2D__square_input__square_kernel"
		)
		assert r.kernel_class == "conv"

	def test_conv_transpose(self) -> None:
		r = classify_kernel(
			"80_conv_transpose_2D__square_input__square_kernel"
		)
		assert r.kernel_class == "conv"

	def test_batchnorm(self) -> None:
		r = classify_kernel("33_BatchNorm")
		assert r.kernel_class == "norm"

	def test_layernorm(self) -> None:
		r = classify_kernel("40_LayerNorm")
		assert r.kernel_class == "norm"

	def test_rmsnorm(self) -> None:
		r = classify_kernel("36_RMSNorm_")
		assert r.kernel_class == "norm"

	def test_hinge_loss(self) -> None:
		r = classify_kernel("100_HingeLoss")
		assert r.kernel_class == "loss"

	def test_cross_entropy(self) -> None:
		r = classify_kernel("95_CrossEntropyLoss")
		assert r.kernel_class == "loss"

	def test_max_pooling(self) -> None:
		r = classify_kernel("42_Max_Pooling_2D")
		assert r.kernel_class == "pooling"

	def test_attention(self) -> None:
		r = classify_kernel("44_ScaledDotProductAttention")
		assert r.kernel_class == "attention"

	def test_typical_bottleneck_matmul(self) -> None:
		r = classify_kernel("1_Square_matrix_multiplication_")
		assert r.typical_bottleneck == "compute_bound"

	def test_typical_bottleneck_elementwise(self) -> None:
		r = classify_kernel("19_ReLU")
		assert r.typical_bottleneck == "memory_bound"

	def test_typical_strategies_matmul(self) -> None:
		r = classify_kernel("1_Square_matrix_multiplication_")
		assert "tf32_tensor_cores" in r.typical_strategies

	def test_unknown_fallback(self) -> None:
		r = classify_kernel("999_SomeRandomOp")
		assert r.kernel_class == "unknown"
		assert r.confidence == 0.0
