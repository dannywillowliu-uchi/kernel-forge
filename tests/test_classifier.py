"""Tests for kernel trait analysis."""

from __future__ import annotations

from kernel_forge.knowledge.classifier import KernelTraits, analyze_traits


class TestAnalyzeTraits:
	def test_matmul_ops(self) -> None:
		t = analyze_traits("1_Square_matrix_multiplication_")
		assert "matmul" in t.dominant_ops

	def test_matmul_compute_bound(self) -> None:
		t = analyze_traits("1_Square_matrix_multiplication_")
		assert t.estimated_bottleneck == "compute_bound"

	def test_matmul_data_reuse(self) -> None:
		t = analyze_traits("1_Square_matrix_multiplication_")
		assert t.has_data_reuse is True

	def test_relu_elementwise(self) -> None:
		t = analyze_traits("19_ReLU")
		assert "elementwise" in t.dominant_ops

	def test_relu_memory_bound(self) -> None:
		t = analyze_traits("19_ReLU")
		assert t.estimated_bottleneck == "memory_bound"

	def test_softmax_reduction(self) -> None:
		t = analyze_traits("23_Softmax")
		assert "softmax" in t.dominant_ops or "reduction" in t.dominant_ops

	def test_conv_ops(self) -> None:
		t = analyze_traits("67_conv_standard_1D")
		assert "conv" in t.dominant_ops

	def test_attention_composite(self) -> None:
		t = analyze_traits(
			"44_ScaledDotProductAttention",
			"torch.nn.functional.scaled_dot_product_attention(q, k, v)",
		)
		assert "attention" in t.dominant_ops

	def test_norm_ops(self) -> None:
		t = analyze_traits(
			"33_BatchNorm",
			"self.bn = nn.BatchNorm2d(num_features)",
		)
		assert "norm" in t.dominant_ops

	def test_loss_ops(self) -> None:
		t = analyze_traits("100_HingeLoss")
		assert "loss" in t.dominant_ops

	def test_shape_analysis_large(self) -> None:
		t = analyze_traits(
			"test",
			"",
			{"input_0": [4096, 4096]},
		)
		assert t.shape_category == "large"
		assert t.shape_aspect == "square"

	def test_shape_analysis_small(self) -> None:
		t = analyze_traits(
			"test",
			"",
			{"input_0": [32, 32]},
		)
		assert t.shape_category == "small"

	def test_composite_detection(self) -> None:
		# Source has both matmul and softmax
		t = analyze_traits(
			"test_fused_attention",
			"x = torch.matmul(q, k); x = F.softmax(x); out = torch.matmul(x, v)",
		)
		assert t.is_composite is True
		assert len(t.dominant_ops) >= 2

	def test_suggestions_advisory(self) -> None:
		t = analyze_traits("1_Square_matrix_multiplication_")
		assert len(t.suggested_strategies) > 0
		# These are suggestions, they exist but don't control anything

	def test_unknown_problem(self) -> None:
		t = analyze_traits("999_SomeRandomThing")
		assert t.confidence < 0.5


class TestTraitSimilarity:
	def test_identical_traits(self) -> None:
		t = KernelTraits(
			dominant_ops=["matmul"],
			estimated_bottleneck="compute_bound",
			has_data_reuse=True,
			shape_category="large",
		)
		assert t.similarity(t) > 0.9

	def test_same_ops_different_shape(self) -> None:
		t1 = KernelTraits(
			dominant_ops=["matmul"],
			estimated_bottleneck="compute_bound",
			has_data_reuse=True,
			shape_category="large",
		)
		t2 = KernelTraits(
			dominant_ops=["matmul"],
			estimated_bottleneck="compute_bound",
			has_data_reuse=True,
			shape_category="small",
		)
		sim = t1.similarity(t2)
		assert sim > 0.7  # mostly similar

	def test_different_ops(self) -> None:
		t1 = KernelTraits(
			dominant_ops=["matmul"],
			estimated_bottleneck="compute_bound",
		)
		t2 = KernelTraits(
			dominant_ops=["elementwise"],
			estimated_bottleneck="memory_bound",
		)
		sim = t1.similarity(t2)
		assert sim < 0.3  # quite different

	def test_partial_op_overlap(self) -> None:
		t1 = KernelTraits(
			dominant_ops=["matmul", "softmax"],
		)
		t2 = KernelTraits(
			dominant_ops=["matmul", "elementwise"],
		)
		sim = t1.similarity(t2)
		# Partial overlap -- should be moderate
		assert 0.2 < sim < 0.8
