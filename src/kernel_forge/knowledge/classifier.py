"""Kernel problem classifier.

Categorizes KernelBench problems into kernel classes based on name
and reference source analysis. Used for cross-problem generalization.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

KERNEL_CLASSES = {
	"matmul": {
		"name_patterns": [
			r"matrix_multiplication",
			r"[Mm]atmul",
			r"Matrix_vector",
			r"Matrix_scalar",
		],
		"source_patterns": [r"torch\.matmul", r"torch\.mm", r"torch\.bmm", r"@ "],
		"description": "Matrix multiplication variants",
		"typical_bottleneck": "compute_bound",
		"typical_strategies": [
			"tf32_tensor_cores",
			"tensor_core_utilization",
			"register_blocking",
		],
	},
	"elementwise": {
		"name_patterns": [
			r"ReLU", r"LeakyReLU", r"Sigmoid", r"Tanh", r"GELU",
			r"Swish", r"SELU", r"ELU", r"HardSigmoid", r"HardTanh",
			r"Softplus", r"Softsign", r"MinGPT",
		],
		"source_patterns": [
			r"torch\.relu", r"torch\.sigmoid", r"torch\.tanh",
			r"F\.relu", r"F\.gelu", r"F\.selu",
		],
		"description": "Elementwise activation functions",
		"typical_bottleneck": "memory_bound",
		"typical_strategies": [
			"kernel_fusion",
			"vectorized_loads",
			"torch_compile_integration",
		],
	},
	"reduction": {
		"name_patterns": [
			r"[Ss]oftmax", r"LogSoftmax", r"FrobeniusNorm",
			r"L1Norm", r"L2Norm",
			r"reduction", r"[Aa]rgmax", r"[Aa]rgmin",
			r"[Mm]ean", r"[Ss]um", r"[Mm]ax_", r"[Mm]in_",
			r"[Pp]roduct", r"cumsum", r"cumprod",
		],
		"source_patterns": [
			r"torch\.softmax", r"torch\.sum", r"torch\.mean",
			r"torch\.norm", r"\.sum\(", r"\.mean\(",
			r"F\.softmax", r"F\.log_softmax",
		],
		"description": "Reduction and normalization operations",
		"typical_bottleneck": "memory_bound",
		"typical_strategies": [
			"warp_level_primitives",
			"shared_memory_tiling",
			"kernel_fusion",
		],
	},
	"conv": {
		"name_patterns": [
			r"conv_",
		],
		"source_patterns": [
			r"nn\.Conv", r"F\.conv",
		],
		"description": "Convolution operations",
		"typical_bottleneck": "compute_bound",
		"typical_strategies": [
			"tensor_core_utilization",
			"torch_compile_integration",
			"shared_memory_tiling",
		],
	},
	"norm": {
		"name_patterns": [
			r"BatchNorm", r"LayerNorm", r"InstanceNorm",
			r"GroupNorm", r"RMSNorm",
		],
		"source_patterns": [
			r"nn\.BatchNorm", r"nn\.LayerNorm",
			r"nn\.GroupNorm", r"nn\.InstanceNorm",
		],
		"description": "Normalization layers",
		"typical_bottleneck": "memory_bound",
		"typical_strategies": [
			"kernel_fusion",
			"warp_level_primitives",
			"vectorized_loads",
		],
	},
	"loss": {
		"name_patterns": [
			r"Loss$", r"Loss_",
		],
		"source_patterns": [
			r"nn\.\w+Loss", r"F\.\w+loss",
			r"cross_entropy", r"mse_loss",
		],
		"description": "Loss functions",
		"typical_bottleneck": "memory_bound",
		"typical_strategies": [
			"kernel_fusion",
			"torch_compile_integration",
		],
	},
	"pooling": {
		"name_patterns": [
			r"[Pp]ooling",
		],
		"source_patterns": [
			r"nn\.\w+Pool", r"F\.\w+pool",
		],
		"description": "Pooling operations",
		"typical_bottleneck": "memory_bound",
		"typical_strategies": [
			"shared_memory_tiling",
			"vectorized_loads",
		],
	},
	"attention": {
		"name_patterns": [
			r"[Aa]ttention",
		],
		"source_patterns": [
			r"scaled_dot_product", r"attention",
		],
		"description": "Attention mechanisms",
		"typical_bottleneck": "memory_bound",
		"typical_strategies": [
			"flash_attention_tiling",
			"kernel_fusion",
			"tensor_core_utilization",
		],
	},
}


@dataclass
class KernelClassification:
	"""Classification result for a kernel problem."""

	kernel_class: str
	confidence: float  # 0.0 to 1.0
	typical_bottleneck: str
	typical_strategies: list[str]
	description: str


def classify_kernel(
	problem_name: str,
	reference_source: str = "",
) -> KernelClassification:
	"""Classify a kernel problem into a kernel class.

	Uses name pattern matching first (high confidence), falls back
	to source code analysis (medium confidence).
	"""
	# Strip number prefix: "1_Square_matrix_multiplication_" -> "Square_matrix_multiplication_"
	clean_name = re.sub(r"^\d+_", "", problem_name)

	best_class = "unknown"
	best_score = 0.0

	for cls_name, cls_info in KERNEL_CLASSES.items():
		score = 0.0

		# Name matching (high weight)
		for pattern in cls_info["name_patterns"]:
			match = re.search(pattern, clean_name)
			if match:
				# Longer pattern matches are more specific -> higher score
				score += 2.0 + len(match.group(0)) / 10.0
				break

		# Source matching (medium weight)
		if reference_source:
			for pattern in cls_info["source_patterns"]:
				if re.search(pattern, reference_source):
					score += 1.0
					break

		if score > best_score:
			best_score = score
			best_class = cls_name

	if best_class == "unknown" or best_score == 0:
		return KernelClassification(
			kernel_class="unknown",
			confidence=0.0,
			typical_bottleneck="mixed",
			typical_strategies=["torch_compile_integration"],
			description="Unclassified kernel",
		)

	info = KERNEL_CLASSES[best_class]
	confidence = min(best_score / 3.0, 1.0)

	return KernelClassification(
		kernel_class=best_class,
		confidence=confidence,
		typical_bottleneck=info["typical_bottleneck"],
		typical_strategies=info["typical_strategies"],
		description=info["description"],
	)
