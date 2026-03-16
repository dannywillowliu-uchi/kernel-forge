"""Kernel trait analysis.

Produces a trait vector describing optimization-relevant characteristics
of a kernel problem. Used as advisory context for the agent and for
similarity matching in the experience store -- NOT as enforced categories.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class KernelTraits:
	"""Optimization-relevant characteristics of a kernel problem.

	This is advisory context, not a hard classification. The agent
	uses these traits as hints alongside profiling data and experience.
	"""

	# Detected operations (may be multiple for fused ops)
	dominant_ops: list[str] = field(default_factory=list)

	# Estimated characteristics (from static analysis, before profiling)
	estimated_bottleneck: str = "unknown"  # compute_bound, memory_bound, unknown
	has_data_reuse: bool = False  # matmul-like ops reuse data
	is_composite: bool = False  # multiple fused operations

	# Shape characteristics
	shape_category: str = "unknown"  # small, medium, large
	shape_aspect: str = "unknown"  # square, tall_skinny, wide, batched

	# Precision
	involves_precision_ops: bool = False  # quantization, casting

	# Strategy hints (suggestions, not requirements)
	suggested_strategies: list[str] = field(default_factory=list)

	# Confidence in the analysis (lower = agent should rely more on profiling)
	confidence: float = 0.0

	def similarity(self, other: KernelTraits) -> float:
		"""Compute trait similarity for experience matching (0.0 to 1.0)."""
		score = 0.0
		total = 0.0

		# Op overlap (weighted heavily)
		if self.dominant_ops and other.dominant_ops:
			shared = set(self.dominant_ops) & set(other.dominant_ops)
			union = set(self.dominant_ops) | set(other.dominant_ops)
			score += 3.0 * len(shared) / max(len(union), 1)
			total += 3.0
		else:
			total += 3.0

		# Bottleneck match
		total += 2.0
		if (
			self.estimated_bottleneck == other.estimated_bottleneck
			and self.estimated_bottleneck != "unknown"
		):
			score += 2.0

		# Data reuse
		total += 1.0
		if self.has_data_reuse == other.has_data_reuse:
			score += 1.0

		# Shape category
		total += 1.0
		if (
			self.shape_category == other.shape_category
			and self.shape_category != "unknown"
		):
			score += 1.0

		return score / max(total, 1.0)

	def summary(self) -> str:
		"""One-line human-readable summary."""
		ops = ", ".join(self.dominant_ops) if self.dominant_ops else "unknown"
		parts = [f"ops=[{ops}]"]
		if self.estimated_bottleneck != "unknown":
			parts.append(f"likely {self.estimated_bottleneck}")
		if self.has_data_reuse:
			parts.append("data reuse")
		if self.shape_category != "unknown":
			parts.append(f"{self.shape_category} {self.shape_aspect}")
		return ", ".join(parts)


# Operation detection patterns
OP_PATTERNS: dict[str, list[str]] = {
	"matmul": [
		r"torch\.matmul", r"torch\.mm\b", r"torch\.bmm",
		r"torch\.addmm", r"@ ", r"F\.linear",
		r"matrix_multiplication", r"[Mm]atmul",
	],
	"conv": [
		r"nn\.Conv\d", r"F\.conv\d",
		r"conv_standard", r"conv_transpose", r"conv_depthwise",
		r"conv_pointwise",
	],
	"reduction": [
		r"\.sum\(", r"\.mean\(", r"\.max\(", r"\.min\(",
		r"\.prod\(", r"\.norm\(",
		r"torch\.sum", r"torch\.mean", r"torch\.norm",
		r"reduction",
	],
	"softmax": [
		r"F\.softmax", r"torch\.softmax", r"F\.log_softmax",
		r"[Ss]oftmax",
	],
	"elementwise": [
		r"torch\.relu", r"F\.relu", r"F\.gelu", r"F\.selu",
		r"torch\.sigmoid", r"torch\.tanh",
		r"ReLU", r"GELU", r"Sigmoid", r"Tanh",
		r"Swish", r"ELU", r"LeakyReLU",
		r"HardSigmoid", r"HardTanh", r"Softplus", r"Softsign",
	],
	"norm": [
		r"nn\.BatchNorm", r"nn\.LayerNorm", r"nn\.GroupNorm",
		r"nn\.InstanceNorm", r"RMSNorm",
	],
	"attention": [
		r"scaled_dot_product", r"[Aa]ttention",
	],
	"loss": [
		r"nn\.\w+Loss", r"F\.\w+loss",
		r"cross_entropy", r"mse_loss",
		r"Loss$", r"Loss_",
	],
	"pooling": [
		r"nn\.\w+Pool", r"F\.\w+pool", r"[Pp]ooling",
	],
	"cumulative": [
		r"cumsum", r"cumprod", r"cummax", r"cummin",
	],
	"indexing": [
		r"torch\.gather", r"torch\.scatter", r"torch\.index_select",
		r"Argmax", r"Argmin",
	],
}

# Ops that typically involve data reuse
REUSE_OPS = {"matmul", "conv", "attention"}

# Ops that are typically memory-bound (low arithmetic intensity)
MEMORY_BOUND_OPS = {
	"elementwise", "reduction", "softmax", "norm",
	"loss", "pooling", "cumulative", "indexing",
}

# Ops that are typically compute-bound
COMPUTE_BOUND_OPS = {"matmul", "conv", "attention"}


def analyze_traits(
	problem_name: str,
	reference_source: str = "",
	input_shapes: dict | None = None,
) -> KernelTraits:
	"""Analyze a kernel problem and produce a trait vector.

	This is best-effort static analysis. The agent should treat these
	as initial hypotheses to be confirmed or overridden by profiling.
	"""
	# Combine name and source for pattern matching
	text = problem_name + "\n" + reference_source
	clean_name = re.sub(r"^\d+_", "", problem_name)

	# Detect operations
	detected_ops: list[str] = []
	for op_name, patterns in OP_PATTERNS.items():
		for pattern in patterns:
			if re.search(pattern, text):
				if op_name not in detected_ops:
					detected_ops.append(op_name)
				break

	# If nothing detected from source, try name-only
	if not detected_ops:
		for op_name, patterns in OP_PATTERNS.items():
			for pattern in patterns:
				if re.search(pattern, clean_name):
					if op_name not in detected_ops:
						detected_ops.append(op_name)
					break

	# Estimate characteristics
	has_reuse = any(op in REUSE_OPS for op in detected_ops)
	is_composite = len(detected_ops) > 1

	# Bottleneck estimate
	compute_ops = [op for op in detected_ops if op in COMPUTE_BOUND_OPS]
	memory_ops = [op for op in detected_ops if op in MEMORY_BOUND_OPS]
	if compute_ops and not memory_ops:
		bottleneck = "compute_bound"
	elif memory_ops and not compute_ops:
		bottleneck = "memory_bound"
	elif compute_ops and memory_ops:
		bottleneck = "mixed"  # composite op, profiling needed
	else:
		bottleneck = "unknown"

	# Shape analysis
	shape_cat = "unknown"
	shape_aspect = "unknown"
	if input_shapes:
		max_dim = 0
		total_elems = 0
		dims_list: list[list[int]] = []
		for shape in input_shapes.values():
			if isinstance(shape, list):
				dims_list.append(shape)
				max_dim = max(max_dim, max(shape) if shape else 0)
				elems = 1
				for d in shape:
					elems *= d
				total_elems += elems

		if total_elems > 10_000_000:
			shape_cat = "large"
		elif total_elems > 100_000:
			shape_cat = "medium"
		else:
			shape_cat = "small"

		# Aspect ratio from first input
		if dims_list and len(dims_list[0]) == 2:
			h, w = dims_list[0]
			ratio = max(h, w) / max(min(h, w), 1)
			if ratio < 2:
				shape_aspect = "square"
			elif h > w:
				shape_aspect = "tall_skinny"
			else:
				shape_aspect = "wide"
		elif dims_list and len(dims_list[0]) >= 3:
			shape_aspect = "batched"

	# Strategy suggestions (advisory, not enforced)
	suggestions: list[str] = []
	if "matmul" in detected_ops:
		suggestions.extend([
			"tf32_tensor_cores",
			"tensor_core_utilization",
			"register_blocking",
		])
	if any(op in detected_ops for op in MEMORY_BOUND_OPS):
		suggestions.extend([
			"kernel_fusion",
			"vectorized_loads",
			"torch_compile_integration",
		])
	if "attention" in detected_ops:
		suggestions.extend([
			"flash_attention_tiling",
			"kernel_fusion",
		])
	if not suggestions:
		suggestions = ["torch_compile_integration"]

	# Confidence based on how much we detected
	confidence = min(
		(len(detected_ops) * 0.3) + (0.2 if shape_cat != "unknown" else 0),
		1.0,
	)

	return KernelTraits(
		dominant_ops=detected_ops,
		estimated_bottleneck=bottleneck,
		has_data_reuse=has_reuse,
		is_composite=is_composite,
		shape_category=shape_cat,
		shape_aspect=shape_aspect,
		involves_precision_ops=False,
		suggested_strategies=suggestions,
		confidence=confidence,
	)
