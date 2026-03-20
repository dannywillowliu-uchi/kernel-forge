import torch
import torch.nn.functional as F
import torch._inductor.config
import torch._dynamo.config
from task import input_t, output_t

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._dynamo.config.cache_size_limit = 64


@torch.compile(mode="max-autotune-no-cudagraphs", fullgraph=True, dynamic=False)
def _trimul_compiled(x, mask, norm_weight, norm_bias,
                     fused_weight_bf16, hidden_dim,
                     to_out_norm_weight, to_out_norm_bias,
                     to_out_weight):
	B, N, _, dim = x.shape
	D = hidden_dim

	# LayerNorm 1
	x = F.layer_norm(x, (dim,), norm_weight, norm_bias)

	# All 5 projections fused in bf16
	projected = F.linear(x.to(torch.bfloat16), fused_weight_bf16)  # [B, N, N, 5*D]

	# Split in [B, N, N, D] layout
	left, right, left_gate, right_gate, out_gate = projected.split(D, dim=-1)

	mask_bf16 = mask.unsqueeze(-1).to(torch.bfloat16)
	left = left * mask_bf16 * left_gate.sigmoid()
	right = right * mask_bf16 * right_gate.sigmoid()
	out_gate_sig = out_gate.sigmoid()

	# Permute for bmm
	left = left.permute(0, 3, 1, 2).contiguous()
	right = right.permute(0, 3, 1, 2).contiguous()

	# bmm
	out = torch.bmm(
		left.reshape(B * D, N, N),
		right.reshape(B * D, N, N).transpose(-1, -2)
	)

	# Back to [B, N, N, D]
	out = out.view(B, D, N, N).permute(0, 2, 3, 1).contiguous()

	# LayerNorm 2 in fp32 + gate + final proj
	out = F.layer_norm(out.float(), (D,), to_out_norm_weight, to_out_norm_bias)
	out = out * out_gate_sig.float()
	out = F.linear(out, to_out_weight)

	return out


def custom_kernel(data: input_t) -> output_t:
	input_tensor, mask, weights, config = data
	dim = config["dim"]
	hidden_dim = config["hidden_dim"]

	fused_weight_bf16 = torch.cat([
		weights["left_proj.weight"],
		weights["right_proj.weight"],
		weights["left_gate.weight"],
		weights["right_gate.weight"],
		weights["out_gate.weight"],
	], dim=0).to(torch.bfloat16)

	out = _trimul_compiled(
		input_tensor, mask,
		weights["norm.weight"], weights["norm.bias"],
		fused_weight_bf16, hidden_dim,
		weights["to_out_norm.weight"], weights["to_out_norm.bias"],
		weights["to_out.weight"],
	)

	return out
