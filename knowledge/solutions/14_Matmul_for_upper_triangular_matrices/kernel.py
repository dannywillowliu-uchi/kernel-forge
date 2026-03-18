import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Enable TF32 globally
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# Optimize cuBLAS for best performance
torch.backends.cuda.preferred_blas_library('cublaslt')

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        self._attn_buf = None
        self._out_buf = None

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        b, h, s, d = Q.shape
        bh = b * h
        scale = 1.0 / math.sqrt(d)
        
        Q_flat = Q.view(bh, s, d)
        K_flat = K.view(bh, s, d)
        V_flat = V.view(bh, s, d)
        
        if self._attn_buf is None or self._attn_buf.shape[0] != bh:
            self._attn_buf = torch.empty(bh, s, s, device=Q.device, dtype=Q.dtype)
            self._out_buf = torch.empty(bh, s, d, device=Q.device, dtype=Q.dtype)
        
        # Use contiguous K transpose for better memory access pattern
        K_t = K_flat.transpose(-2, -1).contiguous()
        
        torch.baddbmm(self._attn_buf, Q_flat, K_t, beta=0.0, alpha=scale, out=self._attn_buf)
        torch.softmax(self._attn_buf, dim=-1, out=self._attn_buf)
        torch.bmm(self._attn_buf, V_flat, out=self._out_buf)
        
        return self._out_buf.view(b, h, s, d)
