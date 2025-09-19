# Minimal, portable upfirdn2d fallback (CPU/CUDA/MPS) using PyTorch ops.
# This is not as fast as the CUDA kernel but is sufficient for testing/inference.
import torch
import torch.nn.functional as F
from typing import Optional

def upfirdn2d(x: torch.Tensor, kernel: Optional[torch.Tensor] = None, up=1, down=1, pad=(0,0,0,0)):
    # x: [N,C,H,W]
    assert x.dim() == 4, "Input must be NCHW"
    n, c, h, w = x.shape
    # Upsample
    if up != 1:
        x = F.interpolate(x, scale_factor=up, mode="nearest")
    # Pad
    if any(pad):
        x = F.pad(x, pad, mode="constant", value=0.0)
    # Filter
    if kernel is not None:
        if kernel.dim() == 1:
            k = torch.outer(kernel, kernel)
        else:
            k = kernel
        k = k / k.sum()
        k = k.to(x.dtype).to(x.device)
        k = k.view(1,1,*k.shape)
        x = F.conv2d(x, k.repeat(c,1,1,1), padding=0, groups=c)
    # Downsample
    if down != 1:
        x = x[:, :, ::down, ::down]
    return x