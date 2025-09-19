import torch
from backbones.op.upfirdn2d import upfirdn2d

x = torch.randn(1, 3, 8, 8)
k1d = torch.tensor([1.0, 2.0, 1.0])
k2d = torch.outer(k1d, k1d)        # <- make it 2D

y = upfirdn2d(x, k2d, up=2, down=2, pad=(1, 1))
print(y.shape)