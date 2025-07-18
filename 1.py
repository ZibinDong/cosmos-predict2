import torch

from cosmos_predict2.models.text2image_dit import Attention

device = "cuda:0"
m = Attention(64, head_dim=16, n_heads=4, backend="transformer_engine")
m = m.to(device)


x = torch.randn((2, 8, 64), device=device)
