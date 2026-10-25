import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()

        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype

        x_float = x.float()

        mean_square = x_float.square().mean(dim=-1, keepdim=True)

        rms = torch.rsqrt(mean_square + self.eps)    

        normalized_x = (x * rms).to(input_dtype)

        return self.gamma * normalized_x
