import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, num_octaves: int):
        super().__init__()
        self.num_octaves = num_octaves
        # self.d_out = 3

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        """Separately encode each channel using a positional encoding. The lowest
        frequency should be 2 * torch.pi, and each frequency thereafter should be
        double the previous frequency. For each frequency, you should encode the input
        signal using both sine and cosine.
        """
        res = []
        for i in range(self.num_octaves):
            res.append(torch.cos(2 * torch.pi* (2 ** i) * samples))
            res.append(torch.sin(2 * torch.pi* (2 ** i) * samples))
        
        res = torch.concat(res, dim=-1)
        
        # self.d_out = res.shape[-1]

        return res

    def d_out(self, dimensionality: int):
        return self.num_octaves * 2 * dimensionality
