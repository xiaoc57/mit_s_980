import torch.nn as nn
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from src.components.positional_encoding import PositionalEncoding

from .field import Field


class FieldMLP(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up an MLP for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/mlp.yaml):

        - positional_encoding_octaves: The number of octaves in the positional encoding.
          If this parameter is None, do not positionally encode the input.
        - num_hidden_layers: The number of hidden linear layers.
        - d_hidden: The dimensionality of the hidden layers.

        Don't forget to add ReLU between your linear layers!
        """

        super().__init__(cfg, d_coordinate, d_out)
        octaves = cfg["positional_encoding_octaves"]
        if octaves is None:
            self.pse = None
        else:
            self.pse = PositionalEncoding(octaves)
        
        num_layer = cfg["num_hidden_layers"]
        num_hidden = cfg["d_hidden"]
        
        if self.pse is None:
            base_layer = nn.Linear(d_coordinate, num_hidden)
        else:
            base_layer = nn.Linear(d_coordinate * 2 * octaves, num_hidden)
        self.mlp = nn.ModuleList()
        
        if num_layer <= 1:
            self.mlp.append(base_layer)
        else:
            self.mlp.append(base_layer)
            for i in range(num_layer):
                self.mlp.append(nn.ReLU())
                self.mlp.append(nn.Linear(num_hidden, num_hidden))
            self.mlp[-1] = nn.Linear(num_hidden, d_out)
        # self.mlp.append(nn.ReLU())
        # self.mlp.append(nn.Linear(num_hidden, d_out))
        self.mlp = nn.Sequential(*self.mlp)
    
    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        if self.pse is None:
            return self.mlp(coordinates)
        else:
            positione = self.pse(coordinates)
            return self.mlp(positione)
