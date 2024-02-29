import torch

from src.components.positional_encoding import PositionalEncoding
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor

from .field import Field
from .field_grid import FieldGrid
from .field_mlp import FieldMLP


class FieldGroundPlan(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a neural ground plan. You should reuse the following components:

        - FieldGrid from  src/field/field_grid.py
        - FieldMLP from src/field/field_mlp.py
        - PositionalEncoding from src/components/positional_encoding.py

        Your ground plan only has to handle the 3D case.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate == 3
        self.fg = FieldGrid(cfg["grid"], d_coordinate-1, d_out)
        octaves = cfg.get("positional_encoding_octaves")
        self.pse = PositionalEncoding(octaves)
        self.fmlp = FieldMLP(cfg["mlp"], d_out+(2 * octaves), d_out)
        

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the ground plan at the specified coordinates. You should:

        - Sample the grid using the X and Y coordinates.
        - Positionally encode the Z coordinates.
        - Concatenate the grid's outputs with the corresponding encoded Z values, then
          feed the result through the MLP.
        """
        xy = coordinates[..., :2]
        z = coordinates[..., 2:]

        o1 = self.fg(xy)
        o2 = self.pse(z)
        out = torch.concat([o1, o2], dim=-1)
        
        out = self.fmlp(out)
        return out
