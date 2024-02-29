import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor
from torch.nn.functional import grid_sample

from .field import Field


class FieldGrid(Field):
    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a grid for the neural field. Your architecture must respect the
        following parameters from the configuration (in config/field/grid.yaml):

        - side_length: the side length in each dimension

        Your architecture only needs to support 2D and 3D grids.
        """
        super().__init__(cfg, d_coordinate, d_out)
        assert d_coordinate in (2, 3)
        
        self.side_length = cfg["side_length"]
        self.d_coordinate = d_coordinate
        self.d_out = d_out
        self.grid = nn.Parameter(Tensor(1, d_out, *([self.side_length] * d_coordinate)))

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Use torch.nn.functional.grid_sample to bilinearly sample from the image grid.
        Remember that your implementation must support either 2D and 3D queries,
        depending on what d_coordinate was during initialization.
        """
        if self.d_coordinate == 3:
            coordinates = rearrange(coordinates * 2.0 - 1.0, "b c -> b 1 1 1 c")
        else:
            coordinates = rearrange(coordinates * 2.0 - 1.0, "b c -> b 1 1 c")
        b = coordinates.shape[0]
        grid = repeat(self.grid, "1 c h w -> b c h w", b = b)
        
        out = grid_sample(grid, coordinates)
        
        out = rearrange(out, "b ... -> b (...)")
        
        return out
