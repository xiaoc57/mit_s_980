import torch

import numpy as np
from einops import rearrange, repeat
from jaxtyping import Float
from omegaconf import DictConfig
from PIL import Image
from torch import Tensor
from torch.nn.functional import grid_sample

from .field_dataset import FieldDataset


class FieldDatasetImage(FieldDataset):
    def __init__(self, cfg: DictConfig) -> None:
        """Load the image in cfg.path into memory here."""

        super().__init__(cfg)
        img = Image.open(cfg["path"]).convert("RGB")
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.image = Tensor(np.array(img))
        self.image = rearrange(self.image, "h w c -> c h w")
        self.image = self.image / 255.0
        
    def query(
        self,
        coordinates: Float[Tensor, "batch d_coordinate"],
    ) -> Float[Tensor, "batch d_out"]:
        """Sample the image at the specified coordinates and return the corresponding
        colors. Remember that the coordinates will be in the range [0, 1].

        You may find the grid_sample function from torch.nn.functional helpful here.
        Pay special attention to grid_sample's expected input range for the grid
        parameter.
        """
        
        # img = img.to(coordinates.device)
        coordinates = rearrange(coordinates * 2.0 - 1.0, "b c -> b 1 1 c")
        img = repeat(self.image, "... -> b ...", b = coordinates.shape[0])
        img = img.to(coordinates.device)
        out = grid_sample(img, coordinates)
        out = rearrange(out, "b c h w -> b (c h w)")
        return out

    @property
    def d_coordinate(self) -> int:
        return 2

    @property
    def d_out(self) -> int:
        return 3

    @property
    def grid_size(self) -> tuple[int, ...]:
        """Return a grid size that corresponds to the image's shape."""

        return (self.image.shape[1], self.image.shape[2])
