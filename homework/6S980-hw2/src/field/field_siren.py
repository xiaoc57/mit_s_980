from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from src.components.sine_layer import SineLayer

from .field import Field


class FieldSiren(Field):
    network: nn.Sequential

    def __init__(
        self,
        cfg: DictConfig,
        d_coordinate: int,
        d_out: int,
    ) -> None:
        """Set up a SIREN network using the sine layers at src/components/sine_layer.py.
        Your network should consist of:

        - An input sine layer whose output dimensionality is 256
        - Two hidden sine layers with width 256
        - An output linear layer
        """
        super().__init__(cfg, d_coordinate, d_out)
        
        self.siren = nn.Sequential(
            SineLayer(d_coordinate, 256, is_first=True),
            SineLayer(256, 256),
            SineLayer(256, 256),
            nn.Linear(256, d_out)
        )

    def forward(
        self,
        coordinates: Float[Tensor, "batch coordinate_dim"],
    ) -> Float[Tensor, "batch output_dim"]:
        """Evaluate the MLP at the specified coordinates."""

        return self.siren(coordinates)
