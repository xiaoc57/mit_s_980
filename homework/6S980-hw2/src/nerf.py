import torch
from einops import rearrange, repeat
from jaxtyping import Float
from omegaconf import DictConfig
from torch import Tensor, nn

from .field.field import Field


class NeRF(nn.Module):
    cfg: DictConfig
    field: Field

    def __init__(self, cfg: DictConfig, field: Field) -> None:
        super().__init__()
        self.cfg = cfg
        self.field = field

    def forward(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
    ) -> Float[Tensor, "batch 3"]:
        """Render the rays using volumetric rendering. Use the following steps:

        1. Generate sample locations along the rays using self.generate_samples().
        2. Evaluate the neural field at the sample locations. The neural field's output
           has four channels: three for RGB color and one for volumetric density. Don't
           forget to map these channels to valid output ranges.
        3. Compute the alpha values for the evaluated volumetric densities using
           self.compute_alpha_values().
        4. Composite these alpha values together with the evaluated colors from.
        """

        xyz, sample_boundaries = self.generate_samples(origins, directions, near, far, 64)
        b = xyz.shape[0]
        xyz = rearrange(xyz, "b h c -> (b h) c")
        color_sigma = self.field(xyz)
        color_sigma = rearrange(color_sigma, "(b h) c -> b h c", b = b)
        color = color_sigma[..., :-1]
        sigma = color_sigma[..., -1:]
        sigma = rearrange(sigma, "b c 1 -> b c")
        alpha = self.compute_alpha_values(sigma, sample_boundaries)
        return self.alpha_composite(alpha, color)

    def generate_samples(
        self,
        origins: Float[Tensor, "batch 3"],
        directions: Float[Tensor, "batch 3"],
        near: float,
        far: float,
        num_samples: int,
    ) -> tuple[
        Float[Tensor, "batch sample 3"],  # xyz sample locations
        Float[Tensor, "batch sample+1"],  # sample boundaries
    ]:
        """For each ray, equally divide the space between the specified near and far
        planes into num_samples segments. Return the segment boundaries (including the
        endpoints at the near and far planes). Also return sample locations, which fall
        at the midpoints of the segments.
        """
        b = origins.shape[0]
        
        halfl = (far - near) / num_samples / 2
        bound = torch.linspace(near, far, num_samples + 1, device=origins.device)
        dt = bound[1:] - halfl
        
        bound = repeat(bound, "... -> b ...", b = b)
        
        origins = repeat(origins, "b c -> b n c", n = num_samples)
        directions = repeat(directions, "b c -> b n c", n = num_samples)
        dt = repeat(dt, "... -> b ... c", b = b, c = 3)
        
        xyz = origins + directions * dt

        return xyz, bound

    def compute_alpha_values(
        self,
        sigma: Float[Tensor, "batch sample"],
        boundaries: Float[Tensor, "batch sample+1"],
    ) -> Float[Tensor, "batch sample"]:
        """Compute alpha values from volumetric densities (values of sigma) and segment
        boundaries.
        """
        d = boundaries[..., 1:] - boundaries[..., :-1] 
        alpha = 1 - torch.exp(-sigma * d)
        return alpha

    def alpha_composite(
        self,
        alphas: Float[Tensor, "batch sample"],
        colors: Float[Tensor, "batch sample 3"],
    ) -> Float[Tensor, "batch 3"]:
        """Alpha-composite the supplied alpha values and colors. You may assume that the
        background is black.
        """

        transmitten = torch.cumprod(1 - alphas, dim=-1)
        w = transmitten * alphas
        w = repeat(w, "... -> ... c", c = colors.shape[-1])
        c = torch.sum(w * colors, dim=1)
        return c
