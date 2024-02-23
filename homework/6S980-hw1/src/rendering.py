from jaxtyping import Float
from torch import Tensor


def render_point_cloud(
    vertices: Float[Tensor, "vertex 3"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    resolution: tuple[int, int] = (256, 256),
) -> Float[Tensor, "batch height width"]:
    """Create a white canvas with the specified resolution. Then, transform the points
    into camera space, project them onto the image plane, and color the corresponding
    pixels on the canvas black.
    """

    import torch
    from einops import rearrange
    from torch import arange, round

    from .geometry import homogenize_points, project, transform_world2cam
    
    v = homogenize_points(vertices)
    cv = transform_world2cam(
        rearrange(v, "v c -> 1 v c"), rearrange(extrinsics, "b r c -> b 1 r c")
        )
    # b v 2
    p = project(cv, rearrange(intrinsics, "b r c -> b 1 r c"))
    
    batch_size = p.shape[0]
    
    canvas = torch.ones(
        [batch_size, resolution[1], resolution[0]], 
        device=p.device)
    
    x_indices = round(p[...,0] * (resolution[0] - 1)).long().clamp(0, resolution[0] - 1)
    y_indices = round(p[...,1] * (resolution[1] - 1)).long().clamp(0, resolution[1] - 1)
    
    b_indices = arange(batch_size, device=p.device).unsqueeze(1).expand_as(x_indices)
    
    xf = x_indices.flatten()
    yf = y_indices.flatten()
    bf = b_indices.flatten()
    
    canvas[bf, yf, xf] = 0
    
    return canvas
    