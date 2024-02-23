from jaxtyping import Float
from torch import Tensor


def homogenize_points(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional points into (n+1)-dimensional homogeneous points."""

    import torch
    s = points.shape
    v = torch.ones(*s[:-1], 1, device=points.device, dtype=points.dtype)
    
    return torch.concat([points, v], dim=-1)



def homogenize_vectors(
    points: Float[Tensor, "*batch dim"],
) -> Float[Tensor, "*batch dim+1"]:
    """Turn n-dimensional vectors into (n+1)-dimensional homogeneous vectors."""

    import torch
    s = points.shape
    v = torch.zeros(*s[:-1], 1, device=points.device, dtype=points.dtype)
    
    return torch.concat([points, v], dim=-1)


def transform_rigid(
    xyz: Float[Tensor, "*#batch 4"],
    transform: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Apply a rigid-body transform to homogeneous points or vectors."""

    import torch
    from einops import einsum
    from torch import matmul
    
    assert torch.allclose(
        matmul(transform, xyz.unsqueeze(-1)).squeeze(-1),
        einsum(transform, xyz, "... i j, ... j -> ... i")
    )
    return einsum(transform, xyz, "... i j, ... j -> ... i")


def transform_world2cam(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D world coordinates to homogeneous
    3D camera coordinates.
    """
    
    import torch
    from einops import einsum
    
    return einsum(torch.inverse(cam2world), xyz, "... i j, ... j -> ... i")


def transform_cam2world(
    xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points or vectors from homogeneous 3D camera coordinates to homogeneous
    3D world coordinates.
    """

    from einops import einsum
    
    return einsum(cam2world, xyz, "... i j, ... j -> ... i")


def project(
    xyz: Float[Tensor, "*#batch 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 2"]:
    """Project homogenized 3D points in camera coordinates to pixel coordinates."""

    import torch
    from einops import einsum
    from torch import matmul
    
    t = xyz[...,:3]
    t = einsum(intrinsics, t, "... i j, ... j -> ... i")
    t[..., 0] = t[..., 0] / t[..., 2]
    t[..., 1] = t[..., 1] / t[..., 2]
    
    proj = matmul(intrinsics, xyz[..., :3].unsqueeze(-1)).squeeze(-1)
    tmp = proj[..., :2] / proj[..., 2:3]
    
    assert torch.allclose(
        tmp,
        t[..., :2]
    )
 
    return t[..., :2]
    
