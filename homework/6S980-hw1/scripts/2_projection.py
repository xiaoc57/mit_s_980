import json

import torch
from einops import repeat
from jaxtyping import install_import_hook

# Add runtime type checking to all imports.
with install_import_hook(("src",), ("beartype", "beartype")):
    from src.provided_code import generate_spin, get_bunny, save_image
    from src.rendering import render_point_cloud

if __name__ == "__main__":
    vertices, faces = get_bunny()

    # Generate a set of camera extrinsics for rendering.
    # NUM_STEPS = 16
    # c2w = generate_spin(NUM_STEPS, 15.0, 2.0)
    with open("D:\\document\\mit_s_980\\homework\\6S980-hw1\\abf149\\metadata.json", 'r') as f:
        metadata = json.load(f)
    
    c2w = torch.Tensor(metadata["extrinsics"])
    k = torch.Tensor(metadata["intrinsics"])
    NUM_STEPS = c2w.shape[0]
    # # Generate a set of camera intrinsics for rendering.
    # k = torch.eye(3, dtype=torch.float32)
    # k[:2, 2] = 0.5
    # k = repeat(k, "i j -> b i j", b=NUM_STEPS)

    # 这里需要构造一个矩阵，使乘后
    
    k1 = torch.Tensor(
        [
            [-1,0,0,0],[0, -1,0,0],[0,0,-1,0],[0,0,0,1]
        ]
    )
    k1 = repeat(k1, "i j -> b i j", b=NUM_STEPS)
    k2 = torch.Tensor(
        [
            [0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]
        ]
    )
    k2 = repeat(k2, "i j -> b i j", b=NUM_STEPS)
    # Render the point cloud.
    
    # assert torch.allclose(
    #     c2w @ k2 @ k1,
    #     torch.inverse(k2 @ k1 @ torch.inverse(c2w) )
    # )
    
    images = render_point_cloud(vertices, c2w @ k1 @ k2 , k)

    
    
    # Save the resulting images.
    for index, image in enumerate(images):
        save_image(image, f"outputs/1_projection/view_{index:0>2}.png")
