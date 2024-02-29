from glob import glob
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from src.provided_code import generate_spin, get_bunny, save_image
from src.rendering import render_point_cloud
from einops import repeat
class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""
    import json
    
    pzl = PuzzleDataset()
    
    with open("D:\\document\\mit_s_980\\homework\\6S980-hw1\\abf149\\metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # metadata = json.load("D:\\document\\mit_s_980\\homework\\6S980-hw1\\abf149\\metadata.json", "r")
    
    pzl["extrinsics"] = Tensor(metadata["extrinsics"])
    pzl["intrinsics"] = Tensor(metadata["intrinsics"])
    
    image_paths = glob("D:\\document\\mit_s_980\\homework\\6S980-hw1\\abf149\\images\\*.png")
    images = []
    
    for path in image_paths:
        # 使用Pillow读取图片
        with Image.open(path) as img:
            # 转换图片为Tensor，并标准化到[0, 1]
            image_tensor = torch.tensor(np.array(img)) / 255.0
            # 如果是灰度图，需要增加一个通道维度
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(0)
            else:
                # 转换为CHW格式
                image_tensor = image_tensor.permute(2, 0, 1)
            images.append(image_tensor)
    
    # 将所有图片堆叠成一个Tensor
    images_tensor = torch.stack(images)
    
    pzl["images"] = images_tensor
    
    return pzl


def convert_dataset(dataset: PuzzleDataset) -> PuzzleDataset:
    """Convert the dataset into OpenCV-style camera-to-world format. As a reminder, this
    format has the following specification:

    - The camera look vector is +Z.
    - The camera up vector is -Y.
    - The camera right vector is +X.
    - The extrinsics are in camera-to-world format, meaning that they transform points
      in camera space to points in world space.

    """

    ex = dataset["extrinsics"]
    
    NUM_STEPS = ex.shape[0]


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
    
    t = torch.Tensor(
        [[0,0,1,0],
        [0,1,0,0],
        [1,0,0,0],
        [0,0,0,1]]
    )
    t = repeat(t, "i j -> b i j", b=NUM_STEPS)
    ex = torch.inverse(t @ torch.inverse(ex))
    
    vertices, _ = get_bunny()
    canvas = render_point_cloud(vertices, ex, dataset["intrinsics"])
    dataset["images"] = canvas
    dataset["extrinsics"] = ex

    return dataset
    


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    return "c2w"


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    return "+x"


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    return "-y"


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    return "-z"


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    return '''
        这道题我做的异常困难，开始的时候我总觉得世界坐标系应该是确定的，然后也无从下手，后来我想直接把它可视化算了，结果画了
        几个坐标轴，好像也不太好。
        最后我是用给的数据集作为参考，github上有两个人的答案我做了参考，还有用了abf149的数据集。
        首先，+x +y或者-z我理解是客观存在的，不存在坐标系的轴是怎么规定的，我现在也很难把它描述清楚，但是有一点我可以确定的是
        首先先要找到正确的外参矩阵，在题目中有一个提示，让我将这些轴的确定总结成为一个矩阵变换的问题，
        有两种轴变换的形式，一种是相反轴，一种是交换轴，还有要枚举的是c2w矩阵还是w2c矩阵，他们说能看出来，
        我不会那种一眼就看出来的方法。。。有人能教我吗？然后试试利用这个输出图像，当图像输出一致的时候说明
        你的一切假设都是正确的，你找到了这个矩阵。然后我利用了题里说的这几点：
        2单位距离原点，点乘是正的，看向原点
        如果知道这个能先（其实不一定有先后吧）求出来看向原点的那个轴，然后能够求出向上的轴，最后根据这两个求另一个，想象出来。
        
        答案对的是abf149，我并不知道对不对，可能大神的东西是对的。
    '''
