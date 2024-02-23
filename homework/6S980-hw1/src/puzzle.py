from glob import glob
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor


class PuzzleDataset(TypedDict):
    extrinsics: Float[Tensor, "batch 4 4"]
    intrinsics: Float[Tensor, "batch 3 3"]
    images: Float[Tensor, "batch height width"]


def load_dataset(path: Path) -> PuzzleDataset:
    """Load the dataset into the required format."""
    import json
    
    pzl = PuzzleDataset()
    
    metadata = json.load("D:\\document\\mit_s_980\\homework\\6S980-hw1\\abf149\\metadata.json", "r")
    
    pzl.extrinsics = Tensor(metadata["extrinsics"])
    pzl.intrinsics = Tensor(metadata["intrinsics"])
    
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
    
    pzl.images = images_tensor
    
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

    raise NotImplementedError("This is your homework.")


def quiz_question_1() -> Literal["w2c", "c2w"]:
    """In what format was your puzzle dataset?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_2() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera look vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_3() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera up vector?"""

    raise NotImplementedError("This is your homework.")


def quiz_question_4() -> Literal["+x", "-x", "+y", "-y", "+z", "-z"]:
    """In your puzzle dataset's format, what was the camera right vector?"""

    raise NotImplementedError("This is your homework.")


def explanation_of_problem_solving_process() -> str:
    """Please return a string (a few sentences) to describe how you solved the puzzle.
    We'll only grade you on whether you provide a descriptive answer, not on how you
    solved the puzzle (brute force, deduction, etc.).
    """

    raise NotImplementedError("This is your homework.")
