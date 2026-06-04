import random
import torch


class RandomFlipRotate:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Random flips along each axis
        if random.random() < 0.5:
            x = torch.flip(x, [0])
        if random.random() < 0.5:
            x = torch.flip(x, [1])
        if random.random() < 0.5:
            x = torch.flip(x, [2])
        # random 90 degree rotations around z-axis
        k = random.randint(0, 3)
        x = torch.rot90(x, k, [0, 1])
        return x
