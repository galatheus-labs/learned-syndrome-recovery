from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from parity import build_parity_matrix
from binvox_rw import read_as_3d_array



class VoxelDataset(Dataset):
    def __init__(self, path: str, cube_size: int, transforms=None, constraint_matrix: torch.Tensor | None = None):
        self.paths = sorted(list(Path(path).glob("**/*.npy")) + list(Path(path).glob("**/*.binvox")))
        self.transforms = transforms
        self.cube = cube_size
        if constraint_matrix is not None:
            self.parity = constraint_matrix
        else:
            self.parity = build_parity_matrix(cube_size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        with open(p, "rb") as f:
            if p.suffix == ".npy":
                arr = np.load(f)
            else:
                voxels = read_as_3d_array(f)
                arr = voxels.data.reshape(-1)
        x = torch.tensor(arr, dtype=torch.uint8).reshape(self.cube, self.cube, self.cube)
        if self.transforms:
            x = self.transforms(x)
        syndrome = (self.parity @ x.reshape(-1)) % 2
        return x.float(), syndrome
