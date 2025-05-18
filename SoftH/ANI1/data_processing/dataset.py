import os

import torch
import numpy as np
from torch.utils.data import Dataset



class ANI1AtomDataset(Dataset):
    """
    PyTorch Dataset for loading ANI-1 atomic data stored in .npz files.

    Each .npz file is expected to contain the following keys:
        - 'mask': A tensor containing atomic numbers or atom type indicators.
        - 'molecule_info': A tensor with per-atom features or coordinates.
        - 'energy': A scalar energy value associated with the molecule.

    This dataset is designed for energy prediction tasks using atom-wise models 
    like ANI.

    Args:
        path (str): Directory path containing .npz files. Each file corresponds 
                    to one molecule.

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            - mask (Tensor): Shape `(num_atoms,)` — atom type indicators.
            - atom_info (Tensor): Shape `(num_atoms, num_features)` — per-atom data.
            - energy (Tensor): Shape `()` — scalar target value for the molecule.

    Example:
        >>> dataset = ANI1AtomDataset("/path/to/npz_files")
        >>> mask, atom_info, energy = dataset[0]
    """

    def __init__(self, path):
        self.npz_files = [
            os.path.join(path, file) for file in os.listdir(path)
        ]

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        data = np.load(self.npz_files[idx])
        mask = torch.tensor(data["mask"], dtype=torch.float32)
        atom_info = torch.tensor(data["molecule_info"], dtype=torch.float32)
        energy = torch.tensor(data["energy"], dtype=torch.float32)

        return mask, atom_info, energy
