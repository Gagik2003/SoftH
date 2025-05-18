import torch.nn as nn


class ANI1EncoderLayer(nn.Module):
    """
    A layer that separates atomic features by element type for the ANI-1 dataset.

    This encoder splits input atom representations into separate groups based on 
    atomic number masks corresponding to hydrogen (H), carbon (C), nitrogen (N), 
    and oxygen (O). The layer assumes the input includes only atoms of these types 
    and uses the atomic number to identify them:
        - H: 1
        - C: 6
        - N: 7
        - O: 8

    Args:
        None

    Inputs:
        x (Tensor): Input tensor of shape `(batch_size, num_atoms, feature_dim)` — 
                    per-atom features for each molecule in the batch.
        mask (Tensor): Atomic number tensor of shape `(1, num_atoms)` — used to 
                       identify atom types in the molecule.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]:
            - x_h (Tensor): Features of hydrogen atoms, shape `(batch_size, num_H, feature_dim)`
            - x_c (Tensor): Features of carbon atoms, shape `(batch_size, num_C, feature_dim)`
            - x_n (Tensor): Features of nitrogen atoms, shape `(batch_size, num_N, feature_dim)`
            - x_o (Tensor): Features of oxygen atoms, shape `(batch_size, num_O, feature_dim)`

    Example:
        >>> layer = ANI1EncoderLayer()
        >>> x = torch.randn(2, 10, 128)  # 2 molecules, 10 atoms each, 128 features
        >>> mask = torch.tensor([[1, 6, 1, 8, 7, 6, 1, 8, 6, 1]])  # shape (1, 10)
        >>> x_h, x_c, x_n, x_o = layer(x, mask)
    """

    def forward(self, x, mask):
        mask = mask[0]
        mask_h = mask == 1
        mask_c = mask == 6
        mask_n = mask == 7
        mask_o = mask == 8

        x_h = x[:, mask_h]
        x_c = x[:, mask_c]
        x_n = x[:, mask_n]
        x_o = x[:, mask_o]

        return x_h, x_c, x_n, x_o
