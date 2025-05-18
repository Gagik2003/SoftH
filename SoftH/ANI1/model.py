import torch.nn as nn

from SoftH.ANI1.encoder import ANI1EncoderLayer
from SoftH.ANI1.softh_atom_layer import SoftHAtomLayer


class ANIModel(nn.Module):
    """
    End-to-end model for processing molecular inputs using element-wise encoding 
    and a SoftH-based atom-wise neural network.

    This model is designed for use with molecular datasets like ANI-1, where atomic 
    features are processed based on element type. It first encodes input atom 
    representations into separate groups (H, C, N, O) using `ANI1EncoderLayer`, then 
    passes each group through an element-specific SoftH-based network (`SoftHAtomLayer`), 
    which outputs a scalar prediction per molecule.

    The architecture is useful for property prediction tasks like energy regression.

    Attributes:
        encoder (ANI1EncoderLayer): Splits atomic features by element type.
        model (SoftHAtomLayer): Processes grouped atomic features and outputs scalar values.

    Inputs:
        x (Tensor): Input atomic features of shape `(batch_size, num_atoms, feature_dim)`.
        mask (Tensor): Atomic number tensor of shape `(1, num_atoms)` used to distinguish atom types.

    Returns:
        Tensor: Predicted scalar values for each molecule, shape `(batch_size,)`.

    Example:
        >>> model = ANIModel()
        >>> x = torch.randn(8, 20, 52)  # batch of 8 molecules, 20 atoms, 52 features
        >>> mask = torch.tensor([[1, 6, 6, 8, 1, 7, 6, 8, 1, 1, 6, 7, 1, 6, 8, 1, 1, 6, 7, 6]])
        >>> output = model(x, mask)  # shape: (8,)
    """

    def __init__(self):
        super().__init__()
        self.encoder = ANI1EncoderLayer()
        self.model = SoftHAtomLayer(52, 64, 1.5, -10.0, 10.0)

    def forward(self, x, mask):
        x_h, x_c, x_n, x_o = self.encoder(x, mask)
        out = self.model(x_h, x_c, x_n, x_o)

        return out
