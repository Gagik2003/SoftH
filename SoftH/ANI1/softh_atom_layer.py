import torch
import torch.nn as nn

from SoftH.layers import SoftH


class SoftHAtomLayer(nn.Module):
    """
    A soft, element-wise feedforward neural network for processing atomic features.

    `SoftHAtomLayer` applies separate soft piecewise transformations (`SoftH` layers)
    for each atom type (H, C, N, O), then aggregates and combines them into a final 
    representation using another `SoftH` layer.

    This architecture is suitable for molecular models where atoms are represented 
    with per-feature embeddings, and you want each element type to be processed 
    distinctly before fusion.

    Args:
        input_dim (int): Number of features per atom.
        n (int): Number of breakpoints in each SoftH layer.
        temperature (float): Temperature parameter for softmax in SoftH.
        a (float): Lower bound for the SoftH breakpoints.
        b (float): Upper bound for the SoftH breakpoints.

    Inputs:
        x_h (Tensor): Hydrogen atom features, shape `(batch_size, num_H, input_dim)`.
        x_c (Tensor): Carbon atom features, shape `(batch_size, num_C, input_dim)`.
        x_n (Tensor): Nitrogen atom features, shape `(batch_size, num_N, input_dim)`.
        x_o (Tensor): Oxygen atom features, shape `(batch_size, num_O, input_dim)`.

    Returns:
        Tensor: Final scalar output for each molecule, shape `(batch_size,)`.

    Example:
        >>> model = SoftHAtomLayer(input_dim=64, n=10, temperature=2.0, a=-1.0, b=1.0)
        >>> out = model(x_h, x_c, x_n, x_o)  # out.shape == (batch_size,)
    """

    def __init__(self, input_dim, n, temperature, a, b):
        super().__init__()

        self.atom_layers = nn.ModuleDict()
        self.atom_layers["H"] = SoftH(input_dim, n, temperature, a, b)
        self.atom_layers["C"] = SoftH(input_dim, n, temperature, a, b)
        self.atom_layers["N"] = SoftH(input_dim, n, temperature, a, b)
        self.atom_layers["O"] = SoftH(input_dim, n, temperature, a, b)

        self.final_layer = SoftH(input_dim * 4, n, temperature, a, b)

    def forward(self, x_h, x_c, x_n, x_o):
        # x_h.shape = (batch_size, seq_length, num_features)
        x_h = self.atom_layers["H"](x_h).sum(dim=1)
        x_c = self.atom_layers["C"](x_c).sum(dim=1)
        x_n = self.atom_layers["N"](x_n).sum(dim=1)
        x_o = self.atom_layers["O"](x_o).sum(dim=1)

        x = torch.cat([x_h, x_c, x_n, x_o], dim=1)
        x = self.final_layer(x)

        return x.sum(dim=1)
