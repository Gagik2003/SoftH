import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftH(nn.Module):
    """
    A feature-wise, differentiable piecewise constant transformation layer.

    This layer applies a soft piecewise constant function independently to each 
    input feature. For each feature, the function is defined by `n` trainable 
    values and `n` fixed breakpoints linearly spaced between [a, b]. Given input 
    `x`, the output is a soft interpolation of these values, weighted by 
    proximity to the breakpoints using a temperature-scaled softmax.

    Args:
        d_model (int): Dimensionality of the last axis of the input (number of features).
        n (int): Number of breakpoints (and corresponding values) per feature.
        temperature (float): Softmax temperature controlling how "sharp" 
                             the interpolation is. Lower = sharper. Default: 1.5.
        a (float): Lower bound of the input domain [a, b]. Default: 0.0.
        b (float): Upper bound of the input domain [a, b]. Default: 1.0.

    Inputs:
        x (Tensor): Input tensor of shape `(*, d_model)` where `*` is any batch shape.

    Returns:
        Tensor: Output tensor of shape `(*, d_model)`, where each feature has undergone 
                a soft, differentiable transformation based on proximity to its breakpoints.

    Example:
        >>> layer = SoftH(d_model=4, n=10, temperature=1.5)
        >>> x = torch.rand(3, 4)  # e.g., batch of 3 vectors of size 4
        >>> y = layer(x)  # shape (3, 4)
    """

    def __init__(self, d_model, n, temperature=1.5, a=0.0, b=1.0):
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.temperature = temperature

        self.values = nn.Parameter(torch.randn(d_model, n))  # (d_model, n)

        breakpoints = torch.linspace(a, b, n).repeat(d_model, 1)  # (d_model, n)
        self.register_buffer("breakpoints", breakpoints.unsqueeze(0))  # (1, d_model, n)

    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, self.d_model)  # (batch_size, d_model)
        x_exp = x.unsqueeze(2)  # (batch_size, d_model, 1)

        distances = torch.abs(x_exp - self.breakpoints)  # (batch_size, d_model, n)
        weights = F.softmax(-self.temperature * distances, dim=-1)  # (batch_size, d_model, n)
        output = torch.sum(weights * self.values.unsqueeze(0), dim=-1)  # (batch_size, d_model)

        return output.view(*original_shape)


class SoftHLayer(nn.Module):
    """
    A learnable, differentiable layer that transforms input features using
    soft piecewise constant functions with shared input-output mapping logic.

    For each input feature and each output unit, this layer defines a 
    soft piecewise constant transformation by using `n` trainable values 
    and `n` fixed breakpoints over the interval [a, b]. Each output is 
    computed as a weighted sum over these values based on the proximity 
    of input features to their corresponding breakpoints, followed by 
    summation over input features.

    Args:
        input_size (int): Number of input features (dimensionality of input vectors).
        output_size (int): Number of output dimensions.
        n (int): Number of breakpoints or bins in each piecewise function.
        temperature (float, optional): Temperature for softmax weighting. Lower values
                                       make the output more similar to hard bin selection.
                                       Default: 1.5.
        a (float, optional): Lower bound of the input domain for breakpoints. Default: -1.0.
        b (float, optional): Upper bound of the input domain for breakpoints. Default: 1.0.

    Inputs:
        x (Tensor): Input tensor of shape `(batch_size, input_size)`.

    Returns:
        Tensor: Output tensor of shape `(batch_size, output_size)`, where each output unit
                applies a learned, differentiable function of all input features.

    Example:
        >>> layer = SoftHLayer(input_size=5, output_size=3, n=10, temperature=2.0)
        >>> x = torch.randn(4, 5)  # batch of 4 samples, 5 features each
        >>> y = layer(x)  # shape: (4, 3)
    """

    def __init__(self, input_size, output_size, n, temperature=1.5, a=-1.0, b=1.0):
        super().__init__()
        self.output_size = output_size
        self.temperature = temperature

        self.values = nn.Parameter(torch.randn(output_size,input_size, n))  # (output_size, input_size, n)

        breakpoints = torch.linspace(a, b, n).unsqueeze(0).repeat(output_size, input_size, 1)  # (output_size, input_size, n)
        self.register_buffer("breakpoints", breakpoints.unsqueeze(0))  # (1, output_size, input_size, n)

    def forward(self, x):
        batch_size, input_size = x.size()  # (batch_size, input_size)

        x_expanded = x.unsqueeze(1).expand(batch_size, self.output_size, input_size)  # (batch_size, output_size, input_size)
        x_expanded = x_expanded.unsqueeze(-1)  # (batch_size, output_size, input_size, 1)

        distances = torch.abs(x_expanded - self.breakpoints)  # (batch_size, output_size, input_size, n)
        weights = F.softmax(-self.temperature * distances, dim=-1)  # (batch_size, output_size, input_size, n)

        x_out = torch.sum(weights * self.values.unsqueeze(0), dim=-1)  # (batch_size, output_size, input_size)

        x_out = x_out.sum(-1)  # (batch_size, output_size)

        return x_out
    

if __name__ == '__main__':
    # Example usage of both layers
    
    input_tensor = torch.randn(4, 5)  # Example input tensor of shape (batch_size, input_size)
    print("Input shape:", input_tensor.shape)

    soft_h = SoftH(d_model=5, n=10, temperature=1.5)
    output_tensor = soft_h(input_tensor)  # Output tensor of shape (batch_size, d_model)
    print("Output shape from SoftH:", output_tensor.shape)

    soft_h_layer = SoftHLayer(input_size=5, output_size=3, n=10, temperature=2.0)
    output_tensor = soft_h_layer(input_tensor)  # Output tensor of shape (batch_size, output_size)
    print("Output shape from SoftHLayer:", output_tensor.shape)
