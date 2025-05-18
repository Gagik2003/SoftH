import torch
import torch.nn as nn


class HFunction(nn.Module):
    """ 
    A trainable piecewise constant function mapping the interval [0, 1] to ℝ.

    The function divides the interval [0, 1] into `n` equal-width bins, and assigns a
    trainable value to each bin. The input `x` is assumed to lie in [0, 1] and is 
    discretized into an index corresponding to one of the bins. The output is the
    trainable value for the selected bin.

    Args:
        n (int): Number of intervals (bins) in the domain [0, 1].

    Inputs:
        x (Tensor): A tensor of shape `(*)` with values in [0, 1].

    Returns:
        Tensor: A tensor of shape `(*)` with values selected from the trainable
                parameter vector of length `n`, based on the bin index.

    Example:
        >>> h = HFunction(n=10)
        >>> x = torch.tensor([0.1, 0.5, 0.95])
        >>> y = h(x)
    """

    def __init__(self, n):
        super(HFunction, self).__init__()
        self.n = n
        self.values = nn.Parameter(torch.randn(n))

    def forward(self, x):
        x = x * self.n
        x = x.long()
        x.clamp_(0, self.n - 1)
        return self.values[x]


class SoftHFunction(nn.Module):
    """
    A differentiable approximation of a piecewise constant function over a real interval [a, b].

    The function defines `n` uniformly spaced breakpoints in [a, b] and associates each with a 
    trainable value. Given input `x`, it computes a soft weighted combination of the trainable 
    values based on distances between `x` and the breakpoints, using a temperature-scaled 
    softmax over negative distances.

    Args:
        n (int): Number of support points (breakpoints) in the range [a, b].
        temperature (float): Softmax temperature parameter. Lower values make the
                             function more "crisp" or closer to a hard selection. Default is 1.5.
        a (float): Lower bound of the domain. Default is 0.0.
        b (float): Upper bound of the domain. Default is 1.0.
        fixed (bool): If True, [a, b] remains fixed during training; otherwise,
                      it adjusts to fit the input range dynamically. Default is False.

    Inputs:
        x (Tensor): Input tensor of shape `(*)`, where each element lies in ℝ.

    Returns:
        Tensor: A tensor of shape `(*)` representing the soft interpolation over the values
                based on proximity to the breakpoints.

    Example:
        >>> f = SoftHFunction(n=20, temperature=5.0)
        >>> x = torch.tensor([0.1, 0.5, 0.95])
        >>> y = f(x)
    """

    def __init__(self, n, temperature=1.5, a=0.0, b=1.0, fixed=False):
        super().__init__()
        self.n = n
        self.temperature = temperature
        self.a = torch.tensor(a)
        self.b = torch.tensor(b)
        self.values = nn.Parameter(torch.randn(n))
        self.breakpoints = torch.linspace(self.a, self.b, self.n)

        self.scale_fn = self.scale_renage if not fixed else lambda x: x

    def scale_renage(self, x):
        if torch.min(x) < self.a:
            self.a = torch.min(x)
            self.breakpoints = torch.linspace(self.a, self.b, self.n)
        if torch.max(x) > self.b:
            self.b = torch.max(x)
            self.breakpoints = torch.linspace(self.a, self.b, self.n)

    def forward(self, x):
        self.scale_fn(x)

        distances = torch.abs(x.unsqueeze(-1) - self.breakpoints)
        weights = torch.softmax(-self.temperature * distances, dim=-1)

        return torch.sum(weights * self.values, dim=-1)


if __name__ == '__main__':
    # Example usage: aproximate a sine function
    num_epochs = 100000

    x = torch.linspace(0., 1., 100)
    y = torch.sin(2 * 3.14159 * x)

    # Using H function
    h = HFunction(32)
    optimizer = torch.optim.Adam(h.parameters(), lr=0.01)
    for i in range(num_epochs):
        optimizer.zero_grad()
        y_hat = h(x)
        loss = ((y - y_hat) ** 2).mean()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f'Loss: {loss.item()}')

    y_hat_h = h(x)
    final_loss_h = ((y - y_hat_h) ** 2).mean()

    # Using SoftH function
    softh = SoftHFunction(32, temperature=1.5, a=0.0, b=1.0)
    optimizer = torch.optim.Adam(softh.parameters(), lr=0.01)
    for i in range(num_epochs):
        optimizer.zero_grad()
        y_hat = softh(x)
        loss = ((y - y_hat) ** 2).mean()
        loss.backward()
        optimizer.step()
        if i % 1000 == 0:
            print(f'Epoch: {i}, Loss: {loss.item()}')

    y_hat_softh = softh(x)
    final_loss_softh = ((y - y_hat_softh) ** 2).mean()

    # Plotting the results
    import matplotlib.pyplot as plt
    plt.plot(x.numpy(), y.numpy(), label='True')
    plt.plot(x.numpy(), y_hat_softh.detach().numpy(), label=f'SoftH: {final_loss_softh.item():.4g}')
    plt.plot(x.numpy(), y_hat_h.detach().numpy(), label=f'H: {final_loss_h.item():.4f}')
    plt.legend()
    plt.show()
