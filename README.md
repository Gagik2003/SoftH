# SoftH

![Alt Text](SoftH.gif)

## Installation

```bash
pip install git+https://github.com/Gagik2003/SoftH.git
```

## SoftH function example
```python
import torch
from SoftH.functions import SoftHFunction

SoftH_fn = SoftHFunction(32)

in_tensor = torch.randn(5)
print("Input shape:", list(in_tensor.shape))
print("Input tensor:", in_tensor)

with torch.no_grad():
    out_tensor = SoftH_fn(in_tensor)
print("Output shape:", list(out_tensor.shape))
print("Output tensor:", out_tensor)
```


## SoftH layer example
```python
import torch
from SoftH.layers import SoftHLayer

SoftH_layer = SoftHLayer(input_size=5, output_size=3, n=32)

in_tensor = torch.randn(100, 5)
print("Input shape:", list(in_tensor.shape))

with torch.no_grad():
    out_tensor = SoftH_layer(in_tensor)
print("Output shape:", list(out_tensor.shape))
```