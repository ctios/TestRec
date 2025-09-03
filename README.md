# Github Actions Test

This package provides custom PyTorch operators with both CPU and CUDA implementations.

## Operators

1. `square_plus`: Computes x² + x for each element in the input tensor
2. `modulo`: Computes x % mod for each element in the input tensor

## Installation

To install the package, run:

```bash
pip install .
```

To install in development mode:

```bash
pip install -e .
```

## Usage

```python
import torch
import torch_ops

# Using square_plus
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
result = torch_ops.square_plus(x)
print(result)  # tensor([2.0, 6.0, 12.0, 20.0])

# Using modulo
x = torch.tensor([10.0, 15.0, 21.0, 7.0])
result = torch_ops.modulo(x, 10)
print(result)  # tensor([0, 1, 1, 7])
```

## Building the wheel

To build a wheel distribution:

```bash
python setup.py bdist_wheel
```

The wheel will be located in the `dist/` directory.