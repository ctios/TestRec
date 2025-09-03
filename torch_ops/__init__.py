import torch
try:
    import _C
    installed = True
except ImportError:
    installed = False

def square_plus(input: torch.Tensor) -> torch.Tensor:
    """
    Computes x^2 + x for each element in the input tensor.
    
    Args:
        input (Tensor): Input tensor
        
    Returns:
        Tensor: Output tensor with same shape as input
    """
    if not installed:
        # Fallback to pure Python implementation
        return input * input + input
    return _C.square_plus(input)

def modulo(input: torch.Tensor, mod: int) -> torch.Tensor:
    """
    Computes x % mod for each element in the input tensor.
    
    Args:
        input (Tensor): Input tensor
        mod (int): Modulus value
        
    Returns:
        Tensor: Output tensor with same shape as input
    """
    if not installed:
        # Fallback to pure Python implementation
        return input.to(torch.int64) % mod
    return _C.modulo(input, mod)

__version__ = "0.0.1"