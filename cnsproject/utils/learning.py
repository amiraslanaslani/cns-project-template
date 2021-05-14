from typing import Callable

import torch


# Always return a constant
def hard_bound(
        lr: float
) -> Callable:
    return lambda w: lr


# Calcualte and returns alpha.(w_ij - bound)
def soft_bound(
        bound: float,
        alpha: float = 1.
) -> Callable:
    result_func: Callable[[torch.Tensor], torch.Tensor] = lambda w: torch.erf((w - bound).abs()) * alpha
    return result_func
