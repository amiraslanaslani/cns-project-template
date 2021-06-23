"""
Module for utility functions.

TODO.

Use this module to implement any required utility function.

Note: You are going to need to implement DoG and Gabor filters. A possible opt
ion would be to write them in this file but it is not a must and you can define\
a separate module/package for them.
"""
from functools import reduce
from operator import mul
from typing import List, Union, Dict, Iterable

import torch


def get_fixed_current(value: float, time: float, dt: float = 1, start=0, end=1) -> torch.Tensor:
    size = int(time / dt)
    current = torch.full((size, ), value)
    current[:start] = 0
    current[-end:] = 0
    return current


def get_random_current(length: int, start: float = 1, coef: float = 0.5):
    random = (torch.rand((length,)) - torch.tensor(.5)) * torch.tensor(coef)
    random[0] += torch.tensor(start)
    return torch.cumsum(random, dim=0).abs()


def population_type(shape: Union[int, tuple], inhibitory_ratio: float=0.2) -> torch.Tensor:
    if isinstance(shape, int):
        shape = (shape,)
    size = reduce(mul, shape)
    is_inhibitory = torch.full((size,), False)
    is_inhibitory[int(size * inhibitory_ratio):] = True
    return is_inhibitory.reshape((*shape,))


def iterlen(iterable: Iterable):
    return sum(1 for e in iterable)


def decorate_all_methods(decorator):  # example: @decorate_all_methods(decorator)
    def decorate(cls):
        for attr in cls.__dict__:
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


class Integer:
    def __init__(self, value):
        self.value = value

    def set_value(self, value):
        self.value = value

    def __repr__(self):
        return str(self.value)

    def get(self):
        return self.value

