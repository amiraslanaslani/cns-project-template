from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Tuple

import torch

from .constants import PI, TWO


# Filter types
ON_CENTER = 'oncenter'
OFF_CENTER = 'offcenter'


def convolve(image: torch.Tensor, filter: torch.Tensor, clip: Tuple[float, float] = None):
    image_shape = image.shape
    filter_shape = filter.shape
    filter_margin_x = int((filter_shape[0] - 1) / 2)
    filter_margin_y = int((filter_shape[1] - 1) / 2)

    conv_x = torch.arange(filter_margin_x, image_shape[0] - filter_margin_x).int()
    conv_y = torch.arange(filter_margin_y, image_shape[1] - filter_margin_y).int()
    result = torch.zeros(conv_x.shape[0], conv_y.shape[0])
    for i, set_i in zip(conv_x, range(conv_x.shape[0])):
        for j, set_j in zip(conv_y, range(conv_y.shape[0])):
            result[set_i][set_j] = (
                    filter * image[
                             i-filter_margin_x:i+filter_margin_x+1,
                             j-filter_margin_y:j+filter_margin_y+1
                    ]
            ).sum()
    if clip is not None:
        result = torch.clamp(result, min=clip[0], max=clip[1])
    return result


class AbstractFilterMaker(ABC):
    @staticmethod
    def make_zero_summed(make_zero_summed: bool, filter_tensor: torch.Tensor):
        if not make_zero_summed:
            return filter_tensor
        n = reduce(mul, filter_tensor.shape)
        per_comp = filter_tensor.sum() / n
        return filter_tensor - per_comp

    @staticmethod
    @abstractmethod
    def filter(cls, **kwargs):
        pass

    @classmethod
    def get(cls, filter_type=ON_CENTER, make_zero_summed: bool = True, **kwargs):
        coef: int
        if filter_type == ON_CENTER:
            coef = 1
        elif filter_type == OFF_CENTER:
            coef = -1
        else:
            raise Exception("Unknown filter type")
        return coef * cls.make_zero_summed(
            make_zero_summed,
            cls.filter(cls, **kwargs)
        )


class Gaussian(AbstractFilterMaker):
    @staticmethod
    def gaussian(x, y, std):
        exp = torch.exp(-(x * x + y * y) / (2 * std * std))
        return exp / (std * torch.sqrt(TWO * PI))

    @staticmethod
    def filter(cls, **kwargs):
        n: int = kwargs['n']
        std = kwargs['std']

        assert n % 2 == 1

        n = torch.tensor(n)
        std = torch.tensor(std)

        n2 = (n - 1) / 2
        arange = torch.arange(-n2, n2 + 1)
        xv, yv = torch.meshgrid([arange, arange])
        return cls.gaussian(xv, yv, std)


class DoG(AbstractFilterMaker):
    @staticmethod
    def filter(cls, **kwargs):
        n = kwargs['n']
        std_2 = kwargs['std_2']
        std_1 = kwargs['std_1']

        assert n % 2 == 1
        assert std_2 > std_1

        g1 = Gaussian.get(make_zero_summed=False, n=n, std=std_1)
        g2 = Gaussian.get(make_zero_summed=False, n=n, std=std_2)
        return g1 - g2


class Gabor(AbstractFilterMaker):
    @staticmethod
    def gabor(x, y, lamb, theta, sigma, gamma):
        xx = x * torch.cos(theta) + y * torch.sin(theta)
        yy = - x * torch.sin(theta) + y * torch.cos(theta)
        exp_term = - (xx * xx + gamma * gamma * yy * yy) / (2 * sigma * sigma)
        cos_term = (2 * PI * xx) / lamb
        return torch.exp(exp_term) * torch.cos(cos_term)

    @staticmethod
    def filter(cls, **kwargs):
        n = kwargs['n']
        lamb = kwargs['wavelen']
        theta = kwargs['theta']
        sigma = kwargs['sigma']
        gamma = kwargs['gamma']

        assert n % 2 == 1

        n = torch.tensor(n)
        lamb = torch.tensor(lamb)
        theta = torch.tensor(theta)
        sigma = torch.tensor(sigma)
        gamma = torch.tensor(gamma)

        n2 = (n - 1) / 2
        arange = torch.arange(-n2, n2 + 1)
        xv, yv = torch.meshgrid([arange, arange])
        return cls.gabor(xv, yv, lamb, theta, sigma, gamma)
