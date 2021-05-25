from abc import ABC, abstractmethod
from functools import reduce
from operator import mul

import torch

from .constants import PI, TWO
from .general import gaussian_2d


# Filter types
ON_CENTER = 'oncenter'
OFF_CENTER = 'offcenter'


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
    def gabor(x, y, lamb, theta, sigma, gama):
        xx = x * torch.cos(theta) + y * torch.sin(theta)
        yy = - x * torch.sin(theta) + y * torch.cos(theta)
        exp_term = - (xx * xx + gama * gama * yy * yy) / (2 * sigma * sigma)
        cos_term = (2 * PI * xx) / lamb
        return torch.exp(exp_term) * torch.cos(cos_term)

    @staticmethod
    def filter(cls, **kwargs):
        n = kwargs['n']
        lamb = kwargs['wavelen']
        theta = kwargs['theta']
        sigma = kwargs['sigma']
        gama = kwargs['gama']

        assert n % 2 == 1

        n = torch.tensor(n)
        lamb = torch.tensor(lamb)
        theta = torch.tensor(theta)
        sigma = torch.tensor(sigma)
        gama = torch.tensor(gama)

        n2 = (n - 1) / 2
        arange = torch.arange(-n2, n2 + 1)
        xv, yv = torch.meshgrid([arange, arange])
        return cls.gabor(xv, yv, lamb, theta, sigma, gama)
