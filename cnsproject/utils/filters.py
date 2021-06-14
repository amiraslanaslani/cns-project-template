from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Tuple, Iterable

import torch

from .constants import PI, TWO


# Filter types
ON_CENTER = 'oncenter'
OFF_CENTER = 'offcenter'


def get_convolve_indices_map(
        image_shape: Iterable[int],
        kernel_shape: Iterable[int],
        stride: int = 1,
        filters_number: int = 1,
        padding: bool = False
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    if padding:
        padd_size = torch.tensor(kernel_shape) if isinstance(kernel_shape, torch.Tensor) else kernel_shape
        padd_size = torch.tensor(padd_size) - 1
        image_shape = tuple(torch.tensor(image_shape) + padd_size)

    size = reduce(mul, image_shape)
    image_template = torch.arange(size).reshape(image_shape)

    filter_margin_x = int((kernel_shape[0] - 1) / 2)
    filter_margin_y = int((kernel_shape[1] - 1) / 2)

    conv_x = torch.arange(filter_margin_x, image_shape[0] - filter_margin_x, step=stride).int()
    conv_y = torch.arange(filter_margin_y, image_shape[1] - filter_margin_y, step=stride).int()

    indices_matrix = []

    for i in conv_x:
        row = []
        for j in conv_y:
            tmp = image_template[
                i - filter_margin_x:i + filter_margin_x + 1,
                j - filter_margin_y:j + filter_margin_y + 1
            ].flatten().tolist()
            row.append([tmp] * filters_number)
        indices_matrix.append(row)
    return torch.tensor(indices_matrix), (conv_x.shape[0], conv_y.shape[0])


def calc_convolution2d_from_indices_matrix(image: torch.Tensor, kernel: torch.Tensor, indices_matrix):
    return (image.flatten()[indices_matrix] * kernel.reshape(kernel.shape[0], -1)).sum(dim=3)


def padding2d(matrix: torch.tensor, padd_x: int, padd_y: int):
    matrix_shape = torch.tensor(matrix.shape)
    kernel_shape = torch.tensor([padd_x, padd_y]) * 2 + 1
    base = torch.zeros(tuple(matrix_shape + kernel_shape - 1))
    base[padd_x:-padd_x, padd_y:-padd_y] = matrix
    return base


def convolve(
        image: torch.Tensor,
        kernel: torch.Tensor,
        clip: Tuple[float, float] = None,
        stride: int = 1,
        two_d_padding=True
):
    if two_d_padding:
        kernel_shape = (torch.tensor(kernel.shape) - 1) // 2
        image = padding2d(image, kernel_shape[0], kernel_shape[1])

    indices_map = get_convolve_indices_map(image.shape, kernel.shape, stride)
    result = calc_convolution2d_from_indices_matrix(image, kernel, indices_map)
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
