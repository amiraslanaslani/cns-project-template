"""
Module for encoding data into spike.
"""

from abc import ABC, abstractmethod
from typing import Optional
from itertools import repeat
from functools import reduce

import torch


class AbstractEncoder(ABC):
    """
    Abstract class to define encoding mechanism.

    You will define the time duration into which you want to encode the data \
    as `time` and define the time resolution as `dt`. All computations will be \
    performed on the CPU by default. To handle computation on both GPU and CPU, \
    make sure to set the device as defined in `device` attribute to all your \
    tensors. You can add any other attributes to the child classes, if needed.

    The computation procedure should be implemented in the `__call__` method. \
    Data will be passed to this method as a tensor for further computations. You \
    might need to define more parameters for this method. The `__call__`  should return \
    the tensor of spikes with the shape (time_steps, \*population.shape).

    Arguments
    ---------
    time : float
        Length of encoded tensor.
    dt : float, Optional
        Simulation time step. The default is 1.0.
    device : str, Optional
        The device to do the computations. The default is "cpu".

    """

    def __init__(
        self,
        time: float,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        **kwargs
    ) -> None:
        self.time = time
        self.dt = dt
        self.device = device

    @abstractmethod
    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        Compute the encoded tensor of the given data.

        Parameters
        ----------
        data : torch.Tensor
            The data tensor to encode.

        Returns
        -------
        None
            It should return the encoded tensor.

        """
        pass


class Time2FirstSpikeEncoder(AbstractEncoder):
    """
    Time-to-First-Spike coding.

    Implement Time-to-First-Spike coding.
    """

    def __init__(
        self,
        time: float,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        d_min: int = 0,
        d_max: int = 255,
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        self.d_min = torch.tensor(d_min, device=self.device)
        self.d_max = torch.tensor(d_max, device=self.device)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = data.clone().detach().to(self.device)
        data = self.d_max + self.d_min - data
        steps = torch.tensor(int(self.time / self.dt), device=self.device)
        scaled_data = (data - self.d_min) * steps / (self.d_max - self.d_min + 1)
        expand_scaled_data = scaled_data.expand(steps, *data.shape)
        steps_list = torch.arange(0, self.time, self.dt, device=self.device)
        reshaped_steps_list = steps_list.reshape(len(steps_list), *tuple(repeat(1, data.dim())))
        return ((expand_scaled_data - reshaped_steps_list).abs() < (self.dt / 2)) + \
               ((expand_scaled_data - reshaped_steps_list) == (self.dt / 2))


class PositionEncoder(AbstractEncoder):
    """
    Position coding.

    Implement Position coding.


    Arguments
    ---------
    peaks : float
        Tensor that includes peak points of encoder. (means of gaussians)
        Result is depends on the order of peaks.

    std : float, Optional
        Standard deviation of gaussians. The default is 1.0.

    """

    def __init__(
        self,
        time: float,
        peaks: torch.Tensor,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        std: Optional[float] = 1.0,
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        self.std = torch.tensor(std, device=self.device)
        self.steps = torch.tensor(int(self.time / self.dt), device=self.device)
        self.peaks = torch.flatten(peaks)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        data = torch.flatten(data)
        result = torch.full((self.steps, self.peaks.size()[0]), False, device=self.device)
        for ndx, peak in enumerate(self.peaks):
            g = self.rev_gaussian(data, peak)
            g = g[g < self.steps * 0.9]
            g = torch.round(g)
            result[g.long(), ndx] = True
        return result

    def rev_gaussian(self, x: torch.Tensor, mu: torch.Tensor):
        power = (- (x - mu) * (x - mu)) / (2 * self.std * self.std)
        return - (self.steps * torch.exp(power)) + self.steps


class PoissonEncoder(AbstractEncoder):
    """
    Poisson coding.

    Implement Poisson coding.
    """

    def __init__(
        self,
        time: float,
        dt: Optional[float] = 1.0,
        device: Optional[str] = "cpu",
        r: float = 10,
        d_min: int = 0,
        d_max: int = 255,
        **kwargs
    ) -> None:
        super().__init__(
            time=time,
            dt=dt,
            device=device,
            **kwargs
        )
        self.d_min = torch.tensor(d_min, device=self.device)
        self.d_max = torch.tensor(d_max, device=self.device)
        self.r = torch.tensor(r, device=self.device)

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        steps = torch.tensor(int(self.time / self.dt), device=self.device)
        data = data.clone().detach().to(self.device)
        r_x = (data - self.d_min) * self.r / self.d_max
        p = r_x * self.dt / self.time
        return torch.rand((steps, *data.shape), device=self.device) < p

