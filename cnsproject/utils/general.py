"""
Module for utility functions.

TODO.

Use this module to implement any required utility function.

Note: You are going to need to implement DoG and Gabor filters. A possible opt
ion would be to write them in this file but it is not a must and you can define\
a separate module/package for them.
"""

from typing import List, Union, Dict

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


def population_type(size: int, inhibitory_ratio: float=0.2) -> torch.Tensor:
    is_inhibitory = torch.full((size,), False)
    is_inhibitory[int(size * inhibitory_ratio):] = True
    return is_inhibitory


# def get_monitor_of_simulation(
#         neural_population: NeuralPopulation,
#         current: torch.Tensor,
#         time: Union[torch.Tensor, float],
#         dt: Union[torch.Tensor, float],
#         state_variables: List[str] = []
# ) -> Monitor:
#     time = torch.tensor(time)
#     dt = torch.tensor(dt)
#
#     monitor = Monitor(neural_population, state_variables=state_variables)
#     monitor.set_time_steps(time, dt)
#     monitor.reset_state_variables()
#
#     for t in range(int(time / dt)):
#         neural_population.forward(current[:, t])
#         monitor.record()
#
#     return monitor
