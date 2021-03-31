
from abc import abstractmethod, ABC
from matplotlib.axes import Axes
from typing import Union
import copy

import torch

from ..network.monitors import Monitor


class AbstractPlotter(ABC):
    @staticmethod
    @abstractmethod
    def plot(ax: Axes, monitor: Monitor, **kwargs) -> Axes:
        pass


class PotentialTimePlotter(AbstractPlotter):
    @staticmethod
    def plot(ax: Axes, monitor: Monitor, time_unit=None, **kwargs) -> Axes:
        time_unit = f"({time_unit})" if time_unit else ""
        ax.set_title("Electric Potential")
        ax.set_xlabel("time" + time_unit)
        ax.set_ylabel("U(time)")
        ax.plot(
            monitor.get("time"),
            monitor.get("u")
        )
        return ax


class CurrentTimePlotter(AbstractPlotter):
    @staticmethod
    def plot(ax: Axes, monitor: Monitor, time_unit=None, **kwargs) -> Axes:
        time_unit = f"({time_unit})" if time_unit else ""
        ax.set_title("Electric Current")
        ax.set_xlabel("time" + time_unit)
        ax.set_ylabel("I(time)")
        ax.plot(
            monitor.get("time"),
            monitor.get("current")
        )
        return ax


class FIPlotter(AbstractPlotter):
    @staticmethod
    def plot(
            ax: Axes,
            monitor: Monitor,
            current_to: Union[int, float] = 20,
            step_size: Union[int, float] = 1,
            current_from: Union[int, float] = 1,
            time_step: Union[int, float, None] = 1,
            time: Union[int, float] = 1000,
            **kwargs
    ) -> Axes:
        ax.set_title("F-I")
        ax.set_xlabel("I(time)")
        ax.set_ylabel("f=1/T")

        spikes = []
        currents = []

        for current in torch.arange(current_from, current_to, step_size):
            neuron = copy.deepcopy(monitor.obj)
            neuron.reset_state_variables()
            if not (time_step is None):
                neuron.set_timestep(time_step)

            monitor = Monitor(
                neuron,
                state_variables=["s"],
                time=int(time / time_step + 1)
            )

            current_value = torch.tensor(current)
            for _ in torch.arange(0, time, time_step):
                neuron.forward(current_value)
                monitor.record()

            del neuron
            spikes_tensor = monitor.get("s")
            # print(spikes_tensor)
            spikes_number = spikes_tensor.sum()
            print(spikes_number)
            spikes.append(spikes_number)
            currents.append(current)

        spikes = torch.tensor(spikes)
        currents = torch.tensor(currents)
        ax.plot(currents, spikes)
        return ax

