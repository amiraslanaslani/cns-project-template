
from abc import abstractmethod, ABC
from matplotlib.axes import Axes
from typing import Union
import copy

import torch

from ..network.monitors import Monitor
from ..network.neural_populations import NeuralPopulation
from ..network.connections import AbstractConnection


class AbstractPlotter(ABC):
    @staticmethod
    @abstractmethod
    def plot(ax: Axes, monitor: Union[Monitor, None], **kwargs) -> Axes:
        pass


class PotentialTimePlotter(AbstractPlotter):
    @staticmethod
    def plot(ax: Axes, monitor: Union[Monitor, None], time_unit=None, spikes: bool = False, **kwargs) -> Axes:
        time_unit = f" ({time_unit})" if time_unit else ""
        ax.set_title("Electric Potential")
        ax.set_xlabel("time" + time_unit)
        ax.set_ylabel("U(time)")
        u_vector = monitor.get("u")
        time_vector = monitor.get("time")
        ax.plot(time_vector, u_vector)

        if spikes:
            PotentialTimePlotter.plot_spikes(
                monitor,
                ax,
                u_vector.min(),
                u_vector.max(),
            )
        return ax

    @staticmethod
    def plot_spikes(monitor: Monitor, ax: Axes, min: float, max: float):
        spike_points = monitor.get("s")
        for spike in spike_points.nonzero(as_tuple=True)[0]:
            ax.vlines(
                spike,
                min,
                max,
                linestyles="dashed",
                colors="red",
                zorder=3,
                linewidth=3
            )


class CurrentTimePlotter(AbstractPlotter):
    @staticmethod
    def plot(ax: Axes, monitor: Union[Monitor, None], time_unit=None, **kwargs) -> Axes:
        time_unit = f" ({time_unit})" if time_unit else ""
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
            monitor: Union[Monitor, None],
            neuron: Union[NeuralPopulation, AbstractConnection, None],
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
            if neuron is None:
                neuron_obj = copy.deepcopy(monitor.obj)
            else:
                neuron_obj = copy.deepcopy(neuron)

            f = FIPlotter.get_frequency_of(
                neuron_obj,
                current,
                time_step,
                time
            )
            spikes.append(f)
            currents.append(current)

        spikes = torch.tensor(spikes)
        currents = torch.tensor(currents)
        ax.plot(currents, spikes)
        return ax

    @staticmethod
    def get_frequency_of(
            neuron: Union[NeuralPopulation, AbstractConnection, None],
            current: Union[float, int],
            time_step: Union[int, float, None],
            time: Union[int, float],

    ):
        neuron.reset_state_variables()
        if not (time_step is None):
            neuron.set_timestep(time_step)

        monitor = Monitor(
            neuron,
            state_variables=["s"]
        )
        monitor.set_time_steps(time, time_step)
        monitor.reset_state_variables()

        current_value = torch.tensor(current)
        for _ in torch.arange(0, time, time_step):
            neuron.forward(current_value)
            monitor.record()

        spikes_tensor = monitor.get("s")
        spikes_number = spikes_tensor.sum()
        return spikes_number

