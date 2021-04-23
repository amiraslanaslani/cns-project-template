
from abc import abstractmethod, ABC
from matplotlib.axes import Axes
from typing import Union
import copy

import torch

from ..network.monitors import Monitor
from ..network.neural_populations import NeuralPopulation, LIFPopulation
from ..network.connections import AbstractConnection


class AbstractPlotter(ABC):
    @staticmethod
    @abstractmethod
    def plot(ax: Axes, monitor: Union[Monitor, None], **kwargs) -> Axes:
        pass


class PotentialTimePlotter(AbstractPlotter):
    @staticmethod
    def plot(
            ax: Axes,
            monitor: Union[Monitor, None],
            time_unit: Union[str, None] = None,
            spikes: bool = False,
            title: Union[str, None] = None,
            **kwargs
    ) -> Axes:
        title_appendible = f" ({title})" if title else ""
        time_unit = f" ({time_unit})" if time_unit else ""
        ax.set_title("Electric Potential" + title_appendible)
        ax.set_xlabel("time" + time_unit)
        ax.set_ylabel("U(time)")
        u_vector = monitor.get("u")
        time_vector = monitor.get("time")
        ax.plot(time_vector, u_vector)
        ax.set_xlim(time_vector.min(), time_vector.max())

        if spikes:
            SpikePlotter.plot(ax, monitor, u_vector.min(), u_vector.max())
        return ax


class SpikePlotter(AbstractPlotter):
    @staticmethod
    def plot(ax: Axes, monitor: Union[Monitor, None], min: float, max: float, **kwargs) -> Axes:
        spike_points = monitor.get("s")
        times = monitor.get("time")
        for spike in spike_points.nonzero(as_tuple=True)[0]:
            ax.vlines(
                times[spike],
                min,
                max,
                linestyles="dashed",
                colors="red",
                zorder=3,
                linewidth=3
            )


class CurrentTimePlotter(AbstractPlotter):
    @staticmethod
    def plot(ax: Axes, monitor: Union[Monitor, None], time_unit: Union[str, None] = None, **kwargs) -> Axes:
        time_unit = f" ({time_unit})" if time_unit else ""
        ax.set_title("Electric Current")
        ax.set_xlabel("time" + time_unit)
        ax.set_ylabel("I(time)")
        time = monitor.get("time")
        ax.set_xlim(time.min(), time.max())
        ax.plot(
            time,
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
            title: Union[str, None] = None,
            **kwargs
    ) -> Axes:
        title_appendible = f" ({title})" if title else ""

        ax.set_title("F-I" + title_appendible)
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
            state_variables=[NeuralPopulation.RB_SPIKES]
        )
        monitor.set_time_steps(time, time_step)
        monitor.reset_state_variables()

        current_value = torch.tensor(current)
        for _ in torch.arange(0, time, time_step):
            neuron.forward(current_value)
            monitor.record()

        spikes_tensor = monitor.get(NeuralPopulation.RB_SPIKES)
        spikes_number = spikes_tensor.sum()
        return spikes_number


class AdaptionTimePlotter(AbstractPlotter):
    @staticmethod
    def plot(
            ax: Axes,
            monitor: Union[Monitor, None],
            time_unit: Union[str, None] = None,
            spikes: bool = False,
            **kwargs
    ) -> Axes:
        time_unit = f" ({time_unit})" if time_unit else ""
        ax.set_title("Adaption")
        ax.set_xlabel("time" + time_unit)
        ax.set_ylabel("W")
        w_vector = monitor.get("w")
        ax.plot(
            monitor.get("time"),
            w_vector
        )

        if spikes:
            SpikePlotter.plot(ax, monitor, w_vector.min(), w_vector.max())
        return ax


class RasterPlotter(AbstractPlotter):
    @staticmethod
    def plot(
            ax: Axes,
            monitor: Union[Monitor, None],
            inhibitories: Union[torch.Tensor, None] = None,
            **kwargs
    ) -> Axes:
        spikes = monitor.get(NeuralPopulation.RB_SPIKES)
        time = monitor.get(LIFPopulation.RB_TIME)

        if inhibitories is None:
            inhibitories = torch.full((spikes.shape[1],), False)

        ax.set_title("Raster Plot")
        ax.set_xlabel("time")
        temp_spikes_exc = spikes.clone().detach()
        temp_spikes_inh = spikes.clone().detach()
        temp_spikes_exc[:, ~ inhibitories] = 0
        temp_spikes_inh[:, inhibitories] = 0
        inh_xs, inh_ys = temp_spikes_inh.nonzero(as_tuple=True)
        exc_xs, hib_ys = temp_spikes_exc.nonzero(as_tuple=True)
        ax.scatter(time[exc_xs], hib_ys, label="Excitatory", s=5)
        ax.scatter(time[inh_xs], inh_ys, label="Inhibitory", s=5)
        ax.set_xlim(time.min(), time.max())
        ax.legend()
        return ax


class ActivityPlotter(AbstractPlotter):
    @staticmethod
    def plot(
            ax: Axes,
            monitor: Union[Monitor, None],
            inhibitories: Union[torch.Tensor, None] = None,
            **kwargs
    ) -> Axes:
        ax.set_title("Activity")
        ax.set_xlabel("time")
        spikes = monitor.get(NeuralPopulation.RB_SPIKES)
        activity = spikes.int().sum(dim=1) / spikes.shape[1]
        time = monitor.get(LIFPopulation.RB_TIME)
        ax.set_xlim(time.min(), time.max())
        ax.plot(activity)
        return ax

