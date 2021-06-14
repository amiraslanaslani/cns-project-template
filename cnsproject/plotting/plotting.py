"""
Module for visualization and plotting.

TODO.

Implement this module in any way you are comfortable with. You are free to use\
any visualization library you like. Providing live plotting and animations is\
also a bonus. The visualizations you will definitely need are as follows:

1. F-I curve.
2. Voltage/current dynamic through time.
3. Raster plot of spikes in a neural population.
4. Convolutional weight demonstration.
5. Weight change through time.
"""

from __future__ import annotations
from typing import Union, Type, Tuple

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np

from .plotters import AbstractPlotter
from ..network.monitors import AbstractMonitor


class Plot:

    def __init__(
            self,
            monitor: Union[AbstractMonitor, None] = None,
            shape: Tuple[int, int] = (1, 1),
            grid_spec: bool = False,
            fig_size: Tuple[int, int] = (10, 10),
            title: Union[str, None] = None
    ):
        self.monitor = monitor
        self.shape = shape
        self.grid_spec = grid_spec
        self.grid_axs = []

        if grid_spec:
            self.fig = plt.figure(figsize=fig_size)
            self.grid = plt.GridSpec(shape[0], shape[1])
            self.axs = None
        else:
            self.grid = None
            self.fig, self.axs = plt.subplots(shape[0], shape[1], figsize=fig_size)
            if isinstance(self.axs, Axes):
                self.axs = np.array(self.axs)
            self.axs = self.axs.reshape(shape[0], shape[1])

        if not (title is None):
            self.fig.suptitle(title, fontweight='bold')

    def plot(
            self,
            plotter: Type[AbstractPlotter],
            x: int = 0,
            y: int = 0,
            x_until: Union[int, None] = None,
            y_until: Union[int, None] = None,
            monitor: AbstractMonitor = None,
            **kwargs
    ) -> Plot:
        if monitor is None:
            monitor_in_use = self.monitor
        else:
            monitor_in_use = monitor

        if self.grid_spec:
            if not x_until:
                x_until = x + 1

            if not y_until:
                y_until = y + 1

            ax = self.fig.add_subplot(self.grid[x:x_until, y:y_until])
            plotter.plot(ax, monitor=monitor_in_use, figure=self.fig, **kwargs)
            self.grid_axs.append(ax)
        else:
            plotter.plot(ax=self.axs[x][y], monitor=monitor_in_use, **kwargs)
        return self

    def make_tight(self) -> Plot:
        self.fig.tight_layout()
        return self

    def show(self=None) -> None:
        plt.show()

    def save(self, fname: str, **kwargs) -> Plot:
        self.fig.savefig(fname, **kwargs)
        return self

