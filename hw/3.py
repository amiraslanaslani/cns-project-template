import sys

sys.path.append('../')

import torch

from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.plotting.plotting import Plot
from cnsproject.plotting.plotters import CurrentTimePlotter, RasterPlotter, ActivityPlotter
from cnsproject.network.connections import DenseConnection, RandomConnection
from cnsproject.utils.general import get_random_current, population_type
from cnsproject.utils.plotting import param2title

def get_param_set(**kwargs):
    return kwargs


torch.manual_seed(593)  # 123 111 593

N = 500
time = 1000
dt = 1
size = int(time / dt)

current = get_random_current(size, start=5)
is_inhibitory = population_type(N, 0.2)

connection_types = [DenseConnection, RandomConnection]

parameters = [
    get_param_set(
        shape=(N,), resistance=7, tau_t=600, threshold=-50,
        u_rest=-70, is_inhibitory=is_inhibitory, j0=10, s0=35
    ),
    get_param_set(
        shape=(N,), resistance=7, tau_t=600, threshold=-50,
        u_rest=-70, is_inhibitory=is_inhibitory, j0=10, s0=30
    ),
    get_param_set(
        shape=(N,), resistance=7, tau_t=400, threshold=-50,
        u_rest=-70, is_inhibitory=is_inhibitory, j0=10, s0=30
    ),
    get_param_set(
        shape=(N,), resistance=7, tau_t=600, threshold=-50,
        u_rest=-70, is_inhibitory=is_inhibitory, j0=5, s0=30
    ),
]

monitor = None
for index, parameter in enumerate(parameters):
    for connection_type in connection_types:
        neuron = LIFPopulation(**parameter)
        neuron.set_timestep(dt)
        connection = connection_type(neuron, j0=parameter['j0'], s0=parameter['s0'])

        monitor = Monitor(neuron, [
                LIFPopulation.RB_POTENTIAL,
                LIFPopulation.RB_TIME,
                LIFPopulation.RB_SPIKES,
                LIFPopulation.RB_CURRENT
            ],
            time=time,
            dt=dt
        )

        for idx, i in enumerate(current):
            neuron.forward(i)
            connection.compute()
            monitor.record()

        title = f"Connection Type: $%s$, Parameters: $%s$" % (
            connection_type.__name__,
            param2title(parameter, {"tau_t": "\\tau", "j0": "J_0", "s0": "\\sigma_0"})
        )
        plot = Plot(monitor, (2, 1), title=title, fig_size=(10, 6))
        plot.plot(RasterPlotter, inhibitories=is_inhibitory)\
            .plot(ActivityPlotter, x=1)\
            .make_tight()
        # plot.save(f"./hw3/img/{connection_type.__name__}-{index}.png")

plot = Plot(monitor, fig_size=(10, 4))
plot.plot(CurrentTimePlotter)
# plot.save(f"./hw3/img/current.png")
plot.show()
