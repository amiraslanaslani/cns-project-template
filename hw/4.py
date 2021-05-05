import sys

from cnsproject.network import Network
from cnsproject.network.connections import RandomConnection
from cnsproject.network.monitors import Monitor

sys.path.append('../')

import torch

from cnsproject.network.neural_populations import LIFPopulation, PopulationVariables
from cnsproject.plotting.plotting import Plot
from cnsproject.utils.general import get_random_current, population_type
from cnsproject.utils.monitor import DummyMonitor
from cnsproject.plotting.plotters import SimplePlotter, RasterPlotter, ActivityPlotter


def get_param_set(**kwargs):
    return kwargs


torch.manual_seed(13)  # 112
SAVE = False

EXC_SIZE = 100
INH_SIZE = 100
exc_params = [
    get_param_set(resistance=5, threshold=-60, tau_t=800, u_rest=-70),
    get_param_set(resistance=5, threshold=-63, tau_t=600, u_rest=-70),
    get_param_set(resistance=5, threshold=-63, tau_t=600, u_rest=-70),
    get_param_set(resistance=5, threshold=-60, tau_t=800, u_rest=-70),
]
inh_params = [
    get_param_set(resistance=5, threshold=-60, tau_t=800, u_rest=-70),
    get_param_set(resistance=5, threshold=-67, tau_t=300, u_rest=-70),
    get_param_set(resistance=5, threshold=-60, tau_t=800, u_rest=-70),
    get_param_set(resistance=5, threshold=-67, tau_t=300, u_rest=-70),
]
j0 = 10
s0 = 30
BINS = 500
TIME = 5000
PROB = .4
monitor_variables = [
    PopulationVariables.RB_POTENTIAL,
    PopulationVariables.RB_TIME,
    PopulationVariables.RB_SPIKES,
    PopulationVariables.RB_CURRENT
]

current_1 = get_random_current(TIME, start=5, coef=.3)
current_2 = current_1.clone().detach()
current_2[int(TIME / 2):] *= 1.05

dm = DummyMonitor({
    "c_1": current_1,
    "c_2": current_2
})
plot = Plot(dm, fig_size=(10, 4))
plot.plot(SimplePlotter, x_name="c_1", xlabel="Time (ms)", ylabel="Current", linewidth=1.5, label="Excitatory 1")\
    .plot(SimplePlotter, x_name="c_2", legend=True, linewidth=0.5, label="Excitatory 2")\
    .make_tight()
if SAVE:
    plot.save(f"hw4/img/current.png")

monitor_1, monitor_2 = None, None
for connection_ratio in [.4, .8]:
    p_set = 1
    for exc_param, inh_param in zip(exc_params, inh_params):
        exc_1 = LIFPopulation(
            shape=(EXC_SIZE,),
            **exc_param
        )

        exc_2 = LIFPopulation(
            shape=(EXC_SIZE,),
            **exc_param
        )

        inh = LIFPopulation(
            shape=(INH_SIZE,),
            is_inhibitory=population_type(size=INH_SIZE, inhibitory_ratio=1.),
            **inh_param
        )

        c_1_1 = RandomConnection(exc_1, j0=j0, s0=s0, probability=connection_ratio)
        c_2_2 = c_1_1.copy(exc_2, exc_2)
        c_3_1 = RandomConnection(inh, exc_1, j0=j0, s0=s0, probability=connection_ratio)
        c_3_2 = c_3_1.copy(inh, exc_2)
        c_1_3 = RandomConnection(exc_1, inh, j0=j0, s0=s0, probability=connection_ratio)
        c_2_3 = c_1_3.copy(exc_2, inh)

        device = "cpu"
        monitor_1 = Monitor(exc_1, monitor_variables, device=device)
        monitor_2 = Monitor(exc_2, monitor_variables, device=device)
        monitor_3 = Monitor(inh, monitor_variables, device=device)

        net = Network(learning=False)
        net.add_layer(exc_1, "exc_1")
        net.add_layer(exc_2, "exc_2")
        net.add_layer(inh, "inh")
        net.add_connection(c_1_1, "exc_1", "exc_1")
        net.add_connection(c_2_2, "exc_2", "exc_2")
        net.add_connection(c_3_1, "inh", "exc_1")
        net.add_connection(c_3_2, "inh", "exc_2")
        net.add_connection(c_1_3, "exc_1", "inh")
        net.add_connection(c_2_3, "exc_2", "inh")
        net.add_monitor(monitor_1, "exc_1")
        net.add_monitor(monitor_2, "exc_2")
        net.add_monitor(monitor_3, "inh")
        # net = net.cuda()

        net.run(TIME, random_factor=0, currents={
            "exc_1": current_1,
            "exc_2": current_2,
        })

        plot_1 = Plot(
            monitor_1,
            (2, 1),
            title=f"Excitatory 1 (Parameter Set {p_set} - Connection Ratio {connection_ratio})",
            fig_size=(10, 6)
        )
        plot_1.plot(RasterPlotter)\
              .plot(ActivityPlotter, x=1, bins=BINS)\
              .make_tight()

        plot_2 = Plot(
            monitor_2,
            (2, 1),
            title=f"Excitatory 2 (Parameter Set {p_set} - Connection Ratio {connection_ratio})",
            fig_size=(10, 6)
        )
        plot_2.plot(RasterPlotter)\
              .plot(ActivityPlotter, x=1, bins=BINS)\
              .make_tight()

        plot_3 = Plot(
            monitor_3,
            (2, 1),
            title=f"Inhibitory (Parameter Set {p_set} - Connection Ratio {connection_ratio})",
            fig_size=(10, 6)
        )
        plot_3.plot(RasterPlotter)\
              .plot(ActivityPlotter, x=1, bins=BINS)\
              .make_tight()

        if SAVE:
            plot_1.save(f"hw4/img/{connection_ratio}-{p_set}-1.png")
            plot_2.save(f"hw4/img/{connection_ratio}-{p_set}-2.png")
            plot_3.save(f"hw4/img/{connection_ratio}-{p_set}-3.png")

        p_set = p_set + 1

# Plot.show()
