import sys

import torch
import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt
from tqdm import trange

from cnsproject.utils.general import get_fixed_current
from cnsproject.utils.learning import hard_bound, soft_bound
from cnsproject.utils.monitor import DummyMonitor

sys.path.append('../')

from cnsproject.encoding.encoders import PositionEncoder, Time2FirstSpikeEncoder, PoissonEncoder
from cnsproject.network.neural_populations import InputPopulation, PopulationVariables, LIFPopulation
from cnsproject.network.network import Network
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection
from cnsproject.plotting.plotting import Plot
from cnsproject.plotting.plotters import SpikeTracePlotter, RasterPlotter, ConnectionWeightsPlotter
from cnsproject.learning.learning_rules import STDP, FlatSTDP

time = 130
device = "cpu"

parameter_sets = [
    {'seed': 6, "title": "Parameter set 1", "name": "FlatSTDP-0", "rule": FlatSTDP, "lr_0": .1, "lr_1": .05, "tau_s": 30, "threshold": -52, "current": 20, "j0": 5},
    # {'seed': 6, "title": "Parameter set 2", "name": "FlatSTDP-1", "rule": FlatSTDP, "lr_0": .1, "lr_1": .05, "tau_s": 30, "threshold": -52, "current": 20, "j0": 7},
    # {'seed': 6, "title": "Parameter set 3", "name": "FlatSTDP-2", "rule": FlatSTDP, "lr_0": .2, "lr_1":  .1, "tau_s": 30, "threshold": -52, "current": 20, "j0": 5},
    # {'seed': 6, "title": "Parameter set 4", "name": "FlatSTDP-3", "rule": FlatSTDP, "lr_0": .1, "lr_1": .05, "tau_s": 20, "threshold": -52, "current": 20, "j0": 5},
    # {'seed': 6, "title": "Parameter set 5", "name": "FlatSTDP-4", "rule": FlatSTDP, "lr_0": .1, "lr_1": .05, "tau_s": 30, "threshold": -54, "current": 20, "j0": 5},
    # {'seed': 35, "title": "Parameter set 1", "name": "STDP-0", "rule": STDP, "lr_0": .1, "lr_1": .09, "tau_s": 19, "threshold": -55, "current": 10, "j0": 5},
    # {'seed': 35, "title": "Parameter set 2", "name": "STDP-1", "rule": STDP, "lr_0": .1, "lr_1": .09, "tau_s": 19, "threshold": -55, "current": 10, "j0": 7},
    # {'seed': 35, "title": "Parameter set 3", "name": "STDP-2", "rule": STDP, "lr_0": .22, "lr_1": .2, "tau_s": 19, "threshold": -55, "current": 10, "j0": 5},
    # {'seed': 35, "title": "Parameter set 4", "name": "STDP-3", "rule": STDP, "lr_0": .1, "lr_1": .09, "tau_s": 17, "threshold": -55, "current": 10, "j0": 5},
    # {'seed': 35, "title": "Parameter set 5", "name": "STDP-4", "rule": STDP, "lr_0": .1, "lr_1": .09, "tau_s": 19, "threshold": -65, "current": 10, "j0": 5},
]

pattern_1 = torch.tensor([1, 1, 1, 1, .05, .05, .05, .05])
pattern_2 = pattern_1.flip(0)
encoder = PoissonEncoder(time, r=13)
spike_dummy_1 = DummyMonitor({PopulationVariables.RB_SPIKES: encoder(pattern_1), PopulationVariables.RB_TIME: torch.arange(0, 130)})
spike_dummy_2 = DummyMonitor({PopulationVariables.RB_SPIKES: encoder(pattern_2), PopulationVariables.RB_TIME: torch.arange(0, 130)})
plot = Plot(shape=(1, 2), fig_size=(10, 3))
plot.plot(RasterPlotter, monitor=spike_dummy_1)\
    .plot(RasterPlotter, monitor=spike_dummy_2, y=1)\
    .make_tight()\
    .save(f"hw6/img/patterns.png")

for parameter_set in parameter_sets:
    torch.manual_seed(parameter_set['seed'])  # 112

    # pattern_1 = torch.tensor([1, 1, 1, 1, .05, .05, .05, .05])
    # pattern_2 = pattern_1.flip(0)

    monitor_vars = [
        PopulationVariables.RB_TIME,
        PopulationVariables.RB_SPIKES,
        PopulationVariables.RB_SPIKE_TRACE
    ]

    encoder = PoissonEncoder(time, r=13)
    spike_train_1 = encoder(pattern_1).to(device)
    spike_train_2 = encoder(pattern_2).to(device)

    inp = InputPopulation(shape=pattern_1.shape, spike_train=spike_train_1)
    out = LIFPopulation(shape=(2,), tau_s=parameter_set['tau_s'], resistance=5, threshold=parameter_set['threshold'], tau_t=800, u_rest=-70)
    connection = DenseConnection(
        inp,
        out,
        learning_rule=parameter_set['rule'],
        learning_rule_device=device,
        weight_decay=0.00005,
        lr=[parameter_set['lr_0'], parameter_set['lr_1']],
        w_min=0.000,
        w_max=5,
        j0=parameter_set['j0'],
        s0=4,
    )
    monitor_inp = Monitor(inp, monitor_vars, device=device)
    monitor_out = Monitor(out, monitor_vars, device=device)
    monitor_connection = Monitor(connection, ["w"], device=device)

    net = Network(learning=True)
    net.add_layer(inp, "input")
    net.add_layer(out, "output")
    net.add_connection(connection, "input", "output")
    net.add_monitor(monitor_inp, "input")
    net.add_monitor(monitor_out, "output")
    net.add_monitor(monitor_connection, "connection")
    net.to(device)
    it = 100

    # net.run(time)

    net.run(total_time=it*time)
    for i in trange(it):
        record = 10
        if i == it - record:
            monitor_out.set_time_steps(time*record)
            monitor_inp.set_time_steps(time*record)

        inp.change_spike_train(spike_train_2 if torch.rand(1) < .5 else spike_train_1)
        net.run(time, resume=True, verbose=0, currents={
            "output": get_fixed_current(10, time)
        })

    plot = Plot(shape=(4, 1), title=parameter_set['title'], fig_size=(17, 11))
    plot.plot(RasterPlotter, monitor=monitor_inp, title="Input Population")\
        .plot(RasterPlotter, monitor=monitor_out, title="Output Population", x=1)\
        .plot(ConnectionWeightsPlotter, monitor=monitor_connection, title="Post neuron 0", post_neuron=0, x=2)\
        .plot(ConnectionWeightsPlotter, monitor=monitor_connection, title="Post neuron 1", post_neuron=1, x=3)\
        .make_tight()\
        .save(f"hw6/img/{parameter_set['name']}.png")

Plot.show()

