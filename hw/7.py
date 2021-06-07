import sys
import random

import torch
from tqdm import trange

from cnsproject.learning.rewards import SimpleReward
from cnsproject.utils.general import Integer
from cnsproject.utils.monitor import DummyMonitor

sys.path.append('../')

from cnsproject.encoding.encoders import PoissonEncoder
from cnsproject.network.neural_populations import InputPopulation, PopulationVariables, LIFPopulation, NeuralPopulation
from cnsproject.network.network import Network
from cnsproject.network.monitors import Monitor
from cnsproject.network.connections import DenseConnection
from cnsproject.plotting.plotting import Plot
from cnsproject.plotting.plotters import RasterPlotter, ConnectionWeightsPlotter, DopaminePlotter
from cnsproject.learning.learning_rules import RSTDP, FlatRSTDP

act_time = 200
emp_time = 100
device = "cpu"

parameter_sets = [
    {"title": "Parameter set 1", "name": "RSTDP-0", "rule": RSTDP, "lr_0": .03, "lr_1": .03, "tau_s": 10, "tau_c": 30, "tau_d": 40},
    {"title": "Parameter set 2", "name": "RSTDP-1", "rule": RSTDP, "lr_0": .03, "lr_1": .03, "tau_s": 10, "tau_c": 30, "tau_d": 20},
    {"title": "Parameter set 3", "name": "RSTDP-2", "rule": RSTDP, "lr_0": .03, "lr_1": .03, "tau_s": 8, "tau_c": 30, "tau_d": 20},
    {"title": "Parameter set 4", "name": "RSTDP-3", "rule": RSTDP, "lr_0": .03, "lr_1": .03, "tau_s": 10, "tau_c": 36, "tau_d": 40},
    {"title": "Parameter set 1", "name": "FlatRSTDP-0", "rule": FlatRSTDP, "lr_0": 0.7, "lr_1": .01, "tau_s": 30, "tau_c": -1, "tau_d": 30},
    {"title": "Parameter set 2", "name": "FlatRSTDP-1", "rule": FlatRSTDP, "lr_0": 0.7, "lr_1": .01, "tau_s": 30, "tau_c": -1, "tau_d": 20},
    {"title": "Parameter set 3", "name": "FlatRSTDP-2", "rule": FlatRSTDP, "lr_0": 0.7, "lr_1": .01, "tau_s": 10, "tau_c": -1, "tau_d": 20},
]

pattern_1 = torch.tensor([1, 1, 1, 1, .0, .0, .0, .0])
pattern_2 = pattern_1.flip(0)

encoder = PoissonEncoder(act_time, r=act_time)
spike_train_1 = encoder(pattern_1).to(device)
spike_train_2 = encoder(pattern_2).to(device)
emp = torch.full((emp_time, pattern_1.shape[0]), False)
spike_train_1 = torch.cat((spike_train_1, emp))
spike_train_2 = torch.cat((spike_train_2, emp))
time = act_time + emp_time

spike_dummy_1 = DummyMonitor({PopulationVariables.RB_SPIKES: spike_train_1, PopulationVariables.RB_TIME: torch.arange(0, time)})
spike_dummy_2 = DummyMonitor({PopulationVariables.RB_SPIKES: spike_train_2, PopulationVariables.RB_TIME: torch.arange(0, time)})
plot = Plot(shape=(1, 2), fig_size=(10, 3))
plot.plot(RasterPlotter, monitor=spike_dummy_1)\
    .plot(RasterPlotter, monitor=spike_dummy_2, y=1)\
    .make_tight()\
    .save(f"hw7/img/patterns.png")


class MyReward(SimpleReward):
    def __init__(self, pattern_id: Integer, out: NeuralPopulation, **kwargs):
        super().__init__(**kwargs)
        self.inp_pattern = pattern_id
        self.out = out

        self.window_size = 1
        self.threshold = 1
        self.spikes = torch.tensor([])

    def da(self):
        this_timesteps_spikes = torch.unsqueeze(out.get(PopulationVariables.RB_SPIKES), 0)
        base = self.spikes[-self.window_size:]
        self.spikes = torch.cat((base, this_timesteps_spikes))
        sums = self.spikes.sum(0)
        should_active = sums[0] if self.inp_pattern.value == 1 else sums[1]
        should_not_active = sums[1] if self.inp_pattern.value == 1 else sums[0]
        threshold = self.threshold
        should_active_is_active = should_active >= threshold
        should_not_active_is_active = should_not_active >= threshold
        if should_active_is_active and not should_not_active_is_active:
            result = 0.02
        elif should_not_active_is_active and not should_active_is_active:
            result = -0.05
        elif should_active_is_active and should_not_active_is_active:
            result = -0.01
        else:
            result = (random.random() * 2 - 1) * 0.003
        return result


for parameter_set in parameter_sets:
    seed = 123
    torch.manual_seed(seed)
    random.seed(seed)

    monitor_vars = [
        PopulationVariables.RB_TIME,
        PopulationVariables.RB_SPIKES,
        PopulationVariables.RB_SPIKE_TRACE
    ]

    pattern_id = Integer(1)
    inp = InputPopulation(shape=pattern_1.shape, spike_train=spike_train_1)
    out = LIFPopulation(shape=(2,), tau_s=parameter_set['tau_s'], resistance=1, threshold=-40, tau_t=15, u_rest=-60)

    connection = DenseConnection(
        inp,
        out,
        learning_rule=parameter_set['rule'],
        learning_rule_device=device,
        weight_decay=0.0000,
        tau_c=parameter_set['tau_c'],
        lr=[parameter_set['lr_0'], parameter_set['lr_1']],
        window_size=100,
        w_min=0.000,
        w_max=5,
        j0=1.5,
        s0=1.5,
    )
    monitor_inp = Monitor(inp, monitor_vars, device=device)
    monitor_out = Monitor(out, monitor_vars, device=device)
    monitor_connection = Monitor(connection, ["w"], device=device)
    monitor_lr = Monitor(connection.learning_rule, ['c'], device=device)

    net = Network(learning=True, reward=MyReward, pattern_id=pattern_id, out=out, tau_d=parameter_set['tau_d'])
    monitor_reward = Monitor(net.reward, ['d'], device=device)
    net.add_layer(inp, "input")
    net.add_layer(out, "output")
    net.add_connection(connection, "input", "output")
    net.add_monitor(monitor_inp, "input")
    net.add_monitor(monitor_out, "output")
    net.add_monitor(monitor_connection, "connection")
    net.add_monitor(monitor_reward, "reward")
    net.add_monitor(monitor_lr, "lr")
    net.to(device)
    it = 20

    net.run(total_time=it*time)
    for i in trange(it):
        record = 10

        if torch.rand(1) < .5:
            spike_train = spike_train_2
            pattern_id.set_value(2)
        else:
            spike_train = spike_train_1
            pattern_id.set_value(1)

        inp.change_spike_train(spike_train)
        net.run(time, resume=True, verbose=0)

    plot = Plot(shape=(5, 1), title=parameter_set['title'], fig_size=(17, 13))
    plot.plot(RasterPlotter, monitor=monitor_inp, title="Input Population")\
        .plot(RasterPlotter, monitor=monitor_out, title="Output Population", x=1)\
        .plot(ConnectionWeightsPlotter, monitor=monitor_connection, title="Post neuron 0", post_neuron=0, x=2) \
        .plot(ConnectionWeightsPlotter, monitor=monitor_connection, title="Post neuron 1", post_neuron=1, x=3) \
        .plot(DopaminePlotter, monitor=monitor_reward, x=4) \
        .make_tight()\
        .save(f"hw7/img/{parameter_set['name']}.png")

Plot.show()
exit()
