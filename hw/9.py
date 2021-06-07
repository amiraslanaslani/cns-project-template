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


img = imread(f'./hw9/img1.jpg')
img = torch.tensor(img.sum(2) / 3).float()

encoder_t2fs = Time2FirstSpikeEncoder(50)
spike_train = encoder_t2fs(img)

monitor_vars = [
    PopulationVariables.RB_TIME,
    PopulationVariables.RB_SPIKES,
    PopulationVariables.RB_SPIKE_TRACE
]

inp = InputPopulation(shape=(150, 150), spike_train=spike_train)
out = LIFPopulation(shape=(3, 3), tau_s=30, resistance=5, threshold=-50, tau_t=800, u_rest=-70)
connection = DenseConnection(
    inp,
    out,
    j0=4,
    s0=4,
)

monitor_inp = Monitor(inp, monitor_vars)
monitor_out = Monitor(out, monitor_vars)
monitor_connection = Monitor(connection, ["w"])
