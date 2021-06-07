import sys

import torch
import numpy as np
from matplotlib.image import imread
from matplotlib import pyplot as plt

sys.path.append('../')

from cnsproject.encoding.encoders import PositionEncoder, Time2FirstSpikeEncoder, PoissonEncoder
from cnsproject.network.neural_populations import InputPopulation, PopulationVariables
from cnsproject.network.network import Network
from cnsproject.network.monitors import Monitor
from cnsproject.plotting.plotting import Plot
from cnsproject.plotting.plotters import RasterPlotter


img = imread('./hw5/resized.png')
img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
img = (img * 255).round().astype(int)
plt.imshow(img, cmap="gray")
img = torch.flatten(torch.tensor(img))

for time in [255, 2000]:
    encoders = [
        Time2FirstSpikeEncoder(time),
        PoissonEncoder(time, r=3),
        PoissonEncoder(time, r=10),
        PositionEncoder(time, peaks=torch.arange(0, 256, 5), std=20),
        PositionEncoder(time, peaks=torch.arange(0, 256, 15), std=20),
        PositionEncoder(time, peaks=torch.arange(0, 256, 5), std=40)
    ]

    titles = [
        f"Time to First Spike Encoder ($T={time}$)",
        f"Poisson Encoder ($T={time}, r=3$)",
        f"Poisson Encoder ($T={time}, r=10$)",
        f"Position Encoder ($T={time}, \\Delta \\mu = 5, STD = 20$)",
        f"Position Encoder ($T={time}, \\Delta \\mu = 15, STD = 20$)",
        f"Position Encoder ($T={time}, \\Delta \\mu = 5, STD = 40$)",
    ]

    file_titles = [
        f"time2fs",
        f"poisson-3",
        f"poisson-10",
        f"position-5-20",
        f"position-15-20",
        f"position-5-40",
    ]

    for title, file_title, encoder in zip(titles, file_titles, encoders):
        spike_train = encoder(img)
        pop = InputPopulation(spike_train.shape[1:], spike_train=spike_train)
        monitor = Monitor(pop, [PopulationVariables.RB_TIME, PopulationVariables.RB_SPIKES])

        net = Network(learning=False)
        net.add_layer(pop, "input")
        net.add_monitor(monitor, "input")
        net.run(time)

        plot = Plot(monitor, fig_size=(10, 5), title=title)
        plot.plot(RasterPlotter)\
            .make_tight()


        plot.save(f"./hw5/img/%s-%s.png" % (time, file_title))

Plot.show()
