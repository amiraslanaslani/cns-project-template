import glob
import sys

import numpy as np
import torch
from PIL import Image
from tqdm import trange

from cnsproject.learning.learning_rules import Conv2dSTDP
from cnsproject.mechanism.decision import WinnerTakeAllInhibition
from cnsproject.utils.connections import convolution_2d_connection, convolution_parameters, \
    output_population_parameters
from cnsproject.utils.filters import DoG, convolve2d
from cnsproject.utils.general import get_fixed_current

sys.path.append('../')

from cnsproject.encoding.encoders import IntensityToLatency
from cnsproject.network.neural_populations import InputPopulation, PopulationVariables, LIFPopulation
from cnsproject.network.network import Network
from cnsproject.network.monitors import Monitor
from cnsproject.plotting.plotting import Plot
from cnsproject.plotting.plotters import ImagePlotter

torch.manual_seed(123)

parameters_set = [
    {'conv_size': 20, 'k': 5, 'lateral_inhibition_r': int(10 / 2)},
    {'conv_size': 40, 'k': 5, 'lateral_inhibition_r': int(10 / 2)},
    {'conv_size': 20, 'k': 7, 'lateral_inhibition_r': int(10 / 2)},
    {'conv_size': 20, 'k': 5, 'lateral_inhibition_r': int(20 / 2)},
]


# Parameters
standard_size = (50, 50)
F = 10
enc_time = 3
i_counter = 2
filter = DoG.get(n=5, std_1=1.2, std_2=1.5)

spike_trains = []
print("Loading dataset . . . ")
for img_f in glob.glob("./hw10/faces/*.jpg"):
    img = Image.open(img_f)
    img = img.resize(standard_size)
    img = np.array(img)

    if img.ndim > 2:
        img = torch.tensor(img.sum(2) / 3).float()
    else:
        img = torch.tensor(img).float()

    convolved_img = convolve2d(img, filter, clip=(0, 255))

    encoder_t2fs = IntensityToLatency(enc_time, d_max=convolved_img.max(), d_min=0)
    spike_train = encoder_t2fs(convolved_img)
    spike_trains.append(spike_train)


for param in parameters_set:
    K = param['k']
    conv_size = param['conv_size']
    lateral_inhibitions_radius = param['lateral_inhibition_r']

    # Network Structure
    inp = InputPopulation(shape=(*spike_trains[0].shape[1:],), spike_train=spike_trains[0])

    out_conv, convolution_connection = convolution_2d_connection(
        inp,
        LIFPopulation,
        output_population_parameters(threshold=-67, u_rest=-70),
        convolution_parameters(
            default_kernels=None,
            injection_coef=1,
            learning_rule=Conv2dSTDP,
            lr=[.08, .06],
            w_min=0.000,
            w_max=5,
            filters=F,
            kernel_size=conv_size
        )
    )

    kwta = WinnerTakeAllInhibition(k=K, lateral_inhibitions_radius=lateral_inhibitions_radius, inactivity_counter=i_counter)
    out_conv.register_mechanism_object(kwta)

    net = Network(learning=True)
    net.add_layer(inp, "input")
    net.add_layer(out_conv, "out_conv")

    net.add_connection(convolution_connection, "out_dog", "out_conv")

    monitor_vars_pop = [
        PopulationVariables.RB_TIME,
        PopulationVariables.RB_SPIKES,
        PopulationVariables.RB_SPIKE_TRACE,
        PopulationVariables.RB_POTENTIAL
    ]

    monitor_inp = Monitor(inp, monitor_vars_pop)
    monitor_out_conv = Monitor(out_conv, monitor_vars_pop)

    monitor_vars_conn = ["w"]

    monitor_conn_conv = Monitor(convolution_connection, monitor_vars_conn)


    net.add_monitor(monitor_inp, "monitor_inp")
    net.add_monitor(monitor_out_conv, "monitor_out_conv")
    net.add_monitor(monitor_conn_conv, "monitor_conn_conv")

    it = len(spike_trains) * 3 + 1
    time = enc_time + 1

    file_name = lambda iter: f"{conv_size}-{K}-{lateral_inhibitions_radius}-{iter}.png"

    net.run(total_time=it*time)
    for i in trange(it):
        if i % len(spike_trains) == 0:
            row_size = int(F / 2)
            title = f"$Convolution~Kernel~Size: {conv_size} \\times {conv_size}, ~~K: {K}, ~Lateral~Inhibition's~Area~Size: {lateral_inhibitions_radius*2} \\times {lateral_inhibitions_radius*2}$"
            plot = Plot(shape=(2, int(F / 2)), fig_size=(17, 8), title=f"Learned Filters ({title})")
            for j, filter in enumerate(convolution_connection.w):
                plot.plot(ImagePlotter, title=f"Filter #{j + 1}", image=filter[0], cmap='gray', x=int(j / row_size), y=j % row_size)
            plot.make_tight().save(f"hw10/img/{file_name(i)}").show()

        kwta.forward()
        inp.change_spike_train(spike_trains[i % len(spike_trains)])
        net.run(time, resume=True, verbose=0, currents={
            "output": get_fixed_current(5, time)
        })


