import sys

import torch
from matplotlib.image import imread

from cnsproject.utils.connections import convolution_2d_connection, convolution_parameters, \
    output_population_parameters, \
    pooling_2d_connection, pooling_parameters
from cnsproject.utils.filters import DoG, ON_CENTER


sys.path.append('../')

from cnsproject.encoding.encoders import Time2FirstSpikeEncoder
from cnsproject.network.neural_populations import InputPopulation, PopulationVariables, LIFPopulation
from cnsproject.network.network import Network
from cnsproject.network.monitors import Monitor
from cnsproject.plotting.plotting import Plot
from cnsproject.plotting.plotters import ImagePlotter


def get_title(s_c, ks_c, s_p, ks_p):
    return "$Stride_{Convolution}: " + str(s_c) +\
        ",~~Kernel~Size_{Convolution}: " + str(ks_c) + "\\times" + str(ks_c) +\
        ",~~Stride_{Pooling}: " + str(s_p) +\
        ",~~Kernel~Size_{Pooling}: " + str(ks_p) + "\\times" + str(ks_p) + "$"


parameters_set = [
    ## base
    {"file": "1", "title": get_title(1, 11, 2, 3), "shape_1": (256, 256), "c_stride": 1, "c_n": 11, "p_stride": 2, "p_size": (3, 3)},
    ## Pooling
    {"file": "2", "title": get_title(1, 11, 4, 3), "shape_1": (256, 256), "c_stride": 1, "c_n": 11, "p_stride": 4, "p_size": (3, 3)},
    {"file": "3", "title": get_title(1, 11, 2, 7), "shape_1": (256, 256), "c_stride": 1, "c_n": 11, "p_stride": 2, "p_size": (7, 7)},
    ## Convolution
    {"file": "4", "title": get_title(2, 11, 2, 3), "shape_1": (256, 256), "c_stride": 2, "c_n": 11, "p_stride": 2, "p_size": (3, 3)},
    {"file": "5", "title": get_title(1, 31, 2, 3), "shape_1": (256, 256), "c_stride": 1, "c_n": 31, "p_stride": 2, "p_size": (3, 3)},
]

time = 6

img = imread('./hw9/img.jpg')
img = torch.tensor(img.sum(2) / 3).float()

encoder_t2fs = Time2FirstSpikeEncoder(5)
spike_train = encoder_t2fs(img)

monitor_vars = [
    PopulationVariables.RB_TIME,
    PopulationVariables.RB_SPIKES,
    PopulationVariables.RB_SPIKE_TRACE,
    PopulationVariables.RB_POTENTIAL
]

plot = Plot(shape=(1, time - 1), fig_size=(17, 4), title="Input Spike Trains")
for i in range(time - 1):
    plot.plot(ImagePlotter, title=f"Input (#{i + 1})", image=spike_train[i], y=i, x=0)
plot.make_tight().save(f"hw9/img/input.png").show()

for p_set in parameters_set:
    conv_kernel = DoG.get(ON_CENTER, n=p_set["c_n"], std_1=2, std_2=3.).unsqueeze(0)
    inp = InputPopulation(shape=(*p_set["shape_1"],), spike_train=spike_train)

    out_conv, convolution_connection = convolution_2d_connection(
        inp,
        LIFPopulation,
        output_population_parameters(threshold=-69, u_rest=-70),
        convolution_parameters(default_kernels=conv_kernel, injection_coef=5, stride=p_set["c_stride"])
    )

    out_pool, pooling_connection = pooling_2d_connection(
        out_conv,
        LIFPopulation,
        output_population_parameters(),
        pooling_parameters(kernel_size=p_set["p_size"], stride=p_set["p_stride"])
    )

    monitor_inp = Monitor(inp, monitor_vars)
    monitor_out_conv = Monitor(out_conv, monitor_vars)
    monitor_out_pool = Monitor(out_pool, monitor_vars)

    net = Network(learning=False)
    net.add_layer(inp, "input")
    net.add_layer(out_conv, "out_conv")
    net.add_layer(out_pool, "out_pool")
    net.add_connection(convolution_connection, "input", "out_conv")
    net.add_connection(pooling_connection, "out_conv", "out_pool")
    net.add_monitor(monitor_inp, "input")
    net.add_monitor(monitor_out_pool, "out_pool")
    net.add_monitor(monitor_out_conv, "out_conv")
    net.run(time)

    spk_conv = monitor_out_conv.get("s").reshape(time, *out_conv.shape[-2:])
    spk_pool = monitor_out_pool.get("s").reshape(time, *out_pool.shape[-2:])

    plot = Plot(shape=(2, time - 1), fig_size=(17, 8), title=p_set['title'])

    for i in range(time - 1):
        plot.plot(ImagePlotter, title=f"Convolution Output (#{i + 1})", image=spk_conv[i + 1], y=i, x=0)

    for i in range(time - 1):
        plot.plot(ImagePlotter, title=f"Pooling Output (#{i + 1})", image=spk_pool[i + 1], y=i, x=1)

    plot.make_tight().save(f"hw9/img/{p_set['file']}.png").show()

