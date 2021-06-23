import sys

from matplotlib.image import imread
import torch

sys.path.append('../')

from cnsproject.plotting.plotters import RasterPlotter, FunctionPlotter, ImagePlotter
from cnsproject.plotting.plotting import Plot
from cnsproject.utils.constants import PI
from cnsproject.utils.filters import DoG, OFF_CENTER, ON_CENTER, convolve2d, Gabor
from cnsproject.encoding.encoders import Time2FirstSpikeEncoder, PoissonEncoder
from cnsproject.utils.monitor import DummyMonitor


filters = [
    ("on-dog-13-1-5", DoG.get(ON_CENTER, n=13, std_1=1., std_2=5.), "DoG Filter - On-Center ($13 \\times 13 ~ Kernel ~-~ \\sigma_1: 1 ,~ \\sigma_2: 5$)"),
    ("on-dog-13-2-5", DoG.get(ON_CENTER, n=13, std_1=2., std_2=5.), "DoG Filter - On-Center ($13 \\times 13 ~ Kernel ~-~ \\sigma_1: 2 ,~ \\sigma_2: 5$)"),
    ("on-dog-13-1-4", DoG.get(ON_CENTER, n=13, std_1=1., std_2=4.), "DoG Filter - On-Center ($13 \\times 13 ~ Kernel ~-~ \\sigma_1: 1 ,~ \\sigma_2: 4$)"),

    ("off-dog-13-1-5", DoG.get(OFF_CENTER, n=13, std_1=1., std_2=5.), "DoG Filter - Off-Center ($13 \\times 13 ~ Kernel ~-~ \\sigma_1: 1 ,~ \\sigma_2: 5$)"),
    ("off-dog-13-2-5", DoG.get(OFF_CENTER, n=13, std_1=2., std_2=5.), "DoG Filter - Off-Center ($13 \\times 13 ~ Kernel ~-~ \\sigma_1: 2 ,~ \\sigma_2: 5$)"),
    ("off-dog-13-1-4", DoG.get(OFF_CENTER, n=13, std_1=1., std_2=4.), "DoG Filter - Off-Center ($13 \\times 13 ~ Kernel ~-~ \\sigma_1: 1 ,~ \\sigma_2: 4$)"),

    ("on-gabor-13-5-000-5-2", Gabor.get(ON_CENTER, n=13, wavelen=5, theta=0, sigma=5, gamma=2), "Gabor Filter - On-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: 0 ,~ \\sigma: 5 ,~ \\gamma: 2$)"),
    ("on-gabor-13-5-000-8-2", Gabor.get(ON_CENTER, n=13, wavelen=5, theta=0, sigma=8, gamma=2), "Gabor Filter - On-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: 0 ,~ \\sigma: 8 ,~ \\gamma: 2$)"),
    ("on-gabor-13-2-000-5-2", Gabor.get(ON_CENTER, n=13, wavelen=2, theta=0, sigma=5, gamma=2), "Gabor Filter - On-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 2 ,~ \\theta: 0 ,~ \\sigma: 5 ,~ \\gamma: 2$)"),
    ("on-gabor-13-5-045-5-2", Gabor.get(ON_CENTER, n=13, wavelen=5, theta=PI/4, sigma=5, gamma=2), "Gabor Filter - On-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: \\pi/4 ,~ \\sigma: 5 ,~ \\gamma: 2$)"),
    ("on-gabor-13-5-090-5-2", Gabor.get(ON_CENTER, n=13, wavelen=5, theta=PI/2, sigma=5, gamma=2), "Gabor Filter - On-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: \\pi/2 ,~ \\sigma: 5 ,~ \\gamma: 2$)"),
    ("on-gabor-13-5-135-5-2", Gabor.get(ON_CENTER, n=13, wavelen=5, theta=3*PI/4, sigma=5, gamma=2), "Gabor Filter - On-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: 3\\pi/4 ,~ \\sigma: 5 ,~ \\gamma: 2$)"),
    ("on-gabor-13-5-000-5-1", Gabor.get(ON_CENTER, n=13, wavelen=5, theta=0, sigma=5, gamma=1), "Gabor Filter - On-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: 0 ,~ \\sigma: 5 ,~ \\gamma: 1$)"),

    ("off-gabor-13-5-000-5-2", Gabor.get(OFF_CENTER, n=13, wavelen=5, theta=0, sigma=5, gamma=2), "Gabor Filter - Off-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: 0 ,~ \\sigma: 5 ,~ \\gamma: 2$)"),
    ("off-gabor-13-5-000-8-2", Gabor.get(OFF_CENTER, n=13, wavelen=5, theta=0, sigma=8, gamma=2), "Gabor Filter - Off-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: 0 ,~ \\sigma: 8 ,~ \\gamma: 2$)"),
    ("off-gabor-13-2-000-5-2", Gabor.get(OFF_CENTER, n=13, wavelen=2, theta=0, sigma=5, gamma=2), "Gabor Filter - Off-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 2 ,~ \\theta: 0 ,~ \\sigma: 5 ,~ \\gamma: 2$)"),
    ("off-gabor-13-5-045-5-2", Gabor.get(OFF_CENTER, n=13, wavelen=5, theta=PI/4, sigma=5, gamma=2), "Gabor Filter - Off-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: \\pi/4 ,~ \\sigma: 5 ,~ \\gamma: 2$)"),
    ("off-gabor-13-5-090-5-2", Gabor.get(OFF_CENTER, n=13, wavelen=5, theta=PI/2, sigma=5, gamma=2), "Gabor Filter - Off-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: \\pi/2 ,~ \\sigma: 5 ,~ \\gamma: 2$)"),
    ("off-gabor-13-5-135-5-2", Gabor.get(OFF_CENTER, n=13, wavelen=5, theta=3*PI/4, sigma=5, gamma=2), "Gabor Filter - Off-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: 3\\pi/4 ,~ \\sigma: 5 ,~ \\gamma: 2$)"),
    ("off-gabor-13-5-000-5-1", Gabor.get(OFF_CENTER, n=13, wavelen=5, theta=0, sigma=5, gamma=1), "Gabor Filter - Off-Center ($13 \\times 13 ~ Kernel ~-~ \\lambda: 5 ,~ \\theta: 0 ,~ \\sigma: 5 ,~ \\gamma: 1$)"),
]

for imgage_file in ['img1', 'img2']:
    img = imread(f'./hw8/{imgage_file}.jpg')
    img = torch.tensor(img.sum(2) / 3).float()

    for file_name, kernel, title in filters:
        kernel_c = convolve2d(img, kernel, clip=(0, 255))

        kernel_flat = torch.flatten(kernel_c)
        encoder_t2fs = Time2FirstSpikeEncoder(255)
        dm_t2fs = DummyMonitor({"s": encoder_t2fs(kernel_flat), "time": torch.arange(256)})

        encoder_pois = PoissonEncoder(255, r=20)
        encoded_pois = encoder_pois(kernel_c)

        plot = Plot(shape=(2, 3), grid_spec=True, fig_size=(15, 10), title=title)
        plot.plot(RasterPlotter, monitor=dm_t2fs, x=0, y=1, y_until=3, title="Time 2 First Spike")\
            .plot(ImagePlotter, image=kernel, x=1, y=0, title="Filter")\
            .plot(ImagePlotter, image=kernel_c, x=0, y=0, title="Image Output", cmap="magma")\
            .plot(FunctionPlotter, function=lambda ax: ax.imshow(encoded_pois[10], cmap="gray"), x=1, y=1, title="Poisson (10th ms)")\
            .plot(FunctionPlotter, function=lambda ax: ax.imshow(encoded_pois[50], cmap="gray"), x=1, y=2, title="Poisson (50th ms)")\
            .make_tight()\
            .save(f"hw8/img/{imgage_file}-{file_name}.png")

Plot.show()
