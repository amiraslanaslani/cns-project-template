import sys

sys.path.append('../')

from cnsproject.network.neural_populations import LIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.plotting.plotting import Plot
from cnsproject.plotting.plotters import CurrentTimePlotter, PotentialTimePlotter, FIPlotter

import torch

save_file = False
pre_file = 1

def get_result_from_vector(
        current_vector: torch.Tensor,
        neuron: LIFPopulation,
        time: int,
        dt: float = 1,
        title=None,
        file: str = ""
):
    neuron.set_time_step(dt)

    monitor = Monitor(neuron, state_variables=["u", "time", "current", "s"])
    monitor.set_time_steps(time, dt)
    monitor.reset_state_variables()
    i = current_vector

    for t in range(int(time / dt)):
        neuron.forward(i[t])
        monitor.record()

    monitor.get("u")

    plot = Plot(monitor, shape=(2, 4), grid_spec=True, fig_size=(10, 5), title=title)
    plot.plot(CurrentTimePlotter, x=0, y=0, y_until=4, time_unit="ms") \
        .plot(PotentialTimePlotter, x=1, y=0, y_until=4, time_unit="ms", spikes=True) \
        .make_tight()

    if save_file:
        plot.save(f"hw1/{pre_file}-{file}.png")

    neuron.reset_state_variables()


def get_result_from_step_function(
        current_value: float,
        neuron: LIFPopulation,
        time,
        noise_value: float = 0,
        dt: float = 1
):
    size = int(time / dt)
    i = torch.zeros(size)
    i[int(size * 0.05):int(size * 0.8)] = current_value

    if noise_value > 0:
        i = i + (noise_value * torch.rand(size)) - (noise_value / 2)

    return get_result_from_vector(i, neuron, time, dt, file=f"U-{current_value}")


def get_result_from_noisy_current(
        neuron: LIFPopulation,
        time: int,
        noise_value: float = 0,
        dt: float = 1
):
    size = int(time / dt)
    noise_value = torch.tensor(noise_value)
    mx = torch.arange(0, 0.25 * size, 0.25)
    nx = torch.sin(mx / torch.tensor(10)) * 40 + mx
    nx = nx + (noise_value * torch.rand(size)) - (noise_value / 2)
    nx = torch.max(nx, torch.tensor(0))
    nx = nx / torch.tensor(20)
    return get_result_from_vector(nx, neuron, time, dt, file=f"Noisy-{noise_value}")


def do_for_neuron(neuron):
    get_result_from_step_function(2, neuron, 1000)
    get_result_from_step_function(5, neuron, 1000)
    get_result_from_step_function(10, neuron, 1000)
    get_result_from_step_function(15, neuron, 1000)

    plot = Plot(fig_size=(5, 5))
    plot.plot(FIPlotter, neuron=neuron, current_to=30) \
        .make_tight()

    if save_file:
        plot.save(f"hw1/{pre_file}-FI.png")

    get_result_from_noisy_current(neuron, 1000, 5)
    get_result_from_noisy_current(neuron, 1000, 50)
    get_result_from_noisy_current(neuron, 1000, 100)
    get_result_from_noisy_current(neuron, 1000, 500)


do_for_neuron(LIFPopulation(
    shape=(1,),
    resistance=5,
    tau_t=100,
    threshold=-50,
    u_rest=-70
))
pre_file = pre_file + 1

do_for_neuron(LIFPopulation(
    shape=(1,),
    resistance=5,
    tau_t=400,
    threshold=-50,
    u_rest=-70
))
pre_file = pre_file + 1

do_for_neuron(LIFPopulation(
    shape=(1,),
    resistance=20,
    tau_t=100,
    threshold=-50,
    u_rest=-70
))
pre_file = pre_file + 1

do_for_neuron(LIFPopulation(
    shape=(1,),
    resistance=5,
    tau_t=100,
    threshold=-30,
    u_rest=-70
))
pre_file = pre_file + 1

do_for_neuron(LIFPopulation(
    shape=(1,),
    resistance=5,
    tau_t=100,
    threshold=-50,
    u_rest=-100
))



Plot.show()
