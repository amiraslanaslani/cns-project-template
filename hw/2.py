import sys
from typing import List

sys.path.append('../')

from cnsproject.network.neural_populations import NeuralPopulation, LIFPopulation, ELIFPopulation, AELIFPopulation
from cnsproject.network.monitors import Monitor
from cnsproject.plotting.plotting import Plot
from cnsproject.plotting.plotters import FIPlotter, CurrentTimePlotter, PotentialTimePlotter, AdaptionTimePlotter
from cnsproject.utils import get_monitor_of_simulation, get_fixed_current

import torch


SAVE = True


def get_result_from_vector(
        current_vector: torch.Tensor,
        neuron: NeuralPopulation,
        time: int,
        dt: float = 1,
        title=None,
        draw_current: bool = False
):
    neuron.set_timestep(dt)

    monitor = get_monitor_of_simulation(
        neuron,
        current_vector,
        time,
        dt,
        [
            AELIFPopulation.RB_POTENTIAL,
            AELIFPopulation.RB_TIME,
            AELIFPopulation.RB_SPIKES,
            AELIFPopulation.RB_CURRENT,
            # AELIFPopulation.RB_ADAPTION
        ]
    )

    plot = Plot(monitor, shape=(1 + (1 if draw_current else 0), 4), grid_spec=True, figsize=(10, 3 * (2 if draw_current else 1)))
    plot.plot(PotentialTimePlotter, x=0, y=0, y_until=4, time_unit="ms", spikes=True, title=f"model: {neuron.__class__.__name__}" + ("" if draw_current else f"I = {current_vector.max()}"))

    if draw_current:
        plot.plot(CurrentTimePlotter, x=1, y=0, y_until=4, time_unit="ms")

    plot.make_tight()
    if SAVE:
        plot.save(title)

    # .plot(AdaptionTimePlotter, x=1, y=0, y_until=4, time_unit="ms") \

    neuron.reset_state_variables()


def do_for_all(
        models,
        params,
        currents_vector: List[torch.Tensor],
        time,
        dt: float = 1
):
    for model in models:
        for param_index, param in enumerate(params):
            neuron = model(**param)
            for current_vector in currents_vector:
                get_result_from_vector(
                    current_vector,
                    neuron,
                    time,
                    dt,
                    f"{model.__name__}-{param_index}-{current_vector.max()}.png"
                )

            plot = Plot(figsize=(5, 5))
            plot.plot(FIPlotter, neuron=neuron, current_to=30, title=f"model: {neuron.__class__.__name__}") \
                .make_tight()

            if SAVE:
                plot.save(f"{model.__name__}-{param_index}-FI.png")


def get_param_set(**kwargs):
    return kwargs


def get_result_from_noisy_current(
        neuron: LIFPopulation,
        time: int,
        noise_value: float = 0,
        dt: float = 1,
        title = ""
):
    size = int(time / dt)
    noise_value = torch.tensor(noise_value)
    mx = torch.arange(0, 0.25 * size, 0.25)
    nx = torch.sin(mx / torch.tensor(10)) * 40 + mx
    nx = nx + (noise_value * torch.rand(size)) - (noise_value / 2)
    nx = torch.max(nx, torch.tensor(0))
    nx = nx / torch.tensor(20)
    return get_result_from_vector(nx, neuron, time, dt, title, draw_current=True)


def do_noisy_for_all(
        models,
        params,
        noise_vector: List[int],
        time,
        dt: float = 1
):
    for model in models:
        for param_index, param in enumerate(params):
            for noise in noise_vector:
                neuron = model(**param)
                get_result_from_noisy_current(
                    neuron,
                    time,
                    noise,
                    dt,
                    f"Noisy-{model.__name__}-{param_index}-{noise}"
                )


data_scale = 1000  # 1000 for ms | 1 for s
time = 1000
dt = 1

parameters_set = [
    get_param_set(
        shape=(1,), resistance=5, tau_t=0.2 * data_scale, threshold=-40, u_rest=-70,
        sharpness=1, theta_rh=-55, tau_w=1 * data_scale, a=2, b=1000 / data_scale,
    ),
    get_param_set(
        shape=(1,), resistance=5, tau_t=0.2 * data_scale, threshold=-40, u_rest=-70,
        sharpness=1, theta_rh=-60, tau_w=1 * data_scale, a=2, b=1000 / data_scale,
    ),
    get_param_set(
        shape=(1,), resistance=5, tau_t=0.2 * data_scale, threshold=-40, u_rest=-70,
        sharpness=5, theta_rh=-55, tau_w=1 * data_scale, a=2, b=1000 / data_scale,
    ),
    get_param_set(
        shape=(1,), resistance=5, tau_t=0.2 * data_scale, threshold=-40, u_rest=-70,
        sharpness=1, theta_rh=-55, tau_w=5 * data_scale, a=2, b=1000 / data_scale,
    ),
    get_param_set(
        shape=(1,), resistance=5, tau_t=0.2 * data_scale, threshold=-40, u_rest=-70,
        sharpness=1, theta_rh=-55, tau_w=1 * data_scale, a=2, b=500 / data_scale,
    ),
    get_param_set(
        shape=(1,), resistance=5, tau_t=0.2 * data_scale, threshold=-40, u_rest=-70,
        sharpness=1, theta_rh=-55, tau_w=1 * data_scale, a=10, b=1000 / data_scale,
    ),
]

currents_vector = [
    get_fixed_current(5, time, dt, start=50, end=200),
    get_fixed_current(10, time, dt, start=50, end=200),
    get_fixed_current(30, time, dt, start=50, end=200),
]

models = [
    ELIFPopulation,
    AELIFPopulation
]

noise_set = [5, 50, 100]

# do_for_all(models, parameters_set, currents_vector, time, dt)
do_noisy_for_all(models, parameters_set, noise_set, time, dt)

Plot.show()
