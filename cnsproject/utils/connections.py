from math import floor
from typing import Iterable, Union, Type, Tuple

from cnsproject.network.connections import Convolutional2dConnection, T2FSMaxPooling2dConnection
from cnsproject.network.neural_populations import NeuralPopulation
from cnsproject.utils.general import iterlen


def convolution_parameters(
        default_kernels=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        injection_coef=1.,
        lr=None,
        **kwargs
) -> dict:
    filter_shape = None if default_kernels is None else default_kernels.shape
    if filter_shape is None:
        if kwargs.get("kernel_size", None) is not None and kwargs.get("filters", None) is not None:
            filters = kwargs.get("filters")
            kernel_size = kwargs.get("kernel_size")
            if isinstance(kernel_size, int):
                filter_shape = (filters, kwargs.get("kernel_size"), kwargs.get("kernel_size"))
            else:
                filter_shape = (filters, *kwargs.get("kernel_size"))

    param_dict = {
        "default_kernels": default_kernels,
        "filter_shape": filter_shape,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "groups": groups,
        "injection_coef": injection_coef,
        "lr": lr,
    }
    param_dict.update(kwargs)
    return param_dict


def pooling_parameters(
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        **kwargs
) -> dict:
    param_dict = {
        "kernel_size": kernel_size,
        "filter_shape": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
    }
    param_dict.update(kwargs)
    return param_dict


def output_population_parameters(
        spike_trace=True,
        additive_spike_trace=True,
        tau_s=15.,
        trace_scale=1.,
        is_inhibitory=None,
        learning=True,
        **kwargs
) -> dict:
    param_dict = {
        "spike_trace": spike_trace,
        "additive_spike_trace": additive_spike_trace,
        "tau_s": tau_s,
        "trace_scale": trace_scale,
        "is_inhibitory": is_inhibitory,
        "learning": learning
    }
    param_dict.update(kwargs)
    return param_dict


def get_convolution_2d_output_shape(
        input_shape,
        filter_shape: Iterable[int],
        stride: Union[int, tuple] = (1, 1),
        padding: Union[int, tuple] = (0, 0),
        dilation: Union[int, tuple] = (1, 1),
        **kwargs
) -> Iterable[int]:
    if isinstance(stride, int):
        stride = (stride, stride)

    if isinstance(padding, int):
        padding = (padding, padding)

    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    if iterlen(filter_shape) < 3:
        channels = 1
    else:
        channels = filter_shape[-3]

    input_size = input_shape[-2:]
    kernel_size = filter_shape[-2:]

    def get_shape(index):
        numer = (input_size[index] + 2 * padding[index] - dilation[index] * (kernel_size[index] - 1) - 1)
        denom = stride[index]
        result = numer / denom
        result = floor(result) + 1
        return result

    h_out = get_shape(0)
    w_out = get_shape(1)
    n = 1
    return n, channels, h_out, w_out


def convolution_2d_connection(
        input_population: NeuralPopulation,
        output_population_type: Type[NeuralPopulation],
        output_population_parameters: dict,  # Contains parameters of output population
        connection_parameters: dict,  # Contains parameters of connection
        **kwarg
) -> Tuple[NeuralPopulation, Convolutional2dConnection]:
    output_shape = get_convolution_2d_output_shape(input_population.shape, **connection_parameters)
    output_population = output_population_type(shape=output_shape, **output_population_parameters)
    connection = Convolutional2dConnection(
        pre=input_population,
        post=output_population,
        **connection_parameters
    )
    return output_population, connection


def pooling_2d_connection(
        input_population: NeuralPopulation,
        output_population_type: Type[NeuralPopulation],
        output_population_parameters: dict,  # Contains parameters of output population
        connection_parameters: dict,  # Contains parameters of connection
        **kwarg
) -> Tuple[NeuralPopulation, Convolutional2dConnection]:
    output_shape = get_convolution_2d_output_shape(input_population.shape, **connection_parameters)
    output_population = output_population_type(shape=output_shape, **output_population_parameters)
    connection = T2FSMaxPooling2dConnection(
        pre=input_population,
        post=output_population,
        **connection_parameters
    )
    return output_population, connection
