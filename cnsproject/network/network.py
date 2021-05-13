"""
Module for spiking neural network construction and simulation.
"""

from typing import Optional, Dict

import torch
from tqdm import trange

from .neural_populations import NeuralPopulation
from .connections import AbstractConnection
from .monitors import Monitor
from ..learning.rewards import AbstractReward
from ..decision.decision import AbstractDecision


class Network(torch.nn.Module):
    """
    The class responsible for creating a neural network and its simulation.

    Examples
    --------
    >>> from cnsproject.network.neural_populations import LIFPopulation
    >>> from cnsproject.network.connections import DenseConnection
    >>> from cnsproject.network.monitors import Monitor
    >>> from cnsproject.network import Network
    >>> inp = InputPopulation(shape=(10,))
    >>> out = LIFPopulation(shape=(2,))
    >>> synapse = DenseConnection(inp, out)
    >>> net = Network(learning=False)
    >>> net.add_layer(inp, "input")
    >>> net.add_layer(out, "output")
    >>> net.add_connection(synapse, "input", "output")
    >>> out_m = Monitor(out, state_variables=["s", "v"])
    >>> syn_m = Monitor(synapse, state_variables=["w"])  # `w` indicates synaptic weights
    >>> net.add_monitor(out_m, "output")
    >>> net.add_monitor(syn_m, "synapse")
    >>> net.run(10)
    Here, we create a simple network with two layers and dense connection. We aim to monitor
    the synaptic weights and output layer's spikes and voltages. We simulate the network for
    10 miliseconds.

    You will need to implement the `run` method. This mthod is responsible for the whole simulation \
    procedure of a spiking neural network. You will have to compute number of time steps using \
    `dt` attribute of the class and `time` parameter of the method. then you will iteratively call \
    the procedures for single step simulation of network objects.

    **NOTE:** If you faced any errors related to importing packages, modify the `__init__.py` files \
    accordingly to solve the problem.

    Arguments
    ---------
    learning: bool, Optional
        Whether to allow weight update and learning. The default is True.
    reward : AbstractReward, Optional
        The class to allow reward modifications in case of reward-modulated
        learning. The default is None.
    decision: AbstractDecision, Optional
        The class to enable decision making. The default is None.

    """

    def __init__(
        self,
        learning: bool = True,
        reward: Optional[AbstractReward] = None,
        decision: Optional[AbstractDecision] = None,
        **kwargs
    ) -> None:
        super().__init__()

        self.layers = {}
        self.connections = {}
        self.monitors = {}

        # Sets `self.training` equals to `learning` and run `train` method
        # recursively for child modules.
        self.train(learning)

        # Make sure that arguments of your reward and decision classes do not
        # share same names. Their arguments are passed to the network as its
        # keyword arguments.
        if not (reward is None):
            self.reward = reward(**kwargs)

        if not (decision is None):
            self.decision = decision(**kwargs)

    def add_layer(self, layer: NeuralPopulation, name: str) -> None:
        """
        Add a neural population to the network.

        Parameters
        ----------
        layer : NeuralPopulation
            The neural population to be added.
        name : str
            Name of the layer for further referencing.

        Returns
        -------
        None

        """
        self.layers[name] = layer
        self.add_module(name, layer)

        layer.train(self.training)

    def add_connection(
        self,
        connection: AbstractConnection,
        pre: str,
        post: str
    ) -> None:
        """
        Add a connection between neural populations to the network. The\
        reference name will be in the format `{pre}_to_{post}`.

        Parameters
        ----------
        connection : AbstractConnection
            The connection to be added.
        pre : str
            Reference name of pre-synaptic population.
        post : str
            Reference name of post-synaptic population.

        Returns
        -------
        None

        """
        self.connections[f"{pre}_to_{post}"] = connection
        self.add_module(f"{pre}_to_{post}", connection)

        connection.train(self.training)

    def add_monitor(self, monitor: Monitor, name: str) -> None:
        """
        Add a monitor on a network object to the network.

        Parameters
        ----------
        monitor : Monitor
            The monitor instance to be added.
        name : str
            Name of the monitor instance for further referencing.

        Returns
        -------
        None

        """
        self.monitors[name] = monitor

    def run(
        self,
        time: float = None,
        dt: float = 1.0,
        inputs: Dict[str, torch.Tensor] = {},  # Spikes
        currents: Dict[str, torch.Tensor] = {},
        one_step: bool = False,
        random_factor: float = 0,
        resume: bool = False,
        total_time: float = None,
        **kwargs
    ) -> None:
        """
        Simulate network for a specific time duration with the possible given\
        input.

        Input to each layer is given to `inputs` parameter. As you see, it is a \
        dictionary of population's name and tensor of input values through time. \
        There is a parameter named `one_step`. This parameter will define how the \
        input is propagated through the network: does it go forward up to the final \
        layer in one time step or it passes from one layer to the next in each \
        step of simulation. You can easily remove it if it is mind-bugling.

        Also, make sure to call `self.reset_state_variables()` before starting the \
        simulation.

        TODO.

        Implement the body of this method.

        Parameters
        ----------
        time : float
            Simulation time.
        dt : float, Optional
            Specify simulation timestep. The default is 1.0.
        inputs : Dict[str, torch.Tensor], optional
            Mapping of input layer names to their input spike tensors. The\
            default is {}.
        currents : Dict[str, torch.Tensor], optional
            Mapping of input layer names to their input current tensors. The\
            default is {}.
        one_step : bool, optional
            Whether to propagate the inputs all the way through the network in\
            a single simulation step. The default is False.
        random_factor : float, optional
            Randomness degree

        Keyword Arguments
        -----------------
        clamp : Dict[str, torch.Tensor]
            Mapping of layer names to boolean masks if neurons should be clamped
            to spiking.
        unclamp : Dict[str, torch.Tensor]
            Mapping of layer names to boolean masks if neurons should be clamped
            not to spiking.
        masks : Dict[str, torch.Tensor]
            Mapping of connection names to boolean masks of the weights to clamp
            to zero.

        **Note:** you can pass the reward and decision methods' arguments as keyword\
        arguments to this function.

        Returns
        -------
        None

        """
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})

        if total_time is None:
            total_time = time

        if not resume:
            # Set time and dt for all instances
            for monitor in self.monitors:
                self.monitors[monitor].set_time_steps(total_time, dt)

            for layer in self.layers:
                self.layers[layer].set_time_step(dt)

            for connection in self.connections:
                self.connections[connection].set_time_step(dt)

        if time:
            time_steps = int(time / dt)
            p_bar = trange(time_steps, unit="timesteps")
            for time_step in p_bar:
                for layer in self.layers:
                    if layer in currents:
                        current = currents[layer][time_step]
                    else:
                        current = torch.tensor(0)
                    self.layers[layer].forward(current=current, random=random_factor)
                for connection in self.connections:
                    self.connections[connection].compute()
                    self.connections[connection].update()
                for monitor in self.monitors:
                    self.monitors[monitor].record()

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        for monitor in self.monitors:
            self.monitors[monitor].reset_state_variables()

    # def train(self, mode: bool = True) -> torch.nn.Module:
    #     """
    #     Set the population's training mode.
    #
    #     Parameters
    #     ----------
    #     mode : bool, optional
    #         Mode of training. `True` turns on the training while `False` turns\
    #         it off. The default is True.
    #
    #     Returns
    #     -------
    #     torch.nn.Module
    #
    #     """
    #     self.learning = mode
    #     return super().train(mode)

