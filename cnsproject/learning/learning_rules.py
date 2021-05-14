"""
Module for learning rules.
"""

from abc import ABC
from typing import Union, Optional, Sequence, Callable

import numpy as np
import torch

from ..network.connections import AbstractConnection
from ..network.neural_populations import PopulationVariables


class AbstractLearningRule(ABC):
    """
    Abstract class for defining learning rules.

    Each learning rule will be applied on a synaptic connection defined as \
    `connection` attribute. It possesses learning rate `lr` and weight \
    decay rate `weight_decay`. You might need to define more parameters/\
    attributes to the child classes.

    Implement the dynamics in `update` method of the classes. Computations \
    for weight decay and clamping the weights has been implemented in the \
    parent class `update` method. So do not invent the wheel again and call \
    it at the end  of the child method.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[Union[float, Callable]], Callable]] = None,
        weight_decay: float = 0.,
        dt: Union[torch.Tensor, float] = 1.,
        device: str = "cpu",
        **kwargs
    ) -> None:
        self.device = device
        self.connection = connection

        if lr is None:
            lr = [1., 1.]
        elif isinstance(lr, int) or isinstance(lr, float) or callable(lr):
            lr = [lr, lr]

        if not callable(lr[0]):
            self.constant_a_plus = lr[0]
            lr[0] = lambda w: self.constant_a_plus

        if not callable(lr[1]):
            self.constant_a_minus = lr[1]
            lr[1] = lambda w: self.constant_a_minus

        self.lr = lr
        self.weight_decay = 1 - weight_decay if weight_decay else 1.
        self.dt = dt if isinstance(dt, torch.Tensor) else torch.tensor(dt, device=self.device)

    def set_time_step(self, dt: Union[float, torch.Tensor]) -> None:
        """
        Time step length setter.

        Parameters
        ----------
        dt : Union[float, torch.Tensor]
            Time step length.

        Returns
        -------
        None

        """
        self.dt = torch.tensor(dt, device=self.device)

    def update(self) -> None:
        """
        Abstract method for a learning rule update.

        Returns
        -------
        None

        """
        if self.weight_decay:
            self.connection.w *= self.weight_decay

        if (
                self.connection.w_min != -np.inf or self.connection.w_max != np.inf
        ) and not isinstance(self.connection, NoOp):
            self.connection.w.clamp_(self.connection.w_min,
                                     self.connection.w_max)


class NoOp(AbstractLearningRule):
    """
    Learning rule with no effect.

    Arguments
    ---------
    connection : AbstractConnection
        The connection on which the learning rule is applied.
    lr : float or sequence of float, Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float
        Define rate of decay in synaptic strength. The default is 0.0.

    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[Union[float, Callable]], Callable]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    def update(self, **kwargs) -> None:
        """
        Only take care about synaptic decay and possible range of synaptic
        weights.

        Returns
        -------
        None

        """
        super().update()


class STDP(AbstractLearningRule):
    """
    Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of STDP learning rule.You might need to implement\
    different update rules based on type of connection.
    """
    def get_spike_trace(self, pre=True):
        return (self.connection.pre if pre else self.connection.post)\
            .get(PopulationVariables.RB_SPIKE_TRACE)

    def __weight_changes(self) -> torch.Tensor:
        post = self.connection.post
        pre = self.connection.pre

        negative_part = (
             self.lr[1](self.connection.w) *
             self.get_spike_trace(pre=False).expand((*pre.shape, *post.shape)) *
             pre.get(PopulationVariables.RB_SPIKES).expand((*post.shape, *pre.shape)).T
         )

        positive_part = (
             self.lr[0](self.connection.w) *
             self.get_spike_trace(pre=True).expand((*post.shape, *pre.shape)).T *
             post.get(PopulationVariables.RB_SPIKES).expand((*pre.shape, *post.shape))
        )

        return positive_part - negative_part

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        self.connection.w += self.dt * (self.__weight_changes())
        super().update()


class FlatSTDP(STDP):
    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[Union[float, Callable]], Callable]] = None,
        weight_decay: float = 0.,
        trace_limit_threshold: float = 0.5,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.trace_limit_threshold = trace_limit_threshold

    """
    Flattened Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-STDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """
    def get_spike_trace(self, pre=True):
        return (super().get_spike_trace(pre) > self.trace_limit_threshold).int()


class RSTDP(AbstractLearningRule):
    """
    Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[Union[float, Callable]], Callable]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        pass


class FlatRSTDP(AbstractLearningRule):
    """
    Flattened Reward-modulated Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-RSTDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        lr: Optional[Union[float, Sequence[Union[float, Callable]], Callable]] = None,
        weight_decay: float = 0.,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        """
        TODO.

        Consider the additional required parameters and fill the body\
        accordingly.
        """

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        pass
