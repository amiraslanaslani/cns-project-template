"""
Module for learning rules.
"""

from abc import ABC
from typing import Union, Optional, Sequence, Callable

import numpy as np
import torch

from .rewards import AbstractReward, SimpleReward, ZeroReward
from ..network.connections import AbstractConnection
from ..network.neural_populations import PopulationVariables, NeuralPopulation


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
            lr = [torch.tensor(1.), torch.tensor(1.)]
        elif isinstance(lr, int) or isinstance(lr, float) or callable(lr):
            lr = [torch.tensor(lr), torch.tensor(lr)]

        if not callable(lr[0]):
            self.constant_a_plus = torch.tensor(lr[0])
            lr[0] = lambda w: self.constant_a_plus

        if not callable(lr[1]):
            self.constant_a_minus = torch.tensor(lr[1])
            lr[1] = lambda w: self.constant_a_minus

        self.lr = lr
        self.weight_decay = 1 - weight_decay if weight_decay else 1.
        self.dt = dt if isinstance(dt, torch.Tensor) else torch.tensor(dt, device=self.device)

        self.reward: AbstractReward
        self.set_reward(kwargs.get("reward", None))

    def set_reward(self, reward: AbstractReward = None):
        if reward is None:
            reward = ZeroReward()
        self.reward = reward

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

    def update(self, **kwargs) -> None:
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
    @classmethod
    def calc_spike_trace(cls, pre: NeuralPopulation, post: NeuralPopulation, is_pre: bool = True, **kwargs):
        return (pre if is_pre else post) \
            .get(PopulationVariables.RB_SPIKE_TRACE)

    @classmethod
    def calc_weight_changes(
            cls,
            pre: NeuralPopulation,
            post: NeuralPopulation,
            lr: Sequence[Callable],
            w: torch.Tensor,
            **kwargs
    ) -> torch.Tensor:
        ltd = (
                lr[1](w) *
                cls.calc_spike_trace(pre, post, is_pre=False, **kwargs).expand((*pre.shape, *post.shape)) *
                pre.get(PopulationVariables.RB_SPIKES).expand((*post.shape, *pre.shape)).T
        )

        ltp = (
                lr[0](w) *
                cls.calc_spike_trace(pre, post, is_pre=True, **kwargs).expand((*post.shape, *pre.shape)).T *
                post.get(PopulationVariables.RB_SPIKES).expand((*pre.shape, *post.shape))
        )
        return ltp - ltd

    def weight_changes(self) -> torch.Tensor:
        return self.calc_weight_changes(
            self.connection.pre,
            self.connection.post,
            self.lr,
            self.connection.w
        )

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the\
        parent method.
        """
        self.connection.w += self.dt * self.weight_changes()
        super().update()


class FlatSTDP(STDP):
    """
    Flattened Spike-Time Dependent Plasticity learning rule.

    Implement the dynamics of Flat-STDP learning rule. You might need to implement\
    different update rules based on type of connection.
    """
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

    @classmethod
    def calc_spike_trace(cls, pre: NeuralPopulation, post: NeuralPopulation, is_pre: bool = True, **kwargs):
        threshold = kwargs.get("trace_limit_threshold", 0.5)
        return (super().calc_spike_trace(pre, post, is_pre, **kwargs) > threshold).int()

    def weight_changes(self) -> torch.Tensor:
        stdp = self.calc_weight_changes(
            self.connection.pre,
            self.connection.post,
            self.lr,
            self.connection.w,
            trace_limit_threshold=self.trace_limit_threshold
        )

        flat = stdp.where(stdp >= 0, self.lr[1](self.connection.w))\
            .where(stdp <= 0, self.lr[0](self.connection.w))
        return flat


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
        tau_c: float = 20,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.tau_c = torch.tensor(tau_c, device=self.device)
        self.c = torch.zeros_like(connection.w)

    def _get_d(self):
        return self.reward.get_dopamine_level()

    def get_spike_trace(self, pre=True):
        return (self.connection.pre if pre else self.connection.post)\
            .get(PopulationVariables.RB_SPIKE_TRACE)

    def _weight_changes(self) -> torch.Tensor:
        return self.c * self._get_d()

    def _do_update(self, **kwargs):
        decay_c = - self.c / self.tau_c
        stdp = STDP.calc_weight_changes(
            self.connection.pre,
            self.connection.post,
            self.lr,
            self.connection.w
        )
        self.c += self.dt * (decay_c + stdp)  # dc/dt = decay_c + STPD
        self.connection.w += self.dt * self._weight_changes()  # ds(w)/dt = c * d

    def update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        self._do_update(**kwargs)
        super().update(**kwargs)


class FlatRSTDP(RSTDP):
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
        trace_limit_threshold: float = 0.5,
        window_size: int = 1000,
        **kwargs
    ) -> None:
        super().__init__(
            connection=connection,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )
        self.trace_limit_threshold = trace_limit_threshold
        self.last_windows = torch.tensor([])
        self.window_size = window_size

    def _do_update(self, **kwargs) -> None:
        """
        TODO.

        Implement the dynamics and updating rule. You might need to call the
        parent method. Make sure to consider the reward value as a given keyword
        argument.
        """
        flat_stdp = FlatSTDP.calc_weight_changes(
            self.connection.pre,
            self.connection.post,
            self.lr,
            self.connection.w
        )

        # Fill Windows
        flat_stdp_unsqueezed = torch.unsqueeze(flat_stdp, 0)
        base = self.last_windows[-self.window_size:]
        self.last_windows = torch.cat((base, flat_stdp_unsqueezed))

        # calculate dc and dw
        if self.last_windows.shape[0] < self.window_size:
            self.c += flat_stdp
        else:
            self.c += flat_stdp - self.last_windows[0]
        self.connection.w += self.dt * self._weight_changes() / self.last_windows.shape[0]
