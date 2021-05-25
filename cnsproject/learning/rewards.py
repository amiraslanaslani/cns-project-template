"""
Module for reward dynamics.

TODO.

Define your reward functions here.
"""

from abc import ABC, abstractmethod

import torch


class AbstractReward(ABC):
    """
    Abstract class to define reward function.

    Make sure to implement the abstract methods in your child class.

    To implement your dopamine functionality, You will write a class \
    inheriting this abstract class. You can add attributes to your \
    child class. The dynamics of dopamine function (DA) will be \
    implemented in `compute` method. So you will call `compute` in \
    your reward-modulated learning rules to retrieve the dopamine \
    value in the desired time step. To reset or update the defined \
    attributes in your reward function, use `update` method and \
    remember to call it your learning rule computations in the \
    right place.
    """
    def __init__(self, device: str = "cpu", **kwargs):
        self.device = device
        self.d = torch.tensor(0., device=self.device)

        self.d_min = kwargs.get('d_min', -0.5)
        self.d_max = kwargs.get('d_max', 0.5)

    def get_dopamine_level(self):
        return self.d

    @abstractmethod
    def compute(self, **kwargs) -> torch.Tensor:
        """
        Compute the reward.

        Returns
        -------
        None
            It should return the computed reward value.

        """
        pass

    def update(self, **kwargs) -> None:
        """
        Update the internal variables.

        Returns
        -------
        None

        """
        # self.d.clamp_(self.d_min, self.d_max)

    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        self.d = torch.tensor(0, device=self.device)


class ZeroReward(AbstractReward):

    def compute(self, **kwargs) -> torch.Tensor:
        return self.d

    def update(self, **kwargs) -> None:
        pass


class SimpleReward(AbstractReward):

    def __init__(self, device: str = "cpu", tau_d: float = 15, **kwargs):
        super().__init__(device)
        self.d_decay = torch.tensor(0, device=self.device)
        self.tau_d = torch.tensor(tau_d, device=self.device)
        self.__zero = torch.tensor(0, device=self.device)

    def da(self):
        return self.__zero

    def compute(self, **kwargs) -> torch.Tensor:
        self.d += self.d_decay + self.da()
        return self.d

    def update(self, **kwargs) -> None:
        self.d_decay = - self.d / self.tau_d
        super().update(**kwargs)

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.d_decay = torch.tensor(0, device=self.device)
