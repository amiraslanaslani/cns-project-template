"""
Module for connections between neural populations.
"""

from abc import ABC, abstractmethod
from typing import Union, Sequence, Callable, Tuple

import torch
from torch.nn.functional import conv2d

from .neural_populations import NeuralPopulation, PopulationVariables


class AbstractConnection(ABC, torch.nn.Module):
    """
    Abstract class for implementing connections.

    Make sure to implement the `compute`, `update`, and `reset_state_variables`\
    methods in your child class.

    You will need to define the populations you want to connect as `pre` and `post`.\
    In case of learning, you will need to define the learning rate (`lr`) and the \
    learning rule to follow. Attribute `w` is reserved for synaptic weights.\
    However, it has not been predefined or allocated, as it depends on the \
    pattern of connectivity. So make sure to define it in child class initializations \
    appropriately to indicate the pattern of connectivity. The default range of \
    each synaptic weight is [0, 1] but it can be controlled by `wmin` and `wmax`. \
    Synaptic strengths might decay in time and do not last forever. To define \
    the decay rate of the synaptic weights, use `weight_decay` attribute. Also, \
    if you want to control the overall input synaptic strength to each neuron, \
    use `norm` argument to normalize the synaptic weights.

    In case of learning, you have to implement the methods `compute` and `update`. \
    You will use the `compute` method to calculate the activity of post-synaptic \
    population based on the pre-synaptic one. Update of weights based on the \
    learning rule will be implemented in the `update` method. If you find this \
    architecture mind-bugling, try your own architecture and make sure to redefine \
    the learning rule architecture to be compatible with this new architecture \
    of yours.

    Arguments
    ---------
    pre : NeuralPopulation
        The pre-synaptic neural population.
    post : NeuralPopulation
        The post-synaptic neural population.
    lr : float or (float, float), Optional
        The learning rate for training procedure. If a tuple is given, the first
        value defines potentiation learning rate and the second one depicts\
        the depression learning rate. The default is None.
    weight_decay : float, Optional
        Define rate of decay in synaptic strength. The default is 0.0.

    Keyword Arguments
    -----------------
    learning_rule : LearningRule
        Define the learning rule by which the network will be trained. The\
        default is NoOp (see learning/learning_rules.py for more details).
    w_min : float
        The minimum possible synaptic strength. The default is 0.0.
    w_max : float
        The maximum possible synaptic strength. The default is 1.0.
    norm : float
        Define a normalization on input signals to a population. If `None`,\
        there is no normalization. The default is None.

    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation = None,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        super().__init__()

        if post is None:
            post = pre

        assert isinstance(pre, NeuralPopulation), \
            "Pre is not a NeuralPopulation instance"
        assert isinstance(post, NeuralPopulation), \
            "Post is not a NeuralPopulation instance"

        self.pre = pre
        self.post = post
        self.lr = lr

        self.weight_decay = weight_decay
        self.dt: torch.Tensor = torch.tensor(1.)

        w = self.get_initial_weights(**kwargs)
        self.register_buffer('w', w)

        self.mask = torch.nn.Parameter(
            self.compute_mask(self.w.shape, **kwargs),
            requires_grad=False
        )
        self.w[~ self.mask] = 0

        from ..learning.learning_rules import NoOp
        learning_rule = kwargs.get('learning_rule', NoOp)
        self.learning_rule = learning_rule(
            connection=self,
            lr=lr,
            weight_decay=weight_decay,
            dt=self.dt,
            device=kwargs.get('learning_rule_device', "cpu"),
            **kwargs
        )
        self.w_min = kwargs.get('w_min', 0.)
        self.w_max = kwargs.get('w_max', 50.)
        self.norm = kwargs.get('norm', None)

    def set_learning_reward(self, reward):
        self.learning_rule.set_reward(reward)

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
        self.dt = torch.tensor(dt)
        self.learning_rule.set_time_step(dt)

    def get_initial_weights(self, **kwargs) -> torch.Tensor:
        j0 = kwargs.get('j0', 5)
        s0 = kwargs.get('s0', 5)

        N = self.post.shape[0]
        w = kwargs.get(
            'weight',
            torch.normal(
                mean=j0 / N,
                std=s0 / N,
                size=(*self.pre.shape, *self.post.shape)
            ).abs()
        )
        w[~ self.pre.is_inhibitory, :] *= -1
        return w

    @abstractmethod
    def compute_mask(self, shape: torch.Tensor, **kwargs) -> torch.Tensor:
        pass

    def compute(self) -> None:
        """
        Compute the post-synaptic neural population activity based on the given\
        spikes of the pre-synaptic population.

        Returns
        -------
        None

        """
        spikes = getattr(self.pre, PopulationVariables.RB_SPIKES)
        half_flatten_w = self.w.reshape((self.pre.n, self.post.n))
        half_flatten_spikes = spikes.reshape((self.pre.n,))
        spikes_effect = (half_flatten_w.t().mul(half_flatten_spikes)).t()
        spikes_effect = spikes_effect.sum(dim=0).reshape(self.post.shape)
        setattr(
            self.post,
            PopulationVariables.RB_POTENTIAL,
            getattr(self.post, PopulationVariables.RB_POTENTIAL) + spikes_effect
        )

    def update(self, **kwargs) -> None:
        """
        Compute connection's learning rule and weight update.

        Keyword Arguments
        -----------------
        learning : bool
            Whether learning is enabled or not. The default is `self.training`.
        mask : torch.ByteTensor
            Define a mask to determine which weights to clamp to zero. Note: Elements that
            equals to False get zero. The default is `self.mask`.

        Returns
        -------
        None

        """
        learning = kwargs.get("learning", self.training)
        if learning:
            self.learning_rule.update(**kwargs)

            # We apply mask only when learning applied. Because weights changed only when learning rule changes it.
            mask = kwargs.get("mask", ~ self.mask)
            if mask is not None:
                self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset all internal state variables.

        Returns
        -------
        None

        """
        pass


class DenseConnection(AbstractConnection):
    """
    Specify a fully-connected synapse between neural populations.

    Implement the dense connection pattern following the abstract connection\
    template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation = None,
        lr: Union[Union[float, Sequence[Union[float, Callable]], Callable]] = None,
        weight_decay: float = 0.0,
        j0: float = 10,
        s0: float = 30,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            j0=j0,
            s0=s0,
            **kwargs
        )

    def compute_mask(self, shape: torch.Tensor, anti_diagonal=False, **kwargs) -> torch.Tensor:
        if anti_diagonal:
            assert shape[0] == shape[1]
            diagonal = 1 - torch.diag(torch.ones(shape[0]))
            return diagonal.bool()
        return torch.full(shape, True)

    def reset_state_variables(self) -> None:
        """
        TODO.

        Reset all the state variables of the connection.
        """
        pass


class RandomConnection(AbstractConnection):
    """
    Specify a random synaptic connection between neural populations.

    Implement the random connection pattern following the abstract connection\
    template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation = None,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        j0: float = 10,
        s0: float = 30,
        **kwargs
    ) -> None:
        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            j0=j0,
            s0=s0,
            **kwargs
        )

    def compute_mask(self, shape: torch.Tensor, prob: torch.Tensor = 0.5, **kwargs) -> torch.Tensor:
        size = shape[0] * shape[1]
        select = int(size * prob)
        indexes = torch.randperm(size)[:select]
        mask = torch.full((size,), False)
        mask[indexes] = True
        mask = mask.reshape(shape[0], shape[1])
        return mask

    def copy(self, pre=None, post=None):
        result = RandomConnection(
            pre=self.pre if pre is None else pre,
            post=self.post if post is None else post,
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        result.w = self.w
        result.w_max = self.w_max
        result.w_min = self.w_min
        result.norm = self.norm
        result.mask = self.mask
        return result

    def reset_state_variables(self) -> None:
        pass


class Convolutional2dConnection(AbstractConnection):
    """
    Specify a convolutional synaptic connection between neural populations.

    Implement the convolutional connection pattern following the abstract\
    connection template.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        lr: Union[float, Sequence[float]] = None,
        weight_decay: float = 0.0,
        default_kernels: torch.Tensor = None,
        stride: int = 1,
        padding: Union[bool, int, Tuple[int]] = False,
        dilation: int = 1,
        groups: int = 1,
        injection_coef: float = 1.,
        **kwargs
    ) -> None:
        if pre.ndim < 3:
            self.in_channels = 1
        else:
            self.in_channels = pre.shape[1]

        if default_kernels.ndim == 2:
            default_kernels = default_kernels.unsqueeze(0)

        if default_kernels.ndim == 3:
            default_kernels = default_kernels.expand((
                *default_kernels.shape[:1],
                self.in_channels // groups,
                *default_kernels.shape[1:]
            ))
        self.initial_kernels = default_kernels

        if isinstance(padding, bool):
            if padding:
                padding = (default_kernels[-1] - 1) // 2
            else:
                padding = 0

        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.coef = injection_coef

        super().__init__(
            pre=pre,
            post=post,
            lr=lr,
            weight_decay=weight_decay,
            **kwargs
        )

    def get_initial_weights(self, **kwargs):
        w = self.initial_kernels
        return w

    def compute_mask(self, shape: torch.Tensor, **kwargs) -> torch.Tensor:
        mask = torch.ones(shape).bool()
        return mask

    def get_convolved(self):
        spikes = getattr(self.pre, PopulationVariables.RB_SPIKES).float()

        for dim in [2, 3]:
            if spikes.ndim == dim:
                spikes = spikes.unsqueeze(0)

        conv = conv2d(
            spikes,
            self.w,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )

        if self.post.shape != conv.shape:
            raise Exception(
                "Post population's dimensions wasn't match with convolution's output. " +
                f"Current shape should be {tuple(conv.shape)} but current shape is {tuple(self.post.shape)}."
            )

        return conv

    def compute(self) -> None:
        conv = self.get_convolved()

        setattr(
            self.post,
            PopulationVariables.RB_POTENTIAL,
            getattr(self.post, PopulationVariables.RB_POTENTIAL) + conv * self.coef
        )

    def update(self, **kwargs) -> None:
        """
        TODO.

        Update the connection weights based on the learning rule computations.
        You might need to call the parent method.
        """
        pass

    def reset_state_variables(self) -> None:
        self.w = self.initial_kernels


class T2FSMaxPooling2dConnection(Convolutional2dConnection):
    """
    Specify a pooling synaptic connection between neural populations.

    Implement the pooling connection pattern following the abstract connection\
    template. Consider a parameter for defining the type of pooling.

    Note: The pooling operation does not support learning. You might need to\
    make some modifications in the defined structure of this class.
    """

    def __init__(
        self,
        pre: NeuralPopulation,
        post: NeuralPopulation,
        kernel_size: Union[float, Sequence[float]],
        stride: int = 1,
        padding: Union[bool, int, Tuple[int]] = False,
        dilation: int = 1,
        **kwargs
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.window_size = kernel_size
        self.active_receptive_fields = torch.ones((*post.shape,))

        super().__init__(
            pre=pre,
            post=post,
            lr=None,
            window_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            default_kernels=torch.ones((*kernel_size,)),
            **kwargs
        )

    def compute(self) -> None:
        conv = self.get_convolved()
        conv = conv > 0
        output_spikes = conv * self.active_receptive_fields.float()
        self.active_receptive_fields *= ~ output_spikes.bool()
        setattr(
            self.post,
            PopulationVariables.RB_SPIKES,
            output_spikes
        )

    def reset_state_variables(self) -> None:
        super().reset_state_variables()
        self.active_receptive_fields = torch.ones(self.output_shape)

