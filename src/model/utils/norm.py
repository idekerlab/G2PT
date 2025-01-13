import torch.nn as nn
import torch
from torch import Tensor, Size
from typing import Union, List, Tuple
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init

import numbers


_shape_t = Union[int, List[int], Size]


def poincare_log_map_zero(x, eps=1e-15):
    """
    Log map from the Poincaré ball to the tangent space at 0.
    x: (..., d)
    Returns: (..., d), the log-space representation in R^d.
    """
    # Euclidean norm of x
    norm_x = torch.norm(x, dim=-1, keepdim=True)  # (..., 1)

    # Handle zero vector separately to avoid division by zero
    # if norm_x < eps, then x is effectively 0 => log_0(0) = 0
    mask = (norm_x > eps)

    # factor = 2 / (1 - ||x||^2)
    # We'll clamp 1 - norm_x^2 to avoid division by zero if x is near boundary
    denom = 1.0 - norm_x.pow(2)
    denom = torch.clamp(denom, min=eps)
    factor = 2.0 / denom

    # log_x = factor * x
    log_x = factor * x

    # Where norm_x <= eps, set log_x = 0
    log_x = torch.where(mask, log_x, torch.zeros_like(log_x))
    return log_x


def poincare_exp_map_zero(v, eps=1e-15):
    """
    Exp map from the tangent space at 0 to the Poincaré ball.
    v: (..., d)
    Returns: (..., d), the point in the Poincaré ball.
    """
    norm_v = torch.norm(v, dim=-1, keepdim=True)  # (..., 1)

    # For v=0 => exp_0(0) = 0
    mask = (norm_v > eps)

    # alpha = tanh(||v|| / 2) / ||v||
    # clamp norm_v to avoid dividing by zero
    half_norm = 0.5 * norm_v
    scale = torch.tanh(torch.clamp(half_norm, max=15.0)) / (norm_v + eps)

    # x = alpha * v
    exp_x = scale * v

    # Where norm_v <= eps, set exp_x=0
    exp_x = torch.where(mask, exp_x, torch.zeros_like(exp_x))

    return exp_x


def euclidian_to_poincare(x: torch.Tensor, eps: float = 1e-15) -> torch.Tensor:
    """
    Maps a Euclidean vector x to the Poincaré disk in the same direction,
    with hyperbolic 'length' = Euclidean norm of x.

    Args:
        x: (batch_size, d) or (d,) shape in Euclidean space.
        eps: small constant for numerical stability.

    Returns:
        y: same shape as x, with ||y|| < 1, so y is in the open unit ball.
           The hyperbolic distance from 0 to y is ||x|| (Euclidean).
    """
    # Euclidean norm of x
    norm_x = x.norm(dim=-1, keepdim=True)  # shape (..., 1)

    # For nonzero x: y = tanh(||x||/2) * (x / ||x||)
    # For x = 0: y = 0
    scale = torch.tanh(norm_x / 2) / (norm_x + eps)  # shape (..., 1)

    y = x * scale  # shape (..., d)

    # Where ||x|| is effectively 0, set y to 0
    mask = (norm_x > eps).type_as(x)
    y = y * mask  # broadcast multiplication, zero out if norm_x <= eps

    return y


def feature_clipping(
        x: torch.Tensor,
        r: float=2.0) -> torch.Tensor:
    """
    Clips the Euclidean norm of x to be at most r.

    x: shape (..., d)
    r: scalar, the effective radius for clipping.

    Returns:
        x_clipped: shape (..., d), same shape as x,
                   with ||x_clipped|| <= r.
    """
    # Norm along the last dimension
    norm_x = torch.norm(x, p=2, dim=-1, keepdim=True)  # (..., 1)

    # scale_factor = min(1, r / norm_x)
    # In PyTorch: elementwise minimum
    scale_factor = r / (norm_x + 1e-15)  # avoid /0
    scale_factor = torch.clamp(scale_factor, max=1.0)  # => <= 1

    # x_clipped
    x_clipped = x * scale_factor

    return x_clipped

class LayerNormNormedScaleOnly(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = nn.LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = torch.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = nn.LayerNorm([C, H, W])
        >>> output = layer_norm(input)

    .. image:: ../_static/img/nn/layer_norm.jpg
        :scale: 50 %

    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LayerNormNormedScaleOnly, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps

        self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)

    def forward(self, input: Tensor) -> Tensor:
        return F.layer_norm(
            input, self.normalized_shape, self.weight/torch.linalg.norm(self.weight), self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class PoincareNorm(Module):
    """
    Rescales vectors so their Euclidean norm is below `max_norm` (< 1).
    """

    def __init__(self, d_model, eps=1e-5, max_norm=0.999):
        super().__init__()
        self.eps = eps

        # Learnable parameters like standard LayerNorm
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.max_norm = max_norm
        self.desired = (self.max_norm - self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: shape (..., d_model)

        # 1) compute mean & variance along the last dim

        x_log = poincare_log_map_zero(x, eps=self.eps)
        mean = x_log.mean(dim=-1, keepdim=True)
        var = x_log.var(dim=-1, keepdim=True, unbiased=False)

        # 2) standard LN transform
        x_hat = (x_log - mean) / torch.sqrt(var + self.eps)
        x_hat = x_hat * self.gamma + self.beta

        norms = x_hat.norm(p=2, dim=-1, keepdim=True)
        exceed_mask = norms > (self.max_norm - self.eps)

        # Avoid division by zero by clamping norms to a small minimum
        norms_safe = torch.clamp(norms, min=1e-12)
        scale_factors = self.desired / norms_safe
        scale_factors = torch.where(exceed_mask, scale_factors, torch.ones_like(norms))
        # Multiply x by the scale factors
        x_hat = x_hat * scale_factors

        # 3) exp-map back
        x_out = poincare_exp_map_zero(x_hat, eps=self.eps)

        return x_out

