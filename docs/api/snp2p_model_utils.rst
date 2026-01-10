SNP2P Model Utilities
=====================

Overview
--------

This page documents model helper layers that are shared across SNP2P
architectures.

Usage and examples
------------------

Example: apply a FiLM layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from src.model.utils import FiLM

   film = FiLM(in_cov=4, hid=16)
   covariates = torch.randn(8, 4)
   features = torch.randn(8, 10, 16)
   modulated = film(features, covariates)

API documentation
-----------------

.. class:: FiLM

   Feature-wise linear modulation layer for injecting covariates.

   :param in_cov: Covariate input dimension.
   :type in_cov: int
   :param hid: Hidden dimension of the modulation.
   :type hid: int


.. class:: MoEHeadPrediction

   Mixture-of-experts head that produces per-position scalar predictions.

   :param hid: Hidden dimension of token embeddings.
   :type hid: int
   :param k_experts: Number of experts.
   :type k_experts: int, optional
   :param top_k: Number of experts to select per token.
   :type top_k: int, optional


.. class:: LayerNormNormedScaleOnly

   Layer normalization variant with normalized scaling weights.

   :param normalized_shape: Shape of the input to normalize.
   :type normalized_shape: int or tuple
   :param eps: Numerical stability term.
   :type eps: float, optional


.. class:: RMSNorm

   Root-mean-square normalization.

   :param dim: Feature dimension.
   :type dim: int
   :param eps: Numerical stability term.
   :type eps: float, optional
   :param elementwise_affine: Whether to learn a scale parameter.
   :type elementwise_affine: bool, optional
   :param memory_efficient: Unused placeholder for compatibility.
   :type memory_efficient: bool, optional


.. class:: BatchNorm1d_BatchOnly_NLC

   Batch-only normalization over ``[B, L, C]`` inputs.

   :param num_features: Number of feature channels.
   :type num_features: int
   :param eps: Numerical stability term.
   :type eps: float, optional
   :param momentum: Momentum for running statistics.
   :type momentum: float, optional
   :param affine: Whether to learn affine parameters.
   :type affine: bool, optional
   :param track_running_stats: Whether to track running mean/variance.
   :type track_running_stats: bool, optional
   :param length: Sequence length used to size running statistics.
   :type length: int, optional
