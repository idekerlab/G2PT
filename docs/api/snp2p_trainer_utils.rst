SNP2P Trainer Utilities
=======================

This page documents loss functions and utility helpers used by the SNP2P
training loop.

.. class:: CCCLoss

   Concordance correlation coefficient (CCC) loss for regression targets.

   :param eps: Numerical stability term.
   :type eps: float, optional
   :param mean_diff: Whether to include mean difference in the denominator.
   :type mean_diff: bool, optional


.. class:: FocalLoss

   Focal loss for imbalanced binary classification.

   :param alpha: Weight for positive examples.
   :type alpha: float, optional
   :param gamma: Focusing parameter.
   :type gamma: float, optional
   :param reduction: Reduction method (``mean`` or ``sum``).
   :type reduction: str, optional


.. class:: VarianceLoss

   Matches the standard deviation of predictions to the targets within a batch.


.. class:: MultiplePhenotypeLoss

   Multi-task loss that applies BCE to binary phenotype indices and MSE to
   quantitative phenotype indices while masking missing values.

   :param bce_cols: Column indices for BCE loss.
   :type bce_cols: list
   :param mse_cols: Column indices for MSE loss.
   :type mse_cols: list
   :param label_smoothing: Optional label smoothing for BCE.
   :type label_smoothing: float, optional


.. class:: BCEWithLogitsLossWithLabelSmoothing

   Binary cross-entropy loss with label smoothing.

   :param alpha: Smoothing factor.
   :type alpha: float, optional
   :param reduction: Reduction method (``mean`` or ``sum``).
   :type reduction: str, optional


.. class:: EarlyStopping

   Early stopping utility that tracks a validation score and restores the best
   weights when training stalls.

   :param patience: Number of epochs to wait for improvement.
   :type patience: int, optional
   :param min_delta: Minimum change to qualify as an improvement.
   :type min_delta: float, optional
   :param mode: ``max`` for metrics to maximize, ``min`` for metrics to minimize.
   :type mode: str, optional
   :param restore_best_weights: Whether to restore the best weights on stop.
   :type restore_best_weights: bool, optional
   :param verbose: Whether to print progress messages.
   :type verbose: bool, optional


.. class:: TrainingEfficiencyManager

   Cache-aware helper that supports adaptive learning rates and transfer
   learning across phenotype combinations.

   :param base_lr: Base learning rate.
   :type base_lr: float, optional
   :param warmup_epochs: Warmup epochs for scheduling.
   :type warmup_epochs: int, optional
