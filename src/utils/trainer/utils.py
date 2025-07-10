import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
import copy
import logging
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import time


class EarlyStopping:
    """
    Early stopping utility to stop training when validation performance stops improving
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'max',
                 restore_best_weights: bool = True,
                 verbose: bool = True):
        """
        Initialize early stopping

        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics to maximize (accuracy), 'min' for metrics to minimize (loss)
            restore_best_weights: Whether to restore model to best weights when stopping
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        # Internal state
        self.wait = 0
        self.best_score = None
        self.best_weights = None
        self.stopped_epoch = 0

        # Set comparison function based on mode
        if mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            self.monitor_op = np.less
            self.min_delta *= -1

        self.logger = logging.getLogger(__name__)

    def __call__(self, score: float, model: torch.nn.Module = None) -> bool:
        """
        Check if training should stop

        Args:
            score: Current validation score
            model: Model to save weights from (if restore_best_weights=True)

        Returns:
            True if training should stop, False otherwise
        """
        current_score = score

        if self.best_score is None:
            self.best_score = current_score
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()

        elif self.monitor_op(current_score, self.best_score + self.min_delta):
            self.best_score = current_score
            self.wait = 0
            if model is not None and self.restore_best_weights:
                self.best_weights = model.state_dict().copy()

            if self.verbose:
                self.logger.info(f"New best score: {current_score:.6f}")

        else:
            self.wait += 1
            if self.verbose:
                self.logger.info(f"No improvement for {self.wait}/{self.patience} epochs. "
                                 f"Current: {current_score:.6f}, Best: {self.best_score:.6f}")

        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = self.wait
            if self.verbose:
                self.logger.info(f"Early stopping triggered after {self.wait} epochs without improvement")
            return True

        return False

    def restore_best_model(self, model: torch.nn.Module):
        """Restore model to best weights"""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                self.logger.info(f"Restored model to best weights (score: {self.best_score:.6f})")
        else:
            self.logger.warning("No best weights to restore")

    def get_best_score(self) -> Optional[float]:
        """Get the best score achieved"""
        return self.best_score

    def reset(self):
        """Reset early stopping state"""
        self.wait = 0
        self.best_score = None
        self.best_weights = None
        self.stopped_epoch = 0


class TrainingEfficiencyManager:
    """Manages efficient training strategies to avoid training from scratch"""

    def __init__(self, base_lr: float = 1e-4, warmup_epochs: int = 5):
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.training_cache = {}  # Cache trained models
        self.performance_cache = {}  # Cache performance results
        self.logger = logging.getLogger(__name__)

    def get_adaptive_lr(self, current_phenotypes: List[str], base_model_performance: float) -> float:
        """
        Dynamically adjust learning rate based on phenotype combination and base performance
        Instead of fixed lr/10 reduction
        """
        num_phenotypes = len(current_phenotypes)

        # Lower LR for more complex phenotype combinations
        complexity_factor = 1.0 / (1.0 + 0.1 * (num_phenotypes - 1))

        # Adjust based on base performance (if already good, use lower LR for fine-tuning)
        performance_factor = 0.5 if base_model_performance > 0.8 else 1.0

        adaptive_lr = self.base_lr * complexity_factor * performance_factor

        self.logger.info(f"Adaptive LR for {num_phenotypes} phenotypes: {adaptive_lr:.2e}")
        return adaptive_lr

    def should_use_transfer_learning(self,
                                     current_phenotypes: List[str],
                                     new_phenotype: str) -> Tuple[bool, Optional[str]]:
        """
        Determine if we should use transfer learning instead of training from scratch
        """
        # Check if we have a cached model for subset of current phenotypes
        for cached_phenotypes in self.training_cache.keys():
            cached_set = set(cached_phenotypes.split('_'))
            current_set = set(current_phenotypes)

            # If cached model contains most of current phenotypes, use transfer learning
            overlap = len(cached_set.intersection(current_set))
            overlap_ratio = overlap / len(current_set) if current_set else 0

            if overlap_ratio >= 0.7:  # 70% overlap threshold
                self.logger.info(f"Using transfer learning from {cached_phenotypes} "
                                 f"(overlap: {overlap_ratio:.2f})")
                return True, cached_phenotypes

        return False, None

    def cache_model(self, phenotypes: List[str], model: nn.Module, performance: float):
        """Cache trained model for future reuse"""
        cache_key = '_'.join(sorted(phenotypes))
        self.training_cache[cache_key] = copy.deepcopy(model.state_dict())
        self.performance_cache[cache_key] = performance

        # Limit cache size to prevent memory issues
        if len(self.training_cache) > 10:
            # Remove oldest entry
            oldest_key = next(iter(self.training_cache))
            del self.training_cache[oldest_key]
            del self.performance_cache[oldest_key]

    def get_cached_model(self, phenotypes: List[str]) -> Optional[nn.Module]:
        """Retrieve cached model if available"""
        cache_key = '_'.join(sorted(phenotypes))
        if cache_key in self.training_cache:
            return self.training_cache[cache_key]
        return None
