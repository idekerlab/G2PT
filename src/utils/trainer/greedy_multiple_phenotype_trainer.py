from . import EarlyStopping
import gc
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
import copy
from torch.utils.data import get_worker_info
from src.utils.data import move_to
from src.utils.trainer import CCCLoss, FocalLoss, VarianceLoss, MultiplePhenotypeLoss
import torch.nn.functional as F
import copy
from .snp2p_trainer import SNP2PTrainer
from typing import List, Tuple, Dict, Optional
import logging




class GreedyMultiplePhenotypeTrainer(SNP2PTrainer):

    def __init__(self, snp2p_model, tree_parser, snp2p_dataloader, device, args, target_phenotype, validation_dataloader=None, fix_system=False, ):
        super(GreedyMultiplePhenotypeTrainer, self).__init__(snp2p_model, tree_parser, snp2p_dataloader, device, args, target_phenotype=target_phenotype, fix_system=fix_system, validation_dataloader=validation_dataloader)

        self.target_phenotype = target_phenotype
        self.improvement_threshold = 0.001
        self.dynamic_phenotype_sampling = True

        self.target_phenotype = target_phenotype

        # Store original datasets to avoid mutation
        #self.original_train_dataset = copy.deepcopy(snp2p_dataloader.dataset)
        #self.original_val_dataset = copy.deepcopy(validation_dataloader.dataset) if validation_dataloader else None

        # History tracking
        self.selection_history = []
        self.logger = logging.getLogger(__name__)


    def greedy_phenotype_selection(self) -> Tuple[float, nn.Module, List[str]]:
        """
        Perform greedy phenotype selection.

        Returns:
            Tuple of (best_performance, best_model, best_phenotype_set)
        """
        # Initially select target_phenotype
        phenotype_set = [self.target_phenotype]

        # Train initial model with target phenotype only
        self.logger.info(f"Training baseline model with {self.target_phenotype} only...")



        model2train = copy.deepcopy(self.snp2p_model)
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model2train.parameters()), lr=self.args.lr,
                                weight_decay=self.args.wd)
        train_dataset = self.snp2p_dataloader.dataset  # .select_phenotypes(phenotypes)
        train_dataset.select_phenotypes(phenotype_set)
        snp2p_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=self.args.jobs, prefetch_factor=2)
        if self.validation_dataloader is not None:
            val_dataset = self.validation_dataloader.dataset  # .dataset#.select_phenotypes(phenotypes)
            val_dataset.select_phenotypes(phenotype_set)
            val_snp2p_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=self.args.jobs, prefetch_factor=2)
        else:
            val_snp2p_dataloader = None

        early_stopping = EarlyStopping(
            patience=5,  # Stop after 10 evaluations without improvement
            min_delta=0.001,  # Minimum improvement threshold
            mode='max',  # Maximizing performance metric
            restore_best_weights=True,
            verbose=True
        )

        best_performance = float('-inf')

        for epoch in range(30):
            # Your training step
            train_loss = self.iter_minibatches(
                model2train, snp2p_dataloader, optimizer, epoch=epoch,
                name=f"Warmup with target phenotypes, {','.join(phenotype_set)}"
            )

            # Evaluate every 5 epochs
            if (epoch % self.args.val_step) == 0:
                self.save_model(model2train, phenotype_set, epoch)
            if val_snp2p_dataloader is not None:
                if (epoch % self.args.val_step) == 0:
                    val_performance = self.evaluate(
                        model2train,
                        dataloader=val_snp2p_dataloader,
                        phenotypes=phenotype_set,
                        epoch=epoch
                    )

                    # Update best and save if improved
                    if val_performance > best_performance:
                        best_performance = val_performance



                    # Check early stopping
                    should_stop = early_stopping(val_performance, model2train)

                    if should_stop:
                        print(f"Early stopping triggered at epoch {epoch}")
                        break

        initial_performance = self.evaluate(model2train, val_snp2p_dataloader, 0, phenotypes=phenotype_set, name=f"Warmup validation with phenotypes, {','.join(phenotype_set) }")

        initial_model = model2train
        self.save_model(initial_model, phenotype_set, epoch='best')
        # Track history
        self.selection_history.append({
            'step': 0,
            'phenotypes': phenotype_set.copy(),
            'performance': initial_performance,
            'added_phenotype': None
        })

        prev_performance = initial_performance
        best_model = initial_model
        best_phenotype_set = phenotype_set.copy()

        step = 1
        while True:
            self.logger.info(f"\nStep {step}: Evaluating phenotype additions...")

            # Find best phenotype to add
            candidate_performance, candidate_model, candidate_phenotype_set, added_phenotype = \
                self.compare_phenotype_combination(best_model, best_phenotype_set, prev_performance, balanced=False)

            improvement = candidate_performance - prev_performance

            if improvement < self.improvement_threshold:
                self.logger.info(
                    f"Improvement {improvement:.4f} below threshold {self.improvement_threshold}. Stopping.")
                break
            else:
                self.logger.info(f"Added {added_phenotype} with improvement {improvement:.4f}")
                prev_performance = candidate_performance
                best_model = candidate_model
                best_phenotype_set = candidate_phenotype_set
                self.save_model(best_model, best_phenotype_set)
                # Track history
                self.selection_history.append({
                    'step': step,
                    'phenotypes': best_phenotype_set.copy(),
                    'performance': candidate_performance,
                    'added_phenotype': added_phenotype,
                    'improvement': improvement
                })

                step += 1

        return prev_performance, best_model, best_phenotype_set

    '''
    def greedy_phenotype_selection(self):

        # initially select target_phenotype

        model = self.snp2p_model
        phenotype_set = [self.target_phenotype]
        initial_performance, initial_model = self.train_with_phenotypes(model, phenotype_set)
        prev_model = initial_model
        prev_performance = initial_performance
        while True:
            best_performance, best_model, best_phenotype_set = self.compare_phenotype_combination(prev_model, phenotype_set, prev_performance)

            if (best_performance - prev_performance) < self.performance_selection_stop_criteria:
                break
            else:
                prev_performance = best_performance
                prev_model = best_model
                phenotype_set = best_phenotype_set

        return best_performance, best_model, best_phenotype_set
    '''

    def compare_phenotype_combination(self, model, phenotype_set, best_performance, balanced=False):
        """
        Compare different phenotype combinations by adding one phenotype at a time.
        """
        best_performance = best_performance
        best_phenotype_set = phenotype_set
        best_model = model
        #orig_model = model.copy()
        #temporal_model = model
        added_phenotype = None
        for phenotype in self.phenotypes:
            if phenotype not in phenotype_set:
                temporal_phenotype_set = phenotype_set + [phenotype]
                temporal_model = copy.deepcopy(model)
                self.logger.info(f"  Testing addition of {phenotype}...")
                for i in range(self.args.epochs):
                # Train with new phenotype set
                    temporal_performance, temporal_model = self.train_with_phenotypes(
                        temporal_model, temporal_phenotype_set, epoch=i, balanced=balanced, train='embedding')

                    self.logger.info(f"    Performance: {temporal_performance:.4f}")

                    if temporal_performance > best_performance:
                        best_performance = temporal_performance
                        best_phenotype_set = temporal_phenotype_set
                        best_model = copy.deepcopy(temporal_model)
                        added_phenotype = phenotype

                temporal_model = best_model
                for i in range(self.args.epochs):
                # Train with new phenotype set
                    temporal_performance, temporal_model = self.train_with_phenotypes(
                        temporal_model, temporal_phenotype_set, epoch=i, balanced=balanced, train='transformer')

                    self.logger.info(f"    Performance: {temporal_performance:.4f}")

                    if temporal_performance > best_performance:
                        best_performance = temporal_performance
                        best_phenotype_set = temporal_phenotype_set
                        best_model = copy.deepcopy(temporal_model)
                        added_phenotype = phenotype

        return best_performance, best_model, best_phenotype_set, added_phenotype

    '''
    def compare_phenotype_combination(self, model, phenotype_set, best_performance):
        best_performance = best_performance
        best_phenotype_set = phenotype_set
        best_model = model
        for phenotype in self.phenotypes:
            if phenotype not in phenotype_set:
                temporal_phenotype_set = phenotype_set + [phenotype]
                temporal_performance, temporal_model = self.train_with_phenotypes(model, temporal_phenotype_set)
                if temporal_performance > best_performance:
                    best_performance = temporal_performance
                    best_phenotype_set = temporal_phenotype_set
                    best_model = temporal_model

        return best_performance, best_model, best_phenotype_set
    

    def find_best_phenotype_addition(self,
                                     current_phenotype_set: List[str],
                                     current_performance: float) -> Tuple[float, nn.Module, List[str], str]:
        """
        Find the best phenotype to add to the current set.

        Returns:
            Tuple of (best_performance, best_model, best_phenotype_set, added_phenotype)
        """
        best_performance = current_performance
        best_phenotype_set = current_phenotype_set
        best_model = None
        best_added_phenotype = None

        # Get remaining phenotypes to try
        remaining_phenotypes = [p for p in self.phenotypes if p not in current_phenotype_set]

        for phenotype in remaining_phenotypes:
            # Create new phenotype set
            temporal_phenotype_set = current_phenotype_set + [phenotype]

            self.logger.info(f"  Testing addition of {phenotype}...")

            # Train model with new phenotype set (from scratch)
            temporal_performance, temporal_model = self.train_with_phenotypes(temporal_phenotype_set)

            self.logger.info(f"    Performance: {temporal_performance:.4f}")

            if temporal_performance > best_performance:
                best_performance = temporal_performance
                best_phenotype_set = temporal_phenotype_set
                best_model = temporal_model
                best_added_phenotype = phenotype

                # Clean up previous best model to save memory
                if best_model is not None and best_model != temporal_model:
                    del best_model
                    torch.cuda.empty_cache()

        return best_performance, best_model, best_phenotype_set, best_added_phenotype
    '''
    def train_with_phenotypes(self, model, phenotypes, epoch=0, balanced=False, train=None):
        train_dataset = self.snp2p_dataloader.dataset#.select_phenotypes(phenotypes)
        if balanced:
            train_dataset.balanced_sampling = self.target_phenotype
        val_dataset = self.validation_dataloader.dataset#.dataset#.select_phenotypes(phenotypes)
        train_dataset.select_phenotypes(phenotypes)
        val_dataset.select_phenotypes(phenotypes)
        snp2p_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=self.args.jobs,
                                      prefetch_factor=2)
        val_snp2p_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=self.args.jobs,
                                      prefetch_factor=2)
        model2train = copy.deepcopy(model)
        if train == 'embedding':
            for name, parameter in model2train.named_parameters():
                if 'embedding' in name:
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False
        elif train == 'transformer':
            for name, parameter in model2train.named_parameters():
                if 'hierarchical_transformer' in name:
                    parameter.requires_grad = True
                else:
                    parameter.requires_grad = False
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model2train.parameters()), lr=self.args.lr/10,
                                     weight_decay=self.args.wd)
        self.iter_minibatches(model2train, snp2p_dataloader, optimizer, epoch=epoch, name=f"Training with phenotypes, {','.join(phenotypes) }, {str(train)}")
        performance = self.evaluate(model2train, val_snp2p_dataloader, 0, phenotypes=phenotypes, name=f"Validation with phenotypes, {','.join(phenotypes) }, {str(train)}")
        return performance, model2train



    def get_selection_summary(self):
        """
        Get a summary of the selection process.
        """
        import pandas as pd

        summary_data = []
        for entry in self.selection_history:
            summary_data.append({
                'step': entry['step'],
                'num_phenotypes': len(entry['phenotypes']),
                'phenotypes': ', '.join(entry['phenotypes']),
                'added_phenotype': entry['added_phenotype'] or 'Baseline',
                'performance': entry['performance'],
                'improvement': entry.get('improvement', 0.0)
            })

        return pd.DataFrame(summary_data)


    def save_model(self, model, phenotype_set, epoch=0):
        if self.args.out:
            output_path = self.args.out + "." + ".".join(phenotype_set) + "." + str(epoch) + ".pt"
            if self.args.distributed:
                if self.args.rank == 0:
                    print("Save to...", output_path)
                    torch.save({"arguments": self.args,
                                "state_dict": model.module.state_dict()},
                               output_path)
            else:
                print("Save to...", output_path)
                torch.save(
                    {"arguments": self.args, "state_dict": model.state_dict()},
                    output_path)


