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
import mlflow
import torch.distributed as dist
from src.utils.tree import SNPTreeParser


class GreedyMultiplePhenotypeTrainer(SNP2PTrainer):

    def __init__(self, snp2p_model, tree_parser, snp2p_dataloader, device, args, target_phenotype,
                 validation_dataloader=None, fix_system=False, pretrained_checkpoint=None):
        super(GreedyMultiplePhenotypeTrainer, self).__init__(snp2p_model, tree_parser, snp2p_dataloader,
                                                             device, args, target_phenotype=target_phenotype,
                                                             fix_system=fix_system,
                                                             validation_dataloader=validation_dataloader)

        self.target_phenotype = target_phenotype
        self.improvement_threshold = 0.0001
        self.dynamic_phenotype_sampling = True
        self.pretrained_checkpoint = pretrained_checkpoint

        # History tracking
        self.selection_history = []
        self.logger = logging.getLogger(__name__)

        # Load pretrained model if provided
        if self.pretrained_checkpoint is not None:
            self.load_pretrained_model()

    @staticmethod
    def map_embeddings(old_embeddings, new_embeddings, old_index, new_index, snp=False):
        """
        Map embeddings from old index to new index.

        Args:
            old_embeddings: torch.Tensor of shape [old_vocab_size, embed_dim]
            old_index: dict mapping token -> old_index
            new_index: dict mapping token -> new_index

        Returns:
            new_embeddings: torch.Tensor of shape [new_vocab_size, embed_dim]
        """
        import torch

        # Get embedding dimension
        embed_dim = old_embeddings.size(1)
        old_vocab_size = len(old_index)
        new_vocab_size = len(new_index)
        #print(old_embeddings.shape)
        #print(new_embeddings.shape)
        # Initialize new embeddings (random or zero)
        # new_embeddings = torch.randn(new_vocab_size, embed_dim) * 0.1  # or torch.zeros()

        # Create reverse mapping for old index (index -> token)
        old_idx_to_token = {idx: token for token, idx in old_index.items()}

        # Map embeddings for common tokens
        for new_idx, token in enumerate(new_index.keys()):
            if token in old_index:
                old_idx = old_index[token]
                new_embeddings[new_idx] = old_embeddings[old_idx]
                if snp:
                    new_embeddings[new_vocab_size + new_idx] = old_embeddings[old_vocab_size + old_idx]
                    new_embeddings[new_vocab_size * 2 + new_idx] = old_embeddings[old_vocab_size * 2 + old_idx]
        return new_embeddings

    def load_pretrained_model(self):
        """Load pretrained model from checkpoint."""
        self.logger.info(f"Loading pretrained model from {self.pretrained_checkpoint}")

        checkpoint = torch.load(self.pretrained_checkpoint, map_location=self.device)

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        new_state_dict = self.snp2p_model.state_dict()
        old_snp_tree_parser = SNPTreeParser(checkpoint['arguments'].onto, checkpoint['arguments'].snp2gene)
        new_snp_tree_parser = self.tree_parser


        # Handle distributed vs non-distributed model loading
        if self.args.distributed:
            # If the checkpoint is from a non-distributed model but we're running distributed
            if not any(key.startswith('module.') for key in state_dict.keys()):
                # Add 'module.' prefix to all keys
                state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        else:
            # If the checkpoint is from a distributed model but we're running non-distributed
            if any(key.startswith('module.') for key in state_dict.keys()):
                # Remove 'module.' prefix from all keys
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        #del state_dict['snp_adapters']

        for key in state_dict.keys():
            if 'system_embedding.weight' in key:
                state_dict[key] = self.map_embeddings(state_dict[key], new_state_dict[key],
                                                                old_snp_tree_parser.sys2ind,
                                                                new_snp_tree_parser.sys2ind)
            if 'gene_embedding.weight' in key:
                state_dict[key] = self.map_embeddings(state_dict[key], new_state_dict[key],
                                                                old_snp_tree_parser.gene2ind,
                                                                new_snp_tree_parser.gene2ind)
            if 'snp_embedding.weight' in key:
                state_dict[key] = self.map_embeddings(state_dict[key], new_state_dict[key],
                                                                old_snp_tree_parser.snp2ind,
                                                                new_snp_tree_parser.snp2ind, snp=True)
            if 'block_embedding.weight' in key:
                state_dict[key] = self.map_embeddings(state_dict[key], new_state_dict[key],
                                                                old_snp_tree_parser.block2ind,
                                                                new_snp_tree_parser.block2ind)
            if 'phenotype_embeddings.weight' in key:
                state_dict[key] = self.map_embeddings(state_dict[key], new_state_dict[key],
                                                                checkpoint['arguments'].pheno2ind,
                                                                self.args.pheno2ind)
            if 'snp_batch_norm.running_mean' in key:
                state_dict[key] = self.map_embeddings(state_dict[key][0], new_state_dict[key][0],
                                                                old_snp_tree_parser.snp2ind,
                                                                new_snp_tree_parser.snp2ind).unsqueeze(0)
            if 'snp_batch_norm.running_var' in key:
                state_dict[key] = self.map_embeddings(state_dict[key][0], new_state_dict[key][0],
                                                                old_snp_tree_parser.snp2ind,
                                                                new_snp_tree_parser.snp2ind).unsqueeze(0)
        # Load the state dict
        self.snp2p_model.load_state_dict(state_dict, strict=False)
        self.logger.info("Pretrained model loaded successfully")
        self.snp2p_model = self.snp2p_model.to(self.device)

    def add_phenotype_as_embedding(self, model, phenotypes):
        """
        Pre-trains embedding layers using phenotype labels.
        """
        self.logger.info(f"Pre-training embedding layer with phenotype labels for: {', '.join(phenotypes)}")

        model_to_train = model.module if self.args.distributed else model

        # Freeze all parameters except for embedding layers
        for name, param in model_to_train.named_parameters():
            if 'embedding' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_to_train.parameters()), lr=self.args.lr / 5)

        train_dataset = self.snp2p_dataloader.dataset
        train_dataset.select_phenotypes(phenotypes)
        dataloader = DataLoader(train_dataset, batch_size=None, num_workers=self.args.jobs, prefetch_factor=2)

        # Run for a few epochs
        for epoch in range(3):
            self.iter_minibatches(
                model, dataloader, optimizer, epoch=epoch,
                name=f"Pre-training embeddings with phenotypes: {', '.join(phenotypes)}"
            )

        # Unfreeze all parameters for the main training
        for param in model_to_train.parameters():
            param.requires_grad = True

    def greedy_phenotype_selection(self, skip_initial_training=False) -> Tuple[float, nn.Module, List[str]]:
        """
        Perform greedy phenotype selection.

        Args:
            skip_initial_training: If True and pretrained model is loaded, skip initial training phase

        Returns:
            Tuple of (best_performance, best_model, best_phenotype_set)
        """
        # Initially select target_phenotype
        phenotype_set = [self.target_phenotype]

        # Determine if we should skip initial training
        skip_training = skip_initial_training and self.pretrained_checkpoint is not None

        if skip_training:
            self.logger.info(f"Skipping initial training - using pretrained model for {self.target_phenotype}")

            # Just evaluate the pretrained model
            if self.args.rank == 0:
                mlflow.start_run(nested=True, run_name=f"Pretrained_Evaluation_{self.target_phenotype}")

            model2train = self.snp2p_model

            # Prepare validation dataloader
            if self.validation_dataloader is not None:
                val_dataset = self.validation_dataloader.dataset
                val_dataset.select_phenotypes(phenotype_set)
                val_snp2p_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=self.args.jobs,
                                                  prefetch_factor=2)

                # Evaluate pretrained model
                initial_performance = self.evaluate(
                    model2train,
                    dataloader=val_snp2p_dataloader,
                    phenotypes=phenotype_set,
                    epoch=0,
                    name=f"Pretrained model evaluation for {self.target_phenotype}"
                )

                if self.args.rank == 0:
                    print(f"Pretrained model performance: {initial_performance:.4f}")
                    mlflow.log_metric("pretrained_performance", initial_performance)
            else:
                initial_performance = 0.0
                self.logger.warning("No validation dataloader provided - cannot evaluate pretrained model")

            if self.args.rank == 0:
                mlflow.end_run()

        else:
            # Train initial model with target phenotype only (original behavior)
            self.logger.info(f"Training baseline model with {self.target_phenotype} only...")

            if self.args.rank == 0:
                mlflow.start_run(nested=True, run_name=f"Baseline_Training_{self.target_phenotype}")
            if self.args.distributed:
                dist.barrier()

            model2train = self.snp2p_model
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model2train.parameters()),
                                    lr=self.args.lr, weight_decay=self.args.wd)

            train_dataset = self.snp2p_dataloader.dataset
            train_dataset.select_phenotypes(phenotype_set)
            snp2p_dataloader = DataLoader(train_dataset, batch_size=None, num_workers=self.args.jobs,
                                          prefetch_factor=2)

            if self.validation_dataloader is not None:
                val_dataset = self.validation_dataloader.dataset
                val_dataset.select_phenotypes(phenotype_set)
                val_snp2p_dataloader = DataLoader(val_dataset, batch_size=None, num_workers=self.args.jobs,
                                                  prefetch_factor=2)
            else:
                val_snp2p_dataloader = None

            early_stopping = EarlyStopping(
                patience=10,
                min_delta=0.001,
                mode='max',
                restore_best_weights=True,
                verbose=True
            )

            best_performance_tracked = float('-inf')
            best_epoch_tracked = 0

            # Use fewer epochs if we have a pretrained model (fine-tuning)
            num_epochs = 15 if self.pretrained_checkpoint is not None else 30

            for epoch in range(num_epochs):
                should_stop = False
                # Training step
                train_loss = self.iter_minibatches(
                    model2train, snp2p_dataloader, optimizer, epoch=epoch,
                    name=f"Warmup with target phenotypes, {','.join(phenotype_set)}"
                )
                if self.args.rank == 0:
                    print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

                # Evaluate every val_step epochs
                if (epoch % self.args.val_step) == 0:
                    self.save_model(model2train, phenotype_set, epoch)

                if val_snp2p_dataloader is not None and (epoch % self.args.val_step) == 0:
                    val_performance = self.evaluate(
                        model2train,
                        dataloader=val_snp2p_dataloader,
                        phenotypes=phenotype_set,
                        epoch=epoch
                    )
                    if self.args.rank == 0:
                        print(f"Validation performance: {val_performance:.4f}")

                    # Check early stopping
                    should_stop = early_stopping(val_performance, model2train)

                    if val_performance == early_stopping.best_score:
                        best_performance_tracked = val_performance
                        best_epoch_tracked = epoch

                    if self.args.distributed:
                        # All-reduce val_performance and should_stop across all ranks
                        val_performance_tensor = torch.tensor(val_performance, device=self.device)
                        dist.all_reduce(val_performance_tensor, op=dist.ReduceOp.MAX)
                        val_performance = val_performance_tensor.item()

                        should_stop_tensor = torch.tensor(int(should_stop), device=self.device)
                        dist.all_reduce(should_stop_tensor, op=dist.ReduceOp.MAX)
                        should_stop = bool(should_stop_tensor.item())

                        # All-reduce early_stopping.best_score
                        best_score_tensor = torch.tensor(early_stopping.best_score, device=self.device)
                        dist.all_reduce(best_score_tensor, op=dist.ReduceOp.MAX)
                        early_stopping.best_score = best_score_tensor.item()

                    if should_stop:
                        if self.args.rank == 0:
                            print(f"Early stopping triggered at epoch {epoch}")
                        break

            initial_performance = early_stopping.best_score
            early_stopping.restore_best_model(model2train)

            if self.args.distributed:
                # All-reduce initial_performance and best_epoch_tracked across all ranks
                initial_performance_tensor = torch.tensor(initial_performance, device=self.device)
                dist.all_reduce(initial_performance_tensor, op=dist.ReduceOp.MAX)
                initial_performance = initial_performance_tensor.item()

                best_epoch_tracked_tensor = torch.tensor(best_epoch_tracked, device=self.device)
                dist.all_reduce(best_epoch_tracked_tensor, op=dist.ReduceOp.MAX)
                best_epoch_tracked = best_epoch_tracked_tensor.item()

            if self.args.rank == 0:
                print(
                    f"Initial training finished. Best performance: {initial_performance:.4f} at epoch {best_epoch_tracked}")

            if self.args.rank == 0:
                mlflow.end_run()

        # Save initial model
        self.save_model(model2train, phenotype_set, epoch='best')

        # Track history
        self.selection_history.append({
            'step': 0,
            'phenotypes': phenotype_set.copy(),
            'performance': initial_performance,
            'added_phenotype': None,
            'pretrained': skip_training
        })

        prev_performance = initial_performance
        best_phenotype_set = phenotype_set.copy()
        best_model = model2train  # Keep track of the best model

        step = 1
        while True:
            self.logger.info(f"Step {step}: Evaluating phenotype additions...")

            # Find best phenotype to add - now also returns the best model
            candidate_performance, candidate_model, candidate_phenotype_set, added_phenotype = \
                self.compare_phenotype_combination(best_model, best_phenotype_set, prev_performance, balanced=False)

            improvement = candidate_performance - prev_performance

            print(f'with Phenotype: {added_phenotype}, {improvement} improved')

            if self.args.distributed:
                improvement_tensor = torch.tensor(improvement, device=self.device)
                dist.all_reduce(improvement_tensor, op=dist.ReduceOp.MAX)
                improvement = improvement_tensor.item()

            if added_phenotype is None or improvement < self.improvement_threshold:
                if added_phenotype is None:
                    self.logger.info(f"No phenotype added, stopping search.")
                    print(f"No phenotype added, stopping search.")
                else:
                    self.logger.info(
                        f"Improvement {improvement:.4f} below threshold {self.improvement_threshold}. Stopping.")
                    print(f"Improvement {improvement:.4f} below threshold {self.improvement_threshold}. Stopping.")
                summary = self.get_selection_summary()
                summary.to_csv(self.args.out+'.selection_history.csv')
                break
            else:
                self.logger.info(f"Added {added_phenotype} with improvement {improvement:.4f}")
                print(f"Added {added_phenotype} with improvement {improvement:.4f}")
                prev_performance = candidate_performance
                best_phenotype_set = candidate_phenotype_set
                best_model = candidate_model  # Update to use the best model for next iteration
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

    def compare_phenotype_combination(self, model, phenotype_set, best_performance, balanced=False):
        """
        Compare different phenotype combinations by adding one phenotype at a time.
        Returns the best performance, best model, best phenotype set, and added phenotype.

        Optimizations:
        1. Validate less frequently (every val_step epochs)
        2. Early stopping for individual candidates
        3. Quick screening phase with fewer epochs
        4. Memory-efficient model handling
        """
        current_best_performance = best_performance
        current_best_phenotype_set = phenotype_set.copy()
        current_added_phenotype = None
        current_best_model = None

        # Quick screening phase - train each candidate for fewer epochs to get rough estimates
        screening_epochs = min(5, self.args.epochs // 3)  # Use 1/3 of total epochs for screening
        candidate_scores = {}

        self.logger.info(f"Starting quick screening with {screening_epochs} epochs...")

        for phenotype in self.phenotypes:
            if phenotype not in phenotype_set:
                temporal_phenotype_set = phenotype_set + [phenotype]
                self.logger.info(f"  Quick screening for {phenotype}...")

                # Create a copy of the model for screening
                screening_model = copy.deepcopy(model)

                # Quick training with fewer epochs
                best_screening_performance = float('-inf')
                for i in range(screening_epochs):
                    performance = self.train_with_phenotypes(
                        screening_model, temporal_phenotype_set, epoch=i,
                        balanced=balanced, train='embedding'
                    )

                    # Only validate every few epochs during screening
                    if i % max(1, screening_epochs // 2) == 0:
                        if self.args.distributed:
                            performance_tensor = torch.tensor(performance, device=self.device)
                            dist.all_reduce(performance_tensor, op=dist.ReduceOp.MAX)
                            performance = performance_tensor.item()

                        best_screening_performance = max(best_screening_performance, performance)

                candidate_scores[phenotype] = best_screening_performance

                # Clean up screening model
                del screening_model
                torch.cuda.empty_cache()

        # Sort candidates by screening performance and only fully train the top candidates
        top_k = min(3, len(candidate_scores))  # Only fully train top 3 candidates
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        top_candidates = [pheno for pheno, _ in sorted_candidates[:top_k]]

        self.logger.info(f"Top candidates from screening: {top_candidates}")

        # Full training for top candidates only
        for phenotype in top_candidates:
            temporal_phenotype_set = phenotype_set + [phenotype]
            self.logger.info(f"  Full training for {phenotype}...")

            if self.args.rank == 0:
                mlflow.start_run(nested=True, run_name=f"Candidate_Training_{'/'.join(temporal_phenotype_set)}")

            # Create a copy of the model for this candidate
            candidate_model = copy.deepcopy(model)

            self.add_phenotype_as_embedding(candidate_model, temporal_phenotype_set)

            # Early stopping for individual candidates
            candidate_early_stopping = EarlyStopping(
                patience=5,  # Smaller patience for candidate evaluation
                min_delta=0.0005,
                mode='max',
                restore_best_weights=True,
                verbose=False
            )

            candidate_best_performance = float('-inf')

            # Phase 1: Embedding training
            for i in range(self.args.epochs):
                temporal_performance = self.train_with_phenotypes(
                    candidate_model, temporal_phenotype_set, epoch=i,
                    balanced=balanced, train='embedding'
                )

                # Validate less frequently
                if i % self.args.val_step == 0:
                    if self.args.distributed:
                        temporal_performance_tensor = torch.tensor(temporal_performance, device=self.device)
                        dist.all_reduce(temporal_performance_tensor, op=dist.ReduceOp.MAX)
                        temporal_performance = temporal_performance_tensor.item()

                    candidate_best_performance = max(candidate_best_performance, temporal_performance)

                    # Early stopping check
                    should_stop = candidate_early_stopping(temporal_performance, candidate_model)

                    if self.args.distributed:
                        should_stop_tensor = torch.tensor(int(should_stop), device=self.device)
                        dist.all_reduce(should_stop_tensor, op=dist.ReduceOp.MAX)
                        should_stop = bool(should_stop_tensor.item())

                    if should_stop:
                        self.logger.info(f"    Early stopping for {phenotype} at epoch {i}")
                        break

                    # Update global best if this candidate is better
                    if temporal_performance > current_best_performance:
                        current_best_performance = temporal_performance
                        current_best_phenotype_set = temporal_phenotype_set
                        current_added_phenotype = phenotype
                        # Save the best model state
                        if current_best_model is not None:
                            del current_best_model
                        current_best_model = copy.deepcopy(candidate_model)
            self.save_model(candidate_model, temporal_phenotype_set)
            # Phase 2: Transformer training (only if embedding phase was promising)
            if candidate_best_performance > best_performance:  # Only continue if showing promise
                candidate_early_stopping.reset()  # Reset for transformer phase

                for i in range(self.args.epochs, self.args.epochs * 2):
                    temporal_performance = self.train_with_phenotypes(
                        candidate_model, temporal_phenotype_set, epoch=i,
                        balanced=balanced, train='transformer'
                    )

                    # Validate less frequently
                    if i % self.args.val_step == 0:
                        if self.args.distributed:
                            temporal_performance_tensor = torch.tensor(temporal_performance, device=self.device)
                            dist.all_reduce(temporal_performance_tensor, op=dist.ReduceOp.MAX)
                            temporal_performance = temporal_performance_tensor.item()

                        candidate_best_performance = max(candidate_best_performance, temporal_performance)

                        # Early stopping check
                        should_stop = candidate_early_stopping(temporal_performance, candidate_model)

                        if self.args.distributed:
                            should_stop_tensor = torch.tensor(int(should_stop), device=self.device)
                            dist.all_reduce(should_stop_tensor, op=dist.ReduceOp.MAX)
                            should_stop = bool(should_stop_tensor.item())

                        if should_stop:
                            self.logger.info(f"    Early stopping for {phenotype} at epoch {i}")
                            break

                        # Update global best if this candidate is better
                        if temporal_performance > current_best_performance:
                            current_best_performance = temporal_performance
                            current_best_phenotype_set = temporal_phenotype_set
                            current_added_phenotype = phenotype
                            # Save the best model state
                            if current_best_model is not None:
                                del current_best_model
                            current_best_model = copy.deepcopy(candidate_model)
                self.save_model(candidate_model, temporal_phenotype_set)
            else:
                self.logger.info(f"    Skipping transformer phase for {phenotype} (insufficient improvement)")

            if self.args.rank == 0:
                mlflow.end_run()

            # Clean up candidate model if it's not the best
            if candidate_model is not current_best_model:
                del candidate_model
                torch.cuda.empty_cache()

        # If no improvement was found, return the original model
        if current_best_model is None:
            current_best_model = model

        return current_best_performance, current_best_model, current_best_phenotype_set, current_added_phenotype

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
        model2train = model
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
        avg_loss = self.iter_minibatches(model2train, snp2p_dataloader, optimizer, epoch=epoch, name=f"Training with phenotypes, {','.join(phenotypes) }, {str(train)}")

        # All ranks should participate in evaluation
        performance = self.evaluate(model2train, val_snp2p_dataloader, epoch, phenotypes=phenotypes, name=f"Validation with phenotypes, {','.join(phenotypes) }, {str(train)}")

        if self.args.rank == 0:
            mlflow.log_metric("train_loss_epoch", avg_loss, step=epoch)

        # All-reduce performance across all ranks
        if self.args.distributed:
            performance_tensor = torch.tensor(performance, device=self.device)
            dist.all_reduce(performance_tensor, op=dist.ReduceOp.MAX)
            performance = performance_tensor.item()
        return performance



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
                    mlflow.log_artifact(output_path)
            else:
                print("Save to...", output_path)
                torch.save(
                    {"arguments": self.args, "state_dict": model.state_dict()},
                    output_path)
                mlflow.log_artifact(output_path)