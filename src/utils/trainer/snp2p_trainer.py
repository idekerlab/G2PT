import gc
import json
import torch
import torch.nn as nn
import mlflow
import torch.distributed as dist
from torch.nn.functional import cosine_similarity
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
import copy
from torch.utils.data import get_worker_info
from src.utils.data import move_to
from src.utils.trainer import CCCLoss, FocalLoss, VarianceLoss, MultiplePhenotypeLoss
from src.utils.trainer.utils import EarlyStopping
import torch.nn.functional as F
import copy

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

def _corrcoef(x):                         # x: [B, P]
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, unbiased=False) + 1e-6)
    return (x.T @ x) / (x.size(0) - 1)    # [P, P]

def correlation_matching_loss(pred, target, lam=0.05):
    """
    pred   : [B, P]  logits or probabilities
    target : [B, P]  0/1 labels (float32)
    """
    c_pred = _corrcoef(pred.detach())     # stop grad through corr
    c_true = _corrcoef(target)
    tri_mask = torch.triu(torch.ones_like(c_pred), diagonal=1).bool()
    diff = torch.abs(c_pred - c_true)[tri_mask]
    return lam * diff.mean()

def linear_temperature_schedule(epoch, total_epochs, T_init=1.0, T_final=0.1):
    return max(T_final, T_init - (epoch / total_epochs) * (T_init - T_final))

def get_param_groups(model, base_lr):
    lora_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora' in name.lower():
            lora_params.append(param)
        else:
            base_params.append(param)

    return [
        {'params': base_params, 'lr': base_lr},
        {'params': lora_params, 'lr': base_lr * 10}
    ]


class SNP2PTrainer(object):

    def __init__(self, snp2p_model, tree_parser, snp2p_dataloader, device, args, target_phenotype, validation_dataloader=None, fix_system=False, pretrain_dataloader=None, label_smoothing=0.0, use_mlflow=False):
        self.args = args
        self.device = device
        self.use_mlflow = use_mlflow
        self.snp2p_model = snp2p_model.to(self.device)
        self.ccc_loss = CCCLoss()
        self.beta = 0.1

        # Initialize gate regularization loss if gating is enabled
        if hasattr(args, 'use_gating') and args.use_gating:
            from src.utils.trainer.gate_losses import GateRegularizationLoss
            self.gate_loss_fn = GateRegularizationLoss(
                sparsity_weight=getattr(args, 'gate_sparsity_weight', 0.01),
                entropy_weight=getattr(args, 'gate_entropy_weight', 0.001),
                coherence_weight=getattr(args, 'gate_coherence_weight', 0.0)
            )
            self.use_gating = True
            print(f"Gate regularization enabled: sparsity={args.gate_sparsity_weight}, "
                  f"entropy={args.gate_entropy_weight}")
        else:
            self.use_gating = False
        '''
        if args.regression:
            self.phenotype_loss = nn.MSELoss()
            self.variance_loss = VarianceLoss()
        else:
            self.phenotype_loss = nn.BCELoss()
        '''
        self.loss_type = args.loss
        if self.loss_type == 'focal':
            self.loss = FocalLoss(alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma)
        else:
            self.loss = MultiplePhenotypeLoss(args.bt_inds, args.qt_inds, label_smoothing=label_smoothing)
        self.phenotypes = args.pheno_ids
        self.qt = args.qt
        self.qt_inds = args.qt_inds
        self.bt = args.bt
        self.bt_inds = args.bt_inds
        self.pheno2type = args.pheno2type
        #self.phenotypes = self.qt + self.bt
        self.optimizer = optim.AdamW(get_param_groups(self.snp2p_model, args.lr),
                                     weight_decay=args.wd)
        self.validation_dataloader = validation_dataloader
        self.snp2p_dataloader = snp2p_dataloader
        self.pretrain_dataloader = pretrain_dataloader
        self.pretrain_epochs = 1
        self.best_model = self.snp2p_model
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tree_parser = tree_parser
        #self.total_train_step = len(
        #    self.snp2p_dataloader) * args.epochs  # + len(self.drug_response_dataloader_cellline)*args.epochs
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=5, factor=0.5)
        self.snp2gene_mask = move_to(torch.tensor(tree_parser.snp2gene_mask, dtype=torch.float32), device)
        self.nested_subtrees_forward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='forward', format='indices')#self.args.input_format)
        self.nested_subtrees_forward = move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='backward', format='indices')#self.args.input_format)
        self.nested_subtrees_backward = move_to(self.nested_subtrees_backward, device)
        self.sys2gene_mask = move_to(
            torch.tensor(tree_parser.sys2gene_mask, dtype=torch.float32), device)
        self.gene2sys_mask = self.sys2gene_mask.T
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        self.fix_system = fix_system

        self.target_phenotype = target_phenotype
        self.mlm = args.mlm
        self.bt_validation_metric = getattr(args, "bt_validation_metric", "aupr")

        n_sys2pad = int(np.ceil(len(tree_parser.sys2ind)/8)*8) - len(tree_parser.sys2ind)
        system_temp_tensor = [np.log(1+len(tree_parser.sys2gene_full[tree_parser.ind2sys[i]])) for i in range(len(tree_parser.sys2ind))] + [10.0] * n_sys2pad
        self.system_temp_tensor = move_to(torch.tensor(system_temp_tensor, dtype=torch.float32), device)
        self.early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        self.grad_clip_norm = getattr(args, "grad_clip_norm", 1.0)
        self.current_epoch = 0
        self.best_epoch = None
        self.last_binary_metrics = {}

    def _broadcast_stop_flag(self, should_stop):
        if not self.args.distributed or not dist.is_initialized():
            return should_stop
        stop_tensor = torch.tensor(
            [1 if should_stop else 0],
            device=self.device,
            dtype=torch.int32,
        )
        dist.broadcast(stop_tensor, src=0)
        return bool(stop_tensor.item())

    def _model_for_rank0_io(self):
        if self.args.distributed and hasattr(self.snp2p_model, "module"):
            return self.snp2p_model.module
        return self.snp2p_model

    def _save_model(self, output_path):
        model_to_save = self._model_for_rank0_io()
        torch.save(
            {"arguments": self.args, "state_dict": model_to_save.state_dict(),
             "training_metadata": {"epoch": self.current_epoch,
                                   "best_epoch": self.best_epoch,
                                   "selection_metric": self.bt_validation_metric,
                                   "best_score": self.early_stopping.best_score,
                                   "seed": getattr(self.args, "seed", None)}},
            output_path,
        )

    def train(self, epochs, output_path=None):
        ccc = False
        '''
        self.dynamic_phenotype_sampling = False
        self.snp2p_dataloader.dataset.dataset.dynamic_phenotype_sampling = False
        for epoch in range(self.pretrain_epochs):
            self.snp2p_model.train()
            #self.pretrain_dataloader.dataset.dynamic_phenotype_sampling = False
            self.pretrain(self.snp2p_dataloader, epoch, name='Pretrain')
            gc.collect()
            torch.cuda.empty_cache()
        

        if not self.args.distributed or (self.args.distributed
                                                         and self.args.rank % torch.cuda.device_count() == 0):
            performance = self.evaluate(self.snp2p_model, self.validation_dataloader, 0, name='Validation', print_importance=False)
            gc.collect()
            torch.cuda.empty_cache()
        self.snp2p_dataloader.dataset.dataset.dynamic_phenotype_sampling = False
        self.dynamic_phenotype_sampling = True
        '''
        #self.best_model = self.snp2p_model
        best_performance = 0

        '''
        for epoch in range(5):
            self.iter_minibatches(self.snp2p_dataloader, epoch, name="SNP Adaptation :", snp_only=True)
        if not self.args.distributed or (self.args.distributed and self.args.rank == 0):
            self.evaluate(self.snp2p_model, self.validation_dataloader, epoch + 1,
                         name="SNP adapdation validation", print_importance=False, snp_only=True)

        if self.args.distributed:
            model = self.snp2p_model.module
        else:
            model = self.snp2p_model
        
        for name, param in model.named_parameters():
            if 'snp_embedding' in name:
                param.requires_grad = False
                print(name, " become freezed")
            if 'block_embedding' in name:
                param.requires_grad = False
                print(name, " become freezed")
        '''
        # Keep the optimizer that the scheduler is attached to. Replacing it
        # here silently caused ReduceLROnPlateau to update an unused optimizer.
        if self.scheduler.optimizer is not self.optimizer:
            raise RuntimeError("Scheduler must control the active training optimizer")

        #if not self.args.distributed or (self.args.distributed
        #                                                 and self.args.rank % torch.cuda.device_count() == 0):
        #    performance = self.evaluate(self.snp2p_model, self.validation_dataloader, 0, phenotypes=self.phenotypes, name='Validation', print_importance=False)
        #    gc.collect()
        #    torch.cuda.empty_cache()

        for epoch in range(self.args.start_epoch, epochs):
            self.current_epoch = epoch + 1
            self.train_epoch(epoch + 1, ccc=ccc)
            gc.collect()
            torch.cuda.empty_cache()

            should_validate = ((epoch + 1) % self.args.val_step == 0 or epoch + 1 == epochs)
            if should_validate:
                should_stop = False
                if self.validation_dataloader is not None:
                    if not self.args.distributed or self.args.rank == 0:
                        eval_model = self._model_for_rank0_io()
                        performance = self.evaluate(eval_model, self.validation_dataloader, epoch + 1,
                                                    name="Validation", print_importance=False, phenotypes=self.phenotypes)
                        if not np.isfinite(performance):
                            raise ValueError("Non-finite checkpoint selection metric")
                        self.scheduler.step(performance)
                        if self.early_stopping(performance, eval_model):
                            print("Early stopping")
                            should_stop = True
                        if self.early_stopping.wait == 0:
                            self.best_epoch = epoch + 1
                            if output_path:
                                self._save_model(f"{output_path}.best")
                        if output_path:
                            record = {"epoch": epoch + 1, "selection_metric": self.bt_validation_metric,
                                      "selection_value": float(performance),
                                      "best_epoch": self.best_epoch,
                                      "learning_rates": [g["lr"] for g in self.optimizer.param_groups],
                                      **self.last_binary_metrics}
                            with open(f"{output_path}.validation.jsonl", "a") as handle:
                                handle.write(json.dumps(record, allow_nan=False) + "\n")
                        torch.cuda.empty_cache()
                        gc.collect()

                    should_stop = self._broadcast_stop_flag(should_stop)

                    if output_path and (not self.args.distributed or self.args.rank == 0):
                        output_path_epoch = f"{output_path}.{epoch + 1}"
                        print("Save to...", output_path_epoch)
                        self._save_model(output_path_epoch)
                        if self.use_mlflow:
                            mlflow.log_artifact(output_path_epoch)
                        print(f"MODEL_PATH:{output_path_epoch}")
                else:
                    should_stop = self._broadcast_stop_flag(False)
                    if output_path and (not self.args.distributed or self.args.rank == 0):
                        output_path_epoch = f"{output_path}.{epoch}"
                        print("Save to...", output_path_epoch)
                        self._save_model(output_path_epoch)
                        if self.use_mlflow:
                            mlflow.log_artifact(output_path_epoch)
                        print(f"MODEL_PATH:{output_path_epoch}")
                if should_stop:
                    break
        if self.args.rank == 0:
            if self.validation_dataloader is not None:
                self._model_for_rank0_io().load_state_dict(self.early_stopping.best_weights)
            if output_path:
                output_path_epoch = f"{output_path}.best"
                print("Save to...", output_path_epoch)
                self._save_model(output_path_epoch)
                if self.use_mlflow:
                    mlflow.log_artifact(output_path_epoch)
                print(f"MODEL_PATH:{output_path_epoch}")
            

    def evaluate(self, model, dataloader, epoch, phenotypes, name="Validation", print_importance=False, snp_only=False):

        if self.args.rank == 0:
            print("Evaluating ", ",".join(phenotypes))
        trues = []
        dataloader_with_tqdm = tqdm(dataloader)
        results = []
        covariates = []
        sys_scores = []
        gene_scores = []
        #model.to(self.device)
        model.eval()
        '''
        if epoch > 5:
            ld = True
        else:
            ld = False
        '''
        with torch.no_grad():
            for i, batch in enumerate(dataloader_with_tqdm):
                trues.append(batch['phenotype'])
                covariates.append(batch['covariates'].detach().cpu().numpy())
                batch = move_to(batch, self.device)
                if 'snp2gene_mask' in batch.keys():
                    snp2gene_mask = batch['snp2gene_mask']
                else:
                    snp2gene_mask = self.snp2gene_mask

                if 'gene2sys_mask' in batch.keys():
                    gene2sys_mask = batch['gene2sys_mask']
                    sys2gene_mask = batch['gene2sys_mask'].T
                else:
                    gene2sys_mask = self.gene2sys_mask
                    sys2gene_mask = self.sys2gene_mask

                if 'hierarchical_mask_forward' in batch.keys():
                    hierarchical_mask_forward = batch['hierarchical_mask_forward']
                else:
                    hierarchical_mask_forward = self.nested_subtrees_forward

                if 'hierarchical_mask_backward' in batch.keys():
                    hierarchical_mask_backward = batch['hierarchical_mask_backward']
                else:
                    hierarchical_mask_backward = self.nested_subtrees_backward

                phenotype_predicted = model(batch['genotype'], batch['covariates'], batch['phenotype_indices'],
                                                   hierarchical_mask_forward,
                                                   hierarchical_mask_backward,
                                                   snp2gene_mask = snp2gene_mask,
                                                                   gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                                                   sys2gene_mask=sys2gene_mask,
                                                                   sys2env=self.args.sys2env,
                                                                   env2sys=self.args.env2sys,
                                                                   sys2gene=self.args.sys2gene,
                                                                   snp_only=snp_only, chunk=False)
                                                       #sys_temp= self.system_temp_tensor)
                #for phenotype_predicted_i, module_name in zip(phenotype_predicted, self.g2p_module_names):
                if len(phenotype_predicted.size())==3:
                    phenotype_predicted = phenotype_predicted[:, :, 0]
                else:
                    phenotype_predicted = phenotype_predicted
                phenotype_predicted_detached = phenotype_predicted.detach().cpu().numpy()
                results.append(phenotype_predicted_detached)
                dataloader_with_tqdm.set_description("%s epoch: %d" % (name, epoch))
                del phenotype_predicted
                del phenotype_predicted_detached
                del batch
        trues = np.concatenate(trues)
        covariates = np.concatenate(covariates)
        results = np.concatenate(results)
        #print(results.shape, trues.shape)
        target_performance = 0.
        for i, pheno in enumerate(phenotypes):
            if self.pheno2type[pheno] == 'bt':
                performance = self.evaluate_binary_phenotype(trues[:, i], results[:, i], covariates, phenotype_name=pheno, epoch=epoch, rank=self.args.rank)
            else:
                performance = self.evaluate_continuous_phenotype(trues[:, i], results[:, i], covariates, phenotype_name=pheno, epoch=epoch, rank=self.args.rank)
            if pheno == self.target_phenotype:
                target_performance = performance

        return target_performance



    def evaluate_continuous_phenotype(self, trues, results, covariates=None, phenotype_name="", epoch=0, rank=0):

        mask = (trues != -9)
        if mask.sum() == 0:
            return 0.
        if rank == 0:
            print("Performance overall for %s" % phenotype_name)
        r_square = metrics.r2_score(trues[mask], results[mask])
        pearson = pearsonr(trues[mask], results[mask])
        spearman = spearmanr(trues[mask], results[mask])
        performance = pearson[0]

        if rank == 0:
            if self.use_mlflow:
                mlflow.log_metric(f"{phenotype_name}_r2", r_square, step=epoch)
            if self.use_mlflow:
                mlflow.log_metric(f"{phenotype_name}_pearson", pearson[0], step=epoch)
            if self.use_mlflow:
                mlflow.log_metric(f"{phenotype_name}_spearman", spearman[0], step=epoch)

        print("R_square: ", r_square)
        print("Pearson R", pearson[0])
        print("Spearman Rho: ", spearman[0])
        if covariates is not None:
            print("Performance female")
            female_indices = (covariates[:, 0] == 1) & mask
            r_square = metrics.r2_score(trues[female_indices], results[female_indices])
            pearson = pearsonr(trues[female_indices], results[female_indices])
            spearman = spearmanr(trues[female_indices], results[female_indices])
            female_performance = pearson[0]

            if rank == 0:
                if self.use_mlflow:
                    mlflow.log_metric(f"female_{phenotype_name}_r2", r_square, step=epoch)
                if self.use_mlflow:
                    mlflow.log_metric(f"female_{phenotype_name}_pearson", pearson[0], step=epoch)
                if self.use_mlflow:
                    mlflow.log_metric(f"female_{phenotype_name}_spearman", spearman[0], step=epoch)

            # print(module_name)
            print("R_square: ", r_square)
            print("Pearson R", pearson[0])
            print("Spearman Rho: ", spearman[0])

            print("Performance male")
            male_indices = (covariates[:, 1] == 1) & mask
            r_square = metrics.r2_score(trues[male_indices], results[male_indices])
            pearson = pearsonr(trues[male_indices], results[male_indices])
            spearman = spearmanr(trues[male_indices], results[male_indices])
            if rank == 0:
                if self.use_mlflow:
                    mlflow.log_metric(f"male_{phenotype_name}_r2", r_square, step=epoch)
                if self.use_mlflow:
                    mlflow.log_metric(f"male_{phenotype_name}_pearson", pearson[0], step=epoch)
                if self.use_mlflow:
                    mlflow.log_metric(f"male_{phenotype_name}_spearman", spearman[0], step=epoch)
            male_performance = pearson[0]
            # print(module_name)
            print("R_square: ", r_square)
            print("Pearson R", pearson[0])
            print("Spearman Rho: ", spearman[0])
            print(" ")
        return performance#(female_performance + male_performance)/2

    #@staticmethod
    @staticmethod
    def _drop_dependent_columns(design_matrix):
        independent_cols = []
        rank = 0
        for col_idx in range(design_matrix.shape[1]):
            candidate_cols = independent_cols + [col_idx]
            candidate_rank = np.linalg.matrix_rank(design_matrix[:, candidate_cols])
            if candidate_rank > rank:
                independent_cols.append(col_idx)
                rank = candidate_rank
        return design_matrix[:, independent_cols]

    def _logit_pseudo_r2(self, trues, results, covariates=None):
        if sm is None:
            print("statsmodels is unavailable; falling back to AUPR for validation metric.")
            return None

        y = np.asarray(trues, dtype=float)
        score = np.asarray(results, dtype=float).reshape(-1, 1)
        design_parts = [np.ones((len(y), 1), dtype=float), score]
        if covariates is not None:
            design_parts.append(np.asarray(covariates, dtype=float))
        design = np.column_stack(design_parts)

        valid = np.isfinite(y) & np.isfinite(design).all(axis=1)
        y = y[valid]
        design = design[valid]
        if len(np.unique(y)) < 2 or design.shape[0] <= design.shape[1]:
            return 0.0

        design = self._drop_dependent_columns(design)
        try:
            fit = sm.Logit(y, design).fit(disp=0, maxiter=100)
            pseudo_r2 = float(fit.prsquared)
        except Exception as exc:
            print(f"Pseudo-R2 validation failed ({exc}); falling back to AUPR.")
            return None
        if not np.isfinite(pseudo_r2):
            return None
        return pseudo_r2

    def evaluate_binary_phenotype(self, trues, results, covariates=None, phenotype_name="", epoch=0, rank=0):
        # Never print participant-level labels or scores to logs.
        mask = (trues != -9)
        if mask.sum() == 0:
            return 0.
        if rank == 0:
            print("Performance overall for %s" % phenotype_name)
        auc_performance = metrics.roc_auc_score(trues[mask], results[mask])
        aupr_performance = metrics.average_precision_score(trues[mask], results[mask])
        pseudo_r2_performance = self._logit_pseudo_r2(
            trues[mask],
            results[mask],
            covariates[mask] if covariates is not None else None,
        )
        performance = aupr_performance
        if self.bt_validation_metric == 'auc':
            performance = auc_performance
        elif self.bt_validation_metric == 'pseudo_r2':
            performance = pseudo_r2_performance if pseudo_r2_performance is not None else aupr_performance
        self.last_binary_metrics = {
            "phenotype": phenotype_name,
            "auc": float(auc_performance), "average_precision": float(aupr_performance),
            "validation_refit_pseudo_r2": pseudo_r2_performance,
        }
        if rank == 0:
            if self.use_mlflow:
                mlflow.log_metric(f"{phenotype_name}_auc", auc_performance, step=epoch)
            if self.use_mlflow:
                mlflow.log_metric(f"{phenotype_name}_aupr", aupr_performance, step=epoch)
            if self.use_mlflow and pseudo_r2_performance is not None:
                mlflow.log_metric(f"{phenotype_name}_pseudo_r2", pseudo_r2_performance, step=epoch)
        print("AUC: ", auc_performance)
        print("AUPR: ", aupr_performance)
        print("Pseudo R2: ", pseudo_r2_performance)
        print("Validation metric (%s): " % self.bt_validation_metric, performance)
        if covariates is not None:

            print("Performance female")
            female_indices = (covariates[:, 0] == 1) & mask
            female_auc_performance = metrics.roc_auc_score(trues[female_indices], results[female_indices])
            female_performance = metrics.average_precision_score(trues[female_indices], results[female_indices])
            print("AUC: ", female_auc_performance)
            print("AUPR: ", female_performance)
            if rank == 0:
                if self.use_mlflow:
                    mlflow.log_metric(f"female_{phenotype_name}_auc", female_auc_performance, step=epoch)
                if self.use_mlflow:
                    mlflow.log_metric(f"female_{phenotype_name}_aupr", female_performance, step=epoch)

            print("Performance male")
            male_indices = (covariates[:, 1] == 1) & mask
            male_auc_performance = metrics.roc_auc_score(trues[male_indices], results[male_indices])
            male_performance = metrics.average_precision_score(trues[male_indices], results[male_indices])
            if rank == 0:
                if self.use_mlflow:
                    mlflow.log_metric(f"male_{phenotype_name}_auc", male_auc_performance, step=epoch)
                if self.use_mlflow:
                    mlflow.log_metric(f"male_{phenotype_name}_aupr", male_performance, step=epoch)
            print("AUC: ", male_auc_performance)
            print("AUPR: ", male_performance)
            print(" ")
        return performance#(male_auc_performance + female_auc_performance)/2


    def train_epoch(self, epoch, ccc=False, sex=False):


        self.snp2p_model.train()
        if self.args.distributed:
            if hasattr(self.snp2p_dataloader.sampler, "set_epoch"):
                self.snp2p_dataloader.sampler.set_epoch(epoch)

        new_temperature = linear_temperature_schedule(epoch, self.args.epochs, T_init=1.0, T_final=0.25)
        new_block_sampling_prob = linear_temperature_schedule(epoch, self.args.epochs, T_init=0.1, T_final=1)
        '''
        if not self.args.distributed:
            self.snp2p_model.set_temperature(new_temperature)
            #self.snp2p_model.block_sampling_prob = new_block_sampling_prob
        else:
            self.snp2p_model.module.set_temperature(new_temperature)
            #self.snp2p_model.module.block_sampling_prob = new_block_sampling_prob
        '''
        avg_loss = self.iter_minibatches(self.snp2p_model, self.snp2p_dataloader, self.optimizer,  epoch, name="Batch", sex=False)
        if self.args.rank == 0:
            print(f"Epoch {epoch}: train_loss_epoch={avg_loss:.4f}")
            if self.use_mlflow:
                mlflow.log_metric("train_loss_epoch", avg_loss, step=epoch)

    def iter_minibatches(self, model, dataloader, optimizer, epoch, name="", snp_only=False, sex=False):
        mean_response_loss = 0.
        mean_ccc_loss = 0.
        mean_score_loss = 0.
        mean_snp_loss = 0.
        worker = get_worker_info()
        dataloader_with_tqdm = tqdm(dataloader)
        for i, batch in enumerate(dataloader_with_tqdm):

            #print(f"Rank {self.args.rank} Getting batch")
            batch = move_to(batch, self.device)

            #print(batch)
            if 'snp2gene_mask' in batch.keys():
                snp2gene_mask = batch['snp2gene_mask']
            else:
                snp2gene_mask = self.snp2gene_mask

            if 'gene2sys_mask' in batch.keys():
                gene2sys_mask = batch['gene2sys_mask']
                sys2gene_mask = batch['gene2sys_mask'].T
            else:
                gene2sys_mask = self.gene2sys_mask
                sys2gene_mask = self.sys2gene_mask

            if 'hierarchical_mask_forward' in batch.keys():
                hierarchical_mask_forward = batch['hierarchical_mask_forward']
            else:
                hierarchical_mask_forward = self.nested_subtrees_forward

            if 'hierarchical_mask_backward' in batch.keys():
                hierarchical_mask_backward = batch['hierarchical_mask_backward']
            else:
                hierarchical_mask_backward = self.nested_subtrees_backward


            #print(f"Rank {self.args.rank}, sent to model")
            if self.mlm:
                phenotype_predicted, snp_predicted = model(batch['genotype'], batch['covariates'], batch['phenotype_indices'],
                                                       hierarchical_mask_forward,
                                                       hierarchical_mask_backward,
                                                       #sys_temp= self.system_temp_tensor,
                                                       snp2gene_mask=snp2gene_mask,
                                                       gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                                       sys2gene_mask=sys2gene_mask,
                                                       sys2env=self.args.sys2env,
                                                       env2sys=self.args.env2sys,
                                                       sys2gene=self.args.sys2gene, snp_only=snp_only,
                                                       chunk=False, predict_snp=True)
            else:
                phenotype_predicted = model(batch['genotype'], batch['covariates'], batch['phenotype_indices'],
                                                   hierarchical_mask_forward,
                                                   hierarchical_mask_backward,
                                                   #sys_temp= self.system_temp_tensor,
                                                   snp2gene_mask=snp2gene_mask,
                                                   gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                                   sys2gene_mask=sys2gene_mask,
                                                   sys2env=self.args.sys2env,
                                                   env2sys=self.args.env2sys,
                                                   sys2gene=self.args.sys2gene, snp_only=snp_only,
                                                   chunk=False, predict_snp=False)
            #print(f"Rank {self.args.rank} pass through model")
            if len(phenotype_predicted.size())==3:
                predictions = phenotype_predicted[:, :, 0]
            else:
                predictions = phenotype_predicted


            loss = self.loss
            # MultiplePhenotypeLoss uses GLOBAL column indices (bce_cols / mse_cols
            # covering all n_total phenotypes).  When a subset batch is used
            # (phenotype-selection training), predictions has only n_selected
            # columns.  Expand both tensors to the full global shape, filling
            # unselected columns with -9 (the "missing" sentinel that the loss
            # already skips via `valid_mask = (target != -9)`).
            if hasattr(loss, 'n_tasks') and predictions.shape[1] < loss.n_tasks:
                pheno_idx = batch['phenotype_indices']
                if pheno_idx.dim() > 1:
                    pheno_idx = pheno_idx[0]   # (n_selected,) from first row
                pheno_idx = pheno_idx.long()
                B = predictions.shape[0]
                full_preds   = predictions.new_full((B, loss.n_tasks), -9.0)
                full_targets = predictions.new_full((B, loss.n_tasks), -9.0)
                full_preds[:, pheno_idx]   = predictions
                full_targets[:, pheno_idx] = batch['phenotype']
                phenotype_loss = loss(full_preds, full_targets)
            else:
                phenotype_loss = loss(predictions, batch['phenotype'])
            #print(f"Rank {self.args.rank}: loss calculated", phenotype_loss)

            
            if self.mlm:
                labels = batch['genotype']['snp_label']
                logits = F.log_softmax(snp_predicted, dim=-1)
                snp_loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            else:
                snp_loss = torch.tensor(0.0, device=self.device)
            

            #phenotype_loss += phenotype_loss #+ correlation_matching_loss(predictions, batch['phenotype'])
            mean_response_loss += float(phenotype_loss)
            mean_snp_loss += float(snp_loss)

            # Add gate regularization loss if enabled
            gate_loss = torch.tensor(0.0, device=self.device)
            if self.use_gating:
                # Get phenotype embedding for gate computation
                # Handle DistributedDataParallel wrapper
                model_module = model.module if hasattr(model, 'module') else model

                # Check if model actually has gating methods (backward compatibility)
                if hasattr(model_module, 'get_phenotype_vector') and hasattr(model_module, 'get_gate_values'):
                    phenotype_vector = model_module.get_phenotype_vector(batch['covariates'])
                    pheno_emb = phenotype_vector.squeeze(1)  # (B, hidden_dims)

                    # Get gate values
                    gates = model_module.get_gate_values(pheno_emb)
                    gate_loss, gate_loss_dict = self.gate_loss_fn(gates)

            #print(pheno_inds, phenotype_loss)
            #if phenotype_loss == 0.0:
            #    phenotype_loss = move_to(torch.tensor(phenotype_loss), device=self.device)
            #if phenotype_loss!=0:
            loss = phenotype_loss + 0.1 * snp_loss + gate_loss
            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip_norm and self.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
            optimizer.step()

            #print(f"Rank {self.args.rank} loss back-propagated")
            dataloader_with_tqdm.set_description(
                "%s Train epoch: %3.f, Phenotype loss: %.3f, SNPLoss: %.3f" % (
                    name, epoch, mean_response_loss / (i + 1), mean_snp_loss / (i + 1)))
            del loss
            del phenotype_loss
            del phenotype_predicted
            del batch

        del mean_ccc_loss
        return mean_response_loss / (i + 1)
