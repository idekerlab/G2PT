import gc
import torch
import torch.nn as nn
import mlflow
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
import torch.nn.functional as F
import copy

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

    def __init__(self, snp2p_model, tree_parser, snp2p_dataloader, device, args, target_phenotype, validation_dataloader=None, fix_system=False, pretrain_dataloader=None, label_smoothing=0.0):
        self.args = args
        self.device = device
        self.snp2p_model = snp2p_model.to(self.device)
        self.ccc_loss = CCCLoss()
        self.beta = 0.1
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
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'max', patience=5, factor=0.5, verbose=True)
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
        self.dynamic_phenotype_sampling = args.dynamic_phenotype_sampling
        self.target_phenotype = target_phenotype

        n_sys2pad = int(np.ceil(len(tree_parser.sys2ind)/8)*8) - len(tree_parser.sys2ind)
        system_temp_tensor = [np.log(1+len(tree_parser.sys2gene_full[tree_parser.ind2sys[i]])) for i in range(len(tree_parser.sys2ind))] + [10.0] * n_sys2pad
        self.system_temp_tensor = move_to(torch.tensor(system_temp_tensor, dtype=torch.float32), device)

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
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.snp2p_model.parameters()), lr=self.args.lr,
                                     weight_decay=self.args.wd)

        #if not self.args.distributed or (self.args.distributed
        #                                                 and self.args.rank % torch.cuda.device_count() == 0):
        #    performance = self.evaluate(self.snp2p_model, self.validation_dataloader, 0, phenotypes=self.phenotypes, name='Validation', print_importance=False)
        #    gc.collect()
        #    torch.cuda.empty_cache()

        for epoch in range(self.args.start_epoch, epochs):
            self.train_epoch(epoch + 1, ccc=ccc)
            gc.collect()
            torch.cuda.empty_cache()

            if (epoch % self.args.val_step) == 0 & (epoch != 0):
                if self.validation_dataloader is not None:
                    if not self.args.distributed or (self.args.distributed and self.args.rank == 0):
                        performance = self.evaluate(self.snp2p_model, self.validation_dataloader, epoch + 1,
                                                    name="Validation", print_importance=False, phenotypes=self.phenotypes)
                        self.scheduler.step(performance)
                        torch.cuda.empty_cache()
                        gc.collect()
                if output_path:
                    output_path_epoch = output_path + ".%d" % epoch
                    if self.args.distributed:
                        if self.args.rank == 0:
                            print("Save to...", output_path_epoch)
                            torch.save({"arguments": self.args,
                                    "state_dict": self.snp2p_model.module.state_dict()},
                                   output_path_epoch)
                            mlflow.log_artifact(output_path_epoch)
                    else:
                        print("Save to...", output_path_epoch)
                        torch.save(
                            {"arguments": self.args, "state_dict": self.snp2p_model.state_dict()},
                            output_path_epoch)
                        mlflow.log_artifact(output_path_epoch)
            

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


    @staticmethod
    def evaluate_continuous_phenotype(trues, results, covariates=None, phenotype_name="", epoch=0, rank=0):

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
            mlflow.log_metric(f"{phenotype_name}_r2", r_square, step=epoch)
            mlflow.log_metric(f"{phenotype_name}_pearson", pearson[0], step=epoch)
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
                mlflow.log_metric(f"female_{phenotype_name}_r2", r_square, step=epoch)
                mlflow.log_metric(f"female_{phenotype_name}_pearson", pearson[0], step=epoch)
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
                mlflow.log_metric(f"male_{phenotype_name}_r2", r_square, step=epoch)
                mlflow.log_metric(f"male_{phenotype_name}_pearson", pearson[0], step=epoch)
                mlflow.log_metric(f"male_{phenotype_name}_spearman", spearman[0], step=epoch)
            male_performance = pearson[0]
            # print(module_name)
            print("R_square: ", r_square)
            print("Pearson R", pearson[0])
            print("Spearman Rho: ", spearman[0])
            print(" ")
        return (female_performance + male_performance)/2

    @staticmethod
    def evaluate_binary_phenotype(trues, results, covariates=None, phenotype_name="", epoch=0, rank=0):
        if rank == 0:
            print(trues[:50])
            print(results[:50])
        mask = (trues != -9)
        if mask.sum() == 0:
            return 0.
        if rank == 0:
            print("Performance overall for %s" % phenotype_name)
        auc_performance = metrics.roc_auc_score(trues[mask], results[mask])
        performance = metrics.average_precision_score(trues[mask], results[mask])
        if rank == 0:
            mlflow.log_metric(f"{phenotype_name}_auc", auc_performance, step=epoch)
            mlflow.log_metric(f"{phenotype_name}_aupr", performance, step=epoch)
        print("AUC: ", auc_performance)
        print("AUPR: ", performance)
        if covariates is not None:

            print("Performance female")
            female_indices = (covariates[:, 0] == 1) & mask
            female_auc_performance = metrics.roc_auc_score(trues[female_indices], results[female_indices])
            female_performance = metrics.average_precision_score(trues[female_indices], results[female_indices])
            print("AUC: ", female_auc_performance)
            print("AUPR: ", female_performance)
            if rank == 0:
                mlflow.log_metric(f"female_{phenotype_name}_auc", female_auc_performance, step=epoch)
                mlflow.log_metric(f"female_{phenotype_name}_aupr", female_performance, step=epoch)

            print("Performance male")
            male_indices = (covariates[:, 1] == 1) & mask
            male_auc_performance = metrics.roc_auc_score(trues[male_indices], results[male_indices])
            male_performance = metrics.average_precision_score(trues[male_indices], results[male_indices])
            if rank == 0:
                mlflow.log_metric(f"male_{phenotype_name}_auc", male_auc_performance, step=epoch)
                mlflow.log_metric(f"male_{phenotype_name}_aupr", male_performance, step=epoch)
            print("AUC: ", male_auc_performance)
            print("AUPR: ", male_performance)
            print(" ")
        return (male_auc_performance + female_auc_performance)/2


    def train_epoch(self, epoch, ccc=False, sex=False):


        self.snp2p_model.train()
        if self.args.distributed:
            if self.args.z_weight!=0:
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
            mlflow.log_metric("train_loss_epoch", avg_loss, step=epoch)

    def iter_minibatches(self, model, dataloader, optimizer, epoch, name="", snp_only=False, sex=False):
        mean_response_loss = 0.
        mean_ccc_loss = 0.
        mean_score_loss = 0.
        mean_snp_loss = 0.
        worker = get_worker_info()
        if self.dynamic_phenotype_sampling:
            num_batches = np.ceil(len(dataloader.dataset))
            dataloader_with_tqdm = tqdm(dataloader, total=num_batches)
        else:
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

            if self.dynamic_phenotype_sampling:
                pheno_inds = batch['phenotype_indices'][0].detach().cpu().tolist()
                phenos = [self.args.ind2pheno[ind] for ind in pheno_inds]
                dynamic_qt_inds = []
                dynamic_bt_inds = []

                for ind, pheno in enumerate(phenos):
                    if self.args.pheno2type[pheno]=='qt':
                        dynamic_qt_inds.append(ind)
                    elif self.args.pheno2type[pheno]=='bt':
                        dynamic_bt_inds.append(ind)
                #print(dynamic_bt_inds, dynamic_qt_inds)
                loss = MultiplePhenotypeLoss(dynamic_bt_inds, dynamic_qt_inds)

            else:
                loss = self.loss
            #print(predictions.size(), batch['phenotype'].size())
            #phenotype_loss = 0
            #print((batch['phenotype']==-9).sum(), batch['phenotype'].size())
            phenotype_loss = loss(predictions, batch['phenotype'])
            #print(f"Rank {self.args.rank}: loss calculated", phenotype_loss)

            '''
            labels = batch['genotype']['snp_label']
            logits = F.log_softmax(snp_predicted, dim=-1)
            snp_loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            print(batch['genotype']['snp'])
            print(labels)
            print(logits)
            print(logits.size())
            print(loss)
            asdf
            '''

            #phenotype_loss += phenotype_loss #+ correlation_matching_loss(predictions, batch['phenotype'])
            mean_response_loss += float(phenotype_loss)
            #mean_snp_loss += float(snp_loss)
            #print(pheno_inds, phenotype_loss)
            #if phenotype_loss == 0.0:
            #    phenotype_loss = move_to(torch.tensor(phenotype_loss), device=self.device)
            #if phenotype_loss!=0:
            loss =  phenotype_loss #+ 0.01 * snp_loss
            optimizer.zero_grad()
            loss.backward()
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
