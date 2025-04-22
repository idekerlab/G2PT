import gc
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
import copy
from src.utils.data import move_to
from src.utils.trainer import CCCLoss, FocalLoss, VarianceLoss, MultiplePhenotypeLoss
import copy


class SNP2PTrainer(object):

    def __init__(self, snp2p_model, tree_parser, snp2p_dataloader, device, args, validation_dataloader=None, fix_system=False, pretrain_dataloader=None):
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
        self.loss = MultiplePhenotypeLoss(args.bt_inds, args.qt_inds)
        self.phenotypes = args.pheno_ids
        self.qt = args.qt
        self.qt_inds = args.qt_inds
        self.bt = args.bt
        self.bt_inds = args.bt_inds
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.snp2p_model.parameters()), lr=args.lr,
                                     weight_decay=args.wd)
        self.validation_dataloader = validation_dataloader
        self.snp2p_dataloader = snp2p_dataloader
        self.pretrain_dataloader = pretrain_dataloader
        self.pretrain_epochs = 1
        self.best_model = self.snp2p_model

        #self.total_train_step = len(
        #    self.snp2p_dataloader) * args.epochs  # + len(self.drug_response_dataloader_cellline)*args.epochs
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10)
        self.nested_subtrees_forward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='forward', format=self.args.input_format)
        self.nested_subtrees_forward = move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='backward', format=self.args.input_format)
        self.nested_subtrees_backward = move_to(self.nested_subtrees_backward, device)
        self.sys2gene_mask = move_to(
            torch.tensor(tree_parser.sys2gene_mask, dtype=torch.bool), device)
        self.gene2sys_mask = self.sys2gene_mask.T

        self.fix_system = fix_system
        self.dynamic_phenotype_sampling = args.dynamic_phenotype_sampling

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
        

        if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                                         and self.args.rank % torch.cuda.device_count() == 0):
            performance = self.evaluate(self.snp2p_model, self.validation_dataloader, 0, name='Validation', print_importance=False)
            gc.collect()
            torch.cuda.empty_cache()
        self.snp2p_dataloader.dataset.dataset.dynamic_phenotype_sampling = False
        self.dynamic_phenotype_sampling = True
        '''
        #self.best_model = self.snp2p_model
        best_performance = 0
        if self.dynamic_phenotype_sampling:
            self.snp2p_dataloader.dataset.dataset.dynamic_phenotype_sampling = True
        for epoch in range(self.args.start_epoch, epochs):
            self.train_epoch(epoch + 1, ccc=ccc)
            gc.collect()
            torch.cuda.empty_cache()

            if (epoch % self.args.val_step) == 0 & (epoch != 0):
                if self.validation_dataloader is not None:
                    if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                                                     and self.args.rank % torch.cuda.device_count() == 0):
                        performance = self.evaluate(self.snp2p_model, self.validation_dataloader, epoch + 1,
                                                    name="Validation", print_importance=False)
                        torch.cuda.empty_cache()
                        gc.collect()
                if output_path:
                    output_path_epoch = output_path + ".%d" % epoch
                    if self.args.multiprocessing_distributed:
                        if self.args.rank % torch.cuda.device_count() == 0:
                            print("Save to...", output_path_epoch)
                            torch.save({"arguments": self.args,
                                    "state_dict": self.snp2p_model.module.state_dict()},
                                   output_path_epoch)
                    else:
                        print("Save to...", output_path_epoch)
                        torch.save(
                            {"arguments": self.args, "state_dict": self.snp2p_model.state_dict()},
                            output_path_epoch)
            #self.scheduler.step()

    def evaluate(self, model, dataloader, epoch, name="Validation", print_importance=False):
        trues = []
        dataloader_with_tqdm = tqdm(dataloader)
        results = []
        covariates = []
        sys_scores = []
        gene_scores = []
        #model.to(self.device)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader_with_tqdm):
                trues.append(batch['phenotype'])
                covariates.append(batch['covariates'].detach().cpu().numpy())
                batch = move_to(batch, self.device)
                phenotype_predicted = model(batch['genotype'], batch['covariates'], batch['phenotype_indices'],
                                                                   self.nested_subtrees_forward,
                                                                   self.nested_subtrees_backward,
                                                                   gene2sys_mask=self.gene2sys_mask,#batch['gene2sys_mask'],
                                                                   sys2gene_mask=self.sys2gene_mask,
                                                                   sys2env=self.args.sys2env,
                                                                   env2sys=self.args.env2sys,
                                                                   sys2gene=self.args.sys2gene,
                                                                   )
                #for phenotype_predicted_i, module_name in zip(phenotype_predicted, self.g2p_module_names):
                if len(phenotype_predicted.size())==3:
                    phenotype_predicted = phenotype_predicted[:, :, 0]
                else:
                    phenotype_predicted = phenotype_predicted
                phenotype_predicted_detached = phenotype_predicted.detach().cpu().numpy()

                #sys_scores.append(sys_score.detach().cpu().numpy())
                #gene_scores.append(gene_score.detach().cpu().numpy())

                results.append(phenotype_predicted_detached)
                dataloader_with_tqdm.set_description("%s epoch: %d" % (name, epoch))
                del phenotype_predicted
                del phenotype_predicted_detached
                del batch
        trues = np.concatenate(trues)
        covariates = np.concatenate(covariates)
        results = np.concatenate(results)

        for t, i in zip(self.qt, self.qt_inds):
            r_square = metrics.r2_score(trues[:, i], results[:, i])
            pearson = pearsonr(trues[:, i], results[:, i])
            spearman = spearmanr(trues[:, i], results[:, i])
            performance = pearson[0]
            #print(module_name)
            print("Performance overall for %s"%t)
            print("R_square: ", r_square)
            print("Pearson R", pearson)
            print("Spearman Rho: ", spearman)

            print("Performance female")
            female_indices = covariates[:, 0]==1
            r_square = metrics.r2_score(trues[female_indices, i], results[female_indices, i])
            pearson = pearsonr(trues[female_indices, i], results[female_indices, i])
            spearman = spearmanr(trues[female_indices, i], results[female_indices, i])
            female_performance = pearson[0]
            #print(module_name)
            print("R_square: ", r_square)
            print("Pearson R", pearson)
            print("Spearman Rho: ", spearman)

            print("Performance male")
            male_indices = covariates[:, 1]==1
            r_square = metrics.r2_score(trues[male_indices, i], results[male_indices, i])
            pearson = pearsonr(trues[male_indices, i], results[male_indices, i])
            spearman = spearmanr(trues[male_indices, i], results[male_indices, i])
            male_performance = pearson[0]
            #print(module_name)
            print("R_square: ", r_square)
            print("Pearson R", pearson)
            print("Spearman Rho: ", spearman)
        for t, i in zip(self.bt, self.bt_inds):
            performance = metrics.average_precision_score(trues[:, i], results[:, i])
            print("Performance overall for %s"%t)
            print("AUPR: ", performance)
            print("Performance female")
            female_indices = covariates[:, 0] == 1
            female_performance = metrics.average_precision_score(trues[female_indices, i], results[female_indices, i])
            print("AUPR: ", female_performance)

            print("Performance male")
            male_indices = covariates[:, 1] == 1
            male_performance = metrics.average_precision_score(trues[male_indices, i], results[male_indices, i])
            print("AUPR: ", male_performance)

        return performance

    def train_epoch(self, epoch, ccc=False, sex=False):
        self.snp2p_model.train()
        if self.args.multiprocessing_distributed:
            if self.args.z_weight!=0:
                self.snp2p_dataloader.sampler.set_epoch(epoch)
        self.iter_minibatches(self.snp2p_dataloader, epoch, name="Batch", ccc=ccc, sex=False)

    def pretrain(self, dataloader, epoch, name=""):
        mean_response_loss = 0.
        mean_ccc_loss = 0.
        mean_score_loss = 0.
        mean_sex_loss = 0.

        dataloader_with_tqdm = tqdm(dataloader)
        for i, batch in enumerate(dataloader_with_tqdm):
            batch = move_to(batch, self.device)
            nested_subtrees_forward = self.nested_subtrees_forward
            nested_subtrees_backward = self.nested_subtrees_backward
            gene2sys_mask = self.gene2sys_mask
            sys2gene_mask = self.sys2gene_mask


            phenotype_predicted = self.snp2p_model(batch['genotype'], batch['covariates'], batch['phenotype_indices'],
                                                   nested_subtrees_forward,
                                                   nested_subtrees_backward,
                                                   gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                                   sys2gene_mask=sys2gene_mask,
                                                   sys2env=self.args.sys2env,
                                                   env2sys=self.args.env2sys,
                                                   sys2gene=self.args.sys2gene)

            phenotype_loss = 0
            phenotype_loss_result = self.loss(phenotype_predicted[:, :, 0], (batch['phenotype']).to(torch.float32))
            phenotype_loss += phenotype_loss_result
            mean_response_loss += float(phenotype_loss_result)
            loss = phenotype_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            dataloader_with_tqdm.set_description(
                "%s Train epoch: %3.f, Phenotype loss: %.3f, CCCLoss: %.3f, SexLoss %.3f" % (
                    name, epoch, mean_response_loss / (i + 1), mean_ccc_loss / (i + 1),
                    mean_sex_loss / (i + 1)))
            del loss
            del phenotype_loss, phenotype_loss_result
            del phenotype_predicted
            del batch

        del mean_response_loss, mean_ccc_loss

    def iter_minibatches(self, dataloader, epoch, name="", ccc=False, sex=False):
        mean_response_loss = 0.
        mean_ccc_loss = 0.
        mean_score_loss = 0.
        mean_sex_loss = 0.
        if self.dynamic_phenotype_sampling:
            num_batches = np.ceil(len(dataloader.dataset.dataset)/(self.args.batch_size*self.args.world_size))
            dataloader_with_tqdm = tqdm(dataloader, total=num_batches)
        else:
            dataloader_with_tqdm = tqdm(dataloader)
        for i, batch in enumerate(dataloader_with_tqdm):
            batch = move_to(batch, self.device)
            #print(batch)
            if self.dynamic_phenotype_sampling:

                nested_subtrees_forward = batch['mask']['subtree_forward']
                nested_subtrees_backward = batch['mask']['subtree_backward']
                gene2sys_mask = self.gene2sys_mask#batch['mask']['gene2sys_mask']
                sys2gene_mask = self.sys2gene_mask#batch['mask']['sys2gene_mask']
            else:
                nested_subtrees_forward = self.nested_subtrees_forward
                nested_subtrees_backward = self.nested_subtrees_backward
                gene2sys_mask = self.gene2sys_mask
                sys2gene_mask = self.sys2gene_mask


            phenotype_predicted = self.snp2p_model(batch['genotype'], batch['covariates'], batch['phenotype_indices'],
                                                   nested_subtrees_forward,
                                                   nested_subtrees_backward,
                                                   gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                                   sys2gene_mask=sys2gene_mask,
                                                   sys2env=self.args.sys2env,
                                                   env2sys=self.args.env2sys,
                                                   sys2gene=self.args.sys2gene)
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

            phenotype_loss = 0
            phenotype_loss_result = loss(predictions, (batch['phenotype']).to(torch.float32))
            phenotype_loss += phenotype_loss_result
            mean_response_loss += float(phenotype_loss_result)
            loss = phenotype_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            dataloader_with_tqdm.set_description(
                "%s Train epoch: %3.f, Phenotype loss: %.3f, CCCLoss: %.3f, SexLoss %.3f" % (
                    name, epoch, mean_response_loss / (i + 1), mean_ccc_loss / (i + 1),
                    mean_sex_loss / (i + 1)))
            del loss
            del phenotype_loss, phenotype_loss_result
            del phenotype_predicted
            del batch

        del mean_response_loss, mean_ccc_loss
