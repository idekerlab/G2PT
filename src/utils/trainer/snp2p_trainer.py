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
from src.utils.trainer import CCCLoss, FocalLoss
import copy


class SNP2PTrainer(object):

    def __init__(self, snp2p_model, snp2p_dataloader, device, args, validation_dataloader=None, fix_system=False):
        self.device = device
        self.snp2p_model = snp2p_model.to(self.device)
        self.ccc_loss = CCCLoss()
        self.beta = 0.1
        if args.regression:
            self.phenotype_loss = nn.MSELoss()
        else:
            self.phenotype_loss = nn.BCELoss()
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.snp2p_model.parameters()), lr=args.lr,
                                     weight_decay=args.wd)
        self.validation_dataloader = validation_dataloader
        self.snp2p_dataloader = snp2p_dataloader
        self.best_model = self.snp2p_model

        self.total_train_step = len(
            self.snp2p_dataloader) * args.epochs  # + len(self.drug_response_dataloader_cellline)*args.epochs
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10)
        self.nested_subtrees_forward = self.snp2p_dataloader.dataset.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='forward', return_indices=True)
        self.nested_subtrees_forward = move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = snp2p_dataloader.dataset.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='backward', return_indices=True)
        self.nested_subtrees_backward = move_to(self.nested_subtrees_backward, device)
        self.sys2gene_mask = move_to(
            torch.tensor(self.snp2p_dataloader.dataset.tree_parser.sys2gene_mask, dtype=torch.bool), device)
        self.gene2sys_mask = self.sys2gene_mask.T
        self.args = args
        self.fix_system = fix_system

    def train(self, epochs, output_path=None):
        ccc = False

        #performance = self.evaluate(self.snp2p_model, self.validation_dataloader, 0, name='Validation', print_importance=False)
        #self.best_model = self.snp2p_model
        best_performance = 0
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
                phenotype_predicted = model(batch['genotype'], batch['covariates'],
                                                                   self.nested_subtrees_forward,
                                                                   self.nested_subtrees_backward,
                                                                   gene2sys_mask=self.gene2sys_mask,#batch['gene2sys_mask'],
                                                                   sys2gene_mask=self.sys2gene_mask,
                                                                   sys2env=self.args.sys2env,
                                                                   env2sys=self.args.env2sys,
                                                                   sys2gene=self.args.sys2gene,
                                                                   )
                #for phenotype_predicted_i, module_name in zip(phenotype_predicted, self.g2p_module_names):
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
        results = np.concatenate(results)[:, 0]
        if print_importance:
            sys_scores = np.concatenate(sys_scores)[:, 0, 0, :]
            gene_scores = np.concatenate(gene_scores)[:, 0, 0, :]
        print(trues[:50])
        print(results[:50])
        if self.args.regression:
            r_square = metrics.r2_score(trues, results)
            pearson = pearsonr(trues, results)
            spearman = spearmanr(trues, results)
            performance = pearson[0]
            #print(module_name)
            print("Performance overall")
            print("R_square: ", r_square)
            print("Pearson R", pearson)
            print("Spearman Rho: ", spearman)

            print("Performance female")
            female_indices = covariates[:, 0]==1
            r_square = metrics.r2_score(trues[female_indices], results[female_indices])
            pearson = pearsonr(trues[female_indices], results[female_indices])
            spearman = spearmanr(trues[female_indices], results[female_indices])
            female_performance = pearson[0]
            #print(module_name)
            print("R_square: ", r_square)
            print("Pearson R", pearson)
            print("Spearman Rho: ", spearman)

            print("Performance male")
            male_indices = covariates[:, 1]==1
            r_square = metrics.r2_score(trues[male_indices], results[male_indices])
            pearson = pearsonr(trues[male_indices], results[male_indices])
            spearman = spearmanr(trues[male_indices], results[male_indices])
            male_performance = pearson[0]
            #print(module_name)
            print("R_square: ", r_square)
            print("Pearson R", pearson)
            print("Spearman Rho: ", spearman)
        else:
            performance = metrics.average_precision_score(trues, results)
            print("Performance overall")
            print("AUPR: ", performance)

            print("Performance female")
            female_indices = covariates[:, 0] == 1
            female_performance = metrics.average_precision_score(trues[female_indices], results[female_indices])
            print("AUPR: ", female_performance)

            print("Performance male")
            male_indices = covariates[:, 1] == 1
            male_performance = metrics.average_precision_score(trues[male_indices], results[male_indices])
            print("AUPR: ", male_performance)
        if print_importance:
            for i, sys in self.snp2p_dataloader.dataset.tree_parser.ind2sys.items():
                female_pearson, _ = pearsonr(results[female_indices], sys_scores[female_indices, i])
                male_pearson, _ = pearsonr(results[male_indices], sys_scores[male_indices, i])
                print(sys, ": ", female_pearson, male_pearson)
            for i, gene in self.snp2p_dataloader.dataset.tree_parser.ind2gene.items():
                female_pearson, _ = pearsonr(results[female_indices], gene_scores[female_indices, i])
                male_pearson, _ = pearsonr(results[male_indices], gene_scores[male_indices, i])
                print(gene, ": ", female_pearson, male_pearson)
        return performance

    def train_epoch(self, epoch, ccc=False, sex=False):
        self.snp2p_model.train()
        if self.args.multiprocessing_distributed:
            self.snp2p_dataloader.sampler.set_epoch(epoch)
        self.iter_minibatches(self.snp2p_dataloader, epoch, name="Batch", ccc=ccc, sex=False)

    def iter_minibatches(self, dataloader, epoch, name="", ccc=False, sex=False):
        mean_response_loss = 0.
        mean_ccc_loss = 0.
        mean_score_loss = 0.
        mean_sex_loss = 0.
        dataloader_with_tqdm = tqdm(dataloader)
        for i, batch in enumerate(dataloader_with_tqdm):
            batch = move_to(batch, self.device)
            phenotype_predicted = self.snp2p_model(batch['genotype'], batch['covariates'],
                                                   self.nested_subtrees_forward,
                                                   self.nested_subtrees_backward,
                                                   gene2sys_mask=self.gene2sys_mask,#batch['gene2sys_mask'],
                                                   sys2gene_mask=self.sys2gene_mask,
                                                   sys2env=self.args.sys2env,
                                                   env2sys=self.args.env2sys,
                                                   sys2gene=self.args.sys2gene)

            phenotype_loss = 0
            ccc_loss = 0
            score_loss = 0
            phenotype_loss_result = self.phenotype_loss(phenotype_predicted[:, 0],
                                                        (batch['phenotype']).to(torch.float32))
            phenotype_loss += phenotype_loss_result
            mean_response_loss += float(phenotype_loss_result)

            if ccc:
                man_indices = batch['covariates'][:, 0] == 1
                ccc_loss_result_man = self.ccc_loss((batch['phenotype']).to(torch.float32)[man_indices], phenotype_predicted[man_indices, 0])
                woman_indices = batch['covariates'][:, 1] == 1
                ccc_loss_result_woman = self.ccc_loss((batch['phenotype']).to(torch.float32)[woman_indices],
                                                    phenotype_predicted[woman_indices, 0])
                ccc_loss += ccc_loss_result_man + ccc_loss_result_woman
                mean_ccc_loss += float((ccc_loss_result_man + ccc_loss_result_woman))/2
            loss = phenotype_loss
            if ccc:
                loss = loss + 0.1 * ccc_loss
            if sex:
                if self.args.multiprocessing_distributed:
                    male_vector = self.snp2p_model.module.covariate_linear_1.weight[:, 0]
                    female_vector = self.snp2p_model.module.covariate_linear_1.weight[:, 1]
                else:
                    male_vector = self.snp2p_model.covariate_linear_1.weight[:, 0]
                    female_vector = self.snp2p_model.covariate_linear_1.weight[:, 1]
                sex_loss = 1 - cosine_similarity(male_vector, female_vector, dim=0)
                # give some allowance to sex loss
                if epoch > 5:
                    sex_loss = 0
                loss = loss + 0.1 * sex_loss
                mean_sex_loss += sex_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            dataloader_with_tqdm.set_description(
                "%s Train epoch: %3.f, Phenotype loss: %.3f, CCCLoss: %.3f, SexLoss %.3f" % (
                    name, epoch, mean_response_loss / (i + 1), mean_ccc_loss / (i + 1),
                    mean_sex_loss / (i + 1)))
            del loss
            del phenotype_loss, ccc_loss, phenotype_loss_result
            del phenotype_predicted
            del batch

        del mean_response_loss, mean_ccc_loss
