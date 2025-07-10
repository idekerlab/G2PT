import argparse
import os
import subprocess
import socket
import warnings

import pandas as pd

import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed


from prettytable import PrettyTable
from src.utils.chunker import MaskBasedChunker, AttentionAwareChunker

from src.model.model.snp2phenotype import SNP2PhenotypeModel

from torch.utils.data.distributed import DistributedSampler
from src.utils.data.dataset import SNP2PCollator, PLINKDataset, PhenotypeSelectionDataset, PhenotypeSelectionDatasetDDP
from src.utils.tree import SNPTreeParser
from src.utils.trainer import GreedyMultiplePhenotypeTrainer
from datetime import timedelta

from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict


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
from torch.utils.data import get_worker_info
from src.utils.data import move_to
from src.utils.trainer import CCCLoss, FocalLoss, VarianceLoss, MultiplePhenotypeLoss
import torch.nn.functional as F
import copy

def evaluate_continuous_phenotype(trues, results, covariates=None, phenotype_name=""):
    mask = (trues != -9)
    if mask.sum() == 0:
        return 0.
    print("Performance overall for %s" % phenotype_name)
    r_square = metrics.r2_score(trues[mask], results[mask])
    pearson = pearsonr(trues[mask], results[mask])
    spearman = spearmanr(trues[mask], results[mask])
    performance = pearson[0]

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
        # print(module_name)
        print("R_square: ", r_square)
        print("Pearson R", pearson[0])
        print("Spearman Rho: ", spearman[0])

        print("Performance male")
        male_indices = (covariates[:, 1] == 1) & mask
        r_square = metrics.r2_score(trues[male_indices], results[male_indices])
        pearson = pearsonr(trues[male_indices], results[male_indices])
        spearman = spearmanr(trues[male_indices], results[male_indices])
        male_performance = pearson[0]
        # print(module_name)
        print("R_square: ", r_square)
        print("Pearson R", pearson[0])
        print("Spearman Rho: ", spearman[0])
        print(" ")
    return performance


def evaluate_binary_phenotype(trues, results, covariates=None, phenotype_name=""):
    print(trues[:50])
    print(results[:50])
    mask = (trues != -9)
    if mask.sum() == 0:
        return 0.
    print("Performance overall for %s" % phenotype_name)
    auc_performance = metrics.roc_auc_score(trues[mask], results[mask])
    performance = metrics.average_precision_score(trues[mask], results[mask])
    print("AUC: ", auc_performance)
    print("AUPR: ", performance)
    if covariates is not None:
        print("Performance female")
        female_indices = (covariates[:, 0] == 1) & mask
        female_auc_performance = metrics.roc_auc_score(trues[female_indices], results[female_indices])
        female_performance = metrics.average_precision_score(trues[female_indices], results[female_indices])
        print("AUC: ", female_auc_performance)
        print("AUPR: ", female_performance)

        print("Performance male")
        male_indices = (covariates[:, 1] == 1) & mask
        male_auc_performance = metrics.roc_auc_score(trues[male_indices], results[male_indices])
        male_performance = metrics.average_precision_score(trues[male_indices], results[male_indices])
        print("AUC: ", male_auc_performance)
        print("AUPR: ", male_performance)
        print(" ")
    return auc_performance


def evaluate(model, dataloader, epoch, phenotypes, name="Validation", print_importance=False, snp_only=False,
             return_prediction=False):
    print("Evaluating ", ",".join(phenotypes))
    trues = []
    dataloader_with_tqdm = tqdm(dataloader)
    results = []
    covariates = []
    sys_scores = []
    gene_scores = []
    # model.to(self.device)
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
            batch = move_to(batch, device)

            snp2gene_mask = batch['snp2gene_mask']
            gene2sys_mask = batch['gene2sys_mask']
            sys2gene_mask = batch['gene2sys_mask'].T
            hierarchical_mask_forward = batch['hierarchical_mask_forward']
            hierarchical_mask_backward = batch['hierarchical_mask_backward']

            phenotype_predicted = model(batch['genotype'], batch['covariates'], batch['phenotype_indices'],
                                        hierarchical_mask_forward,
                                        hierarchical_mask_backward,
                                        snp2gene_mask=snp2gene_mask,
                                        gene2sys_mask=gene2sys_mask,  # batch['gene2sys_mask'],
                                        sys2gene_mask=sys2gene_mask,
                                        sys2env=args.sys2env,
                                        env2sys=args.env2sys,
                                        sys2gene=args.sys2gene,
                                        snp_only=snp_only, chunk=False)
            # sys_temp= self.system_temp_tensor)
            # for phenotype_predicted_i, module_name in zip(phenotype_predicted, self.g2p_module_names):
            if len(phenotype_predicted.size()) == 3:
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
    # print(results.shape, trues.shape)
    target_performance = 0.
    for i, pheno in enumerate(phenotypes):
        if args.pheno2type[pheno] == 'bt':
            performance = evaluate_binary_phenotype(trues[:, i], results[:, i], covariates, phenotype_name=pheno)
        else:
            performance = evaluate_continuous_phenotype(trues[:, i], results[:, i], covariates, phenotype_name=pheno)
        if pheno == args.target_phenotype:
            target_performance = performance
            print("Target performance: ", target_performance)
    if return_prediction:
        return target_performance, trues, results, covariates
    else:
        return target_performance

def initialize_dataset_with_selected_phenotypes(dataset, snp_phenotypes, target_phenotype,  jobs=2, prefetch_factor=2, ):
    dataset.select_phenotypes(snp_phenotypes, target_phenotype)
    snp2p_dataloader = DataLoader(dataset, batch_size=None, num_workers= jobs)
    return snp2p_dataloader

model_dict = torch.load('/cellar/projects//G2PT_T2D/G2PT_output/UKBiobank/Model/model/ont_fold_1_pval_1e-4.DIAMANTE.snp2gene.T2D.lipid.34594039.BP.txtT2D.20.pt')

args = args

tree_parser = SNPTreeParser(args.onto,
                            args.snp2gene,
                            dense_attention=False, multiple_phenotypes=True)

snp2p_dataset = PLINKDataset(tree_parser, args.train_bfile,
                             args.train_cov,
                             args.train_pheno, flip=args.flip,
                             input_format=args.input_format,
                             cov_ids=args.cov_ids, pheno_ids=args.pheno_ids, bt=args.bt, qt=args.qt)

snp2p_collator = SNP2PCollator(tree_parser, input_format=args.input_format)

snp2p_model = SNP2PhenotypeModel(tree_parser, args.hidden_dims,
                                 sys2pheno=args.sys2pheno, gene2pheno=args.gene2pheno, snp2pheno=args.snp2pheno,
                                 interaction_types=args.interaction_types,
                                 dropout=args.dropout, n_covariates=snp2p_dataset.n_cov,
                                 n_phenotypes=snp2p_dataset.n_pheno, activation='softmax', input_format=args.input_format)

new_state_dict = OrderedDict()
for k, v in model_dict['state_dict'].items():
    # remove the "module." prefix
    name = k.replace('module.', '', 1)  # only remove first occurrence
    new_state_dict[name] = v

snp2p_model.load_state_dict(state_dict=new_state_dict)

snp2p_dataloader = initialize_dataset_with_selected_phenotypes(snp2p_dataset, ['T2D'], ['T2D'], jobs=8)


val_snp2p_dataset = PLINKDataset(tree_parser, args.val_bfile, args.val_cov, args.val_pheno, cov_mean_dict=args.cov_mean_dict,
                                 cov_std_dict=args.cov_std_dict, flip=args.flip, input_format=args.input_format,
                                 cov_ids=args.cov_ids, pheno_ids=args.pheno_ids, bt=args.bt, qt=args.qt)

val_snp2p_dataset = PhenotypeSelectionDataset(tree_parser, val_snp2p_dataset, snp2p_collator, batch_size=args.batch_size,#int(args.batch_size/(args.ngpus_per_node*2)),
                                   shuffle=False)
val_snp2p_dataloader = initialize_dataset_with_selected_phenotypes(val_snp2p_dataset, ['T2D'], ['T2D'], jobs=8)