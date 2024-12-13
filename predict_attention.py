import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import pandas as pd

import argparse
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import numpy as np

from prettytable import PrettyTable

from src.model.model.snp2phenotype import SNP2PhenotypeModel

from src.utils.data.dataset import SNP2PDataset, SNP2PCollator, CohortSampler, DistributedCohortSampler, BinaryCohortSampler, DistributedBinaryCohortSampler, PLINKDataset
from src.utils.tree import SNPTreeParser
from src.utils.trainer import SNP2PTrainer

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        return obj.to(device)


def main():
    parser = argparse.ArgumentParser(description='Some beautiful description')
    parser.add_argument('--onto', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--subtree_order', help='Subtree cascading order', nargs='+', default=['default'])
    parser.add_argument('--bfile', help='Training genotype dataset', type=str, default=None)
    parser.add_argument('--cov', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--system_embedding', default=None)
    parser.add_argument('--gene_embedding', default=None)
    parser.add_argument('--snp', help='Mutation information for cell lines', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--snp2gene', help='Gene to ID mapping file', type=str)
    parser.add_argument('--snp2id', help='Gene to ID mapping file', type=str)
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--model', help='trained model')
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--out', help='output csv')
    parser.add_argument('--system_annot', type=str, default=None)

    args = parser.parse_args()

    tree_parser = SNPTreeParser(args.onto, args.snp2gene, by_chr=False, sys_annot_file=args.system_annot)


    g2p_model_dict = torch.load(args.model)
    #g2p_model = g2p_model_dict
    
    #train_df = pd.read_csv(args.train, sep='\t', header=None)
    #val_df = pd.read_csv(args.val, sep='\t', header=None)
    #test_df = pd.read_csv(args.test, sep='\t', header=None)

    cov_df = pd.read_csv(args.cov, sep='\t')

    age_mean = cov_df['AGE'].mean()
    age_std = cov_df['AGE'].std()

    #genotypes = pd.read_csv(args.snp, index_col=0, sep='\t')
    dataset = PLINKDataset(tree_parser, args.bfile, args.cov)
    #dataset = SNP2PDataset(whole_df, genotypes, tree_parser, n_cov=args.n_cov, age_mean=age_mean, age_std=age_std)
    device = torch.device("cuda:%d"%args.cuda)
    whole_collator = SNP2PCollator(tree_parser)
    whole_dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size,
                                          num_workers=args.cpu, collate_fn=whole_collator)

    nested_subtrees_forward = tree_parser.get_nested_subtree_mask(args.subtree_order, direction='forward', return_indices=True)
    nested_subtrees_forward = move_to(nested_subtrees_forward, device)

    nested_subtrees_backward = tree_parser.get_nested_subtree_mask(args.subtree_order, direction='backward', return_indices=True)
    nested_subtrees_backward = move_to(nested_subtrees_backward, device)

    sys2gene_mask = move_to(torch.tensor(tree_parser.sys2gene_mask, dtype=torch.bool), device)
    gene2sys_mask = sys2gene_mask.T

    g2p_model = SNP2PhenotypeModel(tree_parser, hidden_dims=64,
                                         dropout=0.0, n_covariates=args.n_cov,
                                         binary=True, activation='softmax')

    g2p_model.load_state_dict(g2p_model_dict['state_dict'])
    g2p_model = g2p_model.to(device)
    g2p_model = g2p_model.eval()
    
    sys_attentions = []
    gene_attentions = []
    phenotypes = []


    for i, batch in enumerate(tqdm(whole_dataloader)):
        batch = move_to(batch, device)
        with torch.no_grad():
            phenotype_predicted, sys_attention, gene_attention = g2p_model(batch['genotype'], batch['covariates'],
                                            nested_subtrees_forward,
                                            nested_subtrees_backward,
                                            gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                            sys2gene_mask=sys2gene_mask,
                                            sys2env=True,
                                            env2sys=True,
                                            sys2gene=True, 
                                            attention=True)
        phenotypes.append(phenotype_predicted.detach().cpu().numpy())
        sys_attentions.append(sys_attention.detach().cpu().numpy())
        gene_attentions.append(gene_attention.detach().cpu().numpy())
        #phenotypes.append(prediction.detach().cpu().numpy())  
    

    sys_attentions = np.concatenate(sys_attentions)
    gene_attentions = np.concatenate(gene_attentions)

    sys_attentions = sys_attentions[:, 0, 0, :]
    gene_attentions = gene_attentions[:, 0, 0, :]

    sys_attention_df = pd.DataFrame(sys_attentions)
    gene_attention_df = pd.DataFrame(gene_attentions)

    sys_score_cols = [tree_parser.ind2sys[i] for i in range(len(tree_parser.ind2sys))]
    gene_score_cols = [tree_parser.ind2gene[i] for i in range(len(tree_parser.ind2gene))]

    sys_attention_df.columns = sys_score_cols
    gene_attention_df.columns = gene_score_cols

    phenotypes = np.concatenate(phenotypes)[:, 0]
    cov_df["prediction"] = phenotypes

    whole_dataset_with_attentions = pd.concat([cov_df, sys_attention_df, gene_attention_df], axis=1)
    whole_dataset_with_attentions = whole_dataset_with_attentions[whole_dataset_with_attentions.columns[1:]]
    whole_dataset_with_attentions.to_csv(args.out+'.attention.csv', index=False)

    sys_importance_df = pd.DataFrame({'System':sys_score_cols})
    
    if args.system_annot is not None:
        sys_importance_df['System_annot'] = sys_importance_df['System'].map(lambda a: tree_parser.sys_annot_dict[a])

    sys_importance_df['Genes'] = sys_importance_df.System.map(lambda a: ",".join(tree_parser.sys2gene_full[a]))
    sys_importance_df['Size'] = sys_importance_df.System.map(lambda a: len(tree_parser.sys2gene_full[a]))

    whole_dataset_with_attentions_0 = whole_dataset_with_attentions.loc[whole_dataset_with_attentions.SEX==0]
    whole_dataset_with_attentions_1 = whole_dataset_with_attentions.loc[whole_dataset_with_attentions.SEX==1]

    male_sys_corr_dict = {}
    female_sys_corr_dict = {}

    for sys in sys_score_cols:
        male_sys_corr, _ = pearsonr(whole_dataset_with_attentions_0['prediction'], whole_dataset_with_attentions_0[sys])
        female_sys_corr, _ = pearsonr(whole_dataset_with_attentions_1['prediction'], whole_dataset_with_attentions_1[sys])
        male_sys_corr_dict[sys] = male_sys_corr
        female_sys_corr_dict[sys] = female_sys_corr
    
    
    sys_importance_df['man_corr'] = sys_importance_df['System'].map(lambda a: male_sys_corr_dict[a])
    sys_importance_df['woman_corr'] = sys_importance_df['System'].map(lambda a: female_sys_corr_dict[a])

    sys_importance_df['corr_mean'] = (sys_importance_df['man_corr']+sys_importance_df['woman_corr'])/2
    sys_importance_df['corr_mean_abs'] = (sys_importance_df['man_corr'].abs()+sys_importance_df['woman_corr'].abs())/2
    sys_importance_df.to_csv(args.out+'.sys_corr.csv', index=False)


    gene_importance_df = pd.DataFrame({'Gene':gene_score_cols})
    male_gene_corr_dict = {}
    female_gene_corr_dict = {}

    for gene in gene_score_cols:
        male_gene_corr, _ = pearsonr(whole_dataset_with_attentions_0['prediction'], whole_dataset_with_attentions_0[gene])
        female_gene_corr, _ = pearsonr(whole_dataset_with_attentions_1['prediction'], whole_dataset_with_attentions_1[gene])
        male_gene_corr_dict[gene] = male_gene_corr
        female_gene_corr_dict[gene] = female_gene_corr
    
    gene_importance_df['man_corr'] = gene_importance_df['Gene'].map(lambda a: male_gene_corr_dict[a])
    gene_importance_df['woman_corr'] = gene_importance_df['Gene'].map(lambda a: female_gene_corr_dict[a])

    gene_importance_df['corr_mean'] = (gene_importance_df['man_corr']+gene_importance_df['woman_corr'])/2
    gene_importance_df['corr_mean_abs'] = (gene_importance_df['man_corr'].abs()+gene_importance_df['woman_corr'].abs())/2
    gene_importance_df.to_csv(args.out+'.gene_corr.csv', index=False)

    #whole_df.to_csv(args.out, index=False)
    print("Saving to ... ", args.out)


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    main()

