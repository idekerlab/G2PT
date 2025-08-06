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

from src.utils.data.dataset.SNP2PDataset import SNP2PCollator, PLINKDataset, EmbeddingDataset, BlockQueryDataset, BlockDataset, TSVDataset
from src.utils.tree import SNPTreeParser
from src.model.LD_infuser.LDRoBERTa import RoBERTa, RoBERTaConfig
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
    parser.add_argument('--tsv-path', help='Training genotype dataset in tsv format', type=str, default=None)
    parser.add_argument('--cov', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--pheno', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--input-format', default='indices', choices=["indices", "embedding", "block"])
    parser.add_argument('--snp', help='Mutation information for cell lines', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--snp2gene', help='Gene to ID mapping file', type=str)
    parser.add_argument('--snp2id', help='Gene to ID mapping file', type=str)
    parser.add_argument('--cuda', type=int)
    parser.add_argument('--model', help='trained model')
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--prediction-only', action='store_true')
    parser.add_argument('--out', help='output csv')
    parser.add_argument('--system_annot', type=str, default=None)

    parser.add_argument('--cov-effect', default='pre')


    args = parser.parse_args()

    tree_parser = SNPTreeParser(args.onto, args.snp2gene, by_chr=False, sys_annot_file=args.system_annot, multiple_phenotypes=False)


    g2p_model_dict = torch.load(args.model)
    print(g2p_model_dict['arguments'])
    #g2p_model = g2p_model_dict
    
    #train_df = pd.read_csv(args.train, sep='\t', header=None)
    #val_df = pd.read_csv(args.val, sep='\t', header=None)
    #test_df = pd.read_csv(args.test, sep='\t', header=None)

    cov_df = pd.read_csv(args.cov, sep='\t')


    #genotypes = pd.read_csv(args.snp, index_col=0, sep='\t')
    if args.tsv_path:
        dataset = TSVDataset(tree_parser, os.path.join(args.tsv_path, 'genotypes.tsv'), args.cov, args.pheno, cov_ids=g2p_model_dict['arguments'].cov_ids,
                               cov_mean_dict=g2p_model_dict['arguments'].cov_mean_dict, pheno_ids=[], bt=[], qt=[],
                               cov_std_dict=g2p_model_dict['arguments'].cov_std_dict, input_format=g2p_model_dict['arguments'].input_format)
    elif args.input_format == 'indices':
        dataset = PLINKDataset(tree_parser, args.bfile, args.cov, args.pheno, cov_ids=g2p_model_dict['arguments'].cov_ids,
                               cov_mean_dict=g2p_model_dict['arguments'].cov_mean_dict, pheno_ids=[], dynamic_phenotype_sampling=False,
                               cov_std_dict=g2p_model_dict['arguments'].cov_std_dict, input_format=g2p_model_dict['arguments'].input_format)
    elif args.input_format == 'embedding':
        dataset = EmbeddingDataset(tree_parser, args.bfile, embedding=g2p_model_dict['arguments'].embedding, cov=args.cov, pheno=args.pheno,
                               cov_ids=g2p_model_dict['arguments'].cov_ids,
                               cov_mean_dict=g2p_model_dict['arguments'].cov_mean_dict, pheno_ids=[],
                               cov_std_dict=g2p_model_dict['arguments'].cov_std_dict,)
    elif args.input_format == 'block':
        blocks = tree_parser.blocks
        block_bfile_dict = OrderedDict()
        block_model_dict = OrderedDict()
        for chromosome, block in blocks:
            block_bfile = BlockDataset(
                bfile=f'/cellar/users/i5lee/G2PT_T2D/genotype_data/LD_blocks_HapMap/split_block_chr{chromosome}_block{block}')

            try:
                print("Load model weight",
                      f'/cellar/users/i5lee/G2PT_T2D/SNP_embedding/LD_model/ukb_snp_chr{chromosome}.block_{block}.newid.imputed.HapMap.renamed_ld_roberta_epoch_30.pth')
                block_model_weight = torch.load(
                    f'/cellar/users/i5lee/G2PT_T2D/SNP_embedding/LD_model/ukb_snp_chr{chromosome}.block_{block}.newid.imputed.HapMap.renamed_ld_roberta_epoch_30.pth',
                    weights_only=True)
            except:
                print("Failed.. Load model weight",
                      f'/cellar/users/i5lee/G2PT_T2D/SNP_embedding/LD_model/ukb_snp_chr{chromosome}.block_{block}.newid.imputed.HapMap.renamed_ld_roberta_epoch_5.pth')
                block_model_weight = torch.load(
                    f'/cellar/users/i5lee/G2PT_T2D/SNP_embedding/LD_model/ukb_snp_chr{chromosome}.block_{block}.newid.imputed.HapMap.renamed_ld_roberta_epoch_5.pth',
                    weights_only=True)
            block_config = RoBERTaConfig(vocab_size=((block_bfile.n_snps * 3) + 2), hidden_size=64, num_hidden_layers=4,
                                         num_attention_heads=4, intermediate_size=128, max_position_embeddings=2048)
            block_model = RoBERTa(config=block_config, num_classes=((block_bfile.n_snps * 3) + 2), temperature=False)
            unmatched = block_model.load_state_dict(block_model_weight)
            print("Unmatched parameters: ", unmatched)
            print("Load model weight finished")
            block_model_dict[f'chr{chromosome}_block{block}'] = block_model
            block_bfile_dict[(chromosome, block)] = block_bfile
        dataset = BlockQueryDataset(tree_parser, args.train_bfile, block_bfile_dict, args.train_cov,
                                          args.train_pheno,
                                    cov_ids=g2p_model_dict['arguments'].cov_ids,
                                    cov_mean_dict=g2p_model_dict['arguments'].cov_mean_dict, pheno_ids=[], bt=args.bt,
                                    qt=args.qt,
                                    cov_std_dict=g2p_model_dict['arguments'].cov_std_dict)


    args.bt_inds = dataset.bt_inds
    args.qt_inds = dataset.qt_inds
    args.bt = dataset.bt
    args.qt = dataset.qt
    args.pheno_ids = dataset.pheno_ids
    args.pheno2ind = dataset.pheno2ind
    args.ind2pheno = dataset.ind2pheno
    args.pheno2type = dataset.pheno2type

    #dataset = SNP2PDataset(whole_df, genotypes, tree_parser, n_cov=args.n_cov, age_mean=age_mean, age_std=age_std)
    device = torch.device("cuda:%d"%args.cuda)
    whole_collator = SNP2PCollator(tree_parser, input_format=g2p_model_dict['arguments'].input_format)
    whole_dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size,
                                          num_workers=args.cpu, collate_fn=whole_collator)

    nested_subtrees_forward = tree_parser.get_hierarchical_interactions(args.subtree_order, direction='forward', format='indices')
    nested_subtrees_forward = move_to(nested_subtrees_forward, device)

    nested_subtrees_backward = tree_parser.get_hierarchical_interactions(args.subtree_order, direction='backward', format='indices')
    nested_subtrees_backward = move_to(nested_subtrees_backward, device)

    sys2gene_mask = move_to(torch.tensor(tree_parser.sys2gene_mask, dtype=torch.float32), device)
    gene2sys_mask = sys2gene_mask.T
    snp2gene_mask = move_to(torch.tensor(tree_parser.snp2gene_mask, dtype=torch.float32), device)

    g2p_model = SNP2PhenotypeModel(tree_parser, hidden_dims=g2p_model_dict['arguments'].hidden_dims,
                                         dropout=0.0, n_covariates=dataset.n_cov,
                                         activation='softmax', n_phenotypes=dataset.n_pheno,
                                   snp2pheno=g2p_model_dict['arguments'].snp2pheno,
                                   gene2pheno=g2p_model_dict['arguments'].gene2pheno,
                                   sys2pheno=g2p_model_dict['arguments'].sys2pheno,
                                   input_format=g2p_model_dict['arguments'].input_format,
                                   cov_effect=args.cov_effect)

    g2p_model.load_state_dict(g2p_model_dict['state_dict'])
    g2p_model = g2p_model.to(device)
    g2p_model = g2p_model.eval()
    
    sys_attentions = []
    gene_attentions = []
    phenotypes = []


    for i, batch in enumerate(tqdm(whole_dataloader)):
        batch = move_to(batch, device)
        with torch.no_grad():
            if args.prediction_only:
                phenotype_predicted = g2p_model(batch['genotype'], batch['covariates'], batch['phenotype_indices'],
                                                nested_subtrees_forward,
                                                nested_subtrees_backward,
                                                snp2gene_mask=snp2gene_mask,
                                                gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                                sys2gene_mask=sys2gene_mask,
                                                sys2env=g2p_model_dict['arguments'].sys2env,
                                                env2sys=g2p_model_dict['arguments'].env2sys,
                                                sys2gene=g2p_model_dict['arguments'].sys2gene,
                                                attention=False)
                phenotypes.append(phenotype_predicted.detach().cpu().numpy())
            else:
                phenotype_predicted, sys_attention, gene_attention = g2p_model(batch['genotype'], batch['covariates'],
                                                                               batch['phenotype_indices'],
                                                nested_subtrees_forward,
                                                nested_subtrees_backward,
                                                snp2gene_mask=snp2gene_mask,
                                                gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                                sys2gene_mask=sys2gene_mask,
                                                sys2env=g2p_model_dict['arguments'].sys2env,
                                                env2sys=g2p_model_dict['arguments'].env2sys,
                                                sys2gene=g2p_model_dict['arguments'].sys2gene,
                                                attention=True)
                phenotypes.append(phenotype_predicted.detach().cpu().numpy())
                sys_attentions.append(sys_attention[0].detach().cpu().numpy())
                gene_attentions.append(gene_attention[0].detach().cpu().numpy())
        #phenotypes.append(prediction.detach().cpu().numpy())  
    
    phenotypes = np.concatenate(phenotypes)#[:, :, 0]
    cov_df = dataset.cov_df
    #cov_df["prediction"] = phenotypes
    for pheno, ind in dataset.pheno2ind.items():
        cov_df[pheno] = phenotypes[:, ind]
    cov_df.to_csv(args.out + '.prediction.csv', index=False)
    if args.prediction_only:
        print("Prediction-only, prediction done")
        quit()
    sys_attentions = np.concatenate(sys_attentions) # Shape: (num_samples, num_heads, num_phenotypes, num_systems)
    gene_attentions = np.concatenate(gene_attentions) # Shape: (num_samples, num_heads, num_phenotypes, num_genes)

    sys_score_cols = [tree_parser.ind2sys[i] for i in range(len(tree_parser.ind2sys))]
    gene_score_cols = [tree_parser.ind2gene[i] for i in range(len(tree_parser.ind2gene))]

    num_heads = sys_attentions.shape[1] # Assuming num_heads is the second dimension

    for pheno, pheno_ind in args.pheno2ind.items():
        for head_idx in range(num_heads): # Loop through each head
            current_sys_attention = sys_attentions[:, head_idx, pheno_ind, :len(sys_score_cols)]
            sys_attention_df = pd.DataFrame(current_sys_attention, columns=sys_score_cols)

            if g2p_model_dict['arguments'].gene2pheno: # Check if gene2pheno is enabled
                current_gene_attention = gene_attentions[:, head_idx, pheno_ind, :]
                gene_attention_df = pd.DataFrame(current_gene_attention, columns=gene_score_cols)
                combined_attention_df = pd.concat([cov_df, sys_attention_df, gene_attention_df], axis=1)
            else:
                combined_attention_df = pd.concat([cov_df, sys_attention_df], axis=1)

            output_filename = f"{args.out}.{pheno}.head_{head_idx}.csv"
            combined_attention_df.to_csv(output_filename, index=False)
            print(f"Saved attention for phenotype {pheno}, head {head_idx} to {output_filename}")

            # Calculate and save importance scores for each head
            sys_importance_df = pd.DataFrame({'System': sys_score_cols})
            if args.system_annot is not None:
                sys_importance_df['System_annot'] = sys_importance_df['System'].map(lambda a: tree_parser.sys_annot_dict[a])
            sys_importance_df['Genes'] = sys_importance_df.System.map(lambda a: ",".join(tree_parser.sys2gene_full[a]))
            sys_importance_df['Size'] = sys_importance_df.System.map(lambda a: len(tree_parser.sys2gene_full[a]))

            # Calculate correlations for sys attention
            sys_corr_dict = {}
            for sys in sys_score_cols:
                corr, _ = pearsonr(cov_df[pheno], sys_attention_df[sys])
                sys_corr_dict[sys] = corr
            sys_importance_df[f'corr_with_{pheno}'] = sys_importance_df['System'].map(lambda a: sys_corr_dict[a])
            sys_importance_df.to_csv(f"{args.out}.{pheno}.head_{head_idx}.sys_importance.csv", index=False)
            print(f"Saved system importance for phenotype {pheno}, head {head_idx} to {args.out}.{pheno}.head_{head_idx}.sys_importance.csv")

            if g2p_model_dict['arguments'].gene2pheno:
                gene_importance_df = pd.DataFrame({'Gene': gene_score_cols})
                # Calculate correlations for gene attention
                gene_corr_dict = {}
                for gene in gene_score_cols:
                    corr, _ = pearsonr(cov_df[pheno], gene_attention_df[gene])
                    gene_corr_dict[gene] = corr
                gene_importance_df[f'corr_with_{pheno}'] = gene_importance_df['Gene'].map(lambda a: gene_corr_dict[a])
                gene_importance_df.to_csv(f"{args.out}.{pheno}.head_{head_idx}.gene_importance.csv", index=False)
                print(f"Saved gene importance for phenotype {pheno}, head {head_idx} to {args.out}.{pheno}.head_{head_idx}.gene_importance.csv")

        # Handle sum of heads
        sum_sys_attention = sys_attentions[:, :, pheno_ind, :len(sys_score_cols)].sum(axis=1) # Sum across heads
        sys_attention_df_sum = pd.DataFrame(sum_sys_attention, columns=sys_score_cols)

        if g2p_model_dict['arguments'].gene2pheno:
            sum_gene_attention = gene_attentions[:, :, pheno_ind, :].sum(axis=1) # Sum across heads
            gene_attention_df_sum = pd.DataFrame(sum_gene_attention, columns=gene_score_cols)
            combined_attention_df_sum = pd.concat([cov_df, sys_attention_df_sum, gene_attention_df_sum], axis=1)
        else:
            combined_attention_df_sum = pd.concat([cov_df, sys_attention_df_sum], axis=1)

        output_filename_sum = f"{args.out}.{pheno}.head_sum.csv"
        combined_attention_df_sum.to_csv(output_filename_sum, index=False)
        print(f"Saved attention for phenotype {pheno}, sum of heads to {output_filename_sum}")

        # Calculate and save importance scores for sum of heads
        sys_importance_df_sum = pd.DataFrame({'System': sys_score_cols})
        if args.system_annot is not None:
            sys_importance_df_sum['System_annot'] = sys_importance_df_sum['System'].map(lambda a: tree_parser.sys_annot_dict[a])
        sys_importance_df_sum['Genes'] = sys_importance_df_sum.System.map(lambda a: ",".join(tree_parser.sys2gene_full[a]))
        sys_importance_df_sum['Size'] = sys_importance_df_sum.System.map(lambda a: len(tree_parser.sys2gene_full[a]))

        # Calculate correlations for sys attention (sum of heads)
        sys_corr_dict_sum = {}
        for sys in sys_score_cols:
            corr, _ = pearsonr(cov_df[pheno], sys_attention_df_sum[sys])
            sys_corr_dict_sum[sys] = corr
        sys_importance_df_sum[f'corr_with_{pheno}'] = sys_importance_df_sum['System'].map(lambda a: sys_corr_dict_sum[a])
        sys_importance_df_sum.to_csv(f"{args.out}.{pheno}.head_sum.sys_importance.csv", index=False)
        print(f"Saved system importance for phenotype {pheno}, sum of heads to {args.out}.{pheno}.head_sum.sys_importance.csv")

        if g2p_model_dict['arguments'].gene2pheno:
            gene_importance_df_sum = pd.DataFrame({'Gene': gene_score_cols})
            # Calculate correlations for gene attention (sum of heads)
            gene_corr_dict_sum = {}
            for gene in gene_score_cols:
                corr, _ = pearsonr(cov_df[pheno], gene_attention_df_sum[gene])
                gene_corr_dict_sum[gene] = corr
            gene_importance_df_sum[f'corr_with_{pheno}'] = gene_importance_df_sum['Gene'].map(lambda a: gene_corr_dict_sum[a])
            gene_importance_df_sum.to_csv(f"{args.out}.{pheno}.head_sum.gene_importance.csv", index=False)
            print(f"Saved gene importance for phenotype {pheno}, sum of heads to {args.out}.{pheno}.head_sum.gene_importance.csv")

    print("Saving to ... ", args.out)


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    main()

