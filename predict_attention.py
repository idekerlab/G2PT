import argparse
import json
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict
from enum import Enum
from pathlib import Path
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
from src.utils.config import SNP2PConfig
from src.utils.config.data_config import create_dataset_config
from src.utils.config.config import resolve_checkpoint_args
from src.utils.config.model_config import ModelConfig

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def load_config_file(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if config_path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("PyYAML is required to read YAML config files.") from exc
        with config_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    else:
        with config_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config file must parse to a mapping/object.")
    return data


def flatten_config(data: dict) -> dict:
    config = SNP2PConfig.from_mapping(data).to_flat_namespace()
    flat_config = vars(config)
    for key, value in data.items():
        if key not in flat_config and not isinstance(value, dict):
            flat_config[key] = value
    return flat_config


def apply_defaults(args: argparse.Namespace, defaults: dict) -> None:
    for key, value in defaults.items():
        if getattr(args, key, None) is None and value is not None:
            setattr(args, key, value)


def apply_dataset_overrides(args: argparse.Namespace, config_data: dict) -> None:
    dataset_config = config_data.get("dataset", {})
    if not isinstance(dataset_config, dict):
        dataset_config = {}

    def pick_value(keys):
        for key in keys:
            if key in dataset_config and dataset_config[key]:
                return dataset_config[key]
            if key in config_data and config_data[key]:
                return config_data[key]
        return None

    if args.bfile is None:
        args.bfile = pick_value(["bfile", "test_bfile", "val_bfile", "train_bfile"])
    if args.tsv is None:
        args.tsv = pick_value(["tsv", "test_tsv", "val_tsv", "train_tsv"])
    if args.cov is None:
        args.cov = pick_value(["cov", "test_cov", "val_cov", "train_cov"])
    if args.pheno is None:
        args.pheno = pick_value(["pheno", "test_pheno", "val_pheno", "train_pheno"])
    if args.out is None:
        args.out = pick_value(["out", "output", "output_path"])


class PredictionDatasetFactory:
    @staticmethod
    def create_dataset(
        tree_parser,
        dataset_kind,
        dataset_path,
        dataset_config,
        args,
        model_args,
    ):
        if not dataset_path and dataset_kind != "block":
            return None

        if dataset_kind == "tsv":
            dataset_cls = TSVDataset
        elif dataset_kind == "embedding":
            dataset_cls = EmbeddingDataset
        elif dataset_kind == "block":
            blocks = tree_parser.blocks
            block_bfile_dict = OrderedDict()
            block_model_dict = OrderedDict()
            for chromosome, block in blocks:
                block_bfile = BlockDataset(
                    bfile=f"/cellar/users/i5lee/G2PT_T2D/genotype_data/LD_blocks_HapMap/split_block_chr{chromosome}_block{block}"
                )

                try:
                    print(
                        "Load model weight",
                        f"/cellar/users/i5lee/G2PT_T2D/SNP_embedding/LD_model/ukb_snp_chr{chromosome}.block_{block}.newid.imputed.HapMap.renamed_ld_roberta_epoch_30.pth",
                    )
                    block_model_weight = torch.load(
                        f"/cellar/users/i5lee/G2PT_T2D/SNP_embedding/LD_model/ukb_snp_chr{chromosome}.block_{block}.newid.imputed.HapMap.renamed_ld_roberta_epoch_30.pth",
                        weights_only=True,
                    )
                except:
                    print(
                        "Failed.. Load model weight",
                        f"/cellar/users/i5lee/G2PT_T2D/SNP_embedding/LD_model/ukb_snp_chr{chromosome}.block_{block}.newid.imputed.HapMap.renamed_ld_roberta_epoch_5.pth",
                    )
                    block_model_weight = torch.load(
                        f"/cellar/users/i5lee/G2PT_T2D/SNP_embedding/LD_model/ukb_snp_chr{chromosome}.block_{block}.newid.imputed.HapMap.renamed_ld_roberta_epoch_5.pth",
                        weights_only=True,
                    )
                block_config = RoBERTaConfig(
                    vocab_size=((block_bfile.n_snps * 3) + 2),
                    hidden_size=64,
                    num_hidden_layers=4,
                    num_attention_heads=4,
                    intermediate_size=128,
                    max_position_embeddings=2048,
                )
                block_model = RoBERTa(config=block_config, num_classes=((block_bfile.n_snps * 3) + 2), temperature=False)
                unmatched = block_model.load_state_dict(block_model_weight)
                print("Unmatched parameters: ", unmatched)
                print("Load model weight finished")
                block_model_dict[f"chr{chromosome}_block{block}"] = block_model
                block_bfile_dict[(chromosome, block)] = block_bfile
            return BlockQueryDataset(
                tree_parser,
                args.bfile,
                block_bfile_dict,
                args.cov,
                args.pheno,
                cov_ids=dataset_config.cov_ids,
                cov_mean_dict=model_args.cov_mean_dict,
                pheno_ids=[],
                bt=dataset_config.bt,
                qt=dataset_config.qt,
                cov_std_dict=model_args.cov_std_dict,
            )
        else:
            dataset_cls = PLINKDataset

        base_kwargs = dict(
            #tree_parser=tree_parser,
            cov=args.cov,
            pheno=args.pheno,
            cov_mean_dict=model_args.cov_mean_dict,
            cov_std_dict=model_args.cov_std_dict,
            cov_ids=dataset_config.cov_ids,
            pheno_ids=dataset_config.pheno_ids,
            bt=dataset_config.bt,
            qt=dataset_config.qt,
        )
        if dataset_kind == "embedding":
            iid2ind = getattr(model_args, "iid2ind", None)
            if iid2ind is None:
                raise ValueError("Embedding datasets require iid2ind in the checkpoint arguments.")
            base_kwargs.update(embedding=model_args.embedding, iid2ind=iid2ind)
        else:
            base_kwargs.update(
                block=getattr(tree_parser, "block", False),
                input_format=dataset_config.input_format,
            )
        return dataset_cls(tree_parser, dataset_path, **base_kwargs)

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
    parser.add_argument('--config', help='Config file path (json/yaml).', type=str)
    parser.add_argument('--onto', help='Ontology file used to guide the neural network', type=str, default=None)
    parser.add_argument('--subtree_order', help='Subtree cascading order', nargs='+', default=['default'])
    parser.add_argument('--bfile', help='Training genotype dataset', type=str, default=None)
    parser.add_argument('--tsv', help='Training genotype dataset in tsv format', type=str, default=None)
    parser.add_argument('--cov', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--pheno', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--input-format', default=None, choices=["indices", "embedding", "block"])
    parser.add_argument('--snp', help='Mutation information for cell lines', type=str)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--snp2gene', help='Gene to ID mapping file', type=str, default=None)
    parser.add_argument('--snp2id', help='Gene to ID mapping file', type=str)
    parser.add_argument('--cuda', type=int, default=None)
    parser.add_argument('--model', help='trained model', default=None)
    parser.add_argument('--cpu', type=int, default=None)
    parser.add_argument('--prediction-only', action='store_true')
    parser.add_argument('--out', help='output csv', default=None)
    parser.add_argument('--system_annot', type=str, default=None)

    parser.add_argument('--cov-effect', default=None)


    args = parser.parse_args()

    config_data = {}
    if args.config:
        config_data = load_config_file(args.config)
        config_defaults = flatten_config(config_data)
        apply_defaults(args, config_defaults)
        apply_dataset_overrides(args, config_data)

    if args.model is None:
        raise ValueError("Model checkpoint path is required. Use --model or set it in the config.")

    g2p_model_dict = torch.load(args.model, map_location='cuda:0')
    model_args = resolve_checkpoint_args(g2p_model_dict)
    apply_defaults(args, vars(model_args))

    if args.onto is None or args.snp2gene is None:
        raise ValueError("Both --onto and --snp2gene are required (via config, checkpoint, or CLI).")
    if args.cov is None or args.pheno is None:
        raise ValueError("Both --cov and --pheno are required (via config or CLI).")
    if args.out is None:
        raise ValueError("--out is required (via config or CLI).")
    if args.cuda is None:
        args.cuda = 0
    if args.cpu is None:
        args.cpu = 4
    if args.batch_size is None:
        raise ValueError("--batch-size is required (via config or CLI).")

    if args.input_format is None:
        args.input_format = getattr(model_args, "input_format", "indices")
    if args.cov_effect is None:
        args.cov_effect = getattr(model_args, "cov_effect", "pre")

    print(model_args)
    #g2p_model = g2p_model_dict

    if not args.tsv and not args.bfile:
        raise ValueError("Provide --tsv or --bfile (via config or CLI) for genotype inputs.")

    model_config = ModelConfig.from_namespace(args)
    dataset_config = create_dataset_config(args)

    tree_parser = SNPTreeParser(
        model_config.onto,
        model_config.snp2gene,
        by_chr=False,
        sys_annot_file=args.system_annot,
        multiple_phenotypes=False,
    )
    
    #train_df = pd.read_csv(args.train, sep='\t', header=None)
    #val_df = pd.read_csv(args.val, sep='\t', header=None)
    #test_df = pd.read_csv(args.test, sep='\t', header=None)

    cov_df = pd.read_csv(args.cov, sep='\t')


    dataset_path = args.tsv or args.bfile
    if args.tsv:
        dataset_kind = "tsv"
    elif dataset_config.input_format == "embedding":
        dataset_kind = "embedding"
    elif dataset_config.input_format == "block":
        dataset_kind = "block"
    else:
        dataset_kind = "plink"

    dataset = PredictionDatasetFactory.create_dataset(
        tree_parser,
        dataset_kind,
        dataset_path,
        dataset_config,
        args,
        model_args,
    )
    if dataset is None:
        raise ValueError("Failed to create dataset. Check your input paths and format.")


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
    whole_collator = SNP2PCollator(tree_parser, input_format=dataset_config.input_format)
    whole_dataloader = DataLoader(dataset, shuffle=False, batch_size=args.batch_size,
                                          num_workers=args.cpu, collate_fn=whole_collator)

    nested_subtrees_forward = tree_parser.get_hierarchical_interactions(args.subtree_order, direction='forward', format='indices')
    nested_subtrees_forward = move_to(nested_subtrees_forward, device)

    nested_subtrees_backward = tree_parser.get_hierarchical_interactions(args.subtree_order, direction='backward', format='indices')
    nested_subtrees_backward = move_to(nested_subtrees_backward, device)

    sys2gene_mask = move_to(torch.tensor(tree_parser.sys2gene_mask, dtype=torch.float32), device)
    gene2sys_mask = sys2gene_mask.T
    snp2gene_mask = move_to(torch.tensor(tree_parser.snp2gene_mask, dtype=torch.float32), device)

    g2p_model = SNP2PhenotypeModel(tree_parser, hidden_dims=model_args.hidden_dims,
                                         dropout=0.0, n_covariates=dataset.n_cov,
                                         activation='softmax',  phenotypes=dataset.pheno_ids,
                                         ind2pheno=dataset.ind2pheno,
                                   snp2pheno=model_args.snp2pheno,
                                   gene2pheno=model_args.gene2pheno,
                                   sys2pheno=model_args.sys2pheno,
                                   input_format=dataset_config.input_format,
                                   cov_effect=args.cov_effect,
                                   use_moe=model_args.use_moe,
                                   use_hierarchical_transformer=model_args.use_hierarchical_transformer)

    g2p_model.load_state_dict(g2p_model_dict['state_dict'], strict=False)
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
                                                sys2env=model_args.sys2env,
                                                env2sys=model_args.env2sys,
                                                sys2gene=model_args.sys2gene,
                                                attention=False)
                phenotypes.append(phenotype_predicted.detach().cpu().numpy())
            else:
                if model_args.use_hierarchical_transformer:
                    phenotype_predicted, sys_attention, gene_attention = g2p_model(batch['genotype'], batch['covariates'],
                                                                                   batch['phenotype_indices'],
                                                    nested_subtrees_forward,
                                                    nested_subtrees_backward,
                                                    snp2gene_mask=snp2gene_mask,
                                                    gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                                    sys2gene_mask=sys2gene_mask,
                                                    sys2env=model_args.sys2env,
                                                    env2sys=model_args.env2sys,
                                                    sys2gene=model_args.sys2gene,
                                                    score=True)
                    phenotypes.append(phenotype_predicted.detach().cpu().numpy())
                    sys_attentions.append(sys_attention.detach().cpu().numpy())
                    gene_attentions.append(gene_attention.detach().cpu().numpy())
                else:
                    phenotype_predicted, sys_attention, gene_attention = g2p_model(batch['genotype'], batch['covariates'],
                                                                                   batch['phenotype_indices'],
                                                    nested_subtrees_forward,
                                                    nested_subtrees_backward,
                                                    snp2gene_mask=snp2gene_mask,
                                                    gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                                    sys2gene_mask=sys2gene_mask,
                                                    sys2env=model_args.sys2env,
                                                    env2sys=model_args.env2sys,
                                                    sys2gene=model_args.sys2gene,
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
    sys_attentions = np.concatenate(sys_attentions)[..., :len(tree_parser.ind2sys)] # Shape: (num_samples, num_heads, num_phenotypes, num_systems)
    gene_attentions = np.concatenate(gene_attentions)[..., :len(tree_parser.ind2gene)] # Shape: (num_samples, num_heads, num_phenotypes, num_genes)

    sys_score_cols = [tree_parser.ind2sys[i] for i in range(len(tree_parser.ind2sys))]
    gene_score_cols = [tree_parser.ind2gene[i] for i in range(len(tree_parser.ind2gene))]

    num_heads = sys_attentions.shape[1] # Assuming num_heads is the second dimension

    for pheno, pheno_ind in args.pheno2ind.items():
        for head_idx in range(num_heads): # Loop through each head
            current_sys_attention = sys_attentions[:, head_idx, pheno_ind, :len(sys_score_cols)]
            sys_attention_df = pd.DataFrame(current_sys_attention, columns=sys_score_cols)

            if model_args.gene2pheno: # Check if gene2pheno is enabled
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
            sys_corr_female_dict = {}
            sys_corr_male_dict = {}
            sys_corr_mean_abs_dict = {}

            female_indices = cov_df['SEX'] == 0
            male_indices = cov_df['SEX'] == 1

            for sys in sys_score_cols:
                # Overall correlation
                corr, _ = pearsonr(cov_df[pheno], sys_attention_df[sys])
                sys_corr_dict[sys] = corr

                # Female correlation
                if female_indices.sum() > 1:
                    corr_female, _ = pearsonr(cov_df.loc[female_indices, pheno], sys_attention_df.loc[female_indices, sys])
                else:
                    corr_female = np.nan
                sys_corr_female_dict[sys] = corr_female

                # Male correlation
                if male_indices.sum() > 1:
                    corr_male, _ = pearsonr(cov_df.loc[male_indices, pheno], sys_attention_df.loc[male_indices, sys])
                else:
                    corr_male = np.nan
                sys_corr_male_dict[sys] = corr_male
                
                # Mean absolute correlation
                sys_corr_mean_abs_dict[sys] = np.mean([np.abs(corr_female), np.abs(corr_male)])

            sys_importance_df['corr'] = sys_importance_df['System'].map(lambda a: sys_corr_dict[a])
            sys_importance_df['corr_female'] = sys_importance_df['System'].map(lambda a: sys_corr_female_dict[a])
            sys_importance_df['corr_male'] = sys_importance_df['System'].map(lambda a: sys_corr_male_dict[a])
            sys_importance_df['corr_mean_abs'] = sys_importance_df['System'].map(lambda a: sys_corr_mean_abs_dict[a])
            sys_importance_df.to_csv(f"{args.out}.{pheno}.head_{head_idx}.sys_importance.csv", index=False)
            print(f"Saved system importance for phenotype {pheno}, head {head_idx} to {args.out}.{pheno}.head_{head_idx}.sys_importance.csv")

            if model_args.gene2pheno:
                gene_importance_df = pd.DataFrame({'Gene': gene_score_cols})
                # Calculate correlations for gene attention
                gene_corr_dict = {}
                gene_corr_female_dict = {}
                gene_corr_male_dict = {}
                gene_corr_mean_abs_dict = {}

                female_indices = cov_df['SEX'] == 0
                male_indices = cov_df['SEX'] == 1

                for gene in gene_score_cols:
                    # Overall correlation
                    corr, _ = pearsonr(cov_df[pheno], gene_attention_df[gene])
                    gene_corr_dict[gene] = corr

                    # Female correlation
                    if female_indices.sum() > 1:
                        corr_female, _ = pearsonr(cov_df.loc[female_indices, pheno], gene_attention_df.loc[female_indices, gene])
                    else:
                        corr_female = np.nan
                    gene_corr_female_dict[gene] = corr_female

                    # Male correlation
                    if male_indices.sum() > 1:
                        corr_male, _ = pearsonr(cov_df.loc[male_indices, pheno], gene_attention_df.loc[male_indices, gene])
                    else:
                        corr_male = np.nan
                    gene_corr_male_dict[gene] = corr_male

                    # Mean absolute correlation
                    gene_corr_mean_abs_dict[gene] = np.mean([np.abs(corr_female), np.abs(corr_male)])

                gene_importance_df['corr'] = gene_importance_df['Gene'].map(lambda a: gene_corr_dict[a])
                gene_importance_df['corr_female'] = gene_importance_df['Gene'].map(lambda a: gene_corr_female_dict[a])
                gene_importance_df['corr_male'] = gene_importance_df['Gene'].map(lambda a: gene_corr_male_dict[a])
                gene_importance_df['corr_mean_abs'] = gene_importance_df['Gene'].map(lambda a: gene_corr_mean_abs_dict[a])
                gene_importance_df.to_csv(f"{args.out}.{pheno}.head_{head_idx}.gene_importance.csv", index=False)
                print(f"Saved gene importance for phenotype {pheno}, head {head_idx} to {args.out}.{pheno}.head_{head_idx}.gene_importance.csv")

        # Handle sum of heads
        sum_sys_attention = sys_attentions[:, :, pheno_ind, :len(sys_score_cols)].sum(axis=1) # Sum across heads
        sys_attention_df_sum = pd.DataFrame(sum_sys_attention, columns=sys_score_cols)

        if model_args.gene2pheno:
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
        sys_corr_female_dict_sum = {}
        sys_corr_male_dict_sum = {}
        sys_corr_mean_abs_dict_sum = {}

        female_indices = cov_df['SEX'] == 0
        male_indices = cov_df['SEX'] == 1

        for sys in sys_score_cols:
            # Overall correlation
            corr, _ = pearsonr(cov_df[pheno], sys_attention_df_sum[sys])
            sys_corr_dict_sum[sys] = corr

            # Female correlation
            if female_indices.sum() > 1:
                corr_female, _ = pearsonr(cov_df.loc[female_indices, pheno], sys_attention_df_sum.loc[female_indices, sys])
            else:
                corr_female = np.nan
            sys_corr_female_dict_sum[sys] = corr_female

            # Male correlation
            if male_indices.sum() > 1:
                corr_male, _ = pearsonr(cov_df.loc[male_indices, pheno], sys_attention_df_sum.loc[male_indices, sys])
            else:
                corr_male = np.nan
            sys_corr_male_dict_sum[sys] = corr_male

            # Mean absolute correlation
            sys_corr_mean_abs_dict_sum[sys] = np.mean([np.abs(corr_female), np.abs(corr_male)])

        sys_importance_df_sum['corr'] = sys_importance_df_sum['System'].map(lambda a: sys_corr_dict_sum[a])
        sys_importance_df_sum['corr_female'] = sys_importance_df_sum['System'].map(lambda a: sys_corr_female_dict_sum[a])
        sys_importance_df_sum['corr_male'] = sys_importance_df_sum['System'].map(lambda a: sys_corr_male_dict_sum[a])
        sys_importance_df_sum['corr_mean_abs'] = sys_importance_df_sum['System'].map(lambda a: sys_corr_mean_abs_dict_sum[a])
        sys_importance_df_sum.to_csv(f"{args.out}.{pheno}.head_sum.sys_importance.csv", index=False)
        print(f"Saved system importance for phenotype {pheno}, sum of heads to {args.out}.{pheno}.head_sum.sys_importance.csv")

        if model_args.gene2pheno:
            gene_importance_df_sum = pd.DataFrame({'Gene': gene_score_cols})
            # Calculate correlations for gene attention (sum of heads)
            gene_corr_dict_sum = {}
            gene_corr_female_dict_sum = {}
            gene_corr_male_dict_sum = {}
            gene_corr_mean_abs_dict_sum = {}

            female_indices = cov_df['SEX'] == 0
            male_indices = cov_df['SEX'] == 1

            for gene in gene_score_cols:
                # Overall correlation
                corr, _ = pearsonr(cov_df[pheno], gene_attention_df_sum[gene])
                gene_corr_dict_sum[gene] = corr

                # Female correlation
                if female_indices.sum() > 1:
                    corr_female, _ = pearsonr(cov_df.loc[female_indices, pheno], gene_attention_df_sum.loc[female_indices, gene])
                else:
                    corr_female = np.nan
                gene_corr_female_dict_sum[gene] = corr_female

                # Male correlation
                if male_indices.sum() > 1:
                    corr_male, _ = pearsonr(cov_df.loc[male_indices, pheno], gene_attention_df_sum.loc[male_indices, gene])
                else:
                    corr_male = np.nan
                gene_corr_male_dict_sum[gene] = corr_male

                # Mean absolute correlation
                gene_corr_mean_abs_dict_sum[gene] = np.mean([np.abs(corr_female), np.abs(corr_male)])

            gene_importance_df_sum['corr'] = gene_importance_df_sum['Gene'].map(lambda a: gene_corr_dict_sum[a])
            gene_importance_df_sum['corr_female'] = gene_importance_df_sum['Gene'].map(lambda a: gene_corr_female_dict_sum[a])
            gene_importance_df_sum['corr_male'] = gene_importance_df_sum['Gene'].map(lambda a: gene_corr_male_dict_sum[a])
            gene_importance_df_sum['corr_mean_abs'] = gene_importance_df_sum['Gene'].map(lambda a: gene_corr_mean_abs_dict_sum[a])
            gene_importance_df_sum.to_csv(f"{args.out}.{pheno}.head_sum.gene_importance.csv", index=False)
            print(f"Saved gene importance for phenotype {pheno}, sum of heads to {args.out}.{pheno}.head_sum.gene_importance.csv")

    print("Saving to ... ", args.out)


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    main()
