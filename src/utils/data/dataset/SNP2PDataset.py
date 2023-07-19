import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler
from scipy.stats import zscore, skewnorm
import torch
from .. import TreeParser, SNPTreeParser
from torch.nn.utils.rnn import pad_sequence
import time
from random import shuffle


class SNP2PDataset(Dataset):

    def __init__(self, genotype_phenotype, snp_data, tree_parser:SNPTreeParser):
        self.g2p_df = genotype_phenotype
        self.tree_parser = TreeParser
        self.snp_df = snp_data
        self.tree_parser = tree_parser

    def __len__(self):
        return self.g2p_df.shape[0]

    def __getitem__(self, index):
        start = time.time()
        sample_ind, phenotype = self.g2p_df.iloc[index].values
        sample2snp_dict = {}
        heterozygous = np.where(self.snp_df.loc[sample_ind].values == 1.0)[0]
        homozygous = np.where(self.snp_df.loc[sample_ind].values == 2.0)[0]
        #sample2snp_dict['embedding']['heterozygous'] = self.tree_parser.get_snp2gene(total_snps, {1.0: heterozygous, 2.0:homozygous})
        snp_type_dict = {}
        snp_type_dict['heterozygous'] = self.tree_parser.get_snp2gene(heterozygous, {1.0: heterozygous})
        snp_type_dict['homozygous'] = self.tree_parser.get_snp2gene(heterozygous, {1.0: homozygous})
        sample2snp_dict['embedding'] = snp_type_dict
        heterozygous_gene_indices = torch.unique(snp_type_dict['heterozygous']['gene']).tolist()#self.tree_parser.get_snp2gene_indices(heterozygous)
        homozygous_gene_indices = torch.unique(snp_type_dict['homozygous']['gene']).tolist()#self.tree_parser.get_snp2gene_indices(homozygous)
        gene2sys_mask_for_gene = torch.zeros((self.tree_parser.n_systems, self.tree_parser.n_genes), dtype=torch.bool)
        gene2sys_mask_for_gene[:, heterozygous_gene_indices] = 1
        gene2sys_mask_for_gene[:, homozygous_gene_indices] = 1
        sample2snp_dict["gene2sys_mask"] = torch.tensor(self.tree_parser.gene2system_mask, dtype=torch.bool) & gene2sys_mask_for_gene
        result_dict = dict()
        result_dict['genotype'] = sample2snp_dict
        result_dict['phenotype'] = phenotype
        end = time.time()
        result_dict["datatime"] = torch.tensor(end-start)
        return result_dict

class SNP2PCollator(object):

    def __init__(self, tree_parser:SNPTreeParser):
        self.tree_parser = tree_parser
        self.padding_index = {"snp":self.tree_parser.n_snps, "gene":self.tree_parser.n_genes}

    def __call__(self, data):
        start = time.time()
        result_dict = dict()
        genotype_dict = dict()

        if self.tree_parser.by_chr:
            snp_type_dict = {}
            for snp_type in ['heterozygous', 'homozygous']:
                embedding_dict = {}
                for CHR in self.tree_parser.chromosomes:
                    indices_dict = dict()
                    for embedding_type in ["snp", "gene"]:
                        indices_dict[embedding_type] = pad_sequence(
                            [d['genotype']['embedding'][snp_type_dict][CHR][embedding_type] for d in data], batch_first=True,
                            padding_value=self.padding_index[embedding_type]).to(torch.long)

                    chr_snp_max_len = indices_dict["snp"].size(1)
                    chr_gene_max_len = indices_dict["gene"].size(1)
                    mask = torch.stack([d["genotype"]["embedding"][snp_type_dict][CHR]['mask'][:chr_gene_max_len,:chr_snp_max_len] for d in data])
                    indices_dict['mask'] = mask
                    embedding_dict[CHR] = indices_dict
                snp_type_dict[snp_type] = embedding_dict
        else:
            snp_type_dict = {}
            for snp_type in ['heterozygous', 'homozygous']:
                embedding_dict = {}
                for embedding_type in ['snp', 'gene']:
                    embedding_dict[embedding_type] = pad_sequence(
                            [d['genotype']['embedding'][snp_type][embedding_type] for d in data], batch_first=True,
                            padding_value=self.padding_index[embedding_type]).to(torch.long)
                gene_max_len = embedding_dict["gene"].size(1)
                snp_max_len = embedding_dict["snp"].size(1)
                mask = torch.stack(
                        [d["genotype"]["embedding"][snp_type]['mask'][:gene_max_len, :snp_max_len] for d in data])
                embedding_dict['mask'] = mask
                snp_type_dict[snp_type] = embedding_dict
        genotype_dict['embedding'] = snp_type_dict
        genotype_dict['gene2sys_mask'] = torch.stack([d['genotype']['gene2sys_mask'] for d in data])
        result_dict['genotype'] = genotype_dict
        result_dict['phenotype'] = torch.tensor([d['phenotype'] for d in data], dtype=torch.float32)
        end = time.time()
        result_dict['datatime'] = torch.mean(torch.stack([d['datatime'] for d in data]))
        result_dict["time"] = torch.tensor(end - start)
        return result_dict