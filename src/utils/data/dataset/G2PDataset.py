import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler
from scipy.stats import zscore, skewnorm
import torch
from src.utils.tree import TreeParser
from random import shuffle



class G2PDataset(Dataset):

    def __init__(self, genotype_phenotype, cell2genotypes, tree_parser:TreeParser, th=1e-9, gene_weight_type='snp'):
        self.g2p_df = genotype_phenotype
        self.tree_parser = TreeParser
        self.cell2genotype_dict = {genotype: pd.read_csv(mut_data, header=None, index_col=0) for genotype, mut_data in
                                   cell2genotypes.items()}
        self.tree_parser = tree_parser
        self.gene_weight_type = gene_weight_type
        self.th = th

    def __len__(self):
        return self.g2p_df.shape[0]

    def __getitem__(self, index):
        cell_ind, phenotype = self.g2p_df.iloc[index].values
        #print(cell_ind, phenotype)

        cell_mut_dict = {mut_type:self.tree_parser.get_system2genotype_mask(torch.tensor((torch.abs(mut_value.loc[cell_ind].values)>=self.th), dtype=torch.float32))
                         for mut_type, mut_value in self.cell2genotype_dict.items()}
        #print(cell_mut_dict)
        #print(cell_mut_dict['snp'].size(), sum(cell_mut_dict['snp']))
        result_dict = dict()
        result_dict['gene_weight'] = torch.tensor(self.cell2genotype_dict[self.gene_weight_type].loc[cell_ind].values, dtype=torch.float32)
        result_dict['genotype'] = cell_mut_dict
        result_dict['phenotype'] = phenotype
        return result_dict

class G2PCollator(object):

    def __init__(self, genotypes):
        self.genotypes = genotypes

    def __call__(self, data):
        result_dict = dict()
        mutation_dict = dict()
        #result_nested_mask = list()
        for genotype in self.genotypes:
            mutation_dict[genotype] = torch.stack([d['genotype'][genotype] for d in data])
        result_dict['genotype'] = mutation_dict
        result_dict['gene_weight'] = torch.stack([d['gene_weight'] for d in data])
        result_dict['phenotype'] = torch.tensor([d['phenotype'] for d in data], dtype=torch.float32)
        return result_dict