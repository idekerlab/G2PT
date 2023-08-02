import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler
from scipy.stats import zscore, skewnorm
import torch
from src.utils.tree import TreeParser, SNPTreeParser
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import one_hot
import time
from random import shuffle
from ast import literal_eval
from torch.utils.data.distributed import DistributedSampler


class SNP2PDataset(Dataset):

    def __init__(self, genotype_phenotype, snp_data, tree_parser:SNPTreeParser, effective_allele='heterozygous'):
        self.g2p_df = genotype_phenotype
        self.tree_parser = TreeParser
        self.snp_df = snp_data
        self.tree_parser = tree_parser
        self.effective_allele = effective_allele


    def __len__(self):
        return self.g2p_df.shape[0]

    def __getitem__(self, index):
        start = time.time()
        sample_ind, phenotype, sex, *covariates = self.g2p_df.iloc[index].values
        #print(sample_ind, phenotype, sex, covariates)
        sample2snp_dict = {}

        homozygous = [int(i) for i in  (self.snp_df.loc[sample_ind, 'homozygous']).split(",")]#np.where(self.snp_df.loc[sample_ind].values == 2.0)[0]
        type_indices = {1.0:homozygous}
        if self.effective_allele=='heterozygous':
            heterozygous = [int(i) for i in (self.snp_df.loc[sample_ind, 'heterozygous']).split(
                ",")]  # np.where(self.snp_df.loc[sample_ind].values == 1.0)[0]
            total_snps = np.concatenate([heterozygous,homozygous])
            type_indices[2.0] = homozygous
            type_indices[1.0] = heterozygous
        else:
            total_snps = homozygous
        sample2snp_dict['embedding'] = self.tree_parser.get_snp2gene(total_snps, type_indices=type_indices )
        '''
        snp_type_dict = {}
        snp_type_dict['heterozygous'] = self.tree_parser.get_snp2gene(heterozygous, {1.0: heterozygous})
        snp_type_dict['homozygous'] = self.tree_parser.get_snp2gene(heterozygous, {1.0: homozygous})
        sample2snp_dict['embedding'] = snp_type_dict
        #heterozygous_gene_indices = torch.unique(snp_type_dict['heterozygous']['gene']).tolist()#self.tree_parser.get_snp2gene_indices(heterozygous)
        #homozygous_gene_indices = torch.unique(snp_type_dict['homozygous']['gene']).tolist()#self.tree_parser.get_snp2gene_indices(homozygous)
        '''


        homozygous_gene_indices = self.tree_parser.get_snp2gene_indices(homozygous)
        gene2sys_mask_for_gene = torch.zeros((self.tree_parser.n_systems, self.tree_parser.n_genes), dtype=torch.bool)

        gene2sys_mask_for_gene[:, homozygous_gene_indices] = 1
        if self.effective_allele=='heterozygous':
            heterozygous_gene_indices = self.tree_parser.get_snp2gene_indices(heterozygous)
            gene2sys_mask_for_gene[:, heterozygous_gene_indices] = 1

        sample2snp_dict["gene2sys_mask"] = torch.tensor(self.tree_parser.gene2sys_mask, dtype=torch.bool) & gene2sys_mask_for_gene
        result_dict = dict()

        result_dict['phenotype'] = phenotype
        sex_tensor = [0, 0]
        sex_tensor[int(sex)] = 1
        sex = torch.tensor(sex_tensor, dtype=torch.float32)
        covariates = torch.cat([sex, torch.tensor(covariates, dtype=torch.float32)])
        sample2snp_dict["covariates"] = covariates
        result_dict['genotype'] = sample2snp_dict
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
            '''
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
                        [d["genotype"]["embedding"][snp_type]['mask'] for d in data])[:, :gene_max_len, :snp_max_len]
                embedding_dict['mask'] = mask
                #print(mask.sum())
                snp_type_dict[snp_type] = embedding_dict
            '''
        #genotype_dict['embedding'] = snp_type_dict
        embedding_dict = {}
        for embedding_type in ['snp', 'gene']:
            embedding_dict[embedding_type] = pad_sequence(
                [d['genotype']['embedding'][embedding_type] for d in data], batch_first=True,
                padding_value=self.padding_index[embedding_type]).to(torch.long)
        gene_max_len = embedding_dict["gene"].size(1)
        snp_max_len = embedding_dict["snp"].size(1)
        mask = torch.stack(
            [d["genotype"]["embedding"]['mask'] for d in data])[:, :gene_max_len, :snp_max_len]
        embedding_dict['mask'] = mask
        # print(mask.sum())
        genotype_dict['embedding'] = embedding_dict
        genotype_dict['gene2sys_mask'] = torch.stack([d['genotype']['gene2sys_mask'] for d in data])
        genotype_dict['covariates'] = torch.stack([d['genotype']['covariates'] for d in data])
        result_dict['genotype'] = genotype_dict
        result_dict['phenotype'] = torch.tensor([d['phenotype'] for d in data], dtype=torch.float32)
        end = time.time()
        result_dict['datatime'] = torch.mean(torch.stack([d['datatime'] for d in data]))
        result_dict["time"] = torch.tensor(end - start)
        return result_dict

class CohortSampler(Sampler):

    def __init__(self, dataset, n_samples=None, phenotype_index='phenotype', z_weights=1):
        #super(DrugResponseSampler, self).__init__()
        self.indices = dataset.index
        self.num_samples = dataset.shape[0]

        phenotype_values = dataset[phenotype_index]
        #phenotype_mean = np.mean(phenotype_values)
        #phenotype_std = np.std(phenotype_values)
        #weights = np.array(z_weights*np.abs((phenotype_values-phenotype_mean)/np.std(phenotype_std)), dtype=np.int)
        dataset["zscore"] = np.abs(zscore(phenotype_values)*z_weights)
        #self.dataset = result_df.reset_index()[["cellline", "drug", "response", "source", "zscore"]]
        self.weights = torch.tensor(self.dataset.zscore, dtype=torch.double)

    def __iter__(self):
        count = 0
        index = [i for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]
        while count < self.num_samples:
            #print(index[count], type(index[count]))
            #result = index[count].item()
            #print(result, type(result))
            yield index[count].item()
            count += 1

    def __len__(self):
        return self.num_samples



class DistributedCohortSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed = 0, phenotype_index='phenotype', z_weights=1):
        #super(DrugResponseSampler, self).__init__()
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=False)
        self.indices = dataset.index
        self.num_samples = int(dataset.shape[0]/num_replicas)

        phenotype_values = dataset[phenotype_index]
        #phenotype_mean = np.mean(phenotype_values)
        #phenotype_std = np.std(phenotype_values)
        #weights = np.array(z_weights*np.abs((phenotype_values-phenotype_mean)/np.std(phenotype_std)), dtype=np.int)
        dataset["zscore"] = np.abs(zscore(phenotype_values)*z_weights)
        #self.dataset = result_df.reset_index()[["cellline", "drug", "response", "source", "zscore"]]
        self.weights = torch.tensor(self.dataset.zscore.values, dtype=torch.double)

    def __iter__(self):
        count = 0
        index = [i for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]
        while count < self.num_samples:
            #print(index[count], type(index[count]))
            #result = index[count].item()
            #print(result, type(result))
            yield index[count].item()
            count += 1

    def __len__(self):
        return self.num_samples
