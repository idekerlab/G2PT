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

    def __init__(self, genotype_phenotype, snp_data, tree_parser:SNPTreeParser, effective_allele='heterozygous', age_mean=None, age_std=None):
        self.g2p_df = genotype_phenotype
        self.tree_parser = TreeParser
        self.snp_df = snp_data
        self.tree_parser = tree_parser
        self.effective_allele = effective_allele
        if age_mean is None:
            self.age_mean = self.g2p_df[3].mean()
        else:
            self.age_mean = age_mean
        if age_std is None:
            self.age_std = self.g2p_df[3].std()
        else:
            self.age_std = age_std


    def __len__(self):
        return self.g2p_df.shape[0]

    def __getitem__(self, index):
        start = time.time()
        sample_ind, phenotype, sex, age, age_sq, *covariates = self.g2p_df.iloc[index].values
        #print(index, sample_ind, phenotype, sex, covariates)
        sample2snp_dict = {}
        homozygous_a1 = self.snp_df.loc[sample_ind, 'homozygous_a1']
        if type(homozygous_a1)==str:
            homozygous_a1 = [int(i) for i in homozygous_a1.split(",")]
        else:
            homozygous_a1 = []
        #homozygous_a2 = [int(i) for i in (self.snp_df.loc[sample_ind, 'homozygous_a2']).split(",")]
        #heterozygous = [int(i) for i in (self.snp_df.loc[sample_ind, 'heterozygous']).split(",")]
        heterozygous = self.snp_df.loc[sample_ind, 'heterozygous']
        if type(heterozygous) == str:
            heterozygous = [int(i) for i in heterozygous.split(",")]
        else:
            heterozygous = []
        '''
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
        result_dict = dict()
        snp_type_dict = dict()

        snp_type_dict['homozygous_a1'] = self.tree_parser.get_snp2gene(homozygous_a1, {1.0: homozygous_a1})
        #snp_type_dict['homozygous_a0'] = self.tree_parser.get_snp2gene(homozygous_a2, {1.0: homozygous_a2})
        snp_type_dict['heterozygous'] = self.tree_parser.get_snp2gene(heterozygous, {1.0: heterozygous})
        sample2snp_dict['embedding'] = snp_type_dict
        '''
        #homozygous_a2_gene_indices = torch.unique(snp_type_dict['homozygous_a0']['gene']).tolist()
        #homozygous_a1_gene_indices = self.tree_parser.get_snp2gene_indices(homozygous_a1)
        #homozygous_a2_gene_indices = self.tree_parser.get_snp2gene_indices(homozygous_a2)
        #heterozygous_gene_indices = self.tree_parser.get_snp2gene_indices(heterozygous)
        '''
        heterozygous_gene_indices = torch.unique(snp_type_dict['heterozygous']['gene']).tolist()
        homozygous_a1_gene_indices = torch.unique(snp_type_dict['homozygous_a1']['gene']).tolist()
        gene2sys_mask_for_gene = torch.zeros((self.tree_parser.n_systems, self.tree_parser.n_genes), dtype=torch.bool)
        gene2sys_mask_for_gene[:, homozygous_a1_gene_indices] = 1
        gene2sys_mask_for_gene[:, heterozygous_gene_indices] = 1
        result_dict["gene2sys_mask"] = torch.tensor(self.tree_parser.gene2sys_mask, dtype=torch.bool) & gene2sys_mask_for_gene
        result_dict['phenotype'] = phenotype
        sex_age_tensor = [0, 0, 0, 0]
        #sex_age_tensor = [0, 0]
        if int(sex)==-9:
            pass
        else:
            sex_age_tensor[int(sex)] = 1
        sex_age_tensor[2] = (age - self.age_mean)/self.age_std
        sex_age_tensor[3] = (age_sq - self.age_mean**2)/(self.age_std**2)
        sex_age_tensor = torch.tensor(sex_age_tensor, dtype=torch.float32)
        covariates = sex_age_tensor#torch.cat([sex_age_tensor, torch.tensor(covariates, dtype=torch.float32)])
        result_dict['genotype'] = sample2snp_dict
        end = time.time()
        result_dict["datatime"] = torch.tensor(end-start)
        result_dict["covariates"] = covariates

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
            for snp_type in ['heterozygous', 'homozygous_a1', 'homozygous_a0']:
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
            for snp_type in ['heterozygous', 'homozygous_a1']:
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
        '''
        genotype_dict['embedding'] = snp_type_dict
        result_dict['gene2sys_mask'] = torch.stack([d['gene2sys_mask'] for d in data])
        result_dict['genotype'] = genotype_dict
        result_dict['covariates'] = torch.stack([d['covariates'] for d in data])
        result_dict['phenotype'] = torch.tensor([d['phenotype'] for d in data], dtype=torch.float32)
        end = time.time()
        result_dict['datatime'] = torch.mean(torch.stack([d['datatime'] for d in data]))
        result_dict["time"] = torch.tensor(end - start)
        #print(genotype_dict)
        return result_dict

class CohortSampler(Sampler):

    def __init__(self, dataset, n_samples=None, phenotype_index='phenotype', z_weight=1):
        #super(DrugResponseSampler, self).__init__()
        self.indices = dataset.index
        self.num_samples = dataset.shape[0]

        phenotype_values = dataset[phenotype_index]
        #phenotype_mean = np.mean(phenotype_values)
        #phenotype_std = np.std(phenotype_values)
        #weights = np.array(z_weights*np.abs((phenotype_values-phenotype_mean)/np.std(phenotype_std)), dtype=np.int)
        self.weights = np.abs(zscore(phenotype_values)*z_weight)
        #self.dataset = result_df.reset_index()[["cellline", "drug", "response", "source", "zscore"]]
        self.weights = torch.tensor(self.weights, dtype=torch.double)

    def __iter__(self):
        count = 0
        index = [i for i in torch.multinomial(self.weights*10, self.num_samples, replacement=True)]
        while count < self.num_samples:
            #print(index[count], type(index[count]))
            #result = index[count].item()
            #print(result, type(result))
            yield index[count].item()
            count += 1

    def __len__(self):
        return self.num_samples



class DistributedCohortSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed = 0, phenotype_col='phenotype', z_weight=1, sex_col=2):
        #super(DrugResponseSampler, self).__init__()
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=False)
        self.indices = dataset.index
        self.num_samples = int(dataset.shape[0]/num_replicas)
        dataset_sex_0 = dataset.loc[dataset[sex_col]==0]
        phenotype_values = dataset_sex_0[phenotype_col].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        dataset_sex_0['phenotype'] = skewnorm(a, loc, scale*z_weight).pdf(phenotype_values)

        dataset_sex_1 = dataset.loc[dataset[sex_col]==1]
        phenotype_values = dataset_sex_1[phenotype_col].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        dataset_sex_1['phenotype'] = skewnorm(a, loc, scale*z_weight).pdf(phenotype_values)

        dataset_merged = pd.concat([dataset_sex_0, dataset_sex_1]).sort_index()
        #weights = np.array(z_weights*np.abs((phenotype_values-phenotype_mean)/np.std(phenotype_std)), dtype=np.int)
        #self.weights = skewnorm(a, loc, scale*z_weight).pdf(phenotype_values)
        #self.dataset = result_df.reset_index()[["cellline", "drug", "response", "source", "zscore"]]
        self.weights = torch.tensor(dataset_merged['phenotype'].values, dtype=torch.double)
    def __iter__(self):
        count = 0
        index = [i for i in torch.multinomial(self.weights*10, self.num_samples, replacement=True)]
        while count < self.num_samples:
            #print(index[count], type(index[count]))
            #result = index[count].item()
            #print(result, type(result))
            yield index[count].item()
            count += 1

    def __len__(self):
        return self.num_samples
