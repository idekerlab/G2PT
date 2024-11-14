import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from scipy.stats import zscore, skewnorm
import torch
from src.utils.tree import TreeParser, SNPTreeParser
from torch.nn.utils.rnn import pad_sequence
import time
from torch.utils.data.distributed import DistributedSampler
from sgkit.io import plink
import numpy as np


class SNP2PDataset(Dataset):

    def __init__(self, genotype_phenotype, snp_data, tree_parser:SNPTreeParser, age_mean=None, age_std=None, n_cov=4):
        self.g2p_df = genotype_phenotype
        self.tree_parser = TreeParser
        self.snp_df = snp_data
        self.tree_parser = tree_parser
        self.n_cov = n_cov
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

        result_dict = dict()
        snp_type_dict = dict()

        snp_type_dict['homozygous_a1'] = self.tree_parser.get_snp2gene(homozygous_a1, {1.0: homozygous_a1})
        snp_type_dict['heterozygous'] = self.tree_parser.get_snp2gene(heterozygous, {1.0: heterozygous})
        sample2snp_dict['embedding'] = snp_type_dict

        heterozygous_gene_indices = torch.unique(snp_type_dict['heterozygous']['gene']).tolist()
        homozygous_a1_gene_indices = torch.unique(snp_type_dict['homozygous_a1']['gene']).tolist()
        gene2sys_mask_for_gene = torch.zeros((self.tree_parser.n_systems, self.tree_parser.n_genes), dtype=torch.bool)
        gene2sys_mask_for_gene[:, homozygous_a1_gene_indices] = 1
        gene2sys_mask_for_gene[:, heterozygous_gene_indices] = 1
        result_dict["gene2sys_mask"] = torch.tensor(self.tree_parser.gene2sys_mask, dtype=torch.bool) & gene2sys_mask_for_gene
        result_dict['phenotype'] = phenotype
        sex_age_tensor = [0]*self.n_cov
        #sex_age_tensor = [0, 0, 0, 0]
        if int(sex)==-9:
            pass
        else:
            sex_age_tensor[int(sex)] = 1
        if self.n_cov>=4:
            sex_age_tensor[2] = (age - self.age_mean)/self.age_std
            sex_age_tensor[3] = (age_sq - self.age_mean**2)/(self.age_std**2)
        sex_age_tensor = torch.tensor(sex_age_tensor, dtype=torch.float32)
        covariates = sex_age_tensor#torch.cat([sex_age_tensor, torch.tensor(covariates, dtype=torch.float32)])
        result_dict['genotype'] = sample2snp_dict
        end = time.time()
        result_dict["datatime"] = torch.tensor(end-start)
        result_dict["covariates"] = covariates

        return result_dict

class PLINKDataset(Dataset):

    def __init__(self, tree_parser, bfile, cov, age_mean=None, age_std=None):
        self.tree_parser = tree_parser
        plink_data = plink.read_plink(path=bfile)
        self.genotype = pd.DataFrame(plink_data.call_genotype.as_numpy().sum(axis=-1).T)
        self.genotype.index = plink_data.sample_id.values
        self.genotype.columns = plink_data.variant_id.values
        snp_sorted = [snp for snp, i in sorted(list(self.tree_parser.snp2ind.items()), key=lambda a: a[1])]
        self.genotype = self.genotype[snp_sorted]
        self.cov_df = pd.read_csv(cov, sep='\t')
        self.cov_df['FID'] = self.cov_df['FID'].astype(int)
        self.cov_df['IID'] = self.cov_df['IID'].astype(int)
        self.cov_ids = [cov for cov in self.cov_df.columns[2:] if cov!='PHENOTYPE']
        self.n_cov = len(self.cov_ids) + 1 ## +1 for sex cov
        self.has_phenotype = False
        if 'PHENOTYPE' in self.cov_df.columns:
            self.has_phenotype = True

        if 'AGE' in self.cov_ids:
            self.n_cov += 1 # +1 for age^2
            if age_mean is None:
                self.age_mean = self.cov_df['AGE'].mean()
            else:
                self.age_mean = age_mean
            if age_std is None:
                self.age_std = self.cov_df['AGE'].std()
            else:
                self.age_std = age_std
        print("Model will use %d Covariates"%self.n_cov)

    def __len__(self):
        return self.cov_df.shape[0]

    def __getitem__(self, index):
        start = time.time()
        covariates = self.cov_df.iloc[index]
        iid = str(int(covariates['IID']))
        sample2snp_dict = {}
        homozygous_a1 = np.where(self.genotype.loc[iid]==2)[0]
        heterozygous = np.where(self.genotype.loc[iid]==1)[0]
        result_dict = dict()
        snp_type_dict = dict()
        snp_type_dict['homozygous_a1'] = self.tree_parser.get_snp2gene(homozygous_a1, {1.0: homozygous_a1})
        snp_type_dict['heterozygous'] = self.tree_parser.get_snp2gene(heterozygous, {1.0: heterozygous})
        sample2snp_dict['embedding'] = snp_type_dict
        heterozygous_gene_indices = torch.unique(snp_type_dict['heterozygous']['gene']).tolist()
        homozygous_a1_gene_indices = torch.unique(snp_type_dict['homozygous_a1']['gene']).tolist()
        gene2sys_mask_for_gene = torch.zeros((self.tree_parser.n_systems, self.tree_parser.n_genes), dtype=torch.bool)
        gene2sys_mask_for_gene[:, homozygous_a1_gene_indices] = 1
        gene2sys_mask_for_gene[:, heterozygous_gene_indices] = 1
        result_dict["gene2sys_mask"] = torch.tensor(self.tree_parser.gene2sys_mask, dtype=torch.bool) & gene2sys_mask_for_gene
        if self.has_phenotype:
            result_dict['phenotype'] = covariates['PHENOTYPE']
        covariates_tensor = [0]*self.n_cov
        sex = covariates['SEX']
        i_cov = 0
        if int(sex)==-9:
            pass
        else:
            covariates_tensor[int(sex)] = 1
        i_cov += 2
        if 'AGE' in self.cov_ids:
            age = covariates['AGE']
            age_sq = age ** 2
            covariates_tensor[2] = (age - self.age_mean)/self.age_std
            covariates_tensor[3] = (age_sq - self.age_mean**2)/(self.age_std**2)
            i_cov += 2
        for cov_id in self.cov_ids:
            if (cov_id=='SEX') or (cov_id=='AGE'):
                continue
            covariates_tensor[i_cov] = covariates[cov_id]
            i_cov += 1
        covariates_tensor = torch.tensor(covariates_tensor, dtype=torch.float32)
        covariates_tensor = covariates_tensor#torch.cat([sex_age_tensor, torch.tensor(covariates, dtype=torch.float32)])
        result_dict['genotype'] = sample2snp_dict
        end = time.time()
        result_dict["datatime"] = torch.tensor(end-start)
        result_dict["covariates"] = covariates_tensor
        return result_dict

class SNP2PCollator(object):

    def __init__(self, tree_parser:SNPTreeParser):
        self.tree_parser = tree_parser
        self.padding_index = {"snp":self.tree_parser.n_snps, "gene":self.tree_parser.n_genes}

    def __call__(self, data):
        start = time.time()
        result_dict = dict()
        genotype_dict = dict()

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

        genotype_dict['embedding'] = snp_type_dict
        result_dict['gene2sys_mask'] = torch.stack([d['gene2sys_mask'] for d in data])
        result_dict['genotype'] = genotype_dict
        result_dict['covariates'] = torch.stack([d['covariates'] for d in data])
        #print(result_dict['covariates'])
        result_dict['phenotype'] = torch.tensor([d['phenotype'] for d in data], dtype=torch.float32)
        end = time.time()
        result_dict['datatime'] = torch.mean(torch.stack([d['datatime'] for d in data]))
        result_dict["time"] = torch.tensor(end - start)
        #print(genotype_dict)
        return result_dict

class CohortSampler(Sampler):

    def __init__(self, dataset, n_samples=None, phenotype_col='phenotype', z_weight=1, sex_col=2):
        #super(DrugResponseSampler, self).__init__()
        self.indices = dataset.index
        self.num_samples = dataset.shape[0]
        
        phenotype_values = dataset[phenotype_col]
        dataset_sex_0 = dataset.loc[dataset[sex_col]==0]
        phenotype_values = dataset_sex_0[phenotype_col].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        dataset_sex_0['phenotype'] = skewnorm(a, loc, scale*z_weight).pdf(phenotype_values)

        dataset_sex_1 = dataset.loc[dataset[sex_col]==1]
        phenotype_values = dataset_sex_1[phenotype_col].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        dataset_sex_1['phenotype'] = skewnorm(a, loc, scale*z_weight).pdf(phenotype_values)

        dataset_merged = pd.concat([dataset_sex_0, dataset_sex_1]).sort_index()
        #phenotype_mean = np.mean(phenotype_values)
        #phenotype_std = np.std(phenotype_values)
        #weights = np.array(z_weights*np.abs((phenotype_values-phenotype_mean)/np.std(phenotype_std)), dtype=np.int)
        #self.weights = np.abs(zscore(phenotype_values)*z_weight)
        #self.dataset = result_df.reset_index()[["cellline", "drug", "response", "source", "zscore"]]
        #self.weights = torch.tensor(self.weights, dtype=torch.double)
        self.weights = torch.tensor(dataset_merged['phenotype'].values, dtype=torch.double)

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

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed = 0, phenotype_col='phenotype', z_weight=1, sex_col=2):
        #super(DrugResponseSampler, self).__init__()
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=False)
        self.indices = dataset.index
        self.num_samples = int(dataset.shape[0]/num_replicas)
        phenotype_values = dataset[phenotype_col].values
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
        #self.weights = torch.tensor(skewnorm(a, loc, scale*z_weight).pdf(phenotype_values), dtype=torch.double)
        #self.weights = np.abs(zscore(phenotype_values)*z_weight)
        #self.weights = torch.tensor(self.weights, dtype=torch.double)
        #self.dataset = result_df.reset_index()[["cellline", "drug", "response", "source", "zscore"]]
        self.weights = torch.tensor(dataset_merged['phenotype'].values, dtype=torch.double)
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
