import random

import numpy as np
import pandas as pd
import torch
from scipy.stats import skewnorm, zscore
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import BatchSampler, RandomSampler, Sampler



class CohortSampler(Sampler):
    def __init__(self, dataset, n_samples=None, phenotype_col="phenotype", z_weight=1, sex_col=2):
        # super(DrugResponseSampler, self).__init__()
        self.dataset = dataset.cov_df
        self.indices = self.dataset.index
        self.num_samples = self.dataset.shape[0]

        phenotype_values = self.dataset[phenotype_col]
        dataset_sex_0 = self.dataset.loc[self.dataset[sex_col] == 0]
        phenotype_values = dataset_sex_0[phenotype_col].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        dataset_sex_0["phenotype"] = skewnorm(a, loc, scale * z_weight).pdf(phenotype_values)

        dataset_sex_1 = self.dataset.loc[self.dataset[sex_col] == 1]
        phenotype_values = dataset_sex_1[phenotype_col].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        dataset_sex_1["phenotype"] = skewnorm(a, loc, scale * z_weight).pdf(phenotype_values)

        dataset_merged = pd.concat([dataset_sex_0, dataset_sex_1]).sort_index()
        # phenotype_mean = np.mean(phenotype_values)
        # phenotype_std = np.std(phenotype_values)
        # weights = np.array(z_weights*np.abs((phenotype_values-phenotype_mean)/np.std(phenotype_std)), dtype=np.int)
        # self.weights = np.abs(zscore(phenotype_values)*z_weight)
        # self.dataset = result_df.reset_index()[["cellline", "drug", "response", "source", "zscore"]]
        # self.weights = torch.tensor(self.weights, dtype=torch.double)
        self.weights = torch.tensor(dataset_merged["phenotype"].values, dtype=torch.double)

    def __iter__(self):
        count = 0
        index = [i for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]
        while count < self.num_samples:
            # print(index[count], type(index[count]))
            # result = index[count].item()
            # print(result, type(result))
            yield index[count].item()
            count += 1

    def __len__(self):
        return self.num_samples


class BinaryCohortSampler(Sampler):
    def __init__(self, dataset, phenotype_col="PHENOTYPE"):
        # super(DrugResponseSampler, self).__init__()
        self.dataset = dataset.cov_df
        self.indices = self.dataset.index
        self.num_samples = self.dataset.shape[0]
        class_count = np.bincount(self.dataset[phenotype_col].values)
        class_weight = 1.0 / class_count
        sample_weight = class_weight[self.dataset[phenotype_col].values]
        self.weights = torch.tensor(sample_weight, dtype=torch.double)

    def __iter__(self):
        count = 0
        index = [i for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]
        while count < self.num_samples:
            # print(index[count], type(index[count]))
            # result = index[count].item()
            # print(result, type(result))
            yield index[count].item()
            count += 1

    def __len__(self):
        return self.num_samples


class DistributedCohortSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, phenotype_col="PHENOTYPE", z_weight=1, sex_col=2):
        # super(DrugResponseSampler, self).__init__()
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=False)
        self.dataset = dataset.cov_df
        self.indices = self.dataset.index
        self.num_samples = int(self.dataset.shape[0] / num_replicas)
        phenotype_values = self.dataset[phenotype_col].values
        dataset_sex_0 = self.dataset.loc[self.dataset[sex_col] == 0]
        phenotype_values = dataset_sex_0[phenotype_col].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        dataset_sex_0["phenotype"] = skewnorm(a, loc, scale * z_weight).pdf(phenotype_values)

        dataset_sex_1 = self.dataset.loc[dataset[sex_col] == 1]
        phenotype_values = dataset_sex_1[phenotype_col].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        dataset_sex_1["phenotype"] = skewnorm(a, loc, scale * z_weight).pdf(phenotype_values)

        dataset_merged = pd.concat([dataset_sex_0, dataset_sex_1]).sort_index()
        # weights = np.array(z_weights*np.abs((phenotype_values-phenotype_mean)/np.std(phenotype_std)), dtype=np.int)
        # self.weights = torch.tensor(skewnorm(a, loc, scale*z_weight).pdf(phenotype_values), dtype=torch.double)
        # self.weights = np.abs(zscore(phenotype_values)*z_weight)
        # self.weights = torch.tensor(self.weights, dtype=torch.double)
        # self.dataset = result_df.reset_index()[["cellline", "drug", "response", "source", "zscore"]]
        self.weights = torch.tensor(dataset_merged["phenotype"].values, dtype=torch.double)

    def __iter__(self):
        count = 0
        index = [i for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]
        while count < self.num_samples:
            # print(index[count], type(index[count]))
            # result = index[count].item()
            # print(result, type(result))
            yield index[count].item()
            count += 1

    def __len__(self):
        return self.num_samples


class DistributedBinaryCohortSampler(DistributedSampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0, phenotype_col="PHENOTYPE", z_weight=1, sex_col=2):
        # super(DrugResponseSampler, self).__init__()
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=False)
        self.dataset = dataset.cov_df
        self.indices = self.dataset.index
        self.num_samples = int(self.dataset.shape[0] / num_replicas)
        # self.num_case = self.dataset[phenotype_col].sum()
        # self.num_control = self.num_samples - self.num_case
        class_count = np.bincount(self.dataset[phenotype_col].values)
        class_weight = 1.0 / class_count
        sample_weight = class_weight[self.dataset[phenotype_col].values]
        self.weights = torch.tensor(sample_weight, dtype=torch.double)

    def __iter__(self):
        count = 0
        index = [i for i in torch.multinomial(self.weights, self.num_samples, replacement=True)]
        while count < self.num_samples:
            # print(index[count], type(index[count]))
            # result = index[count].item()
            # print(result, type(result))
            yield index[count].item()
            count += 1

    def __len__(self):
        return self.num_samples
