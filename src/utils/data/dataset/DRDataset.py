import pandas as pd
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler
from scipy.stats import zscore, skewnorm
import torch
from src.utils.tree import MutTreeParser
from random import shuffle
from torch.nn.utils.rnn import pad_sequence



def skew_normal_mode(data):
    a, loc, scale = skewnorm.fit(data)
    m, v, s, k = skewnorm.stats(a, loc=loc, scale=scale, moments='mvsk')
    delta = a / np.sqrt(1+a**2)
    u_z = np.sqrt(2/np.pi)*delta
    sigma_z = np.sqrt(1-u_z**2)
    m_z = u_z - s*sigma_z/2 + np.sign(a)*np.exp(-2*np.pi/np.abs(a))/2
    mode = loc + scale*m_z
    return mode, v, s, k

class DrugResponseDataset(Dataset):

    def __init__(self, drug_response, cell2ind, cell2genotypes, compound_encoder, tree_parser:MutTreeParser, with_indices=False):

        self.compound_encoder = compound_encoder
        self.drug_response_df = drug_response #this is dataframe because dataset is split to train and val
        self.compound_grouped = self.drug_response_df.groupby(1)
        self.drug_dict = {drug:self.compound_encoder.encode(drug) for drug in self.drug_response_df[1].unique()}
        self.with_indices = with_indices

        #self.drug_response_mean_dict = {drug: skew_normal_mode(self.compound_grouped.get_group(drug)[2])[0] for drug in self.drug_response_df[1].unique()}
        if self.drug_response_df.shape[1]>2:
            self.drug_response_mean_dict = {drug: self.compound_grouped.get_group(drug)[2].mean() for drug in
                                        self.drug_response_df[1].unique()}
        # the other will be read from csv
        self.cell2ind_df = pd.read_csv(cell2ind, sep='\t', header=None)
        self.cell2ind = {j:i for i, j in self.cell2ind_df.itertuples(0, None)}

        self.cell2genotype_dict = {genotype: pd.read_csv(mut_data, header=None).astype('int32') for genotype, mut_data in cell2genotypes.items()}
        self.tree_parser = tree_parser

        '''
        self.mut_sum_dict = {cell: sum([self.tree_parser.get_system2genotype_mask(torch.tensor(mut_value.iloc[ind].values, dtype=torch.float32))
                         for mut_type, mut_value in self.cell2genotype_dict.items()]) for cell, ind in self.cell2ind.items()}
        self.cell_tree_mask_dict = {cell:[[self.tree_parser.mask_subtree_mask_by_mut(tree_mask, self.mut_sum_dict[cell]) for tree_mask in  sub_tree] for sub_tree in self.nested_subtrees]
                                    for cell, ind in self.cell2ind.items()}
        '''

    def __len__(self):
        return self.drug_response_df.shape[0]

    def __getitem__(self, index):
        if self.drug_response_df.shape[1]==4:
            cell, drug, response, source = self.drug_response_df.iloc[index].values
        elif self.drug_response_df.shape[1]==3:
            cell, drug, response = self.drug_response_df.iloc[index].values
        elif self.drug_response_df.shape[1]==2:
            cell, drug = self.drug_response_df.iloc[index].values

        cell_ind = self.cell2ind[cell]

        if self.with_indices:
            cell_mut_dict = {mut_type:self.tree_parser.get_mut2sys(np.where(mut_value.iloc[cell_ind].values == 1.0)[0],
                                                                   type_indices={1.0: np.where(mut_value.iloc[cell_ind].values == 1.0)[0]})
                             for mut_type, mut_value in self.cell2genotype_dict.items()}
        else:
            cell_mut_dict = {mut_type: self.tree_parser.get_system2genotype_mask(
                torch.tensor(mut_value.iloc[cell_ind].values, dtype=torch.float32))
                             for mut_type, mut_value in self.cell2genotype_dict.items()}
        #cell_tree_mask = self.cell_tree_mask_dict[cell]

        result_dict = dict()
        result_dict['genotype'] = cell_mut_dict
        result_dict['drug'] = self.drug_dict[drug]
        if self.drug_response_df.shape[1]>2:
            #result_dict['nested_system_mask'] = cell_tree_mask
            drug_mean = self.drug_response_mean_dict[drug]
            drug_residual = response - drug_mean
            result_dict["response_mean"] = drug_mean
            result_dict['response_residual'] = drug_residual
        return result_dict

    def get_normal_cellline(self, drug):
        cell_mut_dict = {}
        cell_mut_dict['mutation'] = self.tree_parser.get_mut2sys([], type_indices={1.0: []})
        cell_mut_dict['cna'] = self.tree_parser.get_mut2sys([], type_indices={1.0: []})
        cell_mut_dict['cnd'] = self.tree_parser.get_mut2sys([], type_indices={1.0: []})

        result_dict = dict()
        result_dict['genotype'] = cell_mut_dict
        result_dict['drug'] = self.drug_dict[drug]
        return result_dict

    def get_crispr_celline(self, drug, i):
        result_dict = self.get_normal_cellline(drug)
        result_dict['genotype']['cnd'] = self.tree_parser.get_mut2sys([i], type_indices={1.0: [i]})
        return result_dict



class DrugResponseCollator(object):

    def __init__(self, tree_parser:MutTreeParser, genotypes, compound_encoder, with_indices=False):
        self.tree_parser = tree_parser
        self.genotypes = genotypes
        self.compound_encoder = compound_encoder
        self.compound_type = self.compound_encoder.feature
        self.with_indices = with_indices

    def __call__(self, data):
        result_dict = dict()
        mutation_dict = dict()
        #result_nested_mask = list()
        for genotype in self.genotypes:
            if self.with_indices:
                embedding_dict = {}
                embedding_dict['gene'] = pad_sequence([dr['genotype'][genotype]['gene'] for dr in data], padding_value=self.tree_parser.n_genes, batch_first=True).to(torch.long)
                embedding_dict['sys'] = pad_sequence([dr['genotype'][genotype]['sys'] for dr in data], padding_value=self.tree_parser.n_systems, batch_first=True).to(torch.long)
                gene_max_len = embedding_dict['gene'].size(1)
                sys_max_len = embedding_dict['sys'].size(1)
                embedding_dict['mask'] = torch.stack([dr['genotype'][genotype]['mask'] for dr in data])[:, :sys_max_len, :gene_max_len]

                mutation_dict[genotype] = embedding_dict
            else:
                mutation_dict[genotype] = torch.stack([dr['genotype'][genotype] for dr in data])
        '''
        for i in  range(len(data[0]["nested_system_mask"])):
            result_list = []
            for j in range(len(data[0]["nested_system_mask"][i])):
                result_list.append(torch.stack([dr["nested_system_mask"][i][j] for dr in data]))
            result_nested_mask.append(result_list)
        '''

        result_dict['genotype'] = mutation_dict
        result_dict['drug'] = self.compound_encoder.collate([dr['drug'] for dr in data])
        #result_dict['nested_system_mask'] = result_nested_mask
        if 'response_mean' in data[0].keys():
            result_dict['response_mean'] = torch.tensor([dr['response_mean'] for dr in data])
            result_dict['response_residual'] = torch.tensor([dr['response_residual'] for dr in data])
        return result_dict


class DrugResponseSampler(Sampler):

    def __init__(self, dataset, drug_mean_dict, n_samples=None, group_index='drug', response_index='response', z_weights=1):
        #super(DrugResponseSampler, self).__init__()
        self.indices = dataset.index

        if n_samples is not None:
            self.num_samples = n_samples
        else:
            self.num_samples = dataset.shape[0]
        drug_grouped = dataset.groupby(by=group_index)
        self.drug_mean_dict = drug_mean_dict
        result_dfs = []

        for drug in drug_grouped.groups.keys():
            drug_df = drug_grouped.get_group(drug)
            response_value = drug_df[response_index]
            drug_mean = drug_mean_dict[drug]
            weights = np.array(z_weights*np.abs((response_value-drug_mean)/np.std(response_value)), dtype=np.int)
            drug_df["zscore"] = weights#np.abs(zscore(response_value)*z_weights)
            result_dfs.append(drug_df)
        self.dataset = pd.concat(result_dfs).sort_index()
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


class DrugBatchSampler(BatchSampler):

    def __init__(self, dataset, drug_mean_dict, batch_size=16, group_index='drug', response_index='response', z_weights=1):
        #super(DrugResponseSampler, self).__init__()
        self.indices = dataset.index
        drug_grouped = dataset.groupby(by=group_index)
        self.drug_mean_dict = drug_mean_dict
        self.result_dfs = []
        for drug in drug_grouped.groups.keys():
            drug_df = drug_grouped.get_group(drug)
            response_value = drug_df[response_index]
            mean = drug_mean_dict[drug]
            weights = np.array(z_weights*np.abs((response_value-mean)/np.std(response_value)))
            drug_df["zscore"] = weights#np.abs(zscore(response_value)*z_weights)
            self.result_dfs.append(drug_df)
        self.n_drugs = len(drug_grouped.groups)
        self.batch_size = batch_size

    def __iter__(self):
        count = 0
        shuffle(self.result_dfs)
        while count < self.n_drugs:
            drug_df = self.result_dfs[count]
            weights = torch.tensor(drug_df["zscore"].tolist(), dtype=torch.double)
            index = [i for i in torch.multinomial(weights, self.batch_size, replacement=True)]
            yield drug_df.iloc[index].index.tolist()
            count += 1

    def __len__(self):
        return self.n_drugs


class CellLineBatchSampler(BatchSampler):

    def __init__(self, dataset, drug_mean_dict, batch_size=16, group_index='cellline', drug_index='drug', response_index='response', z_weights=1):
        #super(DrugResponseSampler, self).__init__()
        self.indices = dataset.index

        self.batch_size = batch_size
        self.drug_mean_dict = drug_mean_dict
        '''
        self.result_dfs = []
        for celline in self.cellline_grouped.groups.keys():
            celline_df = self.cellline_grouped.get_group(celline)
            response_value = celline_df[response_index] - celline_df[drug_index].map(lambda a: self.drug_mean_dict[a])
            mean = np.mean(response_value)
            weights = np.array(z_weights*np.abs((response_value-mean)/np.std(response_value)))
            celline_df["zscore"] = weights#np.abs(zscore(response_value)*z_weights)
            self.result_dfs.append(celline_df)
        '''
        drug_dfs = []
        drug_grouped = dataset.groupby(by=drug_index)
        for drug in drug_grouped.groups.keys():
            drug_df = drug_grouped.get_group(drug)
            response_value = drug_df[response_index]
            mean = drug_mean_dict[drug]
            weights = np.array(z_weights*np.abs((response_value-mean)/np.std(response_value)))
            drug_df["zscore"] = weights#np.abs(zscore(response_value)*z_weights)
            drug_dfs.append(drug_df)
        self.df = pd.concat(drug_dfs)
        self.cellline_grouped = self.df.groupby(by=group_index)
        self.celllines = list(self.cellline_grouped.groups.keys())
        self.n_celllines = len(self.celllines)
        self.result_dfs = []
        for cellline in self.cellline_grouped.groups.keys():
            cellline_df = self.cellline_grouped.get_group(cellline)
            self.result_dfs.append(cellline_df)

    def __iter__(self):
        count = 0
        shuffle(self.result_dfs)
        while count < self.n_celllines:
            celline_df = self.result_dfs[count]
            weights = torch.tensor(celline_df["zscore"].tolist(), dtype=torch.double)
            index = [i for i in torch.multinomial(weights, self.batch_size, replacement=True)]
            yield celline_df.iloc[index].index.tolist()
            count += 1

    def __len__(self):
        return self.n_celllines
