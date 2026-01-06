import random

import pandas as pd
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler, RandomSampler
from scipy.stats import zscore, skewnorm
import torch
import torch.nn.functional as F
from src.utils.tree import TreeParser, SNPTreeParser
import time
from torch.utils.data.distributed import DistributedSampler
from sgkit.io import plink
import numpy as np
import os
from collections import OrderedDict
from transformers import DataCollatorForLanguageModeling, PreTrainedTokenizer
from typing import Optional, Tuple, Dict
from .. import pad_indices

class GenotypeDataset(Dataset):
    def __init__(self, tree_parser : SNPTreeParser, cov, pheno=None, cov_mean_dict=None, cov_std_dict=None,
                 cov_ids=(), pheno_ids=(), bt=(), qt=(), dynamic_phenotype_sampling=False,):

        self.tree_parser = tree_parser
        self.n_snp2pad = int(np.ceil((self.tree_parser.n_snps+1) /8)) * 8 - self.tree_parser.n_snps
        self.n_gene2pad = int(np.ceil((self.tree_parser.n_genes+1) / 8)) * 8 - self.tree_parser.n_genes
        self.n_sys2pad = int(np.ceil((self.tree_parser.n_systems+1) / 8)) * 8 - self.tree_parser.n_systems

        self.gene_pad   = self.tree_parser.n_genes
        self.sys_pad    = self.tree_parser.n_systems

        gene_idx = torch.arange(self.tree_parser.n_genes+1, dtype=torch.long)
        gene_idx = torch.cat((gene_idx,
                              torch.full((self.n_gene2pad,), self.gene_pad, dtype=torch.long)))
        self.gene_idx = gene_idx  # (L_gene,)
        sys_idx = torch.arange(self.tree_parser.n_systems+1, dtype=torch.long)
        sys_idx = torch.cat((sys_idx,
                             torch.full((self.n_sys2pad,), self.sys_pad, dtype=torch.long)))
        self.sys_idx = sys_idx  # (L_sys,)
        self.dynamic_phenotype_sampling = dynamic_phenotype_sampling

        self.snp_range = list(range(self.tree_parser.n_snps + self.n_snp2pad))
        self.block_range = list(range(self.tree_parser.n_snps + self.n_snp2pad))
        self.gene_range = list(range(self.tree_parser.n_genes + self.n_gene2pad))
        self.sys_range = list(range(self.tree_parser.n_systems + self.n_sys2pad))

        print('Processing Covariates...')
        if cov is not None:
            print("Loading Covariate file at %s"%cov)
            self.cov_df = pd.read_csv(cov, sep='\t')
            self.cov_df['IID'] = self.cov_df['IID'].astype(str)
            self.cov_df['FID'] = self.cov_df['FID'].astype(str)
            #self.cov_df = self.cov_df.set_index('IID').loc[plink_data.sample_id.values].reset_index()
        else:
            self.cov_df = None # will be initialized in child class

        if len(cov_ids) != 0:
            self.cov_ids = cov_ids
        else:
            self.cov_ids = [cov for cov in self.cov_df.columns[2:] if cov != 'PHENOTYPE']
        self.n_cov = len(self.cov_ids) + 1 ## +1 for sex cov

        if cov_mean_dict is None:
            self.cov_mean_dict = dict()
            self.cov_std_dict = dict()
            for cov in self.cov_ids:
                if not self.is_binary(self.cov_df[cov]):
                    self.cov_mean_dict[cov] = self.cov_df[cov].mean()
                    self.cov_std_dict[cov] = self.cov_df[cov].std()
        else:
            self.cov_mean_dict = cov_mean_dict
            self.cov_std_dict = cov_std_dict

        # Processing Phenotypes
        print('Processing Phenotypes')
        if pheno is not None:
            print("Loading Phenotype file at %s" % pheno)
            self.pheno_df = pd.read_csv(pheno, sep='\t')
            if len(pheno_ids)!=0:
                self.pheno_ids = pheno_ids
            else:
                self.pheno_ids = [pheno for pheno in self.pheno_df.columns[2:]]
            self.pheno_df['IID'] = self.pheno_df['IID'].map(str)
            self.pheno_df['FID'] = self.pheno_df['FID'].map(str)
            self.pheno_df = self.pheno_df.fillna(-9)

        else:
            self.pheno_df = None # will be initialized in child class = self.cov_df[['FID', 'IID', 'PHENOTYPE']]
            self.pheno_df = self.pheno_df.fillna(-9)
        self.pheno2ind = {pheno:i for i, pheno in enumerate(self.pheno_ids)}
        self.ind2pheno = {i:pheno for i, pheno in enumerate(self.pheno_ids)}
        self.pheno2type = {}
        if (len(bt) == 0) & (len(qt) == 0):
            self.qt = self.pheno_ids # all phenotypes will be regression tasks
            self.bt = []
            self.qt_inds = [self.pheno2ind[pheno] for pheno in self.qt]
            self.bt_inds = []
            for pheno in self.qt:
                self.pheno2type[pheno] = 'qt'
        else:
            self.qt = qt
            self.bt = bt
            self.qt_inds = [self.pheno2ind[pheno] for pheno in qt]
            self.bt_inds = [self.pheno2ind[pheno] for pheno in bt]
            for pheno in qt:
                self.pheno2type[pheno] = 'qt'
            for pheno in bt:
                self.pheno2type[pheno] = 'bt'

        self.n_pheno = len(self.pheno_ids)
        self.pheno_range = list(range(self.n_pheno))
        self.has_phenotype = False
        if ('PHENOTYPE' in self.cov_df.columns) or (pheno is not None):
            self.has_phenotype = True
        print("Phenotypes: ", self.pheno_ids)
        print("Phenotype index: ", self.pheno2ind)
        print("Phenotype data type: ", self.pheno2type)

    def is_binary(self, array):
        return np.isin(array, [0, 1]).all()

    def __len__(self):
        return self.cov_df.shape[0]

    def __getitem__(self, index):
        start = time.time()
        covariates = self.cov_df.iloc[index]
        phenotypes = self.pheno_df.iloc[index]
        covariates_tensor = [0] * self.n_cov
        result_dict = {}
        sex = covariates['SEX']
        i_cov = 0
        if (int(sex) == -1) or (int(sex) == -9):
            pass
        else:
            covariates_tensor[int(sex)] = 1
        i_cov += 2
        for cov_id in self.cov_ids:
            if cov_id == 'SEX':
                continue
            if cov_id in self.cov_mean_dict.keys():
                cov_value = (covariates[cov_id] - self.cov_mean_dict[cov_id]) / self.cov_std_dict[cov_id]
            else:
                cov_value = covariates[cov_id]
            covariates_tensor[i_cov] = cov_value
            i_cov += 1
        covariates_tensor = torch.tensor(covariates_tensor, dtype=torch.float32)
        covariates_tensor = covariates_tensor  # torch.cat([sex_age_tensor, torch.tensor(covariates, dtype=torch.float32)])
        phenotype_ind_tensor = torch.tensor([i for i in range(self.n_pheno)], dtype=torch.int)
        result_dict['phenotype_indices'] = phenotype_ind_tensor[self.pheno_range]
        if self.has_phenotype:
            phenotype_tensor = torch.tensor([phenotypes[pheno_id] for pheno_id in self.pheno_ids], dtype=torch.float32)
            result_dict['phenotype'] = phenotype_tensor[self.pheno_range]

        end = time.time()
        result_dict["datatime"] = torch.tensor(end - start)
        result_dict["covariates"] = covariates_tensor
        return result_dict

class TSVDataset(GenotypeDataset):
    def __init__(self, tree_parser : SNPTreeParser, genotype_path, cov, pheno=None, cov_mean_dict=None, cov_std_dict=None, flip=False,
                 input_format='indices', cov_ids=(), pheno_ids=(), bt=(), qt=()):
        super().__init__(tree_parser=tree_parser, cov=cov, pheno=pheno, cov_mean_dict=cov_mean_dict, cov_std_dict=cov_std_dict, cov_ids=cov_ids, pheno_ids=pheno_ids, bt=bt, qt=qt)

        print('Loading Genotype data at %s' % genotype_path)
        genotype_df = pd.read_csv(genotype_path, sep='\t', index_col=0)
        genotype_df.columns = genotype_df.columns.astype(int)
        print('loading done')
        self.input_format = input_format
        genotype = genotype_df.values.astype(np.int8)

        snp_sorted = [snp for snp, i in sorted(list(self.tree_parser.snp2ind.items()), key=lambda a: a[1])]

        self.iid2ind = {str(iid): idx for idx, iid in enumerate(genotype_df.index.values)}
        self.ind2iid = {idx: str(iid) for idx, iid in enumerate(genotype_df.index.values)}

        genotype_df = genotype_df[snp_sorted]
        print("genotype data shape: ", genotype_df.shape)
        genotype = genotype_df.values
        if not flip:
            genotype = 2 - genotype
        else:
            print("Swapping Ref and Alt!")
        self.N, self.n_snps = genotype.shape

        n_snps          = tree_parser.n_snps
        self.snp_offset      = n_snps
        self.snp_pad    = n_snps * 3

        alleles = torch.as_tensor(genotype, dtype=torch.long)
        alleles = torch.clip(alleles, min=0, max=2)
        base = torch.arange(self.n_snps, dtype=torch.long)
        snp_idx = alleles * self.snp_offset + base

        pad = torch.full((self.N, self.n_snp2pad), self.snp_pad, dtype=torch.long)
        snp_idx = torch.cat((snp_idx, pad), dim=1)

        self.snp_idx = snp_idx.contiguous()
        block_pad = self.tree_parser.n_blocks
        block_idx = torch.tensor([self.tree_parser.block2ind[self.tree_parser.snp2block[self.tree_parser.ind2snp[i]]] for i in range(self.tree_parser.n_snps)], dtype=torch.long)

        self.block_idx = torch.cat((block_idx, torch.full((self.n_snp2pad+1,), block_pad, dtype=torch.long)))

        self.flip = flip
        print("From TSV %d variants with %d samples are queried" % (genotype.shape[1], genotype.shape[0]))
        del genotype_df
        del genotype

        if self.cov_df is not None:
            self.cov_df = self.cov_df.set_index('IID').loc[self.ind2iid.values()].reset_index()

        if self.pheno_df is not None:
            self.pheno_df = self.pheno_df.set_index('IID').loc[self.ind2iid.values()].reset_index()

        self.subtree = None
        self.subtree_phenotypes = []

    def __getitem__(self, index):
        result_dict = super().__getitem__(index)
        sample2snp_dict = {}
        sample2snp_dict['block_ind'] = self.block_idx[self.block_range]
        sample2snp_dict['snp'] = self.snp_idx[index, self.snp_range]
        sample2snp_dict['gene'] = self.gene_idx[self.gene_range]
        sample2snp_dict['sys'] = self.sys_idx[self.sys_range]
        result_dict['genotype'] = sample2snp_dict
        return result_dict

class PLINKDataset(GenotypeDataset):
    def __init__(self, tree_parser : SNPTreeParser, bfile=None, cov=None, pheno=None, cov_mean_dict=None, cov_std_dict=None, flip=False,
                 input_format='indices', cov_ids=(), pheno_ids=(), bt=(), qt=()):
        """
        tree_parser: SNP tree parser object
        bfile: PLINK bfile prefix
        cov: covariates tab-delimited text file. If None, covariates will be inferred from bfile.fam file
        pheno: phenotype tab-delimited text file. If None, phenotype will be inferred from bfile.fam file
        cov_mean_dict: dictionary of covariates mean, {cov_id: mean_value}. If None, mean of covariates will be calculated from cov file
        cov_std_dict: dictionary of covariates standard deviation, {cov_id: std_value}. If None, standard deviation of covariates will be calculated from cov file
        flip: If True, it will flip reference and altered allele
        input_format: input data format for model, indices or binary
        cov_ids: If you want to use specific covariates, you can put list of covariate ids. If None, dataset will return all covariates from cov file
        """
        super().__init__(tree_parser=tree_parser, cov=cov, pheno=pheno, cov_mean_dict=cov_mean_dict, cov_std_dict=cov_std_dict, cov_ids=cov_ids, pheno_ids=pheno_ids, bt=bt, qt=qt)

        self.bfile = bfile

        print('Loading PLINK data at %s' % bfile)
        # Processing Genotypes
        plink_data = plink.read_plink(path=bfile)
        #print(f'loading done with{len(plink_data.sample_id.values)} individuals and {len(plink_data.variant_id.values)} SNPs')

        snp_ids = plink_data['variant_id'].values
        snp_contig_mapping = {i:int(chromosome) for i, chromosome in enumerate(plink_data['contig_id'].values)}
        snp_chr = plink_data['variant_position'].values
        snp_pos = [snp_contig_mapping[contig] for contig in plink_data['variant_contig'].values]


        snp_id2chr = {snp: chromosome for snp, chromosome in zip(snp_ids, snp_chr)}
        snp_id2pos = {snp: pos for snp, pos in zip(snp_ids, snp_pos)}

        self.tree_parser.set_snp2chr(snp_id2chr)
        self.tree_parser.set_snp2pos(snp_id2pos)

        self.input_format = input_format
        genotype = plink_data.call_genotype.as_numpy().sum(axis=-1).T.astype(np.int8)

        snp_sorted = [snp for snp, i in sorted(list(self.tree_parser.snp2ind.items()), key=lambda a: a[1])]

        genotype_df = pd.DataFrame(genotype, index=plink_data.sample_id.values, columns=plink_data.variant_id.values)
        #genotype_df = genotype_df[snp_sorted]
        missing_snps = [snp for snp in snp_sorted if snp not in genotype_df.columns]
        print("SNPs not in bfile:", missing_snps)
        genotype_df = genotype_df.reindex(columns=snp_sorted, fill_value=0)


        # Processing Covariates
        if self.cov_df is not None:
            intersection_iid = list(set(genotype_df.index.tolist()).intersection(self.cov_df['IID'].tolist()))
            self.cov_df = self.cov_df.set_index('IID').loc[intersection_iid].reset_index()
            genotype_df = genotype_df.loc[intersection_iid]
        else:
            self.cov_df = pd.DataFrame({'FID': plink_data.sample_family_id.as_numpy(),
                                        'IID': plink_data.sample_id.as_numpy(),
                                        'SEX': plink_data.sample_sex.as_numpy(),
                                        'PHENOTYPE': plink_data.sample_phenotype.as_numpy() })
            self.cov_df = self.cov_df[['FID', 'IID', 'SEX', 'PHENOTYPE']]
            self.cov_df = self.cov_df.loc[self.cov_df.PHENOTYPE!=-1]
            self.cov_df['PHENOTYPE'] = self.cov_df['PHENOTYPE'] - 1
            self.cov_df['IID'] = self.cov_df['IID'].astype(str)
            self.cov_df['FID'] = self.cov_df['FID'].astype(str)
            self.cov_df = self.cov_df.set_index('IID').loc[plink_data.sample_id.values].reset_index()

        if self.pheno_df is not None:
            intersection_iid = list(set(genotype_df.index.tolist()).intersection(self.pheno_df['IID'].tolist()))
            self.pheno_df = self.pheno_df.set_index('IID').loc[intersection_iid].reset_index()
            self.cov_df = self.cov_df.set_index('IID').loc[intersection_iid].reset_index()
            genotype_df = genotype_df.loc[intersection_iid]
        else:
            self.pheno_df = self.cov_df[['FID', 'IID', 'PHENOTYPE']]
            self.pheno_df['IID'] = self.pheno_df['IID'].map(str)
            self.pheno_df = self.pheno_df.set_index('IID').loc[plink_data.sample_id.values].reset_index()
            self.cov_df = self.cov_df.set_index('IID').loc[plink_data.sample_id.values].reset_index()
            self.pheno_ids = ['PHENOTYPE']

        self.iid2ind = {str(iid): idx for idx, iid in enumerate(genotype_df.index.values)}
        self.ind2iid = {idx: str(iid) for idx, iid in enumerate(genotype_df.index.values)}

        #print(genotype_df.head())
        #print(self.cov_df.head())
        #print(self.pheno_df.head())

        self.subtree = None
        self.subtree_phenotypes = []


        print("genotype data shape: ", genotype_df.shape)
        genotype = genotype_df.values
        #print(genotype)
        if not flip:
            genotype = 2 - genotype
        else:
            print("Swapping Ref and Alt!")
        self.N, self.n_snps = genotype.shape

        n_snps          = tree_parser.n_snps               # original SNP catalogue size
        self.snp_offset      = n_snps                           # allele * n_snps + j
        self.snp_pad    = n_snps * 3                  # padding value

        # ---------- SNP indices for all individuals ----------
        alleles = torch.as_tensor(genotype, dtype=torch.long)  # (N × 1 200)
        alleles = torch.clip(alleles, min=0, max=2) # need to be fixed?
        base = torch.arange(self.n_snps, dtype=torch.long)  # [0 … 1 199]
        snp_idx = alleles * self.snp_offset + base  # vectorised

        pad = torch.full((self.N, self.n_snp2pad), self.snp_pad, dtype=torch.long)
        snp_idx = torch.cat((snp_idx, pad), dim=1)  # (N × (1 200 + pad))

        self.snp_idx = snp_idx.contiguous()  # final (N × L_snp)
        if hasattr(self.tree_parser, "n_blocks"):
            #self.padding_index["block"] = self.tree_parser.n_blocks
            self.block = True
        else:
            self.block = False
        if self.block:
            block_pad = self.tree_parser.n_blocks
            block_idx = torch.tensor([self.tree_parser.block2ind[self.tree_parser.snp2block[self.tree_parser.ind2snp[i]]] for i in range(self.tree_parser.n_snps)], dtype=torch.long)
            self.block_idx = torch.cat((block_idx, torch.full((self.n_snp2pad+1,), block_pad, dtype=torch.long)))

        self.flip = flip
        print("From PLINK %d variants with %d samples are queried" % (genotype.shape[1], genotype.shape[0]))
        del genotype_df
        del genotype

    def summary(self):
        print('PLINK data from %s'%self.bfile)
        print('Covariates for prediction %s' % ", ".join(self.cov_ids))
        print("Covariates: ", ", ".join(self.cov_ids))
        if self.flip:
            print("Ref. and Alt. are flipped")
        print("%d of participant with %d of variants"%(self.genotype.shape[0], self.genotype.shape[1]))
        print("The number of covariates:", self.n_cov)

    def sample_population(self, n=100):
        sampled_iids = self.cov_df['IID'].sample(n=n, random_state=42).tolist()
        original_indices = [self.iid2ind[iid] for iid in sampled_iids]
        self.snp_idx = self.snp_idx[original_indices]
        self.cov_df = self.cov_df.set_index('IID').loc[sampled_iids].reset_index()
        self.pheno_df = self.pheno_df.set_index('IID').loc[sampled_iids].reset_index()
        self.iid2ind = {str(iid): idx for idx, iid in enumerate(sampled_iids)}
        self.ind2iid = {idx: str(iid) for idx, iid in enumerate(sampled_iids)}
        self.N = len(sampled_iids)
        print(f"Subsampled to {self.N} individuals.")


    def sample_phenotypes(self, n, seed=None):
        """
        Samples phenotypes. If a seed is provided, the sampling will be deterministic.
        """
        # Use a new random object with the given seed to not affect the global random state
        rand = random.Random(seed) if seed is not None else random
        sampled_phenotypes = rand.sample(self.pheno_ids, n)
        sampled_pheno_inds = sorted([self.pheno2ind[pheno] for pheno in sampled_phenotypes])
        sampled_phenotypes = [self.ind2pheno[ind] for ind in sampled_pheno_inds]
        self.select_phenotypes(phenotypes=sampled_phenotypes)

    def select_phenotypes(self, phenotypes):
        print('Select Phenotypes: ', ', '.join(phenotypes))
        pheno_indices = [self.pheno2ind[pheno] for pheno in phenotypes]
        snp_indices, gene_indices, sys_indices = self.collect_indices_from_phenotypes(phenotypes)
        self.set_range_from_indices(snp_indices, gene_indices, sys_indices, pheno_indices)

    def collect_indices_from_phenotypes(self, phenotypes):
        #print("Sample phenotypes: ", sampled_phenotypes)
        sampled_pheno_inds = sorted([self.pheno2ind[pheno] for pheno in phenotypes])
        self.pheno_range = sampled_pheno_inds
        snp_set = set()
        for pheno in phenotypes:
            snp_set = snp_set.union(self.tree_parser.pheno2snp[pheno])
        snp_set = sorted(list(snp_set))
        gene_set = set()
        for snp in snp_set:
            gene_set = gene_set.union(self.tree_parser.snp2gene[snp])
        gene_set = sorted(list(gene_set))
        sys_set = set()
        for gene in gene_set:
            sys_set = sys_set.union(self.tree_parser.gene2sys_full[gene])
        snp_indices = sorted([self.tree_parser.snp2ind[snp] for snp in snp_set])
        gene_indices = sorted([self.tree_parser.gene2ind[gene] for gene in gene_set])
        sys_indices = sorted([self.tree_parser.sys2ind[sys] for sys in sys_set])
        return snp_indices, gene_indices, sys_indices

    def set_range_from_indices(self, snp_indices, gene_indices, sys_indices, pheno_indices):
        self.snp_range = pad_indices(snp_indices, self.snp_offset)
        self.block_range = pad_indices(snp_indices, self.snp_offset)
        self.gene_range = pad_indices(gene_indices, self.gene_pad)
        self.sys_range = pad_indices(sys_indices, self.sys_pad)
        self.pheno_range = pheno_indices


    def __getitem__(self, index):
        result_dict = super().__getitem__(index)
        sample2snp_dict = {}
        if self.block:
            sample2snp_dict['block_ind'] = self.block_idx[self.block_range]
        sample2snp_dict['snp'] = self.snp_idx[index, self.snp_range]
        sample2snp_dict['gene'] = self.gene_idx[self.gene_range]
        sample2snp_dict['sys'] = self.sys_idx[self.sys_range]
        result_dict['genotype'] = sample2snp_dict
        return result_dict



class EmbeddingDataset(PLINKDataset):

    def __init__(self, tree_parser : SNPTreeParser, bfile, embedding, iid2ind, cov=None, pheno=None, cov_mean_dict=None, cov_std_dict=None,
                 cov_ids=(), pheno_ids=(), bt=(), qt=()):
        super().__init__(tree_parser=tree_parser, bfile=bfile, cov=cov, pheno=pheno, cov_mean_dict=cov_mean_dict, cov_std_dict=cov_std_dict,
                            cov_ids=cov_ids, pheno_ids=pheno_ids, bt=bt, qt=qt)

        self.embedding = embedding#{self.sort_key(p): zarr.open(p, mode="r") for p in embedding_zarrs}

        self.snp2block = {}
        block_pad = self.tree_parser.n_blocks
        self.snp_sorted = [snp for snp, i in sorted(list(self.tree_parser.snp2ind.items()), key=lambda a: a[1])]
        self.snp_indices = [self.tree_parser.snp2ind_all[snp] for snp in self.snp_sorted]
        block_idx = torch.tensor([self.tree_parser.block2ind[self.tree_parser.snp2block[self.tree_parser.ind2snp[i]]] for i in range(self.tree_parser.n_snps)], dtype=torch.long)

        self.block_idx = torch.cat((block_idx, torch.full((self.n_snp2pad,), block_pad, dtype=torch.long)))

        self.iid2ind = iid2ind
        self.ind2iid = {ind: iid for iid, ind in iid2ind.items()}
        self.iid = self.cov_df.IID.tolist()


    def __getitem__(self, index):
        result_dict = super().__getitem__(index)
        iid = self.iid[index]
        ind = self.iid2ind[iid]

        sample2snp_dict = {}
        sample2snp_dict['gene'] = self.gene_idx
        sample2snp_dict['sys'] = self.sys_idx
        sample2snp_dict['block'] = self.block_idx

        #embedding = torch.tensor(self.embedding.oindex[ind, :, :], dtype=torch.float32)
        embedding = torch.load(os.path.join(self.embedding, str(iid)+'.tensor'), weights_only=True)
        embedding = embedding[self.snp_indices, :]
        embedding = F.pad(embedding, (0, 0, 0, self.n_snp2pad), value=0)

        sample2snp_dict['snp'] = self.snp_idx[index]
        sample2snp_dict['embedding'] = embedding
        result_dict['genotype'] = sample2snp_dict

        return result_dict

class SNPTokenizer(PreTrainedTokenizer):
    def __init__(self, vocab: Dict[str, int], max_len: int = None):
        # Set up your attributes before calling super().__init__()
        self.__token_ids = vocab
        self.__id_tokens: Dict[int, str] = {value: key for key, value in vocab.items()}
        super().__init__(max_len=max_len)

    def _tokenize(self, text: str, **kwargs):
        return text.split(' ')

    def _convert_token_to_id(self, token: str) -> int:
        return self.__token_ids[token] if token in self.__token_ids else self.unk_token_id

    def _convert_id_to_token(self, index: int) -> str:
        return self.__id_tokens[index] if index in self.__id_tokens else self.unk_token

    def get_vocab(self) -> Dict[str, int]:
        return self.__token_ids.copy()

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if filename_prefix is None:
            filename_prefix = ''
        vocab_path = Path(save_directory, filename_prefix + 'vocab.json')
        json.dump(self.__token_ids, open(vocab_path, 'w'))
        return (str(vocab_path),)

    @property
    def vocab_size(self) -> int:
        return len(self.__token_ids)

class BlockDataset(Dataset):
    def __init__(self, bfile, flip=False):
        self.bfile = bfile

        print('Loading PLINK data at %s' % bfile)
        # Processing Genotypes
        plink_data = plink.read_plink(path=bfile)
        print('loading done')
        genotype = plink_data.call_genotype.as_numpy().sum(axis=-1).T.astype(np.int8)
        self.iid2ind = {str(iid):idx for idx, iid in enumerate(plink_data.sample_id.values)}
        self.ind2iid = {idx:str(iid) for idx, iid in enumerate(plink_data.sample_id.values)}

        self.snp2ind = {str(snp): idx for idx, snp in enumerate(plink_data.variant_id.values)}
        self.ind2snp = {idx: str(snp) for idx, snp in enumerate(plink_data.variant_id.values)}
        genotype_df = pd.DataFrame(genotype, index=plink_data.sample_id.values, columns=plink_data.variant_id.values)

        genotype = genotype_df.values
        if not flip:
            genotype = 2 - genotype
        else:
            print("Swapping Ref and Alt!")
        self.N, self.n_snps = genotype.shape
        self.snp_offset = self.n_snps                           # allele * n_snps + j
        self.snp_pad    = self.n_snps * 3 + 1                  # padding value
        self.snp_mask =  self.n_snps * 3 + 2

        vocab = {}
        for i, snp in enumerate(self.snp2ind.keys()):
            chrom, pos, ref, alt = snp.split(':')
            vocab[':'.join([chrom, pos, ref, ref])] = i
            vocab[':'.join([chrom, pos, ref, alt])] = i + self.n_snps
            vocab[':'.join([chrom, pos, alt, alt])] = i + self.n_snps * 2
        vocab['[MASK]'] = self.n_snps * 3
        vocab['[PAD]'] = self.n_snps * 3 + 1

        # building a tokenizer..
        tokenizer = SNPTokenizer(vocab=vocab, max_len=1024)

        print("N SNPs: ", self.n_snps)
        print("N tokens: ", len(tokenizer))

        tokenizer.mask_token = "[MASK]"
        tokenizer.pad_token = "[PAD]"
        tokenizer.mask_token_id = vocab["[MASK]"]
        tokenizer.pad_token_id = vocab["[PAD]"]
        self.tokenizer = tokenizer
        # ---------- SNP indices for all individuals ----------

        self.n_snp2pad = int(np.ceil((self.n_snps+1)/8)*8) - self.n_snps
        alleles = torch.as_tensor(genotype, dtype=torch.long)  # (N × 1 200)
        base = torch.arange(self.n_snps, dtype=torch.long)  # [0 … 1 199]
        snp_idx = alleles * self.snp_offset + base  # vectorised
        #if self.n_snp2pad:
        pad = torch.full((self.N, self.n_snp2pad),
                         self.snp_pad, dtype=torch.long)
        snp_idx = torch.cat((snp_idx, pad), dim=1)  # (N × (1 200 + pad))
        self.snp_idx = snp_idx.contiguous()  # final (N × L_snp)

        #self.genotype.index = plink_data.sample_id.values
        #self.genotype.columns = plink_data.variant_id.values
        self.flip = flip

        print("From PLINK %d variants with %d samples are queried" % (genotype.shape[1], genotype.shape[0]))

    def get_individual_block_genotype(self, iid):
        ind = self.iid2ind[iid]
        return self.snp_idx[ind]

    def __getitem__(self, index):
        return self.snp_idx[index]

class BlockQueryDataset(PLINKDataset):
    def __init__(self, tree_parser : SNPTreeParser, bfile, blocks, cov=None, pheno=None, cov_mean_dict=None, cov_std_dict=None, cov_ids=(), pheno_ids=(), bt=(), qt=(), flip=True):
        """
        """
        super().__init__(tree_parser=tree_parser, bfile=bfile, cov=cov, pheno=pheno, cov_mean_dict=cov_mean_dict, cov_std_dict=cov_std_dict, cov_ids=cov_ids, pheno_ids=pheno_ids, bt=bt, qt=qt)
        self.blocks = blocks
        self.iids = self.cov_df.IID.map(str).tolist()
        self.iid2ind = {iid:i for i, iid in enumerate(self.iids)}
        self.ind2iid = {i:iid for i, iid in enumerate(self.iids)}
        block_pad = self.tree_parser.n_blocks
        block_idx = torch.tensor([self.tree_parser.block2ind[self.tree_parser.snp2block[self.tree_parser.ind2snp[i]]] for i in range(self.tree_parser.n_snps)], dtype=torch.long)
        self.block_idx = torch.cat((block_idx, torch.full((self.n_snp2pad,), block_pad, dtype=torch.long)))

    def __getitem__(self, index):
        covariates = self.cov_df.iloc[index]
        iid = covariates.IID
        #iid = self.ind2iid[index]
        sample2snp_dict = {}
        block_dict = OrderedDict()
        result_dict = super().__getitem__(index)
        # sample2snp_dict
        for block_id, block_bfile in self.blocks.items():
            #block_dict[block_id] = {}
            block_dict[block_id] = block_bfile.get_individual_block_genotype(iid)
        sample2snp_dict['block'] = block_dict
        sample2snp_dict['block_ind'] = self.block_idx
        sample2snp_dict['gene'] = self.gene_idx
        sample2snp_dict['sys'] = self.sys_idx
        sample2snp_dict['snp'] = self.snp_idx[index]
        result_dict['genotype'] = sample2snp_dict
        #print(result_dict['genotype'])
        # print(sample2snp_dict['snp'].size(), sample2snp_dict['gene'].size(), sample2snp_dict['sys'].size())
        return result_dict


class SNP2PCollator(object):

    def __init__(self, tree_parser: SNPTreeParser, input_format='indices', pheno_ids = ('PHENOTYPE'), mlm=False, mlm_collator_dict={}):
        self.tree_parser = tree_parser
        self.n_snp2pad = int(np.ceil(self.tree_parser.n_snps/8)) * 8 - self.tree_parser.n_snps
        self.input_format = input_format
        self.padding_index = {"snp": self.tree_parser.n_snps * 3,
                              "gene": self.tree_parser.n_genes,
                              'system': self.tree_parser.n_systems}
        if hasattr(self.tree_parser, "n_blocks"):
            self.padding_index["block"] = self.tree_parser.n_blocks
            self.block = True
        else:
            self.block = False

        self.pheno_ids = pheno_ids
        self.mlm = mlm
        self.mlm_collator_dict = mlm_collator_dict


    def __call__(self, data):
        start = time.time()
        result_dict = dict()
        result_dict['genotype'] = {}
        result_dict['genotype']['snp2gene'] = {}

        result_dict['genotype']['gene'] = torch.stack([d['genotype']['gene'] for d in data])#.long()
        result_dict['genotype']['sys'] = torch.stack([d['genotype']['sys'] for d in data])#.long()
        result_dict['genotype']['gene_indices'] = pad_indices(torch.arange(self.tree_parser.n_genes+1, dtype=torch.long), padding_value=self.tree_parser.n_genes, multiple=8)
        result_dict['genotype']['sys_indices'] = pad_indices(torch.arange(self.tree_parser.n_systems+1, dtype=torch.long), padding_value=self.tree_parser.n_systems, multiple=8)


        if self.input_format == 'embedding':
            result_dict['genotype']['snp'] = torch.stack([d['genotype']['snp'] for d in data])  # .long()#genotype_dict
            result_dict['genotype']['embedding'] = torch.stack([d['genotype']['embedding'] for d in data])
            if self.block:
                result_dict['genotype']['block_ind'] = torch.stack([d['genotype']['block_ind'] for d in data])
        if self.input_format == 'block':
            block_dict = OrderedDict()
            #print(result_dict['genotype'])
            for block_id in data[0]['genotype']['block'].keys():
                block_value_dict = {}
                snp_indices = torch.stack([d['genotype']['block'][block_id] for d in data])
                if self.mlm:
                    snp_indices_mlm = self.mlm_collator_dict[block_id](snp_indices)
                    block_value_dict['snp'] = snp_indices_mlm["input_ids"]
                    block_value_dict['snp_label'] = snp_indices_mlm["labels"]
                else:
                    block_value_dict['snp'] = snp_indices

                block_value_dict['sig_ind'] = torch.tensor(self.tree_parser.block2sig_ind[block_id], dtype=torch.long)
                block_dict[block_id] = block_value_dict
            result_dict['genotype']['snp'] = torch.stack(
                    [d['genotype']['snp'] for d in data])  # .long()#genotype_dict
            result_dict['genotype']['block'] = block_dict
            result_dict['genotype']['block_ind'] = torch.stack([d['genotype']['block_ind'] for d in data])
        else:
            result_dict['genotype']['snp'] = torch.stack([d['genotype']['snp'] for d in data])  # .long()#genotype_dict
            if self.block:
                result_dict['genotype']['block_ind'] = torch.stack([d['genotype']['block_ind'] for d in data])

        if self.mlm:
            masked_snp = result_dict['genotype']['snp'].clone()
            label = result_dict['genotype']['snp'].clone()
            mask_prob = 0.1
            mask = torch.rand(masked_snp.shape, device=masked_snp.device) < mask_prob
            masked_snp[mask] = self.padding_index['snp'] + 1
            label[mask!=True] = -100
            result_dict['genotype']['snp'] = masked_snp
            result_dict['genotype']['snp_label'] = label


        result_dict['covariates'] = torch.stack([d['covariates'] for d in data])
        result_dict['phenotype_indices'] = torch.stack([d['phenotype_indices'] for d in data])
        result_dict['phenotype'] = torch.stack([d['phenotype'] for d in data])
        end = time.time()
        result_dict['datatime'] = torch.mean(torch.stack([d['datatime'] for d in data]))
        result_dict["time"] = torch.tensor(end - start)

        return result_dict


class ChunkSNP2PCollator(SNP2PCollator):

    def __init__(self, tree_parser: SNPTreeParser, chunker, input_format='indices', pheno_ids = ('PHENOTYPE'), mlm=False, mlm_collator_dict={}):
        super().__init__(tree_parser, input_format=input_format, pheno_ids=pheno_ids, mlm=mlm, mlm_collator_dict=mlm_collator_dict)
        self.chunker = chunker
        self.chunks = chunker.create_chunks()
        print("The number of Chunks: ", len(self.chunks))


    def __call__(self, data):
        result_dict = super().__call__(data)
        chunk_list = []
        genotype = result_dict['genotype']
        for chunk in self.chunks:
            chunk_dict = {}
            snp = genotype['snp'][:, chunk['snp_indices']]
            gene = genotype['gene'][:, chunk['gene_indices']]
            sys = genotype['sys'][:, chunk['system_indices']]
            block_ind = genotype['block_ind'][:, chunk['snp_indices']]
            block_ind_padded, n_block_pad = self.pad_batched_indices(block_ind, )
            snp2gene_mask = chunk['snp2gene_submask']
            gene2sys_mask = chunk['gene2sys_submask']
            snp_padded, n_snp_pad = self.pad_batched_indices(snp, padding_value=self.padding_index['snp'])
            gene_padded, n_gene_pad = self.pad_batched_indices(gene, padding_value=self.padding_index['gene'])
            sys_padded, n_sys_pad = self.pad_batched_indices(sys, padding_value=self.padding_index['system'])
            snp2gene_mask_padded = F.pad(snp2gene_mask, (0, n_snp_pad, 0, n_gene_pad), value=-10 ** 4)
            gene2sys_mask_padded = F.pad(gene2sys_mask, (0, n_gene_pad, 0, n_sys_pad), value=-10 ** 4)
            '''
            gene_padded, snp_padded, snp2gene_mask_padded = self.pad_query_key_mask(gene, snp, snp2gene_mask,
                                                               query_padding_index=self.padding_index['gene'],
                                                               key_padding_index=self.padding_index['snp'])
            sys_padded, gene_padded, gene2sys_mask_padded = self.pad_query_key_mask(sys, gene, gene2sys_mask,
                                                               query_padding_index=self.padding_index['system'],
                                                               key_padding_index=self.padding_index['gene'])
            '''
            chunk_dict['snp'] = snp_padded
            chunk_dict['gene'] = gene_padded
            chunk_dict['sys'] = sys_padded
            chunk_dict['block_ind'] = block_ind_padded
            chunk_dict['snp2gene_mask'] = snp2gene_mask_padded
            chunk_dict['gene2sys_mask'] = gene2sys_mask_padded
            chunk_list.append(chunk_dict)
        result_dict['genotype'] = chunk_list
        return result_dict


    #def pad__mask(self, query, key, mask, query_padding_index=0, key_padding_index=0, padding_value=-10 ** 4):
    #    padded_query, n_query_pad = self.pad_indices(query, query_padding_index)
    #    padded_key, n_key_pad = self.pad_indices(key, key_padding_index)
    #    padded_mask = F.pad(mask, (0, n_key_pad, 0, n_query_pad), value=padding_value)
    #    return padded_query, padded_key, padded_mask

    @staticmethod
    def pad_batched_indices(batched_indices, padding_value=0):
        n_indices = batched_indices.size(1)
        n_pad = int(np.ceil((n_indices+1) / 8) * 8) - n_indices
        padded_indices = F.pad(batched_indices, (0, n_pad, 0, 0), value=padding_value)
        return padded_indices, n_pad

    @staticmethod
    def pad_indices(indices, padding_value=0):
        n_indices = indices.size(0)
        n_pad = int(np.ceil((n_indices+1) / 8) * 8) - n_indices
        padded_indices = F.pad(indices, (0, n_pad), value=padding_value)
        return padded_indices, n_pad

class DynamicPhenotypeBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        super().__init__(RandomSampler(dataset),
                         batch_size=batch_size,
                         drop_last=drop_last)
        self.dataset = dataset

    def __iter__(self):
        # super().__iter__() yields lists of indices, one per batch
        for batch_indices in super().__iter__():
            #print(batch_indices)
            # sample once per batch
            #print("Yiedling..")
            n = random.randint(1, self.dataset.n_pheno)
            pheno_list, subtree = self.dataset.sample_phenotypes(
                n=n,
                build_subtree=True
            )
            # stash into dataset so __getitem__ will see it
            self.dataset.subtree = subtree
            self.dataset.subtree_phenotypes = pheno_list

            yield batch_indices



class CohortSampler(Sampler):

    def __init__(self, dataset, n_samples=None, phenotype_col='phenotype', z_weight=1, sex_col=2):
        #super(DrugResponseSampler, self).__init__()
        self.dataset = dataset.cov_df
        self.indices = self.dataset.index
        self.num_samples = self.dataset.shape[0]
        
        phenotype_values = self.dataset[phenotype_col]
        dataset_sex_0 = self.dataset.loc[self.dataset[sex_col]==0]
        phenotype_values = dataset_sex_0[phenotype_col].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        dataset_sex_0['phenotype'] = skewnorm(a, loc, scale*z_weight).pdf(phenotype_values)

        dataset_sex_1 = self.dataset.loc[self.dataset[sex_col]==1]
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


class BinaryCohortSampler(Sampler):

    def __init__(self, dataset, phenotype_col='PHENOTYPE'):
        # super(DrugResponseSampler, self).__init__()
        self.dataset = dataset.cov_df
        self.indices = self.dataset.index
        self.num_samples = self.dataset.shape[0]
        class_count = np.bincount(self.dataset[phenotype_col].values)
        class_weight = 1. / class_count
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

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed = 0, phenotype_col='PHENOTYPE', z_weight=1, sex_col=2):
        #super(DrugResponseSampler, self).__init__()
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=False)
        self.dataset = dataset.cov_df
        self.indices = self.dataset.index
        self.num_samples = int(self.dataset.shape[0]/num_replicas)
        phenotype_values = self.dataset[phenotype_col].values
        dataset_sex_0 = self.dataset.loc[self.dataset[sex_col]==0]
        phenotype_values = dataset_sex_0[phenotype_col].values
        a, loc, scale = skewnorm.fit(phenotype_values)
        dataset_sex_0['phenotype'] = skewnorm(a, loc, scale*z_weight).pdf(phenotype_values)

        dataset_sex_1 = self.dataset.loc[dataset[sex_col]==1]
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

class DistributedBinaryCohortSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed = 0, phenotype_col='PHENOTYPE', z_weight=1, sex_col=2):
        #super(DrugResponseSampler, self).__init__()
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last=False)
        self.dataset = dataset.cov_df
        self.indices = self.dataset.index
        self.num_samples = int(self.dataset.shape[0]/num_replicas)
        #self.num_case = self.dataset[phenotype_col].sum()
        #self.num_control = self.num_samples - self.num_case
        class_count = np.bincount(self.dataset[phenotype_col].values)
        class_weight = 1./class_count
        sample_weight = class_weight[self.dataset[phenotype_col].values]
        self.weights = torch.tensor(sample_weight, dtype=torch.double)

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
