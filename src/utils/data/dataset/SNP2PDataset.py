import random

import pandas as pd
from torch.utils.data.dataset import Dataset
import torch.distributed as dist
from torch.utils.data import IterableDataset, get_worker_info
from torch.utils.data.sampler import Sampler, BatchSampler, RandomSampler
from scipy.stats import zscore, skewnorm
import torch
from src.utils.tree import TreeParser, SNPTreeParser
from torch.nn.utils.rnn import pad_sequence
import time
from torch.utils.data.distributed import DistributedSampler
from sgkit.io import plink
import math
import numpy as np


class SNP2PDataset(Dataset):

    def __init__(self, genotype_phenotype, snp_data, tree_parser: SNPTreeParser, age_mean=None, age_std=None, n_cov=4):
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

        ##heterozygous_gene_indices = torch.unique(snp_type_dict['heterozygous']['gene']).tolist()
        #homozygous_a1_gene_indices = torch.unique(snp_type_dict['homozygous_a1']['gene']).tolist()
        #gene2sys_mask_for_gene = torch.zeros((self.tree_parser.n_systems, self.tree_parser.n_genes), dtype=torch.bool)
        #gene2sys_mask_for_gene[:, homozygous_a1_gene_indices] = 1
        #gene2sys_mask_for_gene[:, heterozygous_gene_indices] = 1
        #result_dict["gene2sys_mask"] = torch.tensor(self.tree_parser.gene2sys_mask, dtype=torch.bool) & gene2sys_mask_for_gene
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

    def __init__(self, tree_parser : SNPTreeParser, bfile, cov=None, pheno=None, cov_mean_dict=None, cov_std_dict=None, flip=False,
                 input_format='indices', cov_ids=(), pheno_ids=(), bt=(), qt=(), dynamic_phenotype_sampling=0):
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
        self.tree_parser = tree_parser
        self.bfile = bfile
        print('Loading PLINK data at %s'%bfile)

        # Processing Genotypes
        plink_data = plink.read_plink(path=bfile)
        self.input_format = input_format

        genotype = plink_data.call_genotype.as_numpy().sum(axis=-1).T.astype(np.int8)
        snp_sorted = [snp for snp, i in sorted(list(self.tree_parser.snp2ind.items()), key=lambda a: a[1])]
        self.ind2iid = {idx:str(iid) for idx, iid in enumerate(plink_data.sample_id.values)}
        #plink_snp_ids = plink_data.variant_id.values
        #plink_snp_id2ind = {snp_id:ind for ind, snp_id in enumerate(plink_snp_ids)}
        #ordered_snp_ind = [tree_parser.snp2ind[col] for col in plink_snp_ids if col in tree_parser.snp2ind.keys()]
        genotype_df = pd.DataFrame(genotype, index=plink_data.sample_id.values, columns=plink_data.variant_id.values)
        genotype_df = genotype_df[snp_sorted]
        genotype = genotype_df.values
        if flip:
            genotype = 2 - genotype
            print("Swapping Ref and Alt!")
        if input_format == 'binary':
            self.genotype == genotype
        else:
            self.genotype = {}
            for i, personal_genotype in enumerate(genotype):
                self.genotype[i] = {}
                self.genotype[i]['homozygous_a1'] = np.where(personal_genotype == 2)[0]
                self.genotype[i]['heterozygous'] = np.where(personal_genotype == 1)[0]


        #self.genotype.index = plink_data.sample_id.values
        #self.genotype.columns = plink_data.variant_id.values
        self.flip = flip
        print("From PLINK %d variants with %d samples are queried" % (genotype.shape[1], genotype.shape[0]))
        del genotype_df
        del genotype

        # Processing Covariates
        print('Processing Covariates...')
        if cov is not None:
            print("Loading Covariate file at %s"%cov)
            self.cov_df = pd.read_csv(cov, sep='\t')
            self.cov_df['IID'] = self.cov_df['IID'].astype(str)
            self.cov_df = self.cov_df.set_index('IID').loc[plink_data.sample_id.values].reset_index()
        else:
            self.cov_df = pd.DataFrame({'FID': plink_data.sample_family_id.as_numpy(),
                                        'IID': plink_data.sample_id.as_numpy(),
                                        'SEX': plink_data.sample_sex.as_numpy(),
                                        'PHENOTYPE': plink_data.sample_phenotype.as_numpy() })
            self.cov_df = self.cov_df[['FID', 'IID', 'SEX', 'PHENOTYPE']]
            self.cov_df = self.cov_df.loc[self.cov_df.PHENOTYPE!=-1]
            self.cov_df['PHENOTYPE'] = self.cov_df['PHENOTYPE'] - 1
            self.cov_df['IID'] = self.cov_df['IID'].astype(str)
            self.cov_df = self.cov_df.set_index('IID').loc[plink_data.sample_id.values].reset_index()
            #self.genotype = self.genotype.loc[self.cov_df.IID]

        self.cov_df['FID'] = self.cov_df['FID'].astype(str)
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
            self.pheno_df = self.pheno_df.set_index('IID').loc[plink_data.sample_id.values].reset_index()
            #if 'PHENOTYPE' not in self.cov_df.columns:
            #self.cov_df.merge(self.pheno_df, left_on=['FID', 'IID'], right_on=['FID', 'IID'])
            #self.genotype = self.genotype.loc[self.cov_df.IID]
        else:
            self.pheno_df = self.cov_df[['FID', 'IID', 'PHENOTYPE']]
            self.pheno_df['IID'] = self.pheno_df['IID'].map(str)
            self.pheno_df = self.pheno_df.set_index('IID').loc[plink_data.sample_id.values].reset_index()
            self.pheno_ids = ['PHENOTYPE']

        self.pheno2ind = {pheno:i for i, pheno in enumerate(self.pheno_ids)}
        self.ind2pheno = {i:pheno for i, pheno in enumerate(self.pheno_ids)}
        if (len(bt)==0) & (len(qt)==0):
            self.qt = self.pheno_ids # all phenotypes will be regression tasks
        else:
            self.qt = qt
            self.bt = bt
            self.qt_inds = [self.pheno2ind[pheno] for pheno in qt]
            self.bt_inds = [self.pheno2ind[pheno] for pheno in bt]
            self.pheno2type = {}
            for pheno in qt:
                self.pheno2type[pheno] = 'qt'
            for pheno in bt:
                self.pheno2type[pheno] = 'bt'


        self.n_pheno = len(self.pheno_ids)
        self.has_phenotype = False
        if ('PHENOTYPE' in self.cov_df.columns) or (pheno is not None):
            self.has_phenotype = True

        if dynamic_phenotype_sampling:
            self.dynamic_phenotype_sampling = True
            self.n_phenotypes_sample = dynamic_phenotype_sampling
        else:
            self.dynamic_phenotype_sampling = False
            self.n_phenotypes_sample = self.n_pheno
        self.subtree = None
        self.subtree_phenotypes = []


    def __len__(self):
        return self.cov_df.shape[0]

    def summary(self):
        print('PLINK data from %s'%self.bfile)
        print('Covariates for prediction %s' % ", ".join(self.cov_ids))
        print("Covariates: ", ", ".join(self.cov_ids))
        if self.flip:
            print("Ref. and Alt. are flipped")
        print("%d of participant with %d of variants"%(self.genotype.shape[0], self.genotype.shape[1]))

    def is_binary(self, array):
        return np.isin(array, [0, 1]).all()

    def sample_population(self, n=100, inplace=False):
        if not inplace:
            sampled_individual = self.cov_df.IID.sample(n=n).tolist()
            return sampled_individual
        else:
            self.genotype = self.genotype.sample(n=n)
            self.cov_df = self.cov_df.set_index("IID").loc[self.genotype.index].reset_index().rename(columns={'index':"IID"})
            self.pheno_df = self.pheno_df.set_index("IID").loc[self.genotype.index].reset_index().rename(columns={'index':"IID"})

    def sample_phenotypes(self, n=2, build_subtree=True):
        sampled_phenotypes = random.sample(self.pheno_ids, n)
        if build_subtree:
            snps_for_phenotypes = set(sum([self.tree_parser.pheno2snp[pheno] for pheno in sampled_phenotypes], []))
            subtree = self.tree_parser.retain_snps(snps_for_phenotypes, inplace=False, verbose=False)
            subtree = subtree.init_ontology_with_snp(subtree.collapse(to_keep=None, min_term_size=5, inplace=False, verbose=False), subtree.snp2gene_df, inplace=False, verbose=False)
            self.subtree = subtree
            self.subtree_phenotypes = sampled_phenotypes
            #print("subtree aligned..", subtree)
            return sampled_phenotypes, subtree
        else:
            return sampled_phenotypes

    def __getitem__(self, index):
        start = time.time()
        covariates = self.cov_df.iloc[index]
        phenotypes = self.pheno_df.iloc[index]
        #print(covariates['IID'] == phenotypes['IID'])
        #djfkd
        iid = str(int(covariates['IID']))
        sample2snp_dict = {}

        result_dict = dict()
        snp_type_dict = dict()
        #print("Subtree", self.subtree)
        if self.dynamic_phenotype_sampling:
            tree_parser = self.subtree
            snp_ind_alias_dict = self.tree_parser.snp2ind
            gene_ind_alias_dict = self.tree_parser.gene2ind
        else:
            tree_parser = self.tree_parser
            snp_ind_alias_dict = None
            gene_ind_alias_dict = None

        if self.input_format=='binary':
            homozygous_a1 = np.where(self.genotype[iid, :] == 2)[0]
            heterozygous = np.where(self.genotype[iid, :] == 1)[0]
            snp_type_dict['homozygous_a1'] = torch.tensor(tree_parser.get_snp2gene_mask({1.0: homozygous_a1}), dtype=torch.int)
            snp_type_dict['heterozygous'] = torch.tensor(tree_parser.get_snp2gene_mask({1.0: heterozygous}), dtype=torch.int)
        else:
            if self.dynamic_phenotype_sampling:
                homozygous_a1 = self.subtree.alias_indices([ind for ind in self.genotype[index]['homozygous_a1'] if self.tree_parser.ind2snp[ind] in self.subtree.snp2ind.keys()], source_ind2id_dict=self.tree_parser.ind2snp, target_id2ind_dict=self.subtree.snp2ind)
                heterozygous = self.subtree.alias_indices([ind for ind in self.genotype[index]['heterozygous'] if self.tree_parser.ind2snp[ind] in self.subtree.snp2ind.keys()], source_ind2id_dict=self.tree_parser.ind2snp, target_id2ind_dict=self.subtree.snp2ind)#self.genotype[index]['heterozygous']
                #print(homozygous_a1, heterozygous)
                snp_type_dict['homozygous_a1'] = tree_parser.get_snp2gene(homozygous_a1, {1.0: homozygous_a1}, snp_ind_alias_dict=snp_ind_alias_dict, gene_ind_alias_dict=gene_ind_alias_dict)
                snp_type_dict['heterozygous'] = tree_parser.get_snp2gene(heterozygous, {1.0: heterozygous}, snp_ind_alias_dict=snp_ind_alias_dict, gene_ind_alias_dict=gene_ind_alias_dict)
            else:
                homozygous_a1 = self.genotype[index]['homozygous_a1']
                heterozygous = self.genotype[index]['heterozygous']
                # print(homozygous_a1, heterozygous)
                snp_type_dict['homozygous_a1'] = self.tree_parser.get_snp2gene(homozygous_a1, {1.0: homozygous_a1})
                snp_type_dict['heterozygous'] = self.tree_parser.get_snp2gene(heterozygous, {1.0: heterozygous})

        sample2snp_dict['embedding'] = snp_type_dict

        covariates_tensor = [0]*self.n_cov
        sex = covariates['SEX']
        i_cov = 0
        if (int(sex)==-1) or (int(sex)==-9):
            pass
        else:
            covariates_tensor[int(sex)] = 1
        i_cov += 2
        for cov_id in self.cov_ids:
            if cov_id=='SEX':
                continue
            if cov_id in self.cov_mean_dict.keys():
                cov_value = (covariates[cov_id]-self.cov_mean_dict[cov_id])/self.cov_std_dict[cov_id]
            else:
                cov_value = covariates[cov_id]
            covariates_tensor[i_cov] = cov_value
            i_cov += 1
        covariates_tensor = torch.tensor(covariates_tensor, dtype=torch.float32)
        covariates_tensor = covariates_tensor#torch.cat([sex_age_tensor, torch.tensor(covariates, dtype=torch.float32)])


        if self.has_phenotype:
            if self.dynamic_phenotype_sampling:
                phenotype_ind_tensor = torch.tensor([self.pheno2ind[pheno] for pheno in self.subtree_phenotypes], dtype=torch.int)
                phenotype_tensor = torch.tensor([phenotypes[pheno_id] for pheno_id in self.subtree_phenotypes],
                                                dtype=torch.float32)
            else:
                phenotype_ind_tensor = torch.tensor([i for i in range(self.n_pheno)], dtype=torch.int)
                phenotype_tensor = torch.tensor([phenotypes[pheno_id] for pheno_id in self.pheno_ids],
                                                dtype=torch.float32)

            result_dict['phenotype_indices'] = phenotype_ind_tensor
            result_dict['phenotype'] = phenotype_tensor

        result_dict['genotype'] = sample2snp_dict
        end = time.time()
        result_dict["datatime"] = torch.tensor(end-start)
        result_dict["covariates"] = covariates_tensor
        return result_dict

class SNP2PCollator(object):

    def __init__(self, tree_parser: SNPTreeParser, input_format='indices'):
        self.tree_parser = tree_parser
        self.input_format = input_format
        self.padding_index = {"snp": self.tree_parser.n_snps, "gene": self.tree_parser.n_genes}

    def __call__(self, data):
        start = time.time()
        result_dict = dict()
        genotype_dict = dict()
        snp_type_dict = {}


        for snp_type in ['heterozygous', 'homozygous_a1']:
            if self.input_format == 'indices':
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
            else:
                snp_type_dict[snp_type] = torch.stack([dr['genotype']["embedding"][snp_type] for dr in data])

        genotype_dict['embedding'] = snp_type_dict
        #result_dict['gene2sys_mask'] = torch.stack([d['gene2sys_mask'] for d in data])
        result_dict['genotype'] = genotype_dict
        result_dict['covariates'] = torch.stack([d['covariates'] for d in data])
        #print(result_dict['covariates'])
        result_dict['phenotype_indices'] = torch.stack([d['phenotype_indices'] for d in data])
        result_dict['phenotype'] = torch.stack([d['phenotype'] for d in data])
        end = time.time()
        result_dict['datatime'] = torch.mean(torch.stack([d['datatime'] for d in data]))
        result_dict["time"] = torch.tensor(end - start)
        #print(genotype_dict)
        return result_dict


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

class DynamicPhenotypeBatchIterableDataset(IterableDataset):
    def __init__(self, dataset, collator, batch_size, shuffle=True):
        super().__init__()
        self.dataset    = dataset      # map‑style dataset, len() defined
        self.collator   = collator
        self.batch_size = batch_size
        self.shuffle    = shuffle

    # OPTIONAL: lets DataLoader len(loader) work
    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        ############ 0.  worker bookkeeping ############
        worker = get_worker_info()          # None if num_workers == 0
        total  = len(self.dataset)          # N samples in underlying ds

        if worker is None:                  # single‑process data‑load
            w_start, w_end   = 0, total
        else:                               # multi‑worker sharding
            per_worker       = math.ceil(total / worker.num_workers)
            w_start          = worker.id * per_worker
            w_end            = min(w_start + per_worker, total)

        # ---- build THIS worker’s index list ----
        indices = list(range(w_start, w_end))
        if self.shuffle:
            random.shuffle(indices)

        ############ 1.  walk in batch‑size chunks ############
        for lo in range(0, len(indices), self.batch_size):
            batch_inds = indices[lo : lo + self.batch_size]

            # if the tail slice is empty (can only happen if len==0)
            if not batch_inds:
                break

            # ----- draw phenotypes & subtree ONCE for this batch -----
            n = random.randint(1, self.dataset.n_pheno - 1)
            pheno_list, subtree = self.dataset.sample_phenotypes(
                n=n, build_subtree=True
            )

            # pull raw samples
            raw_batch = [self.dataset[i] for i in batch_inds]

            # base collate
            collated  = self.collator(raw_batch)

            # ---------- build & attach masks ----------
            masks = {}
            masks['subtree_forward']  = subtree.get_hierarchical_interactions(
                subtree.interaction_types, 'forward', 'indices'
            )
            masks['subtree_backward'] = subtree.get_hierarchical_interactions(
                subtree.interaction_types, 'backward', 'indices'
            )

            # sys↔gene binary masks (shape S×G and its transpose)
            S, G = (self.dataset.tree_parser.n_systems,
                    self.dataset.tree_parser.n_genes)
            sys2gene = torch.zeros((S, G), dtype=torch.bool)
            for sys, genes in subtree.sys2gene.items():
                si = self.dataset.tree_parser.sys2ind[sys]
                for g in genes:
                    gi = self.dataset.tree_parser.gene2ind[g]
                    sys2gene[si, gi] = True

            masks['sys2gene_mask'] = sys2gene
            masks['gene2sys_mask'] = sys2gene.T
            #masks['phenotype_list'] = pheno_list
            collated['mask'] = masks

            yield collated

class DynamicPhenotypeBatchIterableDatasetDDP(IterableDataset):
    def __init__(self, dataset, collator, batch_size, shuffle=True):
        """
        base_ds   : map‑style dataset with __len__ / __getitem__
        collator  : callable(list(raw_samples)) → dict  (no masks yet)
        batch_size: number of samples per batch
        shuffle   : shuffle indices each epoch
        """
        super().__init__()
        self.dataset    = dataset
        self.collator   = collator
        self.batch_size = batch_size
        self.shuffle    = shuffle

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def _index_slice_for_worker_and_rank(self):
        """Return the list of indices this *process* should iterate."""
        N = len(self.dataset)

        # ── split by worker (DataLoader subprocess) ──
        w_info = get_worker_info()
        if w_info is None:
            w_start, w_end, w_workers = 0, N, 1
            w_id = 0
        else:
            per_worker = math.ceil(N / w_info.num_workers)
            w_start    = w_id = w_info.id
            w_start   *= per_worker
            w_end      = min(w_start + per_worker, N)
            w_workers  = w_info.num_workers

        indices = list(range(w_start, w_end))

        # ── split further by DDP rank ──
        if dist.is_initialized():
            rank       = dist.get_rank()
            world      = dist.get_world_size()
            indices    = indices[rank::world]   # stride slicing
        else:
            rank, world = 0, 1

        if self.shuffle and (rank == 0 and w_id == 0):
            random.shuffle(indices)             # shuffle once then broadcast
        # simple broadcast for reproducibility
        if dist.is_initialized():
            idx_tensor = torch.tensor(indices, dtype=torch.int64, device="cpu")
            idx_sizes  = torch.tensor([len(indices)], dtype=torch.int64, device="cpu")
            dist.broadcast(idx_sizes, src=0)
            if rank != 0:
                idx_tensor = torch.empty(idx_sizes.item(), dtype=torch.int64)
            dist.broadcast(idx_tensor, src=0)
            indices = idx_tensor.tolist()

        return indices

    def __iter__(self):
        indices = self._index_slice_for_worker_and_rank()

        for chunk_start in range(0, len(indices), self.batch_size):
            inds = indices[chunk_start : chunk_start + self.batch_size]
            if not inds:
                break

            # 1️⃣  draw phenotypes / subtree once per batch
            n = random.randint(1, self.dataset.n_pheno - 1)
            pheno_list, subtree = self.dataset.sample_phenotypes(
                n=n, build_subtree=True
            )

            # 2️⃣  fetch raw samples
            raw_batch = [self.dataset[i] for i in inds]

            # 3️⃣  basic collation
            batch = self.collator(raw_batch)

            # 4️⃣  build and attach masks
            masks = {
                "subtree_forward":  subtree.get_hierarchical_interactions(
                    subtree.interaction_types, "forward",  "indices"),
                "subtree_backward": subtree.get_hierarchical_interactions(
                    subtree.interaction_types, "backward", "indices")
            }
            S, G = (self.dataset.tree_parser.n_systems,
                    self.dataset.tree_parser.n_genes)
            sys2gene = torch.zeros((S, G), dtype=torch.bool)
            for s, genes in subtree.sys2gene.items():
                si = self.dataset.tree_parser.sys2ind[s]
                for g in genes:
                    gi = self.dataset.tree_parser.gene2ind[g]
                    sys2gene[si, gi] = True

            masks["sys2gene_mask"] = sys2gene
            masks["gene2sys_mask"] = sys2gene.T

            batch["mask"] = masks

            yield batch

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
