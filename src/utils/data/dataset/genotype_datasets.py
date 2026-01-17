import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sgkit.io import plink
from torch.utils.data.dataset import Dataset

from src.utils.tree import SNPTreeParser
from .. import pad_indices


class GenotypeDataset(Dataset):
    def __init__(
        self,
        tree_parser: SNPTreeParser,
        cov,
        pheno=None,
        cov_mean_dict=None,
        cov_std_dict=None,
        cov_ids=(),
        pheno_ids=(),
        bt=(),
        qt=(),
    ):
        self.tree_parser = tree_parser
        self.n_snp2pad = int(np.ceil((self.tree_parser.n_snps + 1) / 8)) * 8 - self.tree_parser.n_snps
        self.n_gene2pad = int(np.ceil((self.tree_parser.n_genes + 1) / 8)) * 8 - self.tree_parser.n_genes
        self.n_sys2pad = int(np.ceil((self.tree_parser.n_systems + 1) / 8)) * 8 - self.tree_parser.n_systems

        self.gene_pad = self.tree_parser.n_genes
        self.sys_pad = self.tree_parser.n_systems

        gene_idx = torch.arange(self.tree_parser.n_genes + 1, dtype=torch.long)
        gene_idx = torch.cat((gene_idx, torch.full((self.n_gene2pad,), self.gene_pad, dtype=torch.long)))
        self.gene_idx = gene_idx  # (L_gene,)
        sys_idx = torch.arange(self.tree_parser.n_systems + 1, dtype=torch.long)
        sys_idx = torch.cat((sys_idx, torch.full((self.n_sys2pad,), self.sys_pad, dtype=torch.long)))
        self.sys_idx = sys_idx  # (L_sys,)

        self.snp_range = list(range(self.tree_parser.n_snps + self.n_snp2pad))
        self.block_range = list(range(self.tree_parser.n_snps + self.n_snp2pad))
        self.gene_range = list(range(self.tree_parser.n_genes + self.n_gene2pad))
        self.sys_range = list(range(self.tree_parser.n_systems + self.n_sys2pad))

        print("Processing Covariates...")
        if cov is not None:
            print("Loading Covariate file at %s" % cov)
            self.cov_df = pd.read_csv(cov, sep="\t")
            self.cov_df["IID"] = self.cov_df["IID"].astype(str)
            self.cov_df["FID"] = self.cov_df["FID"].astype(str)
            # self.cov_df = self.cov_df.set_index('IID').loc[plink_data.sample_id.values].reset_index()
        else:
            self.cov_df = None  # will be initialized in child class

        if len(cov_ids) != 0:
            self.cov_ids = cov_ids
        else:
            self.cov_ids = [cov for cov in self.cov_df.columns[2:] if cov != "PHENOTYPE"]
        self.n_cov = len(self.cov_ids) + 1  # +1 for sex cov

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
        print("Processing Phenotypes")
        if pheno is not None:
            print("Loading Phenotype file at %s" % pheno)
            self.pheno_df = pd.read_csv(pheno, sep="\t")
            if len(pheno_ids) != 0:
                self.pheno_ids = pheno_ids
            else:
                self.pheno_ids = [pheno for pheno in self.pheno_df.columns[2:]]
            self.pheno_df["IID"] = self.pheno_df["IID"].map(str)
            self.pheno_df["FID"] = self.pheno_df["FID"].map(str)
            self.pheno_df = self.pheno_df.fillna(-9)

        else:
            self.pheno_df = None  # will be initialized in child class = self.cov_df[['FID', 'IID', 'PHENOTYPE']]
            self.pheno_df = self.pheno_df.fillna(-9)


        '''
        self.pheno2ind = {pheno: i for i, pheno in enumerate(self.pheno_ids)}
        self.ind2pheno = {i: pheno for i, pheno in enumerate(self.pheno_ids)}
        self.pheno2type = {}
        if (len(bt) == 0) & (len(qt) == 0):
            self.qt = self.pheno_ids  # all phenotypes will be regression tasks
            self.bt = []
            self.qt_inds = [self.pheno2ind[pheno] for pheno in self.qt]
            self.bt_inds = []
            for pheno in self.qt:
                self.pheno2type[pheno] = "qt"
        else:
            self.qt = qt
            self.bt = bt
            self.qt_inds = [self.pheno2ind[pheno] for pheno in qt]
            self.bt_inds = [self.pheno2ind[pheno] for pheno in bt]
            for pheno in qt:
                self.pheno2type[pheno] = "qt"
            for pheno in bt:
                self.pheno2type[pheno] = "bt"

        self.n_pheno = len(self.pheno_ids)
        self.pheno_range = list(range(self.n_pheno))
        self.has_phenotype = False
        if ("PHENOTYPE" in self.cov_df.columns) or (pheno is not None):
            self.has_phenotype = True
        '''
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
        sex = covariates["SEX"]
        i_cov = 0
        if (int(sex) == -1) or (int(sex) == -9):
            pass
        else:
            covariates_tensor[int(sex)] = 1
        i_cov += 2
        for cov_id in self.cov_ids:
            if cov_id == "SEX":
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
        result_dict["phenotype_indices"] = phenotype_ind_tensor[self.pheno_range]
        if self.has_phenotype:
            phenotype_tensor = torch.tensor([phenotypes[pheno_id] for pheno_id in self.pheno_ids], dtype=torch.float32)
            result_dict["phenotype"] = phenotype_tensor[self.pheno_range]

        end = time.time()
        result_dict["datatime"] = torch.tensor(end - start)
        result_dict["covariates"] = covariates_tensor
        return result_dict


class TSVDataset(GenotypeDataset):
    def __init__(
        self,
        tree_parser: SNPTreeParser,
        genotype_path,
        cov,
        pheno=None,
        cov_mean_dict=None,
        cov_std_dict=None,
        flip=False,
        input_format="indices",
        cov_ids=(),
        pheno_ids=(),
        bt=(),
        qt=(),
    ):
        super().__init__(
            tree_parser=tree_parser,
            cov=cov,
            pheno=pheno,
            cov_mean_dict=cov_mean_dict,
            cov_std_dict=cov_std_dict,
            cov_ids=cov_ids,
            pheno_ids=pheno_ids,
            bt=bt,
            qt=qt,
        )

        print("Loading Genotype data at %s" % genotype_path)
        genotype_df = pd.read_csv(genotype_path, sep="\t", index_col=0)
        genotype_df.columns = genotype_df.columns.astype(int)
        print("loading done")
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

        n_snps = tree_parser.n_snps
        self.snp_offset = n_snps
        self.snp_pad = n_snps * 3

        alleles = torch.as_tensor(genotype, dtype=torch.long)
        alleles = torch.clip(alleles, min=0, max=2)
        base = torch.arange(self.n_snps, dtype=torch.long)
        snp_idx = alleles * self.snp_offset + base

        pad = torch.full((self.N, self.n_snp2pad), self.snp_pad, dtype=torch.long)
        snp_idx = torch.cat((snp_idx, pad), dim=1)

        self.snp_idx = snp_idx.contiguous()
        block_pad = self.tree_parser.n_blocks
        block_idx = torch.tensor(
            [self.tree_parser.block2ind[self.tree_parser.snp2block[self.tree_parser.ind2snp[i]]] for i in range(self.tree_parser.n_snps)],
            dtype=torch.long,
        )

        self.block_idx = torch.cat((block_idx, torch.full((self.n_snp2pad + 1,), block_pad, dtype=torch.long)))

        self.flip = flip
        print("From TSV %d variants with %d samples are queried" % (genotype.shape[1], genotype.shape[0]))
        del genotype_df
        del genotype

        if self.cov_df is not None:
            self.cov_df = self.cov_df.set_index("IID").loc[self.ind2iid.values()].reset_index()

        if self.pheno_df is not None:
            self.pheno_df = self.pheno_df.set_index("IID").loc[self.ind2iid.values()].reset_index()

        self.subtree = None
        self.subtree_phenotypes = []

    def __getitem__(self, index):
        result_dict = super().__getitem__(index)
        sample2snp_dict = {}
        sample2snp_dict["block_ind"] = self.block_idx[self.block_range]
        sample2snp_dict["snp"] = self.snp_idx[index, self.snp_range]
        sample2snp_dict["gene"] = self.gene_idx[self.gene_range]
        sample2snp_dict["sys"] = self.sys_idx[self.sys_range]
        result_dict["genotype"] = sample2snp_dict
        return result_dict


class PLINKDataset(GenotypeDataset):
    def __init__(
        self,
        tree_parser: SNPTreeParser,
        bfile=None,
        cov=None,
        pheno=None,
        cov_mean_dict=None,
        cov_std_dict=None,
        flip=False,
        block=False,
        input_format="indices",
        cov_ids=(),
        pheno_ids=(),
        bt=(),
        qt=(),
    ):
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
        super().__init__(
            tree_parser=tree_parser,
            cov=cov,
            pheno=pheno,
            cov_mean_dict=cov_mean_dict,
            cov_std_dict=cov_std_dict,
            cov_ids=cov_ids,
            pheno_ids=pheno_ids,
            bt=bt,
            qt=qt,
        )

        self.bfile = bfile

        print("Loading PLINK data at %s" % bfile)
        # Processing Genotypes
        plink_data = plink.read_plink(path=bfile)
        # print(f'loading done with{len(plink_data.sample_id.values)} individuals and {len(plink_data.variant_id.values)} SNPs')

        snp_ids = plink_data["variant_id"].values
        snp_contig_mapping = {
            i: int(str(chromosome).replace("chr", "")) for i, chromosome in enumerate(plink_data["contig_id"].values)
        }
        snp_chr = plink_data["variant_position"].values
        snp_pos = [snp_contig_mapping[contig] for contig in plink_data["variant_contig"].values]

        snp_id2chr = {snp: chromosome for snp, chromosome in zip(snp_ids, snp_chr)}
        snp_id2pos = {snp: pos for snp, pos in zip(snp_ids, snp_pos)}

        self.tree_parser.set_snp2chr(snp_id2chr)
        self.tree_parser.set_snp2pos(snp_id2pos)

        self.input_format = input_format
        genotype = plink_data.call_genotype.as_numpy().sum(axis=-1).T.astype(np.int8)

        snp_sorted = [snp for snp, i in sorted(list(self.tree_parser.snp2ind.items()), key=lambda a: a[1])]

        genotype_df = pd.DataFrame(genotype, index=plink_data.sample_id.values, columns=plink_data.variant_id.values)
        # genotype_df = genotype_df[snp_sorted]
        missing_snps = [snp for snp in snp_sorted if snp not in genotype_df.columns]
        print("SNPs not in bfile:", missing_snps)
        genotype_df = genotype_df.reindex(columns=snp_sorted, fill_value=0)

        # Processing Covariates
        if self.cov_df is not None:
            intersection_iid = list(set(genotype_df.index.tolist()).intersection(self.cov_df["IID"].tolist()))
            self.cov_df = self.cov_df.set_index("IID").loc[intersection_iid].reset_index()
            genotype_df = genotype_df.loc[intersection_iid]
        else:
            self.cov_df = pd.DataFrame(
                {
                    "FID": plink_data.sample_family_id.as_numpy(),
                    "IID": plink_data.sample_id.as_numpy(),
                    "SEX": plink_data.sample_sex.as_numpy(),
                    "PHENOTYPE": plink_data.sample_phenotype.as_numpy(),
                }
            )
            self.cov_df = self.cov_df[["FID", "IID", "SEX", "PHENOTYPE"]]
            self.cov_df = self.cov_df.loc[self.cov_df.PHENOTYPE != -1]
            self.cov_df["PHENOTYPE"] = self.cov_df["PHENOTYPE"] - 1
            self.cov_df["IID"] = self.cov_df["IID"].astype(str)
            self.cov_df["FID"] = self.cov_df["FID"].astype(str)
            self.cov_df = self.cov_df.set_index("IID").loc[plink_data.sample_id.values].reset_index()

        if self.pheno_df is not None:
            intersection_iid = list(set(genotype_df.index.tolist()).intersection(self.pheno_df["IID"].tolist()))
            self.pheno_df = self.pheno_df.set_index("IID").loc[intersection_iid].reset_index()
            self.cov_df = self.cov_df.set_index("IID").loc[intersection_iid].reset_index()
            genotype_df = genotype_df.loc[intersection_iid]
        else:
            self.pheno_df = self.cov_df[["FID", "IID", "PHENOTYPE"]]
            self.pheno_df["IID"] = self.pheno_df["IID"].map(str)
            self.pheno_df = self.pheno_df.set_index("IID").loc[plink_data.sample_id.values].reset_index()
            self.cov_df = self.cov_df.set_index("IID").loc[plink_data.sample_id.values].reset_index()
            self.pheno_ids = ["PHENOTYPE"]

        self.iid2ind = {str(iid): idx for idx, iid in enumerate(genotype_df.index.values)}
        self.ind2iid = {idx: str(iid) for idx, iid in enumerate(genotype_df.index.values)}

        # print(genotype_df.head())
        # print(self.cov_df.head())
        # print(self.pheno_df.head())

        self.subtree = None
        self.subtree_phenotypes = []

        print("genotype data shape: ", genotype_df.shape)
        genotype = genotype_df.values
        # print(genotype)
        if not flip:
            genotype = 2 - genotype
        else:
            print("Swapping Ref and Alt!")
        self.N, self.n_snps = genotype.shape

        n_snps = tree_parser.n_snps  # original SNP catalogue size
        self.snp_offset = n_snps  # allele * n_snps + j
        self.snp_pad = n_snps * 3  # padding value

        # ---------- SNP indices for all individuals ----------
        alleles = torch.as_tensor(genotype, dtype=torch.long)  # (N × 1 200)
        alleles = torch.clip(alleles, min=0, max=2)  # need to be fixed?
        base = torch.arange(self.n_snps, dtype=torch.long)  # [0 … 1 199]
        snp_idx = alleles * self.snp_offset + base  # vectorised

        pad = torch.full((self.N, self.n_snp2pad), self.snp_pad, dtype=torch.long)
        snp_idx = torch.cat((snp_idx, pad), dim=1)  # (N × (1 200 + pad))

        self.snp_idx = snp_idx.contiguous()  # final (N × L_snp)
        self.block = block
        if block:
            block_pad = self.tree_parser.n_blocks
            block_idx = torch.tensor(
                [self.tree_parser.block2ind[self.tree_parser.snp2block[self.tree_parser.ind2snp[i]]] for i in range(self.tree_parser.n_snps)],
                dtype=torch.long,
            )
            self.block_idx = torch.cat((block_idx, torch.full((self.n_snp2pad + 1,), block_pad, dtype=torch.long)))

        self.flip = flip
        print("From PLINK %d variants with %d samples are queried" % (genotype.shape[1], genotype.shape[0]))
        del genotype_df
        del genotype

    def summary(self):
        print("PLINK data from %s" % self.bfile)
        print("Covariates for prediction %s" % ", ".join(self.cov_ids))
        print("Covariates: ", ", ".join(self.cov_ids))
        if self.flip:
            print("Ref. and Alt. are flipped")
        print("%d of participant with %d of variants" % (self.genotype.shape[0], self.genotype.shape[1]))
        print("The number of covariates:", self.n_cov)

    def sample_population(self, n=100):
        sampled_iids = self.cov_df["IID"].sample(n=n, random_state=42).tolist()
        original_indices = [self.iid2ind[iid] for iid in sampled_iids]
        self.snp_idx = self.snp_idx[original_indices]
        self.cov_df = self.cov_df.set_index("IID").loc[sampled_iids].reset_index()
        self.pheno_df = self.pheno_df.set_index("IID").loc[sampled_iids].reset_index()
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
        print("Select Phenotypes: ", ", ".join(phenotypes))
        pheno_indices = [self.pheno2ind[pheno] for pheno in phenotypes]
        snp_indices, gene_indices, sys_indices = self.collect_indices_from_phenotypes(phenotypes)
        self.set_range_from_indices(snp_indices, gene_indices, sys_indices, pheno_indices)

    def collect_indices_from_phenotypes(self, phenotypes):
        # print("Sample phenotypes: ", sampled_phenotypes)
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
            sample2snp_dict["block_ind"] = self.block_idx[self.block_range]
        sample2snp_dict["snp"] = self.snp_idx[index, self.snp_range]
        sample2snp_dict["gene"] = self.gene_idx[self.gene_range]
        sample2snp_dict["sys"] = self.sys_idx[self.sys_range]
        result_dict["genotype"] = sample2snp_dict
        return result_dict


class EmbeddingDataset(PLINKDataset):
    def __init__(
        self,
        tree_parser: SNPTreeParser,
        bfile,
        embedding,
        iid2ind,
        cov=None,
        pheno=None,
        cov_mean_dict=None,
        cov_std_dict=None,
        cov_ids=(),
        pheno_ids=(),
        bt=(),
        qt=(),
    ):
        super().__init__(
            tree_parser=tree_parser,
            bfile=bfile,
            cov=cov,
            pheno=pheno,
            cov_mean_dict=cov_mean_dict,
            cov_std_dict=cov_std_dict,
            cov_ids=cov_ids,
            pheno_ids=pheno_ids,
            bt=bt,
            qt=qt,
        )

        self.embedding = embedding  # {self.sort_key(p): zarr.open(p, mode="r") for p in embedding_zarrs}

        self.snp2block = {}
        block_pad = self.tree_parser.n_blocks
        self.snp_sorted = [snp for snp, i in sorted(list(self.tree_parser.snp2ind.items()), key=lambda a: a[1])]
        self.snp_indices = [self.tree_parser.snp2ind_all[snp] for snp in self.snp_sorted]
        block_idx = torch.tensor(
            [self.tree_parser.block2ind[self.tree_parser.snp2block[self.tree_parser.ind2snp[i]]] for i in range(self.tree_parser.n_snps)],
            dtype=torch.long,
        )

        self.block_idx = torch.cat((block_idx, torch.full((self.n_snp2pad,), block_pad, dtype=torch.long)))

        self.iid2ind = iid2ind
        self.ind2iid = {ind: iid for iid, ind in iid2ind.items()}
        self.iid = self.cov_df.IID.tolist()

    def __getitem__(self, index):
        result_dict = super().__getitem__(index)
        iid = self.iid[index]
        ind = self.iid2ind[iid]

        sample2snp_dict = {}
        sample2snp_dict["gene"] = self.gene_idx
        sample2snp_dict["sys"] = self.sys_idx
        sample2snp_dict["block"] = self.block_idx

        # embedding = torch.tensor(self.embedding.oindex[ind, :, :], dtype=torch.float32)
        embedding = torch.load(os.path.join(self.embedding, str(iid) + ".tensor"), weights_only=True)
        embedding = embedding[self.snp_indices, :]
        embedding = F.pad(embedding, (0, 0, 0, self.n_snp2pad), value=0)

        sample2snp_dict["snp"] = self.snp_idx[index]
        sample2snp_dict["embedding"] = embedding
        result_dict["genotype"] = sample2snp_dict

        return result_dict
