from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from sgkit.io import plink
from torch.utils.data.dataset import Dataset

from .genotype_datasets import PLINKDataset
from .tokenizers import SNPTokenizer
from src.utils.tree import SNPTreeParser


class BlockDataset(Dataset):
    def __init__(self, bfile, flip=False):
        self.bfile = bfile

        print("Loading PLINK data at %s" % bfile)
        # Processing Genotypes
        plink_data = plink.read_plink(path=bfile)
        print("loading done")
        genotype = plink_data.call_genotype.as_numpy().sum(axis=-1).T.astype(np.int8)
        self.iid2ind = {str(iid): idx for idx, iid in enumerate(plink_data.sample_id.values)}
        self.ind2iid = {idx: str(iid) for idx, iid in enumerate(plink_data.sample_id.values)}

        self.snp2ind = {str(snp): idx for idx, snp in enumerate(plink_data.variant_id.values)}
        self.ind2snp = {idx: str(snp) for idx, snp in enumerate(plink_data.variant_id.values)}
        genotype_df = pd.DataFrame(genotype, index=plink_data.sample_id.values, columns=plink_data.variant_id.values)

        genotype = genotype_df.values
        if not flip:
            genotype = 2 - genotype
        else:
            print("Swapping Ref and Alt!")
        self.N, self.n_snps = genotype.shape
        self.snp_offset = self.n_snps  # allele * n_snps + j
        self.snp_pad = self.n_snps * 3 + 1  # padding value
        self.snp_mask = self.n_snps * 3 + 2

        vocab = {}
        for i, snp in enumerate(self.snp2ind.keys()):
            chrom, pos, ref, alt = snp.split(":")
            vocab[":".join([chrom, pos, ref, ref])] = i
            vocab[":".join([chrom, pos, ref, alt])] = i + self.n_snps
            vocab[":".join([chrom, pos, alt, alt])] = i + self.n_snps * 2
        vocab["[MASK]"] = self.n_snps * 3
        vocab["[PAD]"] = self.n_snps * 3 + 1

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

        self.n_snp2pad = int(np.ceil((self.n_snps + 1) / 8) * 8) - self.n_snps
        alleles = torch.as_tensor(genotype, dtype=torch.long)  # (N × 1 200)
        base = torch.arange(self.n_snps, dtype=torch.long)  # [0 … 1 199]
        snp_idx = alleles * self.snp_offset + base  # vectorised
        # if self.n_snp2pad:
        pad = torch.full((self.N, self.n_snp2pad), self.snp_pad, dtype=torch.long)
        snp_idx = torch.cat((snp_idx, pad), dim=1)  # (N × (1 200 + pad))
        self.snp_idx = snp_idx.contiguous()  # final (N × L_snp)

        # self.genotype.index = plink_data.sample_id.values
        # self.genotype.columns = plink_data.variant_id.values
        self.flip = flip

        print("From PLINK %d variants with %d samples are queried" % (genotype.shape[1], genotype.shape[0]))

    def get_individual_block_genotype(self, iid):
        ind = self.iid2ind[iid]
        return self.snp_idx[ind]

    def __getitem__(self, index):
        return self.snp_idx[index]


class BlockQueryDataset(PLINKDataset):
    def __init__(
        self,
        tree_parser: SNPTreeParser,
        bfile,
        blocks,
        cov=None,
        pheno=None,
        cov_mean_dict=None,
        cov_std_dict=None,
        cov_ids=(),
        pheno_ids=(),
        bt=(),
        qt=(),
        flip=True,
    ):
        """
        """
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
        self.blocks = blocks
        self.iids = self.cov_df.IID.map(str).tolist()
        self.iid2ind = {iid: i for i, iid in enumerate(self.iids)}
        self.ind2iid = {i: iid for i, iid in enumerate(self.iids)}
        block_pad = self.tree_parser.n_blocks
        block_idx = torch.tensor(
            [self.tree_parser.block2ind[self.tree_parser.snp2block[self.tree_parser.ind2snp[i]]] for i in range(self.tree_parser.n_snps)],
            dtype=torch.long,
        )
        self.block_idx = torch.cat((block_idx, torch.full((self.n_snp2pad,), block_pad, dtype=torch.long)))

    def __getitem__(self, index):
        covariates = self.cov_df.iloc[index]
        iid = covariates.IID
        # iid = self.ind2iid[index]
        sample2snp_dict = {}
        block_dict = OrderedDict()
        result_dict = super().__getitem__(index)
        # sample2snp_dict
        for block_id, block_bfile in self.blocks.items():
            # block_dict[block_id] = {}
            block_dict[block_id] = block_bfile.get_individual_block_genotype(iid)
        sample2snp_dict["block"] = block_dict
        sample2snp_dict["block_ind"] = self.block_idx
        sample2snp_dict["gene"] = self.gene_idx
        sample2snp_dict["sys"] = self.sys_idx
        sample2snp_dict["snp"] = self.snp_idx[index]
        result_dict["genotype"] = sample2snp_dict
        # print(result_dict['genotype'])
        # print(sample2snp_dict['snp'].size(), sample2snp_dict['gene'].size(), sample2snp_dict['sys'].size())
        return result_dict
