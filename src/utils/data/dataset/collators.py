import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.tree import SNPTreeParser
from .. import pad_indices


class SNP2PCollator(object):
    def __init__(self, tree_parser: SNPTreeParser, input_format="indices", pheno_ids=("PHENOTYPE"), mlm=False, mlm_collator_dict={}):
        self.tree_parser = tree_parser
        self.n_snp2pad = int(np.ceil(self.tree_parser.n_snps / 8)) * 8 - self.tree_parser.n_snps
        self.input_format = input_format
        self.padding_index = {"snp": self.tree_parser.n_snps * 3, "gene": self.tree_parser.n_genes, "system": self.tree_parser.n_systems}
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
        result_dict["genotype"] = {}
        result_dict["genotype"]["snp2gene"] = {}

        result_dict["genotype"]["gene"] = torch.stack([d["genotype"]["gene"] for d in data])  # .long()
        result_dict["genotype"]["sys"] = torch.stack([d["genotype"]["sys"] for d in data])  # .long()
        result_dict["genotype"]["gene_indices"] = pad_indices(
            torch.arange(self.tree_parser.n_genes + 1, dtype=torch.long), padding_value=self.tree_parser.n_genes, multiple=8
        )
        result_dict["genotype"]["sys_indices"] = pad_indices(
            torch.arange(self.tree_parser.n_systems + 1, dtype=torch.long), padding_value=self.tree_parser.n_systems, multiple=8
        )

        if self.input_format == "embedding":
            result_dict["genotype"]["snp"] = torch.stack([d["genotype"]["snp"] for d in data])  # .long()#genotype_dict
            result_dict["genotype"]["embedding"] = torch.stack([d["genotype"]["embedding"] for d in data])
            if self.block:
                result_dict["genotype"]["block_ind"] = torch.stack([d["genotype"]["block_ind"] for d in data])
        if self.input_format == "block":
            block_dict = OrderedDict()
            # print(result_dict['genotype'])
            for block_id in data[0]["genotype"]["block"].keys():
                block_value_dict = {}
                snp_indices = torch.stack([d["genotype"]["block"][block_id] for d in data])
                if self.mlm:
                    snp_indices_mlm = self.mlm_collator_dict[block_id](snp_indices)
                    block_value_dict["snp"] = snp_indices_mlm["input_ids"]
                    block_value_dict["snp_label"] = snp_indices_mlm["labels"]
                else:
                    block_value_dict["snp"] = snp_indices

                block_value_dict["sig_ind"] = torch.tensor(self.tree_parser.block2sig_ind[block_id], dtype=torch.long)
                block_dict[block_id] = block_value_dict
            result_dict["genotype"]["snp"] = torch.stack([d["genotype"]["snp"] for d in data])  # .long()#genotype_dict
            result_dict["genotype"]["block"] = block_dict
            result_dict["genotype"]["block_ind"] = torch.stack([d["genotype"]["block_ind"] for d in data])
        else:
            result_dict["genotype"]["snp"] = torch.stack([d["genotype"]["snp"] for d in data])  # .long()#genotype_dict
            if self.block:
                result_dict["genotype"]["block_ind"] = torch.stack([d["genotype"]["block_ind"] for d in data])

        if self.mlm:
            masked_snp = result_dict["genotype"]["snp"].clone()
            label = result_dict["genotype"]["snp"].clone()
            mask_prob = 0.1
            mask = torch.rand(masked_snp.shape, device=masked_snp.device) < mask_prob
            masked_snp[mask] = self.padding_index["snp"] + 1
            label[mask != True] = -100
            result_dict["genotype"]["snp"] = masked_snp
            result_dict["genotype"]["snp_label"] = label

        result_dict["covariates"] = torch.stack([d["covariates"] for d in data])
        result_dict["phenotype_indices"] = torch.stack([d["phenotype_indices"] for d in data])
        result_dict["phenotype"] = torch.stack([d["phenotype"] for d in data])
        end = time.time()
        result_dict["datatime"] = torch.mean(torch.stack([d["datatime"] for d in data]))
        result_dict["time"] = torch.tensor(end - start)

        return result_dict


class ChunkSNP2PCollator(SNP2PCollator):
    def __init__(self, tree_parser: SNPTreeParser, chunker, input_format="indices", pheno_ids=("PHENOTYPE"), mlm=False, mlm_collator_dict={}):
        super().__init__(tree_parser, input_format=input_format, pheno_ids=pheno_ids, mlm=mlm, mlm_collator_dict=mlm_collator_dict)
        self.chunker = chunker
        self.chunks = chunker.create_chunks()
        print("The number of Chunks: ", len(self.chunks))

    def __call__(self, data):
        result_dict = super().__call__(data)
        chunk_list = []
        genotype = result_dict["genotype"]
        for chunk in self.chunks:
            chunk_dict = {}
            snp = genotype["snp"][:, chunk["snp_indices"]]
            gene = genotype["gene"][:, chunk["gene_indices"]]
            sys = genotype["sys"][:, chunk["system_indices"]]
            block_ind = genotype["block_ind"][:, chunk["snp_indices"]]
            block_ind_padded, n_block_pad = self.pad_batched_indices(block_ind)
            snp2gene_mask = chunk["snp2gene_submask"]
            gene2sys_mask = chunk["gene2sys_submask"]
            snp_padded, n_snp_pad = self.pad_batched_indices(snp, padding_value=self.padding_index["snp"])
            gene_padded, n_gene_pad = self.pad_batched_indices(gene, padding_value=self.padding_index["gene"])
            sys_padded, n_sys_pad = self.pad_batched_indices(sys, padding_value=self.padding_index["system"])
            snp2gene_mask_padded = F.pad(snp2gene_mask, (0, n_snp_pad, 0, n_gene_pad), value=-10 ** 4)
            gene2sys_mask_padded = F.pad(gene2sys_mask, (0, n_gene_pad, 0, n_sys_pad), value=-10 ** 4)
            """
            gene_padded, snp_padded, snp2gene_mask_padded = self.pad_query_key_mask(gene, snp, snp2gene_mask,
                                                               query_padding_index=self.padding_index['gene'],
                                                               key_padding_index=self.padding_index['snp'])
            sys_padded, gene_padded, gene2sys_mask_padded = self.pad_query_key_mask(sys, gene, gene2sys_mask,
                                                               query_padding_index=self.padding_index['system'],
                                                               key_padding_index=self.padding_index['gene'])
            """
            chunk_dict["snp"] = snp_padded
            chunk_dict["gene"] = gene_padded
            chunk_dict["sys"] = sys_padded
            chunk_dict["block_ind"] = block_ind_padded
            chunk_dict["snp2gene_mask"] = snp2gene_mask_padded
            chunk_dict["gene2sys_mask"] = gene2sys_mask_padded
            chunk_list.append(chunk_dict)
        result_dict["genotype"] = chunk_list
        return result_dict

    # def pad__mask(self, query, key, mask, query_padding_index=0, key_padding_index=0, padding_value=-10 ** 4):
    #    padded_query, n_query_pad = self.pad_indices(query, query_padding_index)
    #    padded_key, n_key_pad = self.pad_indices(key, key_padding_index)
    #    padded_mask = F.pad(mask, (0, n_key_pad, 0, n_query_pad), value=padding_value)
    #    return padded_query, padded_key, padded_mask

    @staticmethod
    def pad_batched_indices(batched_indices, padding_value=0):
        n_indices = batched_indices.size(1)
        n_pad = int(np.ceil((n_indices + 1) / 8) * 8) - n_indices
        padded_indices = F.pad(batched_indices, (0, n_pad, 0, 0), value=padding_value)
        return padded_indices, n_pad

    @staticmethod
    def pad_indices(indices, padding_value=0):
        n_indices = indices.size(0)
        n_pad = int(np.ceil((n_indices + 1) / 8) * 8) - n_indices
        padded_indices = F.pad(indices, (0, n_pad), value=padding_value)
        return padded_indices, n_pad
