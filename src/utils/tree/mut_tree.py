import pandas as pd
import numpy as np
import torch
from . import TreeParser


class MutTreeParser(TreeParser):
    def __init__(self, ontology, gene2ind):
        super(MutTreeParser, self).__init__(ontology, gene2ind)
        print("%d in gene2sys mask"%self.gene2sys_mask.sum())

    def get_mut2sys_mask(self, sys_indices, gene_indices, type_indices=None):

        if len(sys_indices)==0:
            return torch.zeros((self.n_systems, self.n_genes))
        else:
            if type_indices is not None:
                gene2sys_mask = np.copy(self.gene2sys_mask)
                for key, value in type_indices.items():
                    type_mask = np.zeros_like(self.gene2sys_mask)
                    type_mask[:, value] = key

                    gene2sys_mask *= type_mask
                    #print(type_mask, value, key, gene2sys_mask.sum(), self.gene2sys_mask.sum(), self.gene2sys_mask)
            else:
                gene2sys_mask = np.copy(self.gene2sys_mask)
            gene2sys_mask =  torch.tensor(gene2sys_mask)[sys_indices, :]
            gene2sys_mask = gene2sys_mask[:, gene_indices]
            result_mask = torch.zeros((self.n_systems, self.n_genes))
            result_mask[:gene2sys_mask.size(0), :gene2sys_mask.size(1)] = gene2sys_mask
            return result_mask

    def get_mut2sys_embeddings(self, gene_indices):
        if len(gene_indices)==0:
            gene_embedding_indices = torch.tensor([])
            sys_embedding_indices = torch.tensor([])
            return {"gene": gene_embedding_indices, "sys": sys_embedding_indices}
        else:
            gene_embedding_indices = sorted(list(set(sum([[gene]*len(self.gene2sys_dict[gene]) for gene in gene_indices], []))))
            sys_embedding_indices = sorted(list(set(sum([self.gene2sys_dict[gene] for gene in gene_indices], []))))
            return {"gene":torch.tensor(gene_embedding_indices), "sys":torch.tensor(sys_embedding_indices)}

    def get_mut2sys(self, gene_indices, type_indices=None):
        embeddings = self.get_mut2sys_embeddings(gene_indices)
        mask = self.get_mut2sys_mask(embeddings['sys'], embeddings['gene'], type_indices=type_indices)
        return {"gene": embeddings['gene'], 'sys': embeddings['sys'], 'mask': mask}