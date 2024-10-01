import pandas as pd
import numpy as np
import torch
from . import TreeParser


class MutTreeParser(TreeParser):
    def __init__(self, ontology, gene2ind):
        super(MutTreeParser, self).__init__(ontology, gene2ind)
        print("%d in gene2sys mask"%self.gene2sys_mask.sum())

        self.mut2gene_mask = torch.eye(self.n_genes)

    def get_mut2gene_mask(self, mut_indices, type_indices=None):

        if len(mut_indices)==0:
            return torch.zeros((self.n_genes, self.n_genes))
        else:
            if type_indices is not None:
                mut2gene_mask = np.copy(self.mut2gene_mask)
                for key, value in type_indices.items():
                    type_mask = np.zeros_like(self.mut2gene_mask)
                    type_mask[:, value] = key

                    mut2gene_mask *= type_mask
            else:
                mut2gene_mask = np.copy(self.mut2gene_mask)
            mut2gene_mask =  torch.tensor(mut2gene_mask)[mut_indices, :]
            mut2gene_mask = mut2gene_mask[:, mut_indices]
            result_mask = torch.zeros((self.n_genes, self.n_genes))
            result_mask[:mut2gene_mask.size(0), :mut2gene_mask.size(1)] = mut2gene_mask
            return result_mask

    def get_mutation2genotype_mask(self, mut_vector):
        gene2mut_mask =  torch.logical_and(torch.tensor(self.mut2gene_mask, dtype=torch.bool),
                                             mut_vector.unsqueeze(0).expand(self.n_genes, -1).bool())
        return gene2mut_mask.float()
    
    def get_mut2gene_embeddings(self, mut_indices):
        if len(mut_indices)==0:
            mut_embedding_indices = torch.tensor([])
            return {"mut": mut_embedding_indices}
        else:
            mut_embedding_indices = sorted(list(set(sum([[mut] for mut in mut_indices], []))))
            return {"mut": torch.tensor(mut_embedding_indices)}

    def get_mut2gene(self, mut_indices, type_indices=None):
        embeddings = self.get_mut2gene_embeddings(mut_indices)
        mask = self.get_mut2gene_mask(embeddings['mut'], type_indices=type_indices)
        return {"mut": embeddings['mut'], "mask": mask}