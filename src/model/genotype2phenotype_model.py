import torch
import torch.nn as nn

from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import System2Phenotype


class G2PModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, genotypes, hidden_dims, dropout=0.2):
        super(G2PModel, self).__init__(tree_parser, genotypes, hidden_dims, dropout=dropout)

        phenotype_vector = torch.empty(1, hidden_dims)
        nn.init.xavier_normal(phenotype_vector)
        self.phenotype_vector = nn.Parameter(phenotype_vector)

        self.comp2sys_sys_norm = nn.LayerNorm(hidden_dims)
        self.comp2sys_comp_norm = nn.LayerNorm(hidden_dims)

        self.sys2pheno_norm_inner = nn.LayerNorm(hidden_dims)
        self.sys2pheno_norm_outer = nn.LayerNorm(hidden_dims)

        self.gene2pheno_norm_inner = nn.LayerNorm(hidden_dims)
        self.gene2pheno_norm_outer = nn.LayerNorm(hidden_dims)

        self.system2phenotype = System2Phenotype(hidden_dims, 1, hidden_dims * 4, inner_norm=self.sys2pheno_norm_inner,
                                                 outer_norm=self.sys2pheno_norm_outer, dropout=dropout, transform=False)
        self.gene2phenotype = System2Phenotype(hidden_dims, 1, hidden_dims*4, inner_norm=self.gene2pheno_norm_inner,
                                             outer_norm=self.gene2pheno_norm_outer, dropout=dropout, transform=False)
        self.phenotype_norm = nn.LayerNorm(hidden_dims*2)
        self.phenotype_predictor = nn.Linear(hidden_dims*2, 1)

    def forward(self, genotype_dict, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                sys2gene_mask, sys2cell=True, cell2sys=True, sys2gene=True):
        system_embedding, gene_embedding = super(G2PModel, self).forward(genotype_dict, nested_hierarchical_masks_forward,
                                                                         nested_hierarchical_masks_backward, sys2gene_mask,
                                                                        sys2cell=sys2cell, cell2sys=cell2sys, sys2gene=sys2gene)
        phenotype_weighted_by_systems = self.get_system2phenotype(system_embedding, system_mask=None)
        phenotype_weighted_by_genes = self.get_gene2phenotype(gene_embedding, gene_mask=None)

        phenotype_attended = torch.cat([phenotype_weighted_by_systems, phenotype_weighted_by_genes], dim=-1)
        phenotype_prediction = self.phenotype_predictor(phenotype_attended)

        return phenotype_prediction


    def get_system2phenotype(self, system_embedding, system_mask=None, attention=False, score=False):
        batch_size = system_embedding.size(0)
        phenotype_vector = self.phenotype_vector.unsqueeze(0).expand(batch_size, -1, -1)
        sys2phenotype_result = self.system2phenotype.forward(phenotype_vector, system_embedding, system_embedding, mask=system_mask)
        if attention:
            sys2phenotype_attention = self.system2phenotype.get_attention(phenotype_vector, system_embedding, system_embedding)
            sys2phenotype_result = [sys2phenotype_result, sys2phenotype_attention]
            if score:
                sys2phenotype_score = self.system2phenotype.get_score(phenotype_vector, system_embedding, system_embedding)
                sys2phenotype_result += [sys2phenotype_score]
            return sys2phenotype_result
        else:
            if score:
                sys2phenotype_score = self.system2phenotype.get_score(phenotype_vector, system_embedding,
                                                                      system_embedding)
                sys2phenotype_result = [sys2phenotype_result, sys2phenotype_score]
            return sys2phenotype_result


    def get_gene2phenotype(self, gene_embedding, gene_mask=None, attention=False, score=False):
        batch_size = gene_embedding.size(0)
        phenotype_vector = self.phenotype_vector.unsqueeze(0).expand(batch_size, -1, -1)
        gene2phenotype_result = self.gene2phenotype.forward(phenotype_vector, gene_embedding, gene_embedding, mask=gene_mask, norm=True)
        if attention:
            gene2phenotype_attention = self.gene2phenotype.get_attention(phenotype_vector, gene_embedding, gene_embedding)
            gene2phenotype_result = [gene2phenotype_result, gene2phenotype_attention]
            if score:
                gene2phenotype_score = self.gene2phenotype.get_score(phenotype_vector, gene_embedding, gene_embedding)
                gene2phenotype_result += [gene2phenotype_score]
            return gene2phenotype_result
        else:
            if score:
                gene2phenotype_score = self.gene2phenotype.get_score(phenotype_vector, gene_embedding,
                                                                      gene_embedding)
                gene2phenotype_result = [gene2phenotype_result, gene2phenotype_score]
            return gene2phenotype_result