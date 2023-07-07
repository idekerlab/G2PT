import torch.nn as nn
import torch

from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import System2Phenotype

class DrugResponseModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, genotypes, hidden_dims, compound_encoder, dropout=0.2):
        super(DrugResponseModel, self).__init__(tree_parser, genotypes, hidden_dims, dropout=dropout)

        self.compound_encoder = compound_encoder
        self.compound_mapper_1 = nn.Linear(compound_encoder.hidden_layers[-1], hidden_dims)
        self.compound_norm = nn.LayerNorm(hidden_dims)
        self.compound_mapper_2 = nn.Linear(hidden_dims, hidden_dims)

        self.comp2sys_sys_norm = nn.LayerNorm(hidden_dims)
        self.comp2sys_comp_norm = nn.LayerNorm(hidden_dims)

        self.sys2comp_norm_inner = nn.LayerNorm(hidden_dims)
        self.sys2comp_norm_outer = nn.LayerNorm(hidden_dims)

        self.gene2comp_norm_inner = nn.LayerNorm(hidden_dims)
        self.gene2comp_norm_outer = nn.LayerNorm(hidden_dims)

        self.sys2comp = System2Phenotype(hidden_dims, 1, hidden_dims * 4, inner_norm=self.sys2comp_norm_inner,
                                         outer_norm=self.sys2comp_norm_outer, dropout=dropout, transform=True)
        self.gene2comp = System2Phenotype(hidden_dims, 1, hidden_dims*4, inner_norm=self.gene2comp_norm_inner,
                                             outer_norm=self.gene2comp_norm_outer, dropout=dropout, transform=True)
        self.drug_response_predictor = nn.Linear(hidden_dims*2, 1)


    def forward(self, genotype_dict, compound, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                sys2gene_mask, comp2sys_masks=None, sys2cell=True, cell2sys=True, sys2gene=True):
        system_embedding, gene_embedding = super(DrugResponseModel, self).forward(genotype_dict, nested_hierarchical_masks_forward,
                                                                                  nested_hierarchical_masks_backward, sys2gene_mask,
                                                                                  sys2cell=sys2cell, cell2sys=cell2sys, sys2gene=sys2gene)

        compound_embedding = self.get_compound_embedding(compound, unsqueeze=True)

        compound_weighted_by_systems = self.get_system2comp(compound_embedding, system_embedding, system_mask=comp2sys_masks)
        compound_weighted_by_genes = self.get_gene2comp(compound_embedding, gene_embedding, gene_mask=None)

        compound_attended = torch.cat([compound_weighted_by_systems, compound_weighted_by_genes], dim=-1)

        drug_response_prediction = self.drug_response_predictor(compound_attended)

        return drug_response_prediction


    def get_compound_embedding(self, compound, unsqueeze=True): 
        compound_embedding = self.compound_encoder(compound)
        compound_embedding = self.activation(self.compound_mapper_1(compound_embedding))
        #compound_embedding = self.compound_mapper_1(compound_embedding)
        compound_embedding = self.compound_mapper_2(self.compound_norm(compound_embedding))

        if unsqueeze:
            compound_embedding = compound_embedding.unsqueeze(1)

        return compound_embedding

    def get_system2comp(self, compound_embedding, system_embedding, system_mask=None, attention=False, score=False):
        sys2comp_result = self.sys2comp(compound_embedding, system_embedding, system_embedding, mask=system_mask)
        if attention:
            sys2comp_attention = self.sys2comp.get_attention(compound_embedding, system_embedding, system_embedding)
            sys2comp_result = [sys2comp_result, sys2comp_attention]
            if score:
                sys2comp_score = self.sys2comp.get_score(compound_embedding, system_embedding, system_embedding)
                sys2comp_result += [sys2comp_score]
            return sys2comp_result
        else:
            if score:
                sys2comp_score = self.sys2comp.get_score(compound_embedding, system_embedding, system_embedding)
                sys2comp_result = [sys2comp_result, sys2comp_score]
            return sys2comp_result

    def get_gene2comp(self, compound_embedding, gene_embedding, attention=False, score=False, gene_mask=None):
        gene2comp_result = self.gene2comp(compound_embedding, gene_embedding, gene_embedding, mask=gene_mask)
        if attention:
            gene2comp_attention = self.gene2comp.get_attention(compound_embedding, gene_embedding, gene_embedding)
            gene2comp_result = [gene2comp_result, gene2comp_attention]
            if score:
                gene2comp_score = self.gene2comp.get_score(compound_embedding, gene_embedding, gene_embedding)
                gene2comp_result += [gene2comp_score]
            return gene2comp_result
        else:
            if score:
                gene2comp_score = self.gene2comp.get_score(compound_embedding, gene_embedding, gene_embedding)
                gene2comp_result = [gene2comp_result, gene2comp_score]
            return gene2comp_result










