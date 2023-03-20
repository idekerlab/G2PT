import torch
import torch.nn as nn

from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import System2Target


class G2PModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, genotypes, hidden_dims, ontology_adj, dropout=0.2):
        super(G2PModel, self).__init__(tree_parser, genotypes, hidden_dims, ontology_adj, dropout=dropout)

        phenotype_vector = torch.empty(1, hidden_dims)
        nn.init.xavier_normal(phenotype_vector)
        self.phenotype_vector = nn.Parameter(phenotype_vector)

        self.comp2sys_sys_norm = nn.LayerNorm(hidden_dims)
        self.comp2sys_comp_norm = nn.LayerNorm(hidden_dims)

        self.prediction_norm_inner = nn.LayerNorm(hidden_dims)
        self.prediction_norm_outer = nn.LayerNorm(hidden_dims)

        self.system2phenotype = System2Target(hidden_dims, 1, hidden_dims*4, inner_norm=self.prediction_norm_inner,
                                             outer_norm=self.prediction_norm_outer,  dropout=dropout, transform=True)

        self.phenotype_predictor = nn.Linear(hidden_dims, 1)

    def forward(self, genotype_dict, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward, phenotype_masks=None):
        system_embedding = super(G2PModel, self).forward(genotype_dict, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward)
        phenotype_weighted_by_system = self.get_system2phenotype(system_embedding, system_mask=phenotype_masks)
        phenotype_prediction = self.phenotype_predictor(phenotype_weighted_by_system)

        return phenotype_prediction


    def get_system2phenotype(self, system_embedding, system_mask=None, attention=False):
        batch_size = system_embedding.size(0)
        phenotype_vector = self.phenotype_vector.unsqueeze(0).expand(batch_size, -1, -1)
        sys2phenotype_result = self.system2phenotype.forward(phenotype_vector, system_embedding, system_embedding, mask=system_mask, norm=True)
        if attention:
            sys2phenotype_attention = self.system2phenotype.get_attention(phenotype_vector, system_embedding, system_embedding, norm=True)
            return sys2phenotype_result, sys2phenotype_attention
        else:
            return sys2phenotype_result