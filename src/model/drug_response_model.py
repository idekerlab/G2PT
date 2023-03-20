import torch.nn as nn

from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import System2Target

class DrugResponseModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, genotypes, hidden_dims, compound_encoder, dropout=0.2):
        super(DrugResponseModel, self).__init__(tree_parser, genotypes, hidden_dims, dropout=dropout)

        self.compound_encoder = compound_encoder
        self.compound_mapper_1 = nn.Linear(compound_encoder.hidden_layers[-1], hidden_dims)
        self.compound_norm = nn.LayerNorm(hidden_dims)
        self.compound_mapper_2 = nn.Linear(hidden_dims, hidden_dims)

        self.comp2sys_sys_norm = nn.LayerNorm(hidden_dims)
        self.comp2sys_comp_norm = nn.LayerNorm(hidden_dims)

        self.prediction_norm_inner = nn.LayerNorm(hidden_dims)
        self.prediction_norm_outer = nn.LayerNorm(hidden_dims)

        self.sys2comp = System2Target(hidden_dims, 1, hidden_dims*4, inner_norm=self.prediction_norm_inner,
                                             outer_norm=self.prediction_norm_outer,  dropout=dropout, transform=True)

        self.drug_response_predictor = nn.Linear(hidden_dims, 1)


    def forward(self, genotype_dict, compound, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward, comp2sys_masks=None):
        system_embedding = super(DrugResponseModel, self).forward(genotype_dict, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward)

        compound_embedding = self.get_compound_embedding(compound, unsqueeze=True)

        compound_weighted_by_system = self.get_comp2system(compound_embedding, system_embedding, system_maks=comp2sys_masks)
        compound_attended = compound_weighted_by_system
        drug_response_prediction = self.drug_response_predictor(compound_attended)

        return drug_response_prediction


    def get_compound_embedding(self, compound, unsqueeze=True): 
        compound_embedding = self.compound_encoder(compound)
        compound_embedding = self.activation(self.compound_mapper_1(compound_embedding))
        #compound_embedding = self.compound_mapper_1(compound_embedding)
        compound_embedding = self.compound_mapper_2(self.compound_norm(self.activation(compound_embedding)))

        if unsqueeze:
            compound_embedding = compound_embedding.unsqueeze(1)

        return compound_embedding

    def get_comp2system(self, compound_embedding, system_embedding, system_maks=None, attention=False):
        #compound_embedding = self.comp2sys_comp_norm(compound_embedding)
        #system_embedding = self.comp2sys_sys_norm(system_embedding)
        #system_embedding = self.comp2sys_norm_1(self.activation(self.system_mapper_1(system_embedding)))
        #system_embedding = self.comp2sys_norm_2(self.system_mapper_2(system_embedding))

        comp2sys_result = self.sys2comp(compound_embedding, system_embedding, system_embedding, mask=system_maks)
        if attention:
            comp2sys_attention = self.sys2comp.get_attention(compound_embedding, system_embedding, system_embedding)
            return comp2sys_result, comp2sys_attention
        else:
            return comp2sys_result











