import torch.nn as nn
import torch

from src.model.utils import GlobalUpdateNorm
from src.model.tree_conv import TreeConvolution

class DrugResponseModel(nn.Module):

    def __init__(self, n_systems, n_genes, mut_types, compound_encoder, hidden_dims, dropout=0.2):
        super(DrugResponseModel, self).__init__()

        self.n_systems = n_systems
        self.n_genes = n_genes
        self.mut_types = mut_types

        print("Model is initialized with %d systems and %d gene mutations")
        print("Model will consider mutation types;", self.mut_types)

        self.norm = GlobalUpdateNorm(hidden_dims)
        self.system_tree_conv = TreeConvolution(hidden_dims, int(hidden_dims/32), hidden_dims*4, self.norm. dropout)

        self.system_embedding = nn.Embedding(n_systems, hidden_dims)
        self.mut_embedding = nn.ModuleDict({mut_type:nn.Embedding(n_genes, hidden_dims, padding_idx=n_genes)
                                            for mut_type in mut_types})
        self.mut_tree_convs = nn.ModuleDict({mut_type+"_TreeConv": TreeConvolution(hidden_dims, int(hidden_dims/32),
                                                                                   hidden_dims*4, self.norm. dropout)
                                             for mut_type in mut_types})
        self.compound_encoder = compound_encoder
        self.compound_mapper = nn.Linear(compound_encoder.hidden_layers[-1], hidden_dims)
        self.dropout = nn.Dropout(dropout)

        self.drug_response_predictor = nn.Linear(hidden_dims, 1)

    def forward(self, mut_dict, compound, hierarchical_masks):
        batch_size = compound.size(0)

        system_embedding = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        for mut_type in self.mut_types:
            mut_embedding = self.mut_embedding[mut_type].weight.unsqueeze(0).expand(batch_size, -1, -1)
            system_embedding = self.mut_tree_convs[mut_type].forward(system_embedding, mut_embedding, mut_dict)

        drug_embedding = self.norm(self.compound_mapper(self.compound_encoder(compound)))

        for hierarchical_mask in hierarchical_masks:
            system_embedding = self.system_tree_conv(system_embedding, system_embedding, hierarchical_mask)

        drug_response_prediction = self.drug_response_predictor(system_embedding[:, 0])

        return drug_response_prediction







