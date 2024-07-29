import torch.nn as nn
import torch

from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import Genotype2Phenotype
from src.model.hierarchical_transformer import HierarchicalTransformer

class DrugResponseModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, genotypes, hidden_dims, compound_encoder, dropout=0.2, activation='softmax'):
        super(DrugResponseModel, self).__init__(tree_parser, genotypes, hidden_dims, dropout=dropout, activation=activation)


        self.compound_encoder = compound_encoder
        self.compound_mapper_1 = nn.Linear(compound_encoder.hidden_layers[-1], hidden_dims)
        self.compound_norm_1 = nn.LayerNorm(hidden_dims, eps=0.1)
        self.compound_mapper_2 = nn.Linear(hidden_dims, hidden_dims)
        self.compound_norm_2 = nn.LayerNorm(hidden_dims, eps=0.1)

        self.comp2sys_sys_norm = nn.LayerNorm(hidden_dims, eps=0.1)
        self.comp2sys_comp_norm = nn.LayerNorm(hidden_dims, eps=0.1)

        self.sys2comp_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2comp_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)

        self.gene2comp_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2comp_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)

        self.mut2sys = nn.ModuleDict({genotype: HierarchicalTransformer(hidden_dims, 4,
                                                                        hidden_dims * 4, self.mut_update_norm_inner,
                                                                        self.mut_update_norm_outer, dropout,
                                                                        norm_channel_first=self.norm_channel_first,
                                                                        conv_type='genotype')
                                      for genotype in self.genotypes})

        self.sys2comp = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4, inner_norm=self.sys2comp_norm_inner,
                                           outer_norm=self.sys2comp_norm_outer, dropout=dropout, transform=True)
        self.gene2comp = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4, inner_norm=self.gene2comp_norm_inner,
                                            outer_norm=self.gene2comp_norm_outer, dropout=dropout, transform=True)
        self.sigmoid = nn.Sigmoid()
        self.prediction_norm = nn.LayerNorm(hidden_dims * 2, eps=0.1)
        self.drug_response_predictor = nn.Linear(hidden_dims*2, 1)



    def forward(self, genotype_dict, compound, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                sys2gene_mask, comp2sys_masks=None, sys2cell=True, cell2sys=True, sys2gene=True, with_indices=False):
        batch_size = compound.size(0)
        system_embedding, mutation_effect = self.get_mut2system(genotype_dict, with_indices=with_indices, batch_size=batch_size)
        if with_indices:
            system_embedding = system_embedding[:, :-1, :]
        #print(system_embedding[0, :, 0] == system_embedding[1, :, 0])
        if sys2cell:
            system_embedding = self.get_sys2sys(system_embedding, nested_hierarchical_masks_forward, direction='forward', return_updates=False, with_indices=with_indices)
        if cell2sys:
            system_embedding = self.get_sys2sys(system_embedding, nested_hierarchical_masks_backward, direction='backward', return_updates=False, with_indices=with_indices)
        root_embedding = system_embedding[:, 0, :]
        gene_embedding = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
        if sys2gene:
            gene_embedding = self.get_sys2gene(system_embedding, gene_embedding, sys2gene_mask)
        #print(system_embedding[0, :, 0] == system_embedding[1, :, 0])
        compound_embedding = self.get_compound_embedding(compound, unsqueeze=True)
        prediction = self.prediction(self.drug_response_predictor, compound_embedding, system_embedding, gene_embedding)
        return prediction

    def get_mut2system(self, genotype_dict, with_indices=False, batch_size=1):
        if with_indices:
            sys_indices, mut_effects = [], []
            for genotype in self.genotypes:
                sys_indices_genotype, mut_effects_genotype =  self.get_sys_effect(genotype_dict[genotype], self.mut2sys[genotype])
                sys_indices.append(sys_indices_genotype)
                mut_effects.append(mut_effects_genotype)
            sys_indices = torch.cat(sys_indices, dim=-1)
            mut_effects = torch.cat(mut_effects, dim=1)

            mut_effect = torch.zeros_like(self.system_embedding.weight).unsqueeze(0).expand(batch_size, -1, -1)
            results = []
            for b, value in enumerate(mut_effect):
                effect = mut_effect[b].index_add(0, sys_indices[b], mut_effects[b])
                results.append(effect)
            mut_effect = torch.stack(results, dim=0)
            system_embedding = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

            return system_embedding + mut_effect, mut_effect
        else:
            mutation_effects = {}
            system_embedding = self.sys_norm(self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :])
            gene_embedding = self.gene_norm(self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :])
            gene_mask = self.dropout(torch.ones_like(genotype_dict[self.genotypes[0]].sum(0).sum(0)))
            gene_mask = gene_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_systems, -1)
            for genotype in self.genotypes:
                #print(system_embedding.size(), gene_embedding.size(), genotype_dict[genotype].size())
                mutation_effect = self.mut2sys[genotype].forward(system_embedding, gene_embedding,
                                                                     genotype_dict[genotype]*gene_mask, dropout=False)
                mutation_effects[genotype] = mutation_effect
            for genotype in self.genotypes:
                system_embedding = system_embedding + mutation_effects[genotype]
            return system_embedding, mutation_effects

    def get_sys_effect(self, genotype, transformer):
        gene_indices = genotype['gene']

        sys_indices = genotype['sys']
        if len(sys_indices) == 0:
            return None, None
        gene_embedding = self.gene_norm(self.gene_embedding(gene_indices))
        sys_embedding = self.sys_norm(self.system_embedding(sys_indices))
        mask = genotype['mask']
        #print(mask.sum())
        gene_effect_from_embedding = transformer(sys_embedding, gene_embedding, mask)
        return sys_indices, gene_effect_from_embedding

    def prediction(self, predictor, compound_embedding, system_embedding, gene_embedding):
        drug_weighted_by_systems = self.get_system2comp(compound_embedding, system_embedding, system_mask=None)
        drug_weighted_by_genes = self.get_gene2comp(compound_embedding, gene_embedding, gene_mask=None)
        drug_attended = torch.cat([drug_weighted_by_systems, drug_weighted_by_genes], dim=-1)
        drug_prediction = predictor(self.prediction_norm(drug_attended))
        return drug_prediction

    def get_compound_embedding(self, compound, unsqueeze=True): 
        compound_embedding = self.compound_encoder(compound)
        compound_embedding = self.activation(self.compound_mapper_1(compound_embedding))
        compound_embedding = self.compound_norm_2(self.compound_mapper_2(self.compound_norm_1(compound_embedding)))
        if unsqueeze:
            compound_embedding = compound_embedding.unsqueeze(1)
        return compound_embedding

    def get_system2comp(self, compound_embedding, system_embedding, system_mask=None, attention=False, score=False):
        system_embedding = self.sys_norm(system_embedding)
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
        gene_embedding = self.gene_norm(gene_embedding)
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










