import torch.nn as nn
import torch

from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import Genotype2Phenotype
from src.model.hierarchical_transformer import HierarchicalTransformer

class DrugResponseModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, genotypes, hidden_dims, compound_encoder, dropout=0.2, activation='softmax'):
        super(DrugResponseModel, self).__init__(tree_parser, hidden_dims, dropout=dropout, activation=activation)

        self.genotypes = genotypes
        self.n_genes = self.tree_parser.n_genes
        self.mutation_embedding = nn.Embedding(self.n_genes+1, hidden_dims, padding_idx=self.n_genes)
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

        self.mut2gene = nn.ModuleDict({genotype: HierarchicalTransformer(hidden_dims, 4,
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
        self.prediction_norm = nn.LayerNorm(hidden_dims, eps=0.1)
        self.drug_response_predictor = nn.Linear(hidden_dims, 1)

        self.mut_norm = nn.LayerNorm(hidden_dims)


    def forward(self, genotype_dict, compound, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                gene2sys_mask, sys2gene_mask, comp2sys_masks=None, sys2cell=True, cell2sys=True, sys2gene=True, with_indices=False):
        batch_size = compound.size(0)
        gene_embedding, mutation_effects = self.get_mut2gene(genotype_dict, with_indices=with_indices, batch_size=batch_size)
        for genotype in self.genotypes:
            gene_embedding = gene_embedding + self.effect_norm(mutation_effects[genotype])
        system_embedding = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
        system_embedding, gene_effect = self.get_gene2sys(system_embedding, gene_embedding, gene2sys_mask)
        system_embedding = system_embedding + self.effect_norm(gene_effect)
        total_update = self.effect_norm(gene_effect)
        if sys2cell:
            system_embedding, system_effect_forward = self.get_sys2sys(system_embedding,
                                                                       nested_hierarchical_masks_forward,
                                                                       direction='forward', return_updates=True,
                                                                       with_indices=False,
                                                                       update_tensor=total_update)
            total_update = total_update + system_effect_forward
            if not cell2sys:
                system_embedding = system_embedding + total_update
                
        if cell2sys:
            system_embedding, system_effect_backward = self.get_sys2sys(system_embedding,
                                                                        nested_hierarchical_masks_backward,
                                                                        direction='backward', return_updates=True,
                                                                        with_indices=False,
                                                                        update_tensor=total_update)
            total_update = total_update + system_effect_backward
            system_embedding = system_embedding + total_update
        if sys2gene:
            gene_embedding, system_effect_on_gene = self.get_sys2gene(gene_embedding, system_embedding, sys2gene_mask)
            gene_embedding = gene_embedding + self.effect_norm(system_effect_on_gene)
        compound_embedding = self.get_compound_embedding(compound, unsqueeze=True)
        prediction = self.prediction(self.drug_response_predictor, compound_embedding, system_embedding)
        return prediction

    def get_mut2gene(self, genotype_dict, with_indices=False, batch_size=1):
        if with_indices:
            gene_indices, mutation_effects = {}, {}
            for genotype in self.genotypes:
                gene_indices_genotype, mut_effects_genotype =  self.get_gene_effect(genotype_dict[genotype], self.mut2gene[genotype])
                mut_effect = torch.zeros_like(self.gene_embedding.weight).unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
                results = []
                for b, value in enumerate(mut_effect):
                    effect = mut_effect[b].index_add(0, gene_indices_genotype[b], mut_effects_genotype[b])
                    results.append(effect)
                mut_effect = torch.stack(results, dim=0)
                mutation_effects[genotype] = mut_effect
            
            gene_embedding = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]

            return gene_embedding, mutation_effects
        else:
            mutation_effects = {}
            mutation_embedding = self.mut_norm(self.mutation_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :])
            gene_embedding = self.gene_norm(self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :])
            gene_mask = self.dropout(torch.ones_like(genotype_dict[self.genotypes[0]].sum(0).sum(0)))
            gene_mask = gene_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_genes, -1)
            for genotype in self.genotypes:
                mutation_effect = self.mut2gene[genotype].forward(gene_embedding, mutation_embedding,
                                                                     genotype_dict[genotype]*gene_mask, dropout=False)
                mutation_effects[genotype] = mutation_effect
            
            return gene_embedding, mutation_effects
    
    def get_gene_effect(self, genotype, transformer):
        mut_indices = genotype['mut']
        if len(mut_indices) == 0:
            return None, None
        mutation_embedding = self.mut_norm(self.mutation_embedding(mut_indices))
        gene_embedding = self.gene_norm(self.gene_embedding(mut_indices))
        mask = genotype['mask']
        mut_effect_from_embedding = transformer(gene_embedding, mutation_embedding, mask)
        return mut_indices, mut_effect_from_embedding

    def prediction(self, predictor, compound_embedding, system_embedding):
        drug_weighted_by_systems = self.get_system2comp(compound_embedding, system_embedding, system_mask=None)
        #drug_weighted_by_genes = self.get_gene2comp(compound_embedding, gene_embedding, gene_mask=None)
        #drug_attended = torch.cat([drug_weighted_by_systems, drug_weighted_by_genes], dim=-1)
        #drug_prediction = predictor(self.prediction_norm(drug_attended))
        drug_prediction = predictor(self.prediction_norm(drug_weighted_by_systems))
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










