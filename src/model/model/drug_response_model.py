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
        self.predict_norm_genes_systems = nn.LayerNorm(hidden_dims * 2, eps=0.1)
        self.predictor_genes_systems = nn.Linear(hidden_dims*2, 1)
        self.predict_norm_systems = nn.LayerNorm(hidden_dims, eps=0.1)
        self.predictor_systems = nn.Linear(hidden_dims, 1)

        self.mut_norm = nn.LayerNorm(hidden_dims)


    def forward(self, genotype_dict, compound, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                gene2sys_mask, sys2gene_mask, comp2sys_masks=None, sys2cell=True, cell2sys=True, sys2gene=True, 
                gene2drug=True, mut2gene=False, with_indices=False, score=False, attention=False):
        batch_size = compound.size(0)
        if mut2gene:
            gene_embedding = self.get_mut2gene(genotype_dict, with_indices=with_indices, batch_size=batch_size)
            system_embedding = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
            system_embedding, gene_effect = self.get_gene2sys(system_embedding, gene_embedding, gene2sys_mask)
            system_embedding = system_embedding + self.effect_norm(gene_effect)
        else:
            system_embedding = self.get_mut2system(genotype_dict, with_indices=with_indices, batch_size=batch_size)
            gene_embedding = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
        if sys2cell:
            system_embedding = self.get_sys2sys(system_embedding, nested_hierarchical_masks_forward, direction='forward', 
                                                return_updates=False, with_indices=False)
        if cell2sys:
            system_embedding = self.get_sys2sys(system_embedding, nested_hierarchical_masks_backward, direction='backward', 
                                                return_updates=False, with_indices=False)
        if sys2gene:
            gene_embedding, system_effect_on_gene = self.get_sys2gene(gene_embedding, system_embedding, sys2gene_mask)
            gene_embedding = gene_embedding + self.effect_norm(system_effect_on_gene)
        compound_embedding = self.get_compound_embedding(compound, unsqueeze=True)
        if gene2drug:
            prediction = self.prediction(self.predictor_genes_systems, compound_embedding, system_embedding, gene_embedding)
        else:
            prediction = self.prediction(self.predictor_systems, compound_embedding, system_embedding)
        if attention:
            if score:
                system_embedding, system_attention, system_score = self.get_system2comp(compound_embedding, system_embedding, 
                                                                                     attention=True, score=True)
                gene_embedding, gene_attention, gene_score = self.get_gene2comp(compound_embedding, gene_embedding, 
                                                                                attention=True, score=True)
                return prediction, system_attention, gene_attention, system_score, gene_score
            else:
                system_embedding, system_attention = self.get_system2comp(compound_embedding, system_embedding,
                                                                       attention=True, score=False)
                gene_embedding, gene_attention = self.get_gene2comp(comound_embedding, gene_embedding, 
                                                                    attention=True, score=False)
                return prediction, system_attention, gene_attention
        else:
            if score:
                system_embedding, system_score = self.get_system2comp(compound_embedding, system_embedding, 
                                                                   attention=False, score=True)
                gene_embedding, gene_score = self.get_gene2comp(compound_embedding, gene_embedding, 
                                                                attention=False, score=True)
                return prediction, system_score, gene_score
            else:
                return prediction

    def get_mut2gene(self, genotype_dict, with_indices=False, batch_size=1):
        mutation_embedding = self.mut_norm(self.mutation_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :])
        gene_embedding = self.gene_norm(self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :])
        if with_indices:
            gene_indices = {}
            for genotype in self.genotypes:
                gene_indices_genotype, mut_effects_genotype =  self.get_gene_effect(genotype_dict[genotype], self.mut2gene[genotype])
                mut_effect = torch.zeros_like(self.gene_embedding.weight).unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
                results = []
                for b, value in enumerate(mut_effect):
                    effect = mut_effect[b].index_add(0, gene_indices_genotype[b], mut_effects_genotype[b])
                    results.append(effect)
                mut_effect = torch.stack(results, dim=0)
                gene_embedding = gene_embedding + self.effect_norm(mut_effect)

            return gene_embedding
        else:
            gene_mask = self.dropout(torch.ones_like(genotype_dict[self.genotypes[0]].sum(0).sum(0)))
            gene_mask = gene_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_genes, -1)
            for genotype in self.genotypes:
                mut_effect = self.mut2gene[genotype].forward(gene_embedding, mutation_embedding,
                                                             genotype_dict[genotype]*gene_mask, dropout=False)
                gene_embedding = gene_embedding + self.effect_norm(mut_effect)
            
            return gene_embedding

    def get_mut2system(self, genotype_dict, with_indices=False, batch_size=1):
        gene_embedding = self.gene_norm(self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :])
        system_embedding = self.sys_norm(self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :])
        if with_indices:
            sys_indices = {}
            for genotype in self.genotypes:
                sys_indices_genotype, mut_effects_genotype =  self.get_sys_effect(genotype_dict[genotype], self.mut2sys[genotype])
                mut_effect = torch.zeros_like(self.sys_embedding.weight).unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
                results = []
                for b, value in enumerate(mut_effect):
                    effect = mut_effect[b].index_add(0, sys_indices_genotype[b], mut_effects_genotype[b])
                    results.append(effect)
                mut_effect = torch.stack(results, dim=0)
                system_embedding = system_embedding + self.effect_norm(mut_effect)

            return system_embedding
        else:
            gene_mask = self.dropout(torch.ones_like(genotype_dict[self.genotypes[0]].sum(0).sum(0)))
            gene_mask = gene_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, self.n_systems, -1)
            for genotype in self.genotypes:
                mut_effect = self.mut2sys[genotype].forward(system_embedding, gene_embedding, 
                                                            genotype_dict[genotype]*gene_mask, dropout=False)
                system_embedding = system_embedding + self.effect_norm(mut_effect)
        
            return system_embedding
    
    def get_gene_effect(self, genotype, transformer):
        mut_indices = genotype['mut']
        if len(mut_indices) == 0:
            return None, None
        mutation_embedding = self.mut_norm(self.mutation_embedding(mut_indices))
        gene_embedding = self.gene_norm(self.gene_embedding(mut_indices))
        mask = genotype['mask']
        mut_effect_from_embedding = transformer(gene_embedding, mutation_embedding, mask)
        return mut_indices, mut_effect_from_embedding

    def get_sys_effect(self, genotype, transformer):
        gene_indices = genotype['gene']
        sys_indices = genotype['sys']
        if len(sys_indices) == 0:
            return None, None
        gene_embedding = self.gene_norm(self.gene_embedding(gene_indices))
        sys_embedding = self.sys_norm(self.system_embedding(sys_indices))
        mask = genotype['mask']
        gene_effect_from_embedding = transformer(sys_embedding, gene_embedding, mask)
        return sys_indices, gene_effect_from_embedding

    def prediction(self, predictor, compound_embedding, system_embedding, gene_embedding=None):
        drug_weighted_by_systems = self.get_system2comp(compound_embedding, system_embedding, system_mask=None)
        if gene_embedding != None:
            drug_weighted_by_genes = self.get_gene2comp(compound_embedding, gene_embedding, gene_mask=None)
            drug_attended = torch.cat([drug_weighted_by_systems, drug_weighted_by_genes], dim=-1)
            drug_prediction = predictor(self.predict_norm_genes_systems(drug_attended))
        else:
            drug_prediction = predictor(self.predict_norm_systems(drug_weighted_by_systems))
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










