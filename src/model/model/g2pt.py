import torch
import torch.nn as nn
import scipy as sp
import numpy as np

from src.model.hierarchical_transformer import HierarchicalTransformer


class Genotype2PhenotypeTransformer(nn.Module):

    def __init__(self, tree_parser, genotypes, hidden_dims, dropout=0.2, ):
        super(Genotype2PhenotypeTransformer, self).__init__()
        self.hidden_dims = hidden_dims
        self.tree_parser = tree_parser
        self.n_systems = self.tree_parser.n_systems
        self.n_genes = self.tree_parser.n_genes
        self.genotypes = genotypes

        print("Model is initialized with %d systems and %d gene mutations" % (self.n_systems, self.n_genes))
        print("Model will consider mutation types;", self.genotypes)

        self.system_embedding = nn.Embedding(self.n_systems, hidden_dims)
        self.gene_embedding = nn.Embedding(self.n_genes, hidden_dims)

        self.mut_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.mut_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)  # LayerNormNormedScaleOnly(hidden_dims)

        self.sys2cell_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2cell_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)  # LayerNormNormedScaleOnly(hidden_dims)

        self.cell2sys_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.cell2sys_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)  # LayerNormNormedScaleOnly(hidden_dims)

        self.sys2gene_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2gene_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)

        self.norm_channel_first = False

        self.mut2sys = nn.ModuleDict({genotype: HierarchicalTransformer(hidden_dims, 4,
                                                                        hidden_dims * 4, self.mut_update_norm_inner, self.mut_update_norm_outer, dropout,
                                                                        norm_channel_first=self.norm_channel_first, conv_type='genotype')
                                             for genotype in self.genotypes})

        self.sys2env = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4, self.sys2cell_update_norm_inner, self.sys2cell_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='system')
        self.env2sys = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                               self.cell2sys_update_norm_inner, self.cell2sys_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first,
                                               conv_type='system')
        self.sys2gene = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                      self.sys2gene_update_norm_inner, self.sys2gene_update_norm_outer,
                                                      dropout, norm_channel_first=self.norm_channel_first,
                                                      conv_type='system')
        self.gene_norm = nn.LayerNorm(hidden_dims)
        self.sys_norm = nn.LayerNorm(hidden_dims)


        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, genotype_dict, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                sys2gene_mask=None, gene_weight=None, sys2cell=True, cell2sys=True, sys2gene=True):
        batch_size = genotype_dict[self.genotypes[0]].size(0)

        system_embedding = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        gene_embedding = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        system_embedding, mutation_effect = self.get_mut2system(system_embedding, gene_embedding, genotype_dict, )
        if sys2cell:
            system_embedding = self.get_system2system(system_embedding, nested_hierarchical_masks_forward, direction='forward', return_updates=False)
        if cell2sys:
            system_embedding = self.get_system2system(system_embedding, nested_hierarchical_masks_backward, direction='backward', return_updates=False)
        if sys2gene:
            gene_embedding = self.get_system2gene(system_embedding, gene_embedding, sys2gene_mask)
        return system_embedding, gene_embedding


    def get_mut2system(self, system_embedding, mut_embedding, genotype_dict):
        #system_embedding = self.dropout(system_embedding/torch.norm(system_embedding, p=2, dim=-1, keepdim=True))
        #mut_embedding = self.dropout(mut_embedding/torch.norm(mut_embedding, p=2, dim=-1, keepdim=True))
        mutation_effects = {}
        #system_embedding_input = self.dropout(self.sys_norm(system_embedding+self.system_positional_embedding))
        system_embedding_input = self.dropout(self.sys_norm(system_embedding))
        mut_embedding_input = self.dropout(self.gene_norm(mut_embedding))

        for genotype in self.genotypes:
            mutation_effect = self.mut2sys[genotype].forward(system_embedding_input, mut_embedding_input, genotype_dict[genotype])
            mutation_effects[genotype] = mutation_effect
        for genotype in self.genotypes:
            system_embedding = system_embedding + mutation_effects[genotype]
        return system_embedding, mutation_effects

    def get_system2gene(self, system_embedding, gene_embedding, sys2gene_mask):
        system_effect = self.sys2gene.forward(gene_embedding, system_embedding, sys2gene_mask)
        return gene_embedding + system_effect

    def get_system2system(self, system_embedding, nested_hierarchical_masks, direction='forward', return_updates = True):
        #system_embedding = system_embedding
        updated_systems = []
        system_updates = []
        for hierarchical_masks in nested_hierarchical_masks:
            updated_systems_i = []
            system_update_i = []
            for hierarchical_mask in hierarchical_masks:
                if self.norm_channel_first:
                    system_embedding_input = system_embedding
                    system_embedding_input = system_embedding_input.transpose(-1, -2)
                    system_embedding_input = self.sys_norm(system_embedding_input)
                    system_embedding_input = self.dropout(system_embedding_input.transpose(-1, -2))
                else:
                    system_embedding_input = self.sys_norm(system_embedding)
                    system_embedding_input = self.dropout(system_embedding_input)
                if direction=='forward':
                    hitr_result = self.sys2env.forward(system_embedding_input, system_embedding_input, hierarchical_mask)
                else:
                    hitr_result = self.env2sys.forward(system_embedding_input, system_embedding_input, hierarchical_mask)
                system_embedding = system_embedding + hitr_result
                if return_updates:
                    updated_systems_i.append(system_embedding)
                    system_update_i.append(hitr_result)
            updated_systems.append(updated_systems_i)
            system_updates.append(system_update_i)
        if return_updates:
            return updated_systems, system_updates
        else:
            return system_embedding


