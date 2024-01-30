import torch
import torch.nn as nn
import scipy as sp
import numpy as np

from src.model.hierarchical_transformer import HierarchicalTransformer


class Genotype2PhenotypeTransformer(nn.Module):

    def __init__(self, tree_parser, genotypes, hidden_dims, dropout=0.2, activation='softmax'):
        super(Genotype2PhenotypeTransformer, self).__init__()
        self.hidden_dims = hidden_dims
        self.tree_parser = tree_parser
        self.n_systems = self.tree_parser.n_systems
        self.n_genes = self.tree_parser.n_genes
        self.genotypes = genotypes

        print("Model is initialized with %d systems and %d gene mutations" % (self.n_systems, self.n_genes))
        print("Model will consider mutation types;", self.genotypes)

        self.system_embedding = nn.Embedding(self.n_systems+1, hidden_dims, padding_idx=self.n_systems)
        self.gene_embedding = nn.Embedding(self.n_genes+1, hidden_dims, padding_idx=self.n_genes)

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
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='system', activation=activation)
        self.env2sys = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                               self.cell2sys_update_norm_inner, self.cell2sys_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first,
                                               conv_type='system', activation=activation)
        self.sys2gene = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                      self.sys2gene_update_norm_inner, self.sys2gene_update_norm_outer,
                                                      dropout, norm_channel_first=self.norm_channel_first,
                                                      conv_type='system', activation=activation)
        self.gene_norm = nn.LayerNorm(hidden_dims)
        self.sys_norm = nn.LayerNorm(hidden_dims)


        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()


    def get_sys2gene(self, system_embedding, gene_embedding, sys2gene_mask):
        gene_embedding_input = self.gene_norm(gene_embedding)
        system_embedding = self.sys_norm(system_embedding)
        system_effect = self.sys2gene.forward(gene_embedding_input, system_embedding, sys2gene_mask)
        return gene_embedding + system_effect


    def get_sys2sys(self, system_embedding, nested_hierarchical_masks, direction='forward', return_updates = True, with_indices=False, update_tensor=None):
        system_embedding_queries = self.sys_norm(system_embedding)
        system_embedding_keys = self.sys_norm(system_embedding)
        batch_size = system_embedding_queries.size(0)
        feature_size = system_embedding_queries.size(2)
        system_embedding_output = system_embedding
        updated_systems = []
        system_updates = []
        if update_tensor is None:
            update_tensor = torch.zeros_like(system_embedding_queries)#.unsqueeze(0).expand(batch_size, -1, -1)
        i = 0
        for hierarchical_masks in nested_hierarchical_masks:
            updated_systems_j = []
            system_update_j = []
            j = 0
            for hierarchical_mask in hierarchical_masks:
                if with_indices:
                    system_embedding_queries = self.sys_norm(self.system_embedding(hierarchical_mask['query']).unsqueeze(0).expand(batch_size, -1, -1))
                    system_embedding_keys = self.sys_norm(self.system_embedding(hierarchical_mask['key']).unsqueeze(0).expand(batch_size, -1, -1))
                    system_effect_queries = torch.gather(update_tensor, 1, hierarchical_mask['query'].unsqueeze(0).unsqueeze(2).expand(batch_size, -1, feature_size).to(torch.int64))
                    system_embedding_queries = system_embedding_queries + system_effect_queries
                    system_embedding_queries = self.sys_norm(system_embedding_queries)
                    system_effect_keys = torch.gather(update_tensor, 1, hierarchical_mask['key'].unsqueeze(0).unsqueeze(2).expand(batch_size, -1, feature_size).to(torch.int64))
                    system_embedding_keys = system_embedding_keys + system_effect_keys
                    system_embedding_keys = self.sys_norm(system_embedding_keys)
                    mask = hierarchical_mask['mask']
                else:
                    mask = hierarchical_mask
                if direction=='forward':
                    #print(system_embedding_queries.size(), system_embedding_keys.size(), mask.size())
                    hitr_result = self.sys2env.forward(system_embedding_queries, system_embedding_keys, mask)
                else:
                    hitr_result = self.env2sys.forward(system_embedding_queries, system_embedding_keys, mask)
                if not with_indices:
                    system_embedding_output = system_embedding_output + hitr_result
                    system_embedding_queries = self.sys_norm(system_embedding_output)
                    system_embedding_keys = self.sys_norm(system_embedding_output)
                else:
                    results = []
                    for b, value in enumerate(update_tensor):
                        results.append(
                            update_tensor[b].index_add(0, hierarchical_mask['query'], hitr_result[b]))
                    sysetem_effect = torch.stack(results, dim=0)
                    update_tensor = update_tensor + sysetem_effect
                if return_updates:
                    updated_systems_j.append(system_embedding_output)
                    updated_systems_j.append(hitr_result)
            updated_systems.append(updated_systems_j)
            system_updates.append(updated_systems_j)
            i += 1
        if return_updates:
            return system_embedding + update_tensor, update_tensor
        else:
            if with_indices:
                return system_embedding + update_tensor
            else:
                system_embedding = system_embedding_output
                return system_embedding


