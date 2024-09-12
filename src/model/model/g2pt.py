import torch
import torch.nn as nn
import scipy as sp
import numpy as np

from src.model.hierarchical_transformer import HierarchicalTransformer


class Genotype2PhenotypeTransformer(nn.Module):

    def __init__(self, tree_parser, hidden_dims, subtree_order=('default', ), dropout=0.2, activation='softmax'):
        super(Genotype2PhenotypeTransformer, self).__init__()
        self.hidden_dims = hidden_dims
        self.tree_parser = tree_parser
        self.n_systems = self.tree_parser.n_systems
        self.n_genes = self.tree_parser.n_genes
        print("Model is initialized with %d systems and %d gene mutations" % (self.n_systems, self.n_genes))

        print('\n')
        print("self.n_systems:", self.n_systems)
        print("hidden_dims:", hidden_dims)
        print('\n')
        self.system_embedding = nn.Embedding(self.n_systems+1, hidden_dims, padding_idx=self.n_systems)
        self.gene_embedding = nn.Embedding(self.n_genes+1, hidden_dims, padding_idx=self.n_genes)

        self.mut_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.mut_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)  # LayerNormNormedScaleOnly(hidden_dims)

        self.sys2env_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2env_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)  # LayerNormNormedScaleOnly(hidden_dims)

        self.env2sys_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.env2sys_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)  # LayerNormNormedScaleOnly(hidden_dims)

        self.sys2gene_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2gene_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)

        self.norm_channel_first = False

        self.sys2env = nn.ModuleList([HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                              self.sys2env_update_norm_inner,
                                                              self.sys2env_update_norm_outer,
                                                              dropout, norm_channel_first=self.norm_channel_first,
                                                              conv_type='system', activation='softmax') for subtree in
                                      subtree_order])
        self.env2sys = nn.ModuleList([HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                              self.env2sys_update_norm_inner,
                                                              self.env2sys_update_norm_outer,
                                                              dropout, norm_channel_first=self.norm_channel_first,
                                                              conv_type='system', activation='softmax') for subtree in
                                      subtree_order])
        self.effect_norm = nn.LayerNorm(hidden_dims)
        self.sys2gene = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                      self.sys2gene_update_norm_inner, self.sys2gene_update_norm_outer,
                                                      dropout, norm_channel_first=self.norm_channel_first,
                                                      conv_type='system', activation='softmax')
        self.gene_norm = nn.LayerNorm(hidden_dims)
        self.sys_norm = nn.LayerNorm(hidden_dims)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def get_gene2sys(self, system_embedding, gene_embedding, gene2sys_mask):
        system_embedding_input = self.sys_norm(system_embedding)
        gene_embedding_input_input = self.gene_norm(gene_embedding)
        system_effect = self.gene2sys.forward(system_embedding_input, gene_embedding_input_input, gene2sys_mask)
        return system_embedding,  system_effect

    def get_sys2gene(self, gene_embedding, system_embedding, sys2gene_mask):
        gene_embedding_input = self.gene_norm(gene_embedding)
        system_embedding_input = self.sys_norm(system_embedding)
        system_effect = self.sys2gene.forward(gene_embedding_input, system_embedding_input, sys2gene_mask)
        return gene_embedding,  system_effect

    def get_sys2sys(self, system_embedding, nested_hierarchical_masks, direction='forward', return_updates = True, with_indices=False, update_tensor=None):
        batch_size = system_embedding.size(0)
        feature_size = system_embedding.size(2)
        system_embedding_output = torch.clone(system_embedding)
        update_result = torch.zeros_like(system_embedding)
        if direction=='forward':
            sys2sys = self.sys2env
        else:
            sys2sys = self.env2sys
        for hierarchical_masks, hitr in zip(nested_hierarchical_masks, sys2sys):
            for hierarchical_mask in hierarchical_masks:
                if with_indices:
                    system_embedding_queries = self.system_embedding(hierarchical_mask['query']).unsqueeze(0).expand(batch_size, -1, -1)
                    system_embedding_keys = self.system_embedding(hierarchical_mask['key']).unsqueeze(0).expand(batch_size, -1, -1)
                    system_effect_queries = torch.index_select(update_tensor, 1,
                                                         hierarchical_mask['query'].to(torch.int64))
                    system_effect_keys = torch.index_select(update_tensor, 1,
                                                      hierarchical_mask['key'].to(torch.int64))
                    system_embedding_queries = system_embedding_queries + system_effect_queries
                    system_embedding_keys = system_embedding_keys + system_effect_keys
                    system_embedding_queries = self.sys_norm(system_embedding_queries)
                    system_embedding_keys = self.sys_norm(system_embedding_keys)
                    mask = hierarchical_mask['mask']
                else:
                    system_embedding_queries = self.sys_norm(system_embedding_output)
                    system_embedding_keys = self.sys_norm(system_embedding_output)
                    mask = hierarchical_mask


                hitr_result = hitr.forward(system_embedding_queries, system_embedding_keys, mask)

                if with_indices:
                    results = []
                    for b, value in enumerate(update_tensor):
                        results.append(
                            update_tensor[b].index_add(0, hierarchical_mask['query'], hitr_result[b]))
                    system_effect = torch.stack(results, dim=0)

                    #update_tensor = update_tensor + system_effect
                    #update_result = update_result + system_effect
                    update_tensor = update_tensor + self.effect_norm(system_effect)
                    update_result = update_result + self.effect_norm(system_effect)
                else:
                    system_embedding_output = system_embedding_output + self.effect_norm(hitr_result)
                    update_tensor = update_tensor + self.effect_norm(hitr_result)
                    update_result = update_result + self.effect_norm(hitr_result)
        if return_updates:
            # return original system embeddings and update tensor separately
            return system_embedding, update_result
        else:
            # return original system embeddings + update
            if with_indices:
                return system_embedding + update_tensor
            else:
                return system_embedding_output
