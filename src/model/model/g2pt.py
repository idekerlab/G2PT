import torch
import torch.nn as nn
import scipy as sp
import numpy as np

from src.model.hierarchical_transformer import HierarchicalTransformer


class Genotype2PhenotypeTransformer(nn.Module):

    def __init__(self, tree_parser, hidden_dims, subtree_order=['default'], dropout=0.2, activation='softmax'):
        super(Genotype2PhenotypeTransformer, self).__init__()
        self.hidden_dims = hidden_dims
        self.tree_parser = tree_parser
        self.n_systems = self.tree_parser.n_systems
        self.n_genes = self.tree_parser.n_genes
        self.subtree_order = subtree_order

        print("Model is initialized with %d systems and %d genes" % (self.n_systems, self.n_genes))

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
        self.sys2env_list = nn.ModuleList([HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                                   self.sys2cell_update_norm_inner,
                                                                   self.sys2cell_update_norm_outer,
                                                                   dropout, norm_channel_first=self.norm_channel_first,
                                                                   conv_type='system', activation=activation) for
                                           subtree in self.subtree_order])
        # note that env2sys list is reversed from subtree order
        self.env2sys_list = nn.ModuleList([HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                                   self.sys2cell_update_norm_inner,
                                                                   self.sys2cell_update_norm_outer,
                                                                   dropout, norm_channel_first=self.norm_channel_first,
                                                                   conv_type='system', activation=activation) for
                                           subtree in self.subtree_order])
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


    def get_sys2sys(self, system_embedding, nested_hierarchical_masks, direction='forward', return_updates = True, with_indices=False, update_tensor=0):
        system_embedding_queries = self.sys_norm(system_embedding)
        system_embedding_keys = self.sys_norm(system_embedding)
        batch_size = system_embedding_queries.size(0)
        feature_size = system_embedding_queries.size(2)
        system_embedding_output = torch.clone(system_embedding)
        i = 0
        for hierarchical_masks in nested_hierarchical_masks:
            for hierarchical_mask, sys2env, env2sys in zip(hierarchical_masks, self.sys2env_list, self.env2sys_list):
                if with_indices:
                    system_embedding_queries = self.system_embedding(hierarchical_mask['query'].unsqueeze(0).expand(batch_size, -1, -1))
                    system_embedding_keys = self.system_embedding(hierarchical_mask['key'].unsqueeze(0).expand(batch_size, -1, -1))
                    system_effect_queries = torch.gather(update_tensor, 1, hierarchical_mask['query'].unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1).to(torch.int64))
                    system_effect_keys = torch.gather(update_tensor, 1, hierarchical_mask['key'].unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1).to(torch.int64))
                    system_embedding_queries = system_embedding_queries + system_effect_queries
                    system_embedding_keys = system_embedding_keys + system_effect_keys
                    system_embedding_queries = self.sys_norm(system_embedding_queries)
                    system_embedding_keys = self.sys_norm(system_embedding_keys)
                    mask = hierarchical_mask['mask']
                else:
                    mask = hierarchical_mask
                if direction=='forward':
                    #print(system_embedding_queries.size(), system_embedding_keys.size())
                    hitr_result = self.sys2env.forward(system_embedding_queries, system_embedding_keys, mask)
                else:
                    hitr_result = self.env2sys.forward(system_embedding_queries, system_embedding_keys, mask)
                if not with_indices:
                    results = []
                    for b, value in enumerate(update_tensor):
                        results.append(
                            update_tensor[b].index_add(0, hierarchical_mask['query'], hitr_result[b]))
                    sysetem_effect = torch.stack(results, dim=0)
                    update_tensor = update_tensor + sysetem_effect
                else:
                    system_embedding_output = system_embedding_output + hitr_result
                    system_embedding_queries = self.sys_norm(system_embedding_output)
                    system_embedding_keys = self.sys_norm(system_embedding_output)
                    update_tensor += hitr_result
        if return_updates:
            if with_indices:
                return system_embedding + update_tensor, update_tensor
            else:
                return system_embedding_output, update_tensor
        else:
            if with_indices:
                return system_embedding + update_tensor
            else:
                system_embedding = system_embedding_output
                return system_embedding


