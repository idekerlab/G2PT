import torch
import torch.nn as nn
import scipy as sp
import numpy as np

from src.model.utils import PoincareNorm

from src.model.hierarchical_transformer import HierarchicalTransformer


class Genotype2PhenotypeTransformer(nn.Module):

    def __init__(self, tree_parser, hidden_dims, interaction_types=('default', ), dropout=0.2, activation='softmax',
                 input_format='indices', poincare=False):
        super(Genotype2PhenotypeTransformer, self).__init__()
        self.input_format = input_format
        self.hidden_dims = hidden_dims
        self.tree_parser = tree_parser
        self.n_systems = self.tree_parser.n_systems
        self.n_genes = self.tree_parser.n_genes
        self.interaction_types = interaction_types
        print("Model is initialized with %d systems and %d gene mutations" % (self.n_systems, self.n_genes))
        self.poincare = poincare

        self.system_embedding = nn.Embedding(int(np.ceil((self.tree_parser.n_systems + 1) / 8) * 8), hidden_dims, padding_idx=self.n_systems)
        self.gene_embedding = nn.Embedding(int(np.ceil((self.tree_parser.n_genes + 1) / 8) * 8), hidden_dims, padding_idx=self.n_genes)

        self.mut_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.mut_update_norm_outer = nn.LayerNorm(hidden_dims)  # LayerNormNormedScaleOnly(hidden_dims)

        self.sys2env_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.sys2env_update_norm_outer = nn.LayerNorm(hidden_dims)  # LayerNormNormedScaleOnly(hidden_dims)

        self.env2sys_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.env2sys_update_norm_outer = nn.LayerNorm(hidden_dims)  # LayerNormNormedScaleOnly(hidden_dims)

        self.sys2gene_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.sys2gene_update_norm_outer = nn.LayerNorm(hidden_dims)

        self.norm_channel_first = False

        self.sys2env = nn.ModuleDict({interaction_type:HierarchicalTransformer(hidden_dims, 4, hidden_dims,
                                                              self.sys2env_update_norm_inner,
                                                              self.sys2env_update_norm_outer,
                                                              dropout, norm_channel_first=self.norm_channel_first,
                                                              conv_type='system', activation='softmax', poincare=poincare)
                                      for interaction_type in
                                      interaction_types})
        self.env2sys = nn.ModuleDict({interaction_type:HierarchicalTransformer(hidden_dims, 4, hidden_dims,
                                                              self.env2sys_update_norm_inner,
                                                              self.env2sys_update_norm_outer,
                                                              dropout, norm_channel_first=self.norm_channel_first,
                                                              conv_type='system', activation='softmax', poincare=poincare)
                                      for interaction_type in
                                      interaction_types})

        self.sys2gene = HierarchicalTransformer(hidden_dims, 4, hidden_dims,
                                                      self.sys2gene_update_norm_inner, self.sys2gene_update_norm_outer,
                                                      dropout, norm_channel_first=self.norm_channel_first,
                                                      conv_type='system', activation='softmax', poincare=poincare)
        self.gene_norm = nn.LayerNorm(hidden_dims)
        self.sys_norm = nn.LayerNorm(hidden_dims)
        self.effect_norm = nn.LayerNorm(hidden_dims)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def moebius_add(self, x: torch.Tensor, y: torch.Tensor, eps=1e-15):
        """
        x, y: shape (..., d) (broadcastable)
        Returns the Möbius (hyperbolic) addition x ⊕ y in the Poincaré disk.
        """
        xy = 2 * (x * y).sum(dim=-1, keepdim=True)  # 2 * <x,y>
        x_sq = (x * x).sum(dim=-1, keepdim=True)  # ||x||^2
        y_sq = (y * y).sum(dim=-1, keepdim=True)  # ||y||^2

        denom = 1 + xy + x_sq * y_sq / 4.0  # We'll reorganize for numeric stability
        denom = torch.clamp(denom, min=eps)

        numerator = (1 + xy + x_sq) * y + (1 - y_sq) * x
        return numerator / denom

    def get_gene2sys(self, system_embedding, gene_embedding, gene2sys_mask):
        system_embedding_input = self.dropout(self.sys_norm(system_embedding))
        gene_embedding_input_input = self.dropout(self.gene_norm(gene_embedding))
        gene_effect = self.gene2sys.forward(system_embedding_input, gene_embedding_input_input, gene2sys_mask)
        return gene_effect

    def get_sys2gene(self, gene_embedding, system_embedding, sys2gene_mask):
        gene_embedding_input = self.dropout(self.gene_norm(gene_embedding))
        system_embedding_input = self.dropout(self.sys_norm(system_embedding))
        system_effect = self.sys2gene.forward(gene_embedding_input, system_embedding_input, sys2gene_mask)
        return system_effect

    def get_sys2sys(self, system_embedding, nested_hierarchical_masks,
                    direction: str = "forward") -> torch.Tensor:
        """
        Args
        ----
        system_embedding : (B, S, H)  – current system-level embeddings
        nested_hierarchical_masks : iterable of dicts produced by the dataloader
        direction : 'forward' | 'backward'
        Returns
        -------
        (B, S, H)  – updated system embeddings
        """

        B, S, H = system_embedding.shape  # batch, #systems, hidden
        flat_size = B * S
        device = system_embedding.device
        dtype = system_embedding.dtype

        # choose the interaction module
        sys2sys_layer = self.sys2env if direction == "forward" else self.env2sys

        # ---- flatten once; KEEP THIS BUFFER READ-ONLY AFTER WE TAKE A VIEW ----
        #system_flat = system_embedding.view(flat_size, H)  # (B·S, H)

        # updates will be accumulated here to avoid in-place writes on the view
        delta = torch.zeros_like(system_embedding)  # (B·S, H)

        for hierarchical_masks in nested_hierarchical_masks:
            for inter_type, hmask in hierarchical_masks.items():
                hitr = sys2sys_layer[inter_type]

                q_idx = hmask["query"]  # (n_q,)
                k_idx = hmask["key"]  # (n_k,)
                attn_mask = hmask["mask"]  # Bool / Float  – n_q × n_k

                # -------- gather the *partial* embeddings (no big temporaries) ----
                sys_q = torch.index_select(system_embedding, 1, q_idx)  # (B, n_q, H)
                sys_k = torch.index_select(system_embedding, 1, k_idx)  # (B, n_k, H)

                sys_q = self.dropout(self.sys_norm(sys_q))
                sys_k = self.dropout(self.sys_norm(sys_k))

                # hitr.forward returns (B, n_q, H)
                effect = self.effect_norm(hitr.forward(sys_q, sys_k, attn_mask))

                # -------- flat indices for B·n_q rows -----------------------------
                #   idx = q_idx + b*S   (broadcasted over batch dimension)
                #flat_idx = (q_idx.unsqueeze(0) +
                #            torch.arange(B, device=device)[:, None] * S).reshape(-1)  # (B·n_q,)
                #flat_src = effect.reshape(-1, H)  # (B·n_q, H)

                # safe in-place add into the *delta* buffer
                delta[:, hmask['query_indices'], :] = delta[:, hmask['query_indices'], :] + effect[:,:hmask['query_indices'].size(0), :]#.index_add_(0, flat_idx, flat_src)
        # combine read-only base + accumulated deltas, reshape back
        #updated = (system_flat + delta_flat).view(B, S, H)
        updated = system_embedding + delta
        return updated


    '''
    def get_sys2sys(self, system_embedding, nested_hierarchical_masks, direction='forward', return_updates=True, input_format='indices', update_tensor=None):
        batch_size = system_embedding.size(0)
        feature_size = system_embedding.size(2)
        system_embedding_output = torch.clone(system_embedding)

        update_result = torch.zeros_like(system_embedding)
        if direction=='forward':
            sys2sys = self.sys2env
        else:
            sys2sys = self.env2sys
        for hierarchical_masks in nested_hierarchical_masks:
            hitr_result = 0.
            for interaction_type, hierarchical_mask in hierarchical_masks.items():
                hitr = sys2sys[interaction_type]
                ## Calculate hitr by the propagation independently
                system_embedding_queries = self.system_embedding(hierarchical_mask['query']).unsqueeze(
                    0).expand(batch_size, -1, -1)
                system_embedding_keys = self.system_embedding(hierarchical_mask['key']).unsqueeze(0).expand(
                    batch_size, -1, -1)
                system_effect_queries = torch.index_select(update_tensor, 1,
                                                           hierarchical_mask['query'].to(torch.int64))
                system_effect_keys = torch.index_select(update_tensor, 1,
                                                        hierarchical_mask['key'].to(torch.int64))

                system_embedding_queries = system_embedding_queries + system_effect_queries
                system_embedding_keys = system_embedding_keys + system_effect_keys
                system_embedding_queries = self.dropout(self.sys_norm(system_embedding_queries))
                system_embedding_keys = self.dropout(self.sys_norm(system_embedding_keys))
                mask = hierarchical_mask['mask']
                hitr_result = self.effect_norm(hitr.forward(system_embedding_queries, system_embedding_keys, mask))

                query_indices = hierarchical_mask['query'].unsqueeze(0).expand(batch_size, -1).unsqueeze(-1).expand(-1, -1, self.hidden_dims)
                update_tensor = update_tensor.scatter_add(1, query_indices, hitr_result)  # + self.effect_norm(system_effect)
                update_result = update_result.scatter_add(1, query_indices, hitr_result)  # + self.effect_norm(system_effect)
        if return_updates:
            # return original system embeddings and update tensor separately
            return update_result
        else:
            # return original system embeddings + update
            if input_format == 'indices':
                return system_embedding + update_tensor
            else:
                return system_embedding_output
    '''