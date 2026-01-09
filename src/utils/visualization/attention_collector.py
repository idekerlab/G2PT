import torch
import networkx as nx
import pandas as pd
import numpy as np
import gc
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from src.utils.tree import SNPTreeParser
from src.utils.data.dataset import SNP2PCollator
from .sankey_utils import make_all_column_names




class AttentionCollector(object):

    def __init__(self, tree_parser : SNPTreeParser, model, dataset, device='cuda:0', subtree_order=['default'], num_workers=8, batch_size=16):
        self.tree_parser = tree_parser
        self.model = model
        self.dataset = dataset
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.model = model.eval()
        self.collator = SNP2PCollator(tree_parser)
        self.dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, collate_fn=self.collator)
        self.nested_subtrees_forward = self.tree_parser.get_hierarchical_interactions(
            subtree_order, direction='forward', format='indices')
        self.nested_subtrees_forward = self.move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = self.tree_parser.get_hierarchical_interactions(
            subtree_order, direction='backward', format='indices')
        self.nested_subtrees_backward = self.move_to(self.nested_subtrees_backward, device)
        self.sys2gene_mask = self.move_to(torch.tensor(self.tree_parser.sys2gene_mask, dtype=torch.float32), device)
        self.gene2sys_mask = self.sys2gene_mask.T

        self.snp2gene_mask = self.move_to(torch.tensor(self.tree_parser.snp2gene_mask, dtype=torch.float32), device)
        self.samples = self.dataset.cov_df.IID

    def forward(self, target_gos, target_genes, target_snps):

        target_go_inds, target_gene_inds, target_snp_inds = self.tree_parser.get_target_indices(target_gos, target_genes, target_snps)
        sys2sys_forward_partial, sys2sys_backward_partial, sys2ind_partial, all_edges_forward = self.tree_parser.get_partial_sys2sys_masks(target_gos)
        sys2sys_forward_partial = self.move_to(sys2sys_forward_partial, self.device)
        sys2sys_backward_partial = self.move_to(sys2sys_backward_partial, self.device)


        column_names = make_all_column_names(self.tree_parser, target_gos, target_genes)
        multi_idx = pd.MultiIndex.from_tuples(
            column_names,
            names=["direction", "query", "key", "module"]  # optional, but nice for readability
        )
        attention_result_df = {i:pd.DataFrame(columns=multi_idx, index=self.samples) for i in range(4)}

        sample_ind = 0
        for i, batch in enumerate(tqdm(self.dataloader)):
            with torch.no_grad():
                batch = self.move_to(batch, self.device)
                batch_size = batch['genotype']['snp'].size(0)

                snp_embedding, _ = self.model.get_snp_embedding(batch['genotype'])
                gene_embedding = self.model.gene_embedding(batch['genotype']['gene'])
                system_embedding = self.model.system_embedding(batch['genotype']['sys'])

                gene_norm = self.model.gene_norm(gene_embedding)

                snp2gene_score = self.model.snp2gene.get_score(
                    gene_norm, self.model.snp_norm(snp_embedding), self.snp2gene_mask)
                snp2gene_score_np = snp2gene_score.detach().cpu().numpy()

                snp_effect_on_gene = self.model.get_snp2gene(
                    gene_embedding, snp_embedding, self.snp2gene_mask)
                gene_embedding = gene_embedding + self.model.effect_norm(snp_effect_on_gene)

                system_norm = self.model.sys_norm(system_embedding)
                gene_norm = self.model.gene_norm(gene_embedding)
                gene2sys_score = self.model.gene2sys.get_score(
                    system_norm, gene_norm, self.gene2sys_mask)
                gene2sys_score_np = gene2sys_score.detach().cpu().numpy()

                gene_effect_on_system = self.model.get_gene2sys(
                    system_embedding, gene_embedding, self.gene2sys_mask)
                system_embedding_total = self.model.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
                system_embedding_total[:, batch['genotype']['sys_indices']] = (
                    system_embedding_total[:, batch['genotype']['sys_indices']]
                    + self.model.effect_norm(gene_effect_on_system)
                )

                system_embedding_total = self.model.get_sys2sys(
                    system_embedding_total, self.nested_subtrees_forward, direction='forward')
                system_embedding_total = self.model.get_sys2sys(
                    system_embedding_total, self.nested_subtrees_backward, direction='backward')

                system_embedding = system_embedding_total[:, batch['genotype']['sys_indices']]
                sys2gene_score = self.model.sys2gene.get_score(
                    self.model.gene_norm(gene_embedding),
                    self.model.sys_norm(system_embedding),
                    self.sys2gene_mask)
                sys2gene_score_np = sys2gene_score.detach().cpu().numpy()

                interaction_type = next(iter(self.model.sys2env.keys()))
                system_embedding_partial = system_embedding_total[:, target_go_inds, :]
                sys2env_score = self.model.sys2env[interaction_type].get_score(
                    self.model.sys_norm(system_embedding_partial),
                    self.model.sys_norm(system_embedding_partial),
                    sys2sys_forward_partial)
                env2sys_score = self.model.env2sys[interaction_type].get_score(
                    self.model.sys_norm(system_embedding_partial),
                    self.model.sys_norm(system_embedding_partial),
                    sys2sys_backward_partial)

                batch_size_int = int(batch_size)
                for head in range(4):
                    attention_result_df[head] = self.assign_snp2gene_attention(
                        attention_result_df[head],
                        self.move_to(batch, 'cpu'),
                        snp2gene_score_np[:, head, :, :],
                        target_gene_inds,
                        target_snp_inds,
                        sample_ind)

                    for j, go in zip(target_go_inds, target_gos):
                        for k, gene in zip(target_gene_inds, target_genes):
                            if gene in self.tree_parser.sys2gene[go]:
                                col_name = ('forward', go, gene, 'gene2sys')
                                attention_result_df[head].loc[
                                    self.samples[sample_ind: sample_ind + batch_size_int], col_name] = (
                                    gene2sys_score_np[:, head, j, k])

                    for j, go in zip(target_go_inds, target_gos):
                        for k, gene in zip(target_gene_inds, target_genes):
                            if gene in self.tree_parser.sys2gene[go]:
                                col_name = ('backward', go, gene, 'sys2gene')
                                attention_result_df[head].loc[
                                    self.samples[sample_ind: sample_ind + batch_size_int], col_name] = (
                                    sys2gene_score_np[:, head, k, j])

                    for query, key in all_edges_forward:
                        col_name_forward = ('forward', query, key, 'sys2env')
                        forward_score_result = sys2env_score[:, head, sys2ind_partial[query], sys2ind_partial[key]].detach().cpu().numpy()
                        attention_result_df[head].loc[
                            self.samples[sample_ind: sample_ind + batch_size_int], col_name_forward] = forward_score_result

                        col_name_backward = ('backward', query, key, 'env2sys')
                        backward_score_result = env2sys_score[:, head, sys2ind_partial[key], sys2ind_partial[query]].detach().cpu().numpy()
                        attention_result_df[head].loc[
                            self.samples[sample_ind: sample_ind + batch_size_int], col_name_backward] = backward_score_result

                del system_embedding, gene_embedding, system_embedding_total
                del snp_embedding
                del snp2gene_score_np, gene2sys_score_np, sys2gene_score_np
                gc.collect()
                sample_ind = sample_ind + batch_size_int

        return attention_result_df


    def assign_snp2gene_attention(self, result_df, batch, snp2gene_attention, target_gene_inds, target_snp_inds, start_ind):

        i_temp = start_ind
        snp_padding = self.tree_parser.n_snps * 3
        for snps, genes, attn in zip(batch['genotype']['snp'],
                                     batch['genotype']['gene'],
                                     snp2gene_attention):
            snps_np = snps.numpy()
            genes_np = genes.numpy()
            for j, gene_idx in enumerate(genes_np):
                if gene_idx == self.tree_parser.n_genes:
                    continue
                gene = self.tree_parser.ind2gene[int(gene_idx)]
                if int(gene_idx) not in target_gene_inds:
                    continue
                for k, snp_idx in enumerate(snps_np):
                    if snp_idx >= snp_padding:
                        continue
                    snp_ind = int(snp_idx % self.tree_parser.n_snps)
                    if snp_ind not in target_snp_inds:
                        continue
                    snp = self.tree_parser.ind2snp[snp_ind]
                    col_name_hom = ('forward', gene, snp, 'homozygous')
                    col_name_het = ('forward', gene, snp, 'heterozygous')
                    if col_name_hom in result_df.columns:
                        result_df.loc[self.samples[i_temp], col_name_hom] = attn[j, k]
                    if col_name_het in result_df.columns:
                        result_df.loc[self.samples[i_temp], col_name_het] = attn[j, k]
            i_temp += 1
        return result_df

    def move_to(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device))
            return res
        else:
            return obj.to(device)

