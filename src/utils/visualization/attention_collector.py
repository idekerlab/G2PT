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
        self.nested_subtrees_forward = self.tree_parser.get_nested_subtree_mask(
            subtree_order, direction='forward', format='indices')
        self.nested_subtrees_forward = self.move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = self.tree_parser.get_nested_subtree_mask(
            subtree_order, direction='backward', format='indices')
        self.nested_subtrees_backward = self.move_to(self.nested_subtrees_backward, device)
        self.sys2gene_mask = self.move_to(torch.tensor(self.tree_parser.sys2gene_mask, dtype=torch.bool), device)
        self.gene2sys_mask = self.sys2gene_mask.T
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
                # batches.append(batch)
                batch = self.move_to(batch, self.device)
                batch_size, _ = batch['genotype']['embedding']['homozygous_a1']['snp'].size()
                batch_size_int = int(batch_size)
                # Get SNP2Gene attention
                a, b, homozygous_snp2gene_attention = self.model.get_snp_effects(
                    batch['genotype']['embedding']['homozygous_a1'], self.model.snp2gene_homozygous, attention=False,
                    score=True)
                homozygous_snp2gene_attention = homozygous_snp2gene_attention.detach().cpu().numpy()
                c, d, heterozygous_snp2gene_attention = self.model.get_snp_effects(
                    batch['genotype']['embedding']['heterozygous'], self.model.snp2gene_heterozygous, attention=False,
                    score=True)
                heterozygous_snp2gene_attention = heterozygous_snp2gene_attention.detach().cpu().numpy()

                for head in range(4):
                    attention_result_df[head] = self.assign_snp2gene_attention(attention_result_df[head], self.move_to(batch, 'cpu'),
                                                                            heterozygous_snp2gene_attention[:, head, :, :],
                                                                            homozygous_snp2gene_attention[:, head, :, :],
                                                                            target_gene_inds, target_snp_inds,
                                                                            sample_ind)
                # Update gene embedding
                gene_embedding, snp_effect_on_gene = self.model.get_snp2gene(genotype=batch['genotype'])
                gene_embedding = gene_embedding[:, :-1, :] + self.model.effect_norm(snp_effect_on_gene[:, :-1, :])
                gene_embedding = self.model.gene_norm(gene_embedding)
                # Update system embedding
                system_embedding = self.model.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
                system_embedding_input = self.model.sys_norm(system_embedding)
                system_embedding, gene_effect_on_system = self.model.get_gene2sys(system_embedding_input,
                                                                                  gene_embedding,
                                                                                  self.gene2sys_mask)

                gene2sys_score = self.model.gene2sys.get_score(system_embedding, self.model.gene_norm(gene_embedding),
                                                               self.gene2sys_mask)
                gene2sys_score = gene2sys_score.detach().cpu().numpy()
                for head in range(4):
                    for j, go in zip(target_go_inds, target_gos):
                        for k, gene in zip(target_gene_inds, target_genes):
                            if gene in self.tree_parser.sys2gene[go]:
                                col_name = ('forward', go, gene, 'gene2sys')  # "_".join([gene, go, 'gene2sys'])
                                #print(attention_result_df[head].loc[self.samples[sample_ind : sample_ind+batch_size_int], col_name].shape)
                                attention_result_df[head].loc[self.samples[sample_ind : sample_ind+batch_size_int], col_name] = gene2sys_score[
                                                                      :, head, j, k]

                # Update by sys2sys module
                total_update = self.model.effect_norm(gene_effect_on_system)
                system_embedding, system_effect_forward = self.model.get_sys2sys(system_embedding,
                                                                                 self.nested_subtrees_forward,
                                                                                 direction='forward',
                                                                                 return_updates=True,
                                                                                 input_format='indices',
                                                                                 update_tensor=total_update)
                total_update_sys2env = total_update + system_effect_forward
                system_embedding, system_effect_backward = self.model.get_sys2sys(system_embedding,
                                                                                  self.nested_subtrees_backward,
                                                                                  direction='backward',
                                                                                  return_updates=True,
                                                                                  input_format='indices',
                                                                                  update_tensor=total_update_sys2env)
                total_update_env2sys = total_update_sys2env + system_effect_backward
                system_embedding_sys2sys = system_embedding + total_update
                gene_embedding, system_effect_on_gene = self.model.get_sys2gene(gene_embedding,
                                                                                system_embedding_sys2sys,
                                                                                self.sys2gene_mask)
                gene_embedding = gene_embedding + self.model.effect_norm(system_effect_on_gene)
                sys2gene_score = self.model.sys2gene.get_score(gene_embedding,
                                                               self.model.sys_norm(system_embedding_sys2sys),
                                                               self.sys2gene_mask)
                sys2gene_score = sys2gene_score.detach().cpu().numpy()
                for head in range(4):
                    for j, go in zip(target_go_inds, target_gos):
                        for k, gene in zip(target_gene_inds, target_genes):
                            if gene in self.tree_parser.sys2gene[go]:
                                col_name = ('backward', go, gene, 'sys2gene')  # "_".join([gene, go, 'gene2sys'])
                                attention_result_df[head].loc[self.samples[sample_ind : sample_ind+batch_size_int], col_name] = sys2gene_score[
                                                                      :, head, k, j]

                system_embedding_partial = self.model.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:,
                                           target_go_inds, :]
                for query, key in all_edges_forward:
                    query_forward = self.model.sys_norm(
                        system_embedding_partial + total_update_sys2env[:, target_go_inds, :])
                    key_forward = self.model.sys_norm(
                        system_embedding_partial + total_update_sys2env[:, target_go_inds, :])
                    hitr_result_score_forward = self.model.sys2env[0].get_score(query_forward, key_forward,
                                                                                sys2sys_forward_partial)
                    col_name = ('forward', query, key, 'sys2env')
                    for head in range(4):
                        forward_score_result = hitr_result_score_forward[:, head, sys2ind_partial[query], sys2ind_partial[key]].detach().cpu().numpy()
                        attention_result_df[head].loc[self.samples[sample_ind : sample_ind + batch_size_int], col_name] = forward_score_result
                    query_backward = self.model.sys_norm(
                        system_embedding_partial + total_update_sys2env[:, target_go_inds, :])
                    key_backward = self.model.sys_norm(
                        system_embedding_partial + total_update_env2sys[:, target_go_inds, :])
                    hitr_result_score_backward = self.model.env2sys[0].get_score(query_backward, key_backward,
                                                                                 sys2sys_backward_partial)
                    col_name = ('backward', query, key, 'env2sys')
                    for head in range(4):
                        backward_score_result = hitr_result_score_backward[:, head, sys2ind_partial[key],sys2ind_partial[query]].detach().cpu().numpy()
                        attention_result_df[head].loc[self.samples[sample_ind :sample_ind + batch_size_int], col_name] = backward_score_result


                del system_embedding, gene_embedding
                del a, b, c, d
                del homozygous_snp2gene_attention, heterozygous_snp2gene_attention
                del gene2sys_score, sys2gene_score
                gc.collect()
                sample_ind = sample_ind + batch_size_int

        return attention_result_df


    def assign_snp2gene_attention(self, result_df, batch, heterozygous_attention, homozygous_attention, target_gene_inds, target_snp_inds, start_ind):

        i_temp = start_ind
        for snps, genes, mask, attn in zip(batch['genotype']['embedding']['heterozygous']['snp'],
                                           batch['genotype']['embedding']['heterozygous']['gene'],
                                           batch['genotype']['embedding']['heterozygous']['mask'],
                                           heterozygous_attention):
            for j, k in zip(*np.where(mask.numpy())):
                if (genes[j]==self.tree_parser.n_genes) or (snps[k]==self.tree_parser.n_snps):
                    continue
                gene = self.tree_parser.ind2gene[int(genes[j])]
                snp = self.tree_parser.ind2snp[int(snps[k])]
                col_name = ('forward', gene, snp, 'heterozygous')
                if col_name in result_df.columns:
                    result_df.loc[self.samples[i_temp], col_name] = attn[j, k]
            i_temp += 1

        i_temp = start_ind
        for snps, genes, mask, attn in zip(batch['genotype']['embedding']['homozygous_a1']['snp'],
                                           batch['genotype']['embedding']['homozygous_a1']['gene'],
                                           batch['genotype']['embedding']['homozygous_a1']['mask'],
                                           homozygous_attention):
            for j, k in zip(*np.where(mask.numpy())):
                if (genes[j]==self.tree_parser.n_genes) or (snps[k]==self.tree_parser.n_snps):
                    continue
                gene = self.tree_parser.ind2gene[int(genes[j])]
                snp = self.tree_parser.ind2snp[int(snps[k])]
                col_name = ('forward', gene, snp, 'homozygous')
                if col_name in result_df.columns:
                    result_df.loc[self.samples[i_temp], col_name] = attn[j, k]
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

