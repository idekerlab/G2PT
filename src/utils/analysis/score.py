import torch
import numpy as np
import pandas as pd

class ScoreAnalyzer(object):

    def __init__(self, drug_response_model, tree_parser, subtree_order=['default']):

        self.drug_response_model = drug_response_model.to("cpu")
        self.tree_parser = tree_parser

        self.nested_subtrees_forward = self.tree_parser.get_nested_subtree_mask(subtree_order, direction='forward')
        self.nested_subtrees_backward = self.tree_parser.get_nested_subtree_mask(subtree_order, direction='backward')
        self.system_embedding_tensor = self.drug_response_model.system_embedding.weight.unsqueeze(0)[:, :-1, :]
        self.gene_embedding_tensor = drug_response_model.gene_embedding.weight.unsqueeze(0)[:, :-1, :]
        self.system_embedding = self.drug_response_model.system_embedding.weight.detach().cpu().numpy()[:-1, :]
        self.gene_embedding = self.drug_response_model.gene_embedding.weight.detach().cpu().numpy()[:-1, :]


    def build_score_matrix(self, genotype, drug, skip_mut2sys=False):
        if skip_mut2sys:
            mutation_updated = self.system_embedding_tensor
        else:
            mutation_updated, mutation_updates_dict = self.drug_response_model.get_mut2system(genotype)
            mutation_updated = mutation_updated[:, :-1, :]
        tree_updated_forward = self.drug_response_model.get_sys2sys(mutation_updated,
                                                                                           self.nested_subtrees_forward,
                                                                                           direction='forward',
                                                                                           return_updates=False)
        tree_updated_backward = self.drug_response_model.get_sys2sys(
            tree_updated_forward, self.nested_subtrees_backward, direction='backward', return_updates=False)

        gene_updated = self.drug_response_model.get_sys2gene(tree_updated_backward, self.gene_embedding_tensor, torch.tensor(self.tree_parser.sys2gene_mask))

        drug_embedding = self.drug_response_model.get_compound_embedding(drug)

        _, original_score = self.drug_response_model.get_system2comp(drug_embedding, self.system_embedding_tensor, score=True)
        original_score = original_score[0, 0, :, :].detach().numpy()
        _, mutation_score = self.drug_response_model.get_system2comp(drug_embedding, mutation_updated, score=True)
        mutation_score = mutation_score[0, 0, :, :].detach().numpy()
        _, forward_score = self.drug_response_model.get_system2comp(drug_embedding, tree_updated_forward, score=True)
        forward_score = forward_score[0, 0, :, :].detach().numpy()
        _, backward_score = self.drug_response_model.get_system2comp(drug_embedding, tree_updated_backward, score=True)
        backward_score = backward_score[0, 0, :, :].detach().numpy()
        #print(original_score.shape, mutation_score.shape, forward_score.shape, backward_score.shape)
        sys_result_df = pd.DataFrame(np.concatenate([original_score, mutation_score, forward_score, backward_score], axis=0).T)
        sys_result_df.index = sys_result_df.index.map(lambda a: self.tree_parser.ind2sys[a])
        sys_result_df.columns = ["original", "mut2sys", "sys2env", "env2sys"]

        _, orig_gene_score = self.drug_response_model.get_gene2comp(drug_embedding, self.gene_embedding_tensor, score=True)
        orig_gene_score = orig_gene_score[0, 0, :, :].detach().numpy()
        _, updated_gene_score = self.drug_response_model.get_gene2comp(drug_embedding, gene_updated, score=True)
        updated_gene_score = updated_gene_score[0, 0, :, :].detach().numpy()

        gene_result_df = pd.DataFrame(np.concatenate([orig_gene_score, updated_gene_score], axis=0).T)
        gene_result_df.index = gene_result_df.index.map(lambda a: self.tree_parser.ind2gene[a])
        gene_result_df.columns = ["original", "sys2gene"]
        return sys_result_df, gene_result_df

    def get_systems_without_effect(self, genotype, drug, source='original', target='env2sys'):
        score_matrix = self.build_score_matrix(genotype, drug)
        return  score_matrix.loc[(score_matrix[target]-score_matrix[source])==0]


