import torch
import numpy as np
import pandas as pd

class ScoreAnalyzer(object):

    def __init__(self, drug_response_model, tree_parser, subtree_order=['default']):

        self.drug_response_model = drug_response_model.to("cpu")
        self.tree_parser = tree_parser

        self.nested_subtrees_forward = self.tree_parser.get_nested_subtree_mask(subtree_order, direction='forward')
        self.nested_subtrees_backward = self.tree_parser.get_nested_subtree_mask(subtree_order, direction='backward')
        self.system_embedding_tensor = self.drug_response_model.system_embedding.weight.unsqueeze(0)
        self.gene_embedding_tensor = drug_response_model.gene_embedding.weight.unsqueeze(0)
        self.system_embedding = self.drug_response_model.system_embedding.weight.detach().cpu().numpy()
        self.gene_embedding = self.drug_response_model.gene_embedding.weight.detach().cpu().numpy()


    def build_score_matrix(self, genotype, drug):
        mutation_updated, mutation_updates_dict = self.drug_response_model.get_mut2system(self.system_embedding_tensor,
                                                                                          self.gene_embedding_tensor,
                                                                                          genotype)
        tree_updated_forward = self.drug_response_model.get_system2system(mutation_updated,
                                                                                           self.nested_subtrees_forward,
                                                                                           direction='forward',
                                                                                           return_updates=False)
        tree_updated_backward = self.drug_response_model.get_system2system(
            tree_updated_forward, self.nested_subtrees_backward, direction='backward', return_updates=False)

        drug_embedding = self.drug_response_model.get_compound_embedding(drug)

        original_score = self.drug_response_model.sys2comp.get_score(drug_embedding, self.system_embedding_tensor, self.system_embedding_tensor)[0, 0, :, :].detach().numpy()
        mutation_score = self.drug_response_model.sys2comp.get_score(drug_embedding, mutation_updated, mutation_updated)[0, 0, :, :].detach().numpy()
        forward_score = self.drug_response_model.sys2comp.get_score(drug_embedding, tree_updated_forward, tree_updated_forward)[0, 0, :, :].detach().numpy()
        backward_score = self.drug_response_model.sys2comp.get_score(drug_embedding, tree_updated_backward, tree_updated_backward)[0, 0, :, :].detach().numpy()
        result_df = pd.DataFrame(np.concatenate([original_score, mutation_score, forward_score, backward_score], axis=0).T)

        result_df.index = result_df.index.map(lambda a: self.tree_parser.ind2system[a])
        result_df.columns = ["original", "mut2sys", "sys2env", "env2sys"]

        return result_df

    def get_systems_without_effect(self, genotype, drug, source='original', target='env2sys'):
        score_matrix = self.build_score_matrix(genotype, drug)
        return  score_matrix.loc[(score_matrix[target]-score_matrix[source])==0]


