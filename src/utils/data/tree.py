import pandas as pd
import numpy as np
import networkx as nx
import torch

class TreeParser(object):

    def __init__(self, ontology, gene2ind):
        self.ontology = pd.read_csv(ontology, sep='\t', names=['parent', 'child', 'interaction'])
        gene2ind = pd.read_csv(gene2ind, sep='\t', names=['index', 'gene'])
        self.gene2ind = {gene:index for index, gene in zip(gene2ind['index'], gene2ind['gene'])}
        self.ind2gene = {index:gene for index, gene in zip(gene2ind['index'], gene2ind['gene'])}
        self.system_df = self.ontology.loc[self.ontology['interaction'] != 'gene']
        self.gene2system_df = self.ontology.loc[self.ontology['interaction'] == 'gene']
        systems = np.unique(self.system_df[['parent', 'child']].values)
        self.system2ind = {system: i for i, system in enumerate(systems)}
        self.ind2system = {i:system for system, i in self.system2ind.items()}
        print(self.system2ind)
        print(self.gene2ind)
        self.gene2system_mask = np.zeros((len(self.system2ind), len(self.gene2ind)))

        self.n_systems = len(self.system2ind)
        self.n_genes = len(self.gene2ind)

        for system, gene in zip(self.gene2system_df['parent'], self.gene2system_df['child']):
            self.gene2system_mask[self.system2ind[system], self.gene2ind[gene]] = 1
        self.system2gene_mask = self.gene2system_mask.T
        self.subtree_types = self.system_df['interaction'].unique()
        self.system_graph = nx.from_pandas_edgelist(self.system_df, create_using=nx.DiGraph(), source='parent',
                                                    target='child')
        print("Building descendant dict")
        self.descendant_dict = {system: list(nx.descendants(self.system_graph, system))+[system] for system in systems}
        self.descendant_dict_ind = {self.system2ind[key]:[self.system2ind[descendant] for descendant in value]
                                    for key, value in self.descendant_dict.items()}
        print("Subtree types: ", self.subtree_types)
        self.subtree_dfs = {subtree_type:self.system_df.loc[self.system_df['interaction']==subtree_type]
                            for subtree_type in self.subtree_types}
        self.subtree_graphs = {subtree_type: nx.from_pandas_edgelist(self.subtree_dfs[subtree_type], create_using=nx.DiGraph(),
                                                                     source='parent', target='child')
                               for subtree_type in self.subtree_types}

        self.subtree_reverse_graphs = {subtree_type: nx.from_pandas_edgelist(self.subtree_dfs[subtree_type], create_using=nx.DiGraph(),
                                                                     source='child', target='parent')
                               for subtree_type in self.subtree_types}
        self.subtree_reverse_roots = {subtree_type:set([node for node in self.subtree_reverse_graphs[subtree_type].nodes()
                                                if self.subtree_reverse_graphs[subtree_type].out_degree(node)==0])
                              for subtree_type in self.subtree_types}

        self.gene2gene_mask = np.zeros((len(self.gene2ind), len(self.gene2ind)))
        self.gene2system_graph = nx.from_pandas_edgelist(self.gene2system_df, create_using=nx.DiGraph(),
                                                         source='parent', target='child')
        for node in self.gene2system_graph.nodes:
            if self.gene2system_graph.in_degree(node) == 0:
                children_genes = [target for source, target in self.gene2system_graph.out_edges(node)]
                for child_i in children_genes:
                    for child_j in children_genes:
                        self.gene2gene_mask[self.gene2ind[child_i], self.gene2ind[child_j]] = 1

        #self.subtree_depths = {len(nx.dag_longest_path(self.subtree_graphs[subtree_type]))
        #                       for subtree_type in self.subtree_types}

    def get_subtree_types(self):
        return self.subtree_types

    def get_gene2ind(self, gene):
        if type(gene)==str:
            return self.gene2ind[gene]
        elif type(gene)==list:
            return [self.gene2ind[g] for g in gene]

    def get_system2ind(self, system):
        if type(system)==str:
            return self.system2ind[system]
        elif type(system)==list:
            return [self.system2ind[sys] for sys in system]

    def get_ind2system(self, system):
        if type(system)==int:
            return self.ind2system[system]
        elif type(system)==list:
            return [self.ind2system[sys] for sys in system]

    def get_system_hierarchies(self, system, interaction_type):
        subtree = self.subtree_reverse_graphs[interaction_type]
        subtree_root = self.subtree_reverse_roots[interaction_type]
        return [path for path in nx.all_simple_paths(subtree, system, subtree_root)]


    def get_parent_system_of_gene(self, gene):
        systems = [system for system in self.gene2system_df.loc[self.gene2system_df["child"] == gene]["parent"]]
        return systems

    def get_gene_hierarchies(self, gene, interaction_type):
        systems = self.get_parent_system_of_gene(gene)
        system_hierarchy_dict = {system: self.get_system_hierarchies(system, interaction_type) for system in systems}
        return system_hierarchy_dict

    def get_system2genotype_mask(self, mut_vector):
        system2mut_mask =  torch.logical_and(torch.tensor(self.gene2system_mask, dtype=torch.bool),
                                             mut_vector.unsqueeze(0).expand(self.n_systems, -1).bool())
        return system2mut_mask.float()


    def get_subtree_mask(self, interaction_type, direction='forward'):
        sub_tree = self.subtree_graphs[interaction_type]
        sub_tree_roots = set([node for node in sub_tree.nodes() if sub_tree.in_degree(node)==0])
        cur_subtree_dfs = self.subtree_dfs[interaction_type]
        cur_parents = sub_tree_roots
        result_masks = []
        while True:
            parent_subtree_dfs = cur_subtree_dfs.loc[cur_subtree_dfs['parent'].map(lambda a: a in cur_parents)]
            mask = np.zeros((len(self.system2ind), len(self.system2ind)))
            for parent, child in zip(parent_subtree_dfs['parent'], parent_subtree_dfs['child']):
                if direction=='forward':
                    mask[self.system2ind[parent], self.system2ind[child]] = 1
                elif direction=='backward':
                    mask[self.system2ind[child], self.system2ind[parent]] = 1
            cur_subtree_dfs = cur_subtree_dfs.loc[cur_subtree_dfs['parent'].map(lambda a: a not in cur_parents)]
            cur_parents = set(parent_subtree_dfs['child'])
            result_masks.append(torch.tensor(mask, dtype=torch.float32))
            if cur_subtree_dfs.empty:
                break
        if direction=='forward':
            result_masks.reverse()
        return result_masks

    def get_nested_subtree_mask(self, subtree_order, direction='forward'):
        nested_subtrees = [self.get_subtree_mask(subtree_type, direction=direction) for subtree_type in subtree_order]
        return nested_subtrees

    def mask_subtree_mask_by_mut(self, tree_mask, mut_vector):
        non_zero_index_tree = np.where(tree_mask!=0)
        result_mask = np.zeros_like(tree_mask)
        mut_genes = np.where(np.array(mut_vector)!=0)[0]
        mut_systems = np.where(self.gene2system_mask[:, mut_genes]!=0)[0]
        for x, y in zip(*non_zero_index_tree):
            if np.any([mut_system in self.descendant_dict_ind[y] for mut_system in mut_systems]):
                result_mask[x, y] =1
        return torch.tensor(result_mask, dtype=torch.float32)