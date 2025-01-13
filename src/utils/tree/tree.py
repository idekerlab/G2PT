import pandas as pd
import numpy as np
import networkx as nx
import torch

class TreeParser(object):

    def __init__(self, ontology, dense_attention=False, sys_annot_file=None):
        self.ontology = pd.read_csv(ontology, sep='\t', names=['parent', 'child', 'interaction'])
        self.dense_attention = dense_attention
        #gene2ind = pd.read_csv(gene2ind, sep='\t', names=['index', 'gene'])
        self.system_df = self.ontology.loc[self.ontology['interaction'] != 'gene']
        self.gene2sys_df = self.ontology.loc[self.ontology['interaction'] == 'gene']
        sys2gene_grouped_by_sys = self.gene2sys_df.groupby('parent')
        sys2gene_grouped_by_gene = self.gene2sys_df.groupby('child')
        sys2gene_dict = {sys: sys2gene_grouped_by_sys.get_group(sys)['child'].values.tolist() for sys in
                         sys2gene_grouped_by_sys.groups.keys()}

        self.sys2gene = {sys: sys2gene_grouped_by_sys.get_group(sys)['child'].values.tolist() for sys in
                         sys2gene_grouped_by_sys.groups.keys()}
        self.gene2sys = {gene: sys2gene_grouped_by_gene.get_group(gene)['parent'].values.tolist() for gene in
                         sys2gene_grouped_by_gene.groups.keys()}

        genes = self.gene2sys_df['child'].unique()
        self.gene2ind = {gene: index for index, gene in enumerate(genes)}
        self.ind2gene = {index: gene for index, gene in enumerate(genes)}
        systems = np.unique(self.system_df[['parent', 'child']].values)
        self.sys2ind = {system: i for i, system in enumerate(systems)}
        self.ind2sys = {i:system for system, i in self.sys2ind.items()}
        self.n_systems = len(self.sys2ind)
        self.n_genes = len(self.gene2ind)
        print("%d Systems are queried"%self.n_systems)
        print("%d Genes are queried"%self.n_genes)

        self.system2system_mask = np.zeros((len(self.sys2ind), len(self.sys2ind)))

        for parent_system, child_system in zip(self.system_df['parent'], self.system_df['child']):
            self.system2system_mask[self.sys2ind[parent_system], self.sys2ind[child_system]] = 1

        self.gene2sys_mask = np.zeros((len(self.sys2ind), len(self.gene2ind)))
        self.sys2gene_dict = {self.sys2ind[system]: [] for system in systems}
        self.gene2sys_dict = {gene: [] for gene in range(self.n_genes)}
        for system, gene in zip(self.gene2sys_df['parent'], self.gene2sys_df['child']):
            #print(system, gene)
            self.gene2sys_mask[self.sys2ind[system], self.gene2ind[gene]] = 1.
            self.sys2gene_dict[self.sys2ind[system]].append(self.gene2ind[gene])
            self.gene2sys_dict[self.gene2ind[gene]].append(self.sys2ind[system])
        print("Total %d Gene-System interactions are queried"%self.gene2sys_mask.sum())
        if self.dense_attention:
            self.gene2sys_mask = torch.ones_like(torch.tensor(self.gene2sys_mask))
        self.sys2gene_mask = self.gene2sys_mask.T
        self.subtree_types = self.system_df['interaction'].unique()
        self.system_graph = nx.from_pandas_edgelist(self.system_df, create_using=nx.DiGraph(), source='parent',
                                                    target='child')
        #self.system_graph_revered = nx.from_pandas_edgelist(self.system_df, create_using=nx.DiGraph(), source='child',
        #                                            target='parent')
        self.node_height_dict = self.compute_node_heights()

        for sys in systems:
            if sys in sys2gene_dict.keys():
                continue
            else:
                sys2gene_dict[sys] = []
        for sys in systems:
            self.add_child_genes_to_parents(self.system_graph, sys, sys2gene_dict)
        self.sys2gene_full = sys2gene_dict

        self.gene2sys_full = {}
        for sys, genes in self.sys2gene_full.items():
            for gene in genes:
                if gene in self.gene2sys_full.keys():
                    self.gene2sys_full[gene].append(sys)
                else:
                    self.gene2sys_full[gene] = [sys]

        if sys_annot_file:
            sys_descriptions = pd.read_csv(sys_annot_file, header=None, names=['Term', 'Term_Description'], index_col=0, sep='\t')

            self.sys_annot_dict = sys_descriptions.to_dict()["Term_Description"]
        else:
            self.sys_annot_dict = None


        print("Building descendant dict")
        self.descendant_dict = {system: list(nx.descendants(self.system_graph, system))+[system] for system in systems}
        self.descendant_dict_ind = {self.sys2ind[key]:[self.sys2ind[descendant] for descendant in value]
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
        self.gene2sys_graph = nx.from_pandas_edgelist(self.gene2sys_df, create_using=nx.DiGraph(),
                                                         source='parent', target='child')
        for node in self.gene2sys_graph.nodes:
            if self.gene2sys_graph.in_degree(node) == 0:
                children_genes = [target for source, target in self.gene2sys_graph.out_edges(node)]
                for child_i in children_genes:
                    for child_j in children_genes:
                        self.gene2gene_mask[self.gene2ind[child_i], self.gene2ind[child_j]] = 1

    def summary(self, system=True, gene=True):
        if system:
            print("The number of systems : %d"%self.n_systems)
            print(" ")
            print("System Index: ")
            for i in range(len(self.ind2sys)):
                if self.sys_annot_dict:
                    print(i, ":", self.ind2sys[i], self.sys_annot_dict[self.ind2sys[i]])
                else:
                    print(i, ":", self.ind2sys[i])
            print(" ")
        if gene:
            print("The number of genes   : %d" % self.n_genes)
            print("The number of gene-system connections: %d" % self.gene2sys_mask.sum())
            print("Gene Index: ")
            for i in range(len(self.ind2gene)):
                print(i, ":", self.ind2gene[i], ' -> ', ",".join(self.gene2sys[self.ind2gene[i]]))
            print(" ")

    def add_child_genes_to_parents(self, tree, node, gene_dict):
        """
        Recursively adds child genes to parent nodes.

        :param tree: NetworkX DiGraph representing the tree.
        :param node: Current node to process.
        :param gene_dict: Dictionary mapping nodes to their gene sets.
        """
        # Base case: if the node is a leaf, return its genes
        if tree.out_degree(node) == 0:
            return gene_dict[node]

        # Initialize the set of genes for the current node with its own genes
        genes = set(gene_dict[node])

        # Recurse for all children and add their genes
        for child in tree.successors(node):
            child_genes = self.add_child_genes_to_parents(tree, child, gene_dict)
            genes.update(child_genes)

        # Update the gene set for the current node
        gene_dict[node] = genes
        return genes

    def compute_node_heights(self):
        # Ensure the graph is a tree (i.e., has a single root)
        if not nx.is_directed_acyclic_graph(self.system_graph):
            raise ValueError("The input graph must be a directed acyclic graph (DAG).")

        # Start by identifying leaf nodes
        leaf_nodes = [node for node in self.system_graph.nodes if self.system_graph.out_degree(node) == 0]

        # Dictionary to store node heights
        heights = {node: 0 for node in leaf_nodes}  # Leaf nodes have height 0

        # Process nodes in reverse topological order
        for node in reversed(list(nx.topological_sort(self.system_graph))):
            if node not in heights:  # Non-leaf nodes
                # Height is 1 + max height of its children
                heights[node] = 1 + max(heights[child] for child in self.system_graph.successors(node))
        return heights

    def get_descendants_sorted_by_height(self, node):
        # Compute the height of all nodes
        node_heights = self.compute_node_heights()
        # Get all descendants of the given node
        descendants = nx.descendants(self.system_graph, node)
        # Sort descendants by height (ascending or descending)
        sorted_descendants = sorted(descendants, key=lambda x: node_heights[x])
        return sorted_descendants

    def write_gmt(self, output_path):
        f = open(output_path, 'w')
        for sys, genes in self.sys2gene_full.items():
            if self.sys_annot_dict is not None:
                lines = "\t".join([sys, self.sys_annot_dict[sys]] + list(genes))
            else:
                lines = "\t".join([sys]+list(genes))
            f.write(lines+'\n')
            f.flush()
        f.close()

    def get_subtree_types(self):
        return self.subtree_types

    def get_gene2ind(self, gene):
        if type(gene)==str:
            return self.gene2ind[gene]
        elif type(gene)==list:
            return [self.gene2ind[g] for g in gene]

    def get_system2ind(self, system):
        if type(system)==str:
            return self.sys2ind[system]
        elif type(system)==list:
            return [self.sys2ind[sys] for sys in system]

    def get_ind2system(self, system):
        if type(system)==int:
            return self.ind2sys[system]
        elif type(system)==list:
            return [self.ind2sys[sys] for sys in system]

    def get_system_hierarchies(self, system, interaction_type):
        subtree = self.subtree_reverse_graphs[interaction_type]
        subtree_root = self.subtree_reverse_roots[interaction_type]
        return [path for path in nx.all_simple_paths(subtree, system, subtree_root)]


    def get_parent_system_of_gene(self, gene):
        systems = [system for system in self.gene2sys_df.loc[self.gene2sys_df["child"] == gene]["parent"]]
        return systems

    def get_gene_hierarchies(self, gene, interaction_type):
        systems = self.get_parent_system_of_gene(gene)
        system_hierarchy_dict = {system: self.get_system_hierarchies(system, interaction_type) for system in systems}
        return system_hierarchy_dict

    def get_system2genotype_mask(self, mut_vector):
        system2mut_mask =  torch.logical_and(torch.tensor(self.gene2sys_mask, dtype=torch.bool),
                                             mut_vector.unsqueeze(0).expand(self.n_systems, -1).bool())
        return system2mut_mask.float()


    def get_subtree_mask(self, interaction_type, direction='forward', format='indices'):
        sub_tree = self.subtree_graphs[interaction_type]
        sub_tree_roots = set([node for node in sub_tree.nodes() if sub_tree.in_degree(node)==0])
        cur_subtree_dfs = self.subtree_dfs[interaction_type]
        #cur_parents = sub_tree_roots
        result_masks = []
        #result_indices = []
        cur_parents = set(sub_tree_roots)
        while cur_parents:
            mask = np.zeros((len(self.sys2ind), len(self.sys2ind)))
            queries = []
            keys = []
            new_parents = set()

            for parent in cur_parents:
                children = list(sub_tree.successors(parent))
                for child in children:
                    if direction == 'forward':
                        mask[self.sys2ind[parent], self.sys2ind[child]] = 1
                        if format == 'indices':
                            queries.append(self.sys2ind[parent])
                            keys.append(self.sys2ind[child])
                    elif direction == 'backward':
                        mask[self.sys2ind[child], self.sys2ind[parent]] = 1
                        if format == 'indices':
                            queries.append(self.sys2ind[child])
                            keys.append(self.sys2ind[parent])
                    new_parents.add(child)
            if mask.sum() == 0:
                break
            if format == 'indices':
                result_mask = mask[queries, :]
                result_mask = result_mask[:, keys]
                result_mask = torch.tensor(result_mask, dtype=torch.float32)
                if self.dense_attention:
                    result_mask = torch.ones_like(result_mask)
                result_masks.append({"query": torch.tensor(queries, dtype=torch.long), "key": torch.tensor(keys, dtype=torch.long), 'mask': result_mask})
            else:
                result_mask = torch.tensor(mask, dtype=torch.float32)
                if self.dense_attention:
                    result_mask = torch.ones_like(result_mask)
                result_masks.append(result_mask)

            cur_parents = new_parents

        if direction == 'forward':
            result_masks.reverse()
        return result_masks

    def get_nested_subtree_mask(self, subtree_order, direction='forward', format='indices', sys_list=None):
        nested_subtrees = [self.get_subtree_mask(subtree_type, direction=direction, format=format) for subtree_type in subtree_order]
        return nested_subtrees

    def mask_subtree_mask_by_mut(self, tree_mask, mut_vector):
        non_zero_index_tree = np.where(tree_mask!=0)
        result_mask = np.zeros_like(tree_mask)
        mut_genes = np.where(np.array(mut_vector)!=0)[0]
        mut_systems = np.where(self.gene2sys_mask[:, mut_genes]!=0)[0]
        for x, y in zip(*non_zero_index_tree):
            if np.any([mut_system in self.descendant_dict_ind[y] for mut_system in mut_systems]):
                result_mask[x, y] =1
        return torch.tensor(result_mask, dtype=torch.float32)



