import pandas as pd
import numpy as np
import networkx as nx
import torch
import copy
import torch.nn.functional as F
from itertools import product

class TreeParser(object):

    def __init__(self, ontology, dense_attention=False, sys_annot_file=None):
        ontology = pd.read_csv(ontology, sep='	', names=['parent', 'child', 'interaction'])
        ontology['parent'] = ontology['parent'].astype(str)
        ontology['child'] = ontology['child'].astype(str)
        self.dense_attention = dense_attention
        if sys_annot_file:
            sys_descriptions = pd.read_csv(sys_annot_file, header=None, names=['Term', 'Term_Description'], index_col=0, sep='	')

            self.sys_annot_dict = sys_descriptions.to_dict()["Term_Description"]
        else:
            self.sys_annot_dict = None
        self.init_ontology(ontology)

    def init_ontology(self, ontology_df, inplace=True, verbose=True):
        # If inplace is False, work on a deep copy of self.
        if not inplace:
            obj = copy.deepcopy(self)
        else:
            obj = self

        # Initialize ontology and various attributes
        print("Processing Ontology dataframe...")
        obj.ontology = ontology_df
        obj.sys_df = ontology_df.loc[ontology_df['interaction'] != 'gene']
        obj.gene2sys_df = ontology_df.loc[ontology_df['interaction'] == 'gene']
        obj.sys_graph = nx.from_pandas_edgelist(obj.sys_df, create_using=nx.DiGraph(),
                                                 source='parent', target='child', edge_attr='interaction')
        obj.interaction_types = obj.sys_df['interaction'].unique()
        obj.interaction_dict = { (row['parent'], row['child']): row['interaction']
                                 for i, row in obj.sys_df.iterrows() }

        # Define system and gene nodes explicitly
        system_nodes = sorted(list(set(obj.sys_df['parent'].unique()) | set(obj.sys_df['child'].unique())))
        gene_nodes = sorted(list(obj.gene2sys_df['child'].unique()))

        print("Building system and gene indices dictionary..")
        obj.sys2ind = { system: i for i, system in enumerate(system_nodes) }
        obj.ind2sys = { i: system for system, i in obj.sys2ind.items() }
        obj.gene2ind = { gene: index for index, gene in enumerate(gene_nodes) }
        obj.ind2gene = { index: gene for index, gene in enumerate(gene_nodes) }

        obj.n_systems = len(obj.sys2ind)
        obj.n_genes = len(obj.gene2ind)

        sys2gene_grouped_by_sys = obj.gene2sys_df.groupby('parent')
        sys2gene_grouped_by_gene = obj.gene2sys_df.groupby('child')

        obj.sys2gene_dict = { sys: sys2gene_grouped_by_sys.get_group(sys)['child'].values.tolist()
                          for sys in sys2gene_grouped_by_sys.groups.keys() if sys in obj.sys2ind}

        # Initialize sys2gene and gene2sys
        obj.sys2gene = copy.deepcopy(obj.sys2gene_dict)
        obj.gene2sys = {}
        for sys, genes in obj.sys2gene.items():
            for gene in genes:
                if gene in obj.gene2sys:
                    obj.gene2sys[gene].append(sys)
                else:
                    obj.gene2sys[gene] = [sys]

        obj.system2system_mask = np.full((len(obj.sys2ind), len(obj.sys2ind)), -10**4)
        for parent_system, child_system in zip(obj.sys_df['parent'], obj.sys_df['child']):
            obj.system2system_mask[obj.sys2ind[parent_system], obj.sys2ind[child_system]] = 0

        obj.gene2sys_mask = np.full((int(np.ceil((obj.n_systems+1)/8)*8), (int(np.ceil((obj.n_genes+1)/8)*8))), -10**4)#np.zeros((len(obj.sys2ind), len(obj.gene2ind)))
        obj.sys2gene_dict = {system: [] for system in system_nodes}
        obj.gene2sys_dict = {gene: [] for gene in gene_nodes}
        print("Creating masks..")
        for system, gene in zip(obj.gene2sys_df['parent'], obj.gene2sys_df['child']):
            if system in obj.sys2ind and gene in obj.gene2ind:
                obj.gene2sys_mask[obj.sys2ind[system], obj.gene2ind[gene]] = 0.
                obj.sys2gene_dict[system].append(gene)
                obj.gene2sys_dict[gene].append(system)

        if obj.dense_attention:
            obj.gene2sys_mask = torch.ones_like(torch.tensor(obj.gene2sys_mask))
        obj.sys2gene_mask = obj.gene2sys_mask.T
        obj.subtree_types = obj.sys_df['interaction'].unique()

        obj.node_height_dict = obj.compute_node_heights()

        systems = list(obj.sys2ind.keys())
        sys2gene_full = {sys: obj.sys2gene.get(sys, []) for sys in systems}

        for sys in reversed(list(nx.topological_sort(obj.sys_graph))):
            # Get genes of the current system
            current_genes = set(sys2gene_full.get(sys, []))
            # Add genes from all its children
            for child in obj.sys_graph.successors(sys):
                current_genes.update(sys2gene_full.get(child, []))
            sys2gene_full[sys] = list(current_genes)
        obj.sys2gene_full = sys2gene_full

        obj.gene2sys_full = {}
        for sys, genes in obj.sys2gene_full.items():
            for gene in genes:
                if gene in obj.gene2sys_full:
                    obj.gene2sys_full[gene].append(sys)
                else:
                    obj.gene2sys_full[gene] = [sys]

        obj.descendant_dict = { system: list(nx.descendants(obj.sys_graph, system)) + [system]
                                for system in systems }
        obj.descendant_dict_ind = { obj.sys2ind[key]: [obj.sys2ind[descendant] for descendant in value]
                                    for key, value in obj.descendant_dict.items() }


        obj.gene2gene_mask = np.zeros((len(obj.gene2ind), len(obj.gene2ind)))
        obj.gene2sys_graph = nx.from_pandas_edgelist(obj.gene2sys_df, create_using=nx.DiGraph(),
                                                     source='parent', target='child')
        for node in obj.gene2sys_graph.nodes:
            if obj.gene2sys_graph.in_degree(node) == 0:
                children_genes = [target for source, target in obj.gene2sys_graph.out_edges(node)]
                for child_i in children_genes:
                    for child_j in children_genes:
                        obj.gene2gene_mask[obj.gene2ind[child_i], obj.gene2ind[child_j]] = 1
        # If not modifying self, return the updated copy.
        if verbose:
            print("%d Systems are queried" % obj.n_systems)
            print("%d Genes are queried" % obj.n_genes)
            #print("Total %d Gene-System interactions are queried" % obj.gene2sys_mask.sum())
            print("Building descendant dict")
            print("Subtree types: ", obj.subtree_types)
        if not inplace:
            return obj

    def build_mask(self, ordered_query, ordered_key, query2key_dict, interaction_value=0, mask_value=-10**4):
        mask = np.full((int(np.ceil((len(ordered_query)+1)/8)*8), (int(np.ceil((len(ordered_key)+1)/8)*8))), mask_value)
        query2ind = {q: i for i, q in enumerate(ordered_query)}
        ind2query = {i: q for i, q in enumerate(ordered_query)}
        key2ind = {k: i for i, k in enumerate(ordered_key)}
        ind2key = {i: k for i, k in enumerate(ordered_key)}
        for i, query in enumerate(ordered_query):
            if query in query2key_dict.keys():
                keys = query2key_dict[query]
                for key in keys:
                    mask[i, key2ind[key]] = interaction_value

        return query2ind, ind2query, key2ind, ind2key, mask

    def summary(self, system=True, gene=True):
        """
        Print a summary of the systems and genes in the ontology.

        Parameters:
        ----------
        system : bool, optional
            Whether to include system summary. Default is True.
        gene : bool, optional
            Whether to include gene summary. Default is True.
        """
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
            interaction_type = tree.edges[(node, child)]['interaction']
            #if (interaction_type == 'is_a') | (interaction_type == 'part_of') | (interaction_type == 'default'):
            child_genes = self.add_child_genes_to_parents(tree, child, gene_dict)
            genes.update(child_genes)

        # Update the gene set for the current node
        gene_dict[node] = genes
        return genes

    def delete_parent_genes_from_child(self, tree, node, gene_dict):
        """
        Recursively removes parent genes from child nodes.

        :param tree: NetworkX DiGraph representing the tree.
        :param node: Current node to process.
        :param gene_dict: Dictionary mapping nodes to their gene sets.
        """
        # Base case: if the node is a leaf, return its genes
        if node not in gene_dict.keys():
            return set()

        if tree.out_degree(node) == 0:
            return set(gene_dict[node])

        # Initialize the set of genes for the current node with its own genes
        genes = set(gene_dict[node])

        # Recurse for all children
        for child in tree.successors(node):
            # Recursively process the child's genes
            interaction_type = tree.edges[(node, child)]['interaction']
            #if (interaction_type == 'is_a') | (interaction_type == 'part_of') | (interaction_type == 'default'):
            child_genes = self.delete_parent_genes_from_child(tree, child, gene_dict)
            # Remove parent genes from the child's genes
            child_genes.difference_update(genes)
            # Update the child's gene set in the dictionary
            gene_dict[child] = child_genes

        # Update the gene set for the current node in the dictionary
        gene_dict[node] = genes
        return genes

    

    def collapse(self, to_keep=None, min_term_size=2, verbose=True, inplace=False):
        """
        Remove redundant and empty terms while preserving hierarchical relations. Each child of a removed term T is
        connected to every parent of T. This operation is commutative, meaning the order of removal does not matter.
        This function is a Python implementation of the collapseRedundantNodes script in the ddot package.

        Parameters
        ----------
        to_keep : list, optional
            Systems to retain. Default is None.
        min_term_size : int, optional
            Minimum term size to retain. Default is 2.
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        if not inplace:
            obj = copy.deepcopy(self)
        else:
            obj = self

        # Propagate genes up the hierarchy
        systems = list(obj.sys2ind.keys())
        sys2gene_dict = {sys: obj.sys2gene.get(sys, []) for sys in systems}

        # Create a temporary graph for propagation
        temp_graph = obj.sys_graph.copy()

        for sys in systems:
            if sys not in sys2gene_dict:
                sys2gene_dict[sys] = []

        # Propagate genes from children to parents
        for sys in reversed(list(nx.topological_sort(temp_graph))):
            # Get genes of the current system
            current_genes = set(sys2gene_dict.get(sys, []))
            # Add genes from all its children
            for child in temp_graph.successors(sys):
                current_genes.update(sys2gene_dict.get(child, []))
            sys2gene_dict[sys] = list(current_genes)
        obj.sys2gene_full = sys2gene_dict

        # Determine systems to collapse
        # 1. Find redundant systems (same set of genes in sys2gene_full)
        hash_to_terms = {}
        for term, genes in obj.sys2gene_full.items():
            h = hash(frozenset(genes))
            if h not in hash_to_terms:
                hash_to_terms[h] = []
            hash_to_terms[h].append(term)

        to_collapse_redundant = []
        for h, terms in hash_to_terms.items():
            if len(terms) > 1:
                # Keep one term, collapse the others.
                # Sorting by name provides deterministic behavior.
                terms.sort()
                to_collapse_redundant.extend(terms[1:])

        # 2. Find small systems (few genes in sys2gene_full)
        to_collapse_small = [
            term for term, genes in obj.sys2gene_full.items()
            if len(genes) < min_term_size
        ]

        to_collapse = list(set(to_collapse_redundant + to_collapse_small))

        # 3. If to_keep is specified, do not collapse them
        if to_keep is not None:
            to_collapse = [term for term in to_collapse if term not in to_keep]

        if verbose:
            print(f"Systems to collapse: {len(to_collapse)}")

        # Create a new graph for the collapsed ontology
        collapsed_sys_graph = obj.sys_graph.copy()

        for node in to_collapse:
            if node in collapsed_sys_graph:
                parents = list(collapsed_sys_graph.predecessors(node))
                children = list(collapsed_sys_graph.successors(node))
                for p in parents:
                    for c in children:
                        if not collapsed_sys_graph.has_edge(p, c):
                            collapsed_sys_graph.add_edge(p, c)
                collapsed_sys_graph.remove_node(node)

        # Rebuild the ontology DataFrame from the collapsed graph
        interactions = []
        for u, v, data in collapsed_sys_graph.edges(data=True):
            interactions.append((u, v, data.get('interaction', 'default')))

        # Add gene interactions, ensuring collapsed systems are excluded
        for sys, genes in obj.sys2gene.items():
            if sys in collapsed_sys_graph:
                for gene in genes:
                    interactions.append((sys, gene, 'gene'))

        collapsed_ontology_df = pd.DataFrame(interactions, columns=['parent', 'child', 'interaction'])

        # Reinitialize the TreeParser object with the new collapsed ontology
        obj.init_ontology(collapsed_ontology_df, inplace=True, verbose=verbose)

        if not inplace:
            return obj

    def sys_graph_to_ontology_table(self, sys_graph, sys2gene):
        """
        Convert a system graph and system-to-gene mapping to an ontology DataFrame.

        Parameters:
        ----------
        sys_graph : NetworkX DiGraph
            Graph representing the system relationships.
        sys2gene : dict
            Dictionary mapping systems to their genes.

        Returns:
        -------
        DataFrame
            Ontology table containing parent-child interactions.
        """
        interactions = [(parent, child, sys_graph.edges[(parent, child)]['interaction']) for parent, child in sys_graph.edges]
        for sys, genes in sys2gene.items():
            for gene in genes:
                interactions.append((sys, gene, 'gene'))
        ontology_df = pd.DataFrame(interactions, columns=['parent', 'child', 'interaction'])
        return  ontology_df

    def retain_genes(self, gene_list, inplace=False):
        """
        retain genes in input gene list and rebuild ontology

        Parameters:
        ----------
        gene_list : list, tuple
            list of genes to retain
        """
        gene2sys_to_keep = self.gene2sys_df.loc[self.gene2sys_df.child.isin(gene_list)]
        ontology_df_new = pd.concat([self.sys_df, gene2sys_to_keep])
        if inplace:
            self.init_ontology(ontology_df_new, inplace=inplace)
        else:
            return self.init_ontology(ontology_df_new, inplace=inplace)

    def compute_node_heights(self):
        """
        Compute the heights of nodes in the ontology graph.

        Returns:
        -------
        dict
            Dictionary mapping nodes to their heights.
        """
        # Ensure the graph is a tree (i.e., has a single root)
        if not nx.is_directed_acyclic_graph(self.sys_graph):
            raise ValueError("The input graph must be a directed acyclic graph (DAG).")

        # Start by identifying leaf nodes
        leaf_nodes = [node for node in self.sys_graph.nodes if self.sys_graph.out_degree(node) == 0]

        # Dictionary to store node heights
        heights = {node: 0 for node in leaf_nodes}  # Leaf nodes have height 0

        # Process nodes in reverse topological order
        for node in reversed(list(nx.topological_sort(self.sys_graph))):
            if node not in heights:  # Non-leaf nodes
                # Height is 1 + max height of its children
                heights[node] = 1 + max(heights[child] for child in self.sys_graph.successors(node))
        return heights


    def get_descendants_sorted_by_height(self, node):
        """
        Get descendants of a node sorted by their height.

        Parameters:
        ----------
        node : str
            Node for which to retrieve sorted descendants.

        Returns:
        -------
        list
            List of descendants sorted by height.
        """
        # Compute the height of all nodes
        node_heights = self.compute_node_heights()
        # Get all descendants of the given node
        descendants = nx.descendants(self.sys_graph, node)
        # Sort descendants by height (ascending or descending)
        sorted_descendants = sorted(descendants, key=lambda x: node_heights[x])
        return sorted_descendants

    def save_ontology(self, out_dir):
        """
        Write the ontology to a GMT file format.

        Parameters:
        ----------
        output_path : str
            Path to the output GMT file.
        """
        self.ontology.to_csv(out_dir, sep='\t', index=False, header=None)

    def write_gmt(self, out_dir):
        """
        Write the ontology to a GMT file format.

        Parameters:
        ----------
        out_dir : str
            Path to the output GMT file.
        """
        f = open(out_dir, 'w')
        for sys, genes in self.sys2gene_full.items():
            if self.sys_annot_dict is not None:
                lines = "\t".join([sys, self.sys_annot_dict[sys]] + list(genes))
            else:
                lines = "\t".join([sys]+list(genes))
            f.write(lines+'\n')
            f.flush()
        f.close()

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


    def get_hierarchical_interactions(self, interaction_types, direction='forward', format='indices', sys_ind_alias_dict=None):
        """
        Compute hierarchical interaction masks across the system graph.

        This function traverses the system graph starting from the root nodes (nodes with no incoming
        edges) and iteratively computes interaction masks for each hierarchical level. At each level,
        it groups interactions by the provided interaction types and aggregates the edge information
        into mask matrices. Depending on the `direction` parameter, the function treats the connection
        between nodes either in the forward direction (parent-to-child) or in the backward direction
        (child-to-parent). The output format is flexible; by default, it returns index-based dictionaries
        that include corresponding query indices, key indices, and mask tensors. When an alias mapping
        dictionary is supplied, the indices are transformed accordingly.

        Parameters
        ----------
        interaction_types : iterable
            A collection of interaction types to be processed. Only interactions whose type is present
            in this iterable will be included in the masks.
        direction : str, optional
            The direction in which to traverse and record the interactions. Use 'forward' (default) to
            record interactions from parent to child, or 'backward' to reverse the roles of parent and child.
        format : str, optional
            The output format for the interaction masks. If set to 'indices' (default), the function
            returns, for each hierarchical level, a dictionary where each key (an interaction type) maps to
            a sub-dictionary containing:
                - "query": A torch.LongTensor of query indices.
                - "key": A torch.LongTensor of key indices.
                - "mask": A torch.FloatTensor representing the corresponding interaction mask.
            Any value other than 'indices' causes the function to return, for each level, a dictionary mapping
            each interaction type to a torch.FloatTensor mask.
        sys_ind_alias_dict : dict, optional
            If provided, this dictionary maps system indices to alternative alias indices. The function applies
            this mapping (via the `alias_indices` method) to the query and key indices in the result when the
            format is 'indices'.

        Returns
        -------
        list
            A list of hierarchical interaction masks for each level of the system graph traversal.
            Each element corresponds to a hierarchical level and is structured as follows:
                - When `format` is 'indices': a dictionary where keys are interaction types and values are
                  sub-dictionaries with "query", "key", and "mask" tensors.
                - Otherwise: a dictionary mapping each interaction type to a torch.FloatTensor mask.

        Notes
        -----
        - The traversal begins from all root nodes of the system graph (nodes with in_degree == 0) and
          continues iteratively until no more edges contribute to the masks.
        - If the `direction` is 'forward', the resulting list is reversed at the end to maintain the proper
          hierarchical order.
        - When the instance attribute `dense_attention` is True, all computed masks are replaced with tensors
          of ones.
        """
        tree_roots = set([node for node in self.sys_graph.nodes() if self.sys_graph.in_degree(node)==0])
        result_masks = []
        cur_parents = set(tree_roots)
        while cur_parents:
            masks = {interaction_type:np.full((len(self.sys2ind), len(self.sys2ind)), -10**4) for interaction_type in interaction_types}
            queries = {interaction_type: [] for interaction_type in interaction_types}
            keys = {interaction_type: [] for interaction_type in interaction_types}
            new_parents = set()

            for parent in cur_parents:
                children = list(self.sys_graph.successors(parent))
                for child in children:
                    edge_type = self.interaction_dict[(parent, child)]
                    if direction == 'forward':
                        masks[edge_type][self.sys2ind[parent], self.sys2ind[child]] = 0
                        if format == 'indices':
                            queries[edge_type].append(self.sys2ind[parent])
                            keys[edge_type].append(self.sys2ind[child])
                    elif direction == 'backward':
                        masks[edge_type][self.sys2ind[child], self.sys2ind[parent]] = 0
                        if format == 'indices':
                            queries[edge_type].append(self.sys2ind[child])
                            keys[edge_type].append(self.sys2ind[parent])
                    new_parents.add(child)
            if sum([len(np.where(mask==0)) for mask in masks.values()]) == 0:
                break
            if format == 'indices':
                result_dict = {}

                for interaction_type, mask in masks.items():
                    if len(queries[interaction_type]) == 0:
                        continue
                    queries[interaction_type] = sorted(list(set(queries[interaction_type])))
                    keys[interaction_type] = sorted(list(set(keys[interaction_type])))
                    result_mask = mask[queries[interaction_type], :]
                    result_mask = result_mask[:, keys[interaction_type]]
                    result_mask = torch.tensor(result_mask, dtype=torch.float32)
                    query_padded, key_padded, result_mask = self.pad_query_key_mask(queries[interaction_type],
                                                                                    keys[interaction_type],
                                                                                    result_mask,
                                                                                    query_padding_index=self.n_systems,
                                                                                    key_padding_index=self.n_systems)
                    if self.dense_attention:
                        result_mask = torch.ones_like(result_mask)
                    if direction=='forward':
                        result_dict[interaction_type] = {"query": torch.tensor(query_padded, dtype=torch.long),
                                                         "key": torch.tensor(key_padded, dtype=torch.long),
                                                         "query_indices": torch.tensor(queries[interaction_type], dtype=torch.long),
                                                         "key_indices": torch.tensor(keys[interaction_type], dtype=torch.long),
                                                         "mask": result_mask}
                    else:
                        result_dict[interaction_type] = {"query": torch.tensor(key_padded, dtype=torch.long),
                                                         "key": torch.tensor(query_padded, dtype=torch.long),
                                                         "query_indices": torch.tensor(keys[interaction_type], dtype=torch.long),
                                                         "key_indices": torch.tensor(queries[interaction_type], dtype=torch.long),
                                                         "mask": result_mask.T}
                result_masks.append(result_dict)
            else:
                result_mask = {torch.tensor(mask, dtype=torch.float32) for mask in masks}
                if self.dense_attention:
                    result_mask = {interaction_type: torch.ones_like(mask) for interaction_type, mask in masks.items()}
                result_masks.append(result_mask)
            cur_parents = new_parents
        if direction == 'forward':
            result_masks.reverse()
        return result_masks

    @staticmethod
    def pad_query_key_mask(query, key, tensor, query_padding_index=0, key_padding_index=0, padding_value=-10 ** 4):
        n_query, n_key = len(query), len(key)
        n_query_pad = int(np.ceil(n_query / 8) * 8) - n_query
        n_key_pad = int(np.ceil(n_key / 8) * 8) - n_key
        padded_tensor = F.pad(tensor, (0, n_key_pad, 0, n_query_pad), value=padding_value)
        padded_query = query + [query_padding_index] * n_query_pad
        padded_key = key + [key_padding_index] * n_key_pad
        return padded_query, padded_key, padded_tensor

    def get_nested_subtree_mask(self, subtree_order, direction='forward', format='indices'):
        nested_subtrees = [self.get_subtree_mask(subtree_type, direction=direction, format=format) for subtree_type in subtree_order]
        return nested_subtrees

    def get_target_components(self, target_go):
        if self.tree_parser.node_height_dict[target_go] != 0:
            target_gos = self.tree_parser.get_descendants_sorted_by_height(target_go) + [target_go]
        else:
            target_gos = [target_go]

        target_genes = self.tree_parser.sys2gene_full[target_go]
        return target_gos, target_genes

    def get_target_indices(self, target_gos, target_genes):
        """Fetch integer indices from parser for the target GO, gene"""
        target_go_inds = [self.sys2ind[go] for go in target_gos]
        target_gene_inds = [self.gene2ind[g] for g in target_genes]
        return target_go_inds, target_gene_inds

    def get_partial_sys2sys_masks(self, target_gos):
        """Build partial adjacency masks and edges from GO subgraph."""
        sys2ind_partial = {go: i for i, go in enumerate(target_gos)}
        all_paths = self.get_paths_from_node_to_leaves(self.sys_graph, target_gos[-1])
        all_edges_forward = self.get_edges_from_paths(all_paths)

        sys2sys_forward_partial = torch.zeros(size=(len(target_gos), len(target_gos)))
        sys2sys_backward_partial = torch.zeros(size=(len(target_gos), len(target_gos)))

        for target, source in all_edges_forward:
            sys2sys_forward_partial[sys2ind_partial[target], sys2ind_partial[source]] = 1
            sys2sys_backward_partial[sys2ind_partial[source], sys2ind_partial[target]] = 1

        return sys2sys_forward_partial, sys2sys_backward_partial, sys2ind_partial, all_edges_forward

    def get_paths_from_node_to_leaves(self, G, start_node):
        all_paths = []
        for node in G.nodes:
            if G.out_degree(node) == 0:  # If it's a leaf node
                paths = list(nx.all_simple_paths(G, start_node, node))
                all_paths.extend(paths)
        return all_paths

    def get_edges_from_paths(self, paths):
        edges = set()
        for path in paths:
            edges.update([(path[i], path[i + 1]) for i in range(len(path) - 1)])
        return list(edges)

    def _compute_gene_order_from_snps(self):
        """
        Compute new gene order based on SNP chromosome/block structure
        """
        # SNPs are already ordered by chromosome and block
        # Group genes by their first appearing SNP
        gene_to_first_snp = {}
        for snp_idx, snp_id in enumerate(self.ind2snp.values()):
            # Get genes connected to this SNP
            connected_genes = [self.ind2gene[gene_idx]
                               for gene_idx in range(self.n_genes)
                               if self.snp2gene_mask[gene_idx, snp_idx] != -10 ** 4]
            for gene in connected_genes:
                if gene not in gene_to_first_snp:
                    gene_to_first_snp[gene] = snp_idx
        # Sort genes by their first appearing SNP index
        ordered_genes = sorted(gene_to_first_snp.keys(),
                               key=lambda g: gene_to_first_snp[g])
        # Add any genes not connected to SNPs at the end
        all_genes = set(self.ind2gene.values())
        unconnected_genes = all_genes - set(ordered_genes)
        ordered_genes.extend(sorted(unconnected_genes))
        return ordered_genes

    def _create_gene_remapping(self, new_gene_order):
        """
        Create mapping from old gene indices to new gene indices
        """
        old_to_new = {}
        new_to_old = {}

        for new_idx, gene_id in enumerate(new_gene_order):
            old_idx = self.gene2ind[gene_id]
            old_to_new[old_idx] = new_idx
            new_to_old[new_idx] = old_idx

        return {
            'old_to_new': old_to_new,
            'new_to_old': new_to_old,
            'new_gene_order': new_gene_order
        }

    def create_subset_index_mapping(self, subset_system_indices):
        """
        Create index mapping between full system indices and subset indices.

        Parameters
        ----------
        subset_systems : list
            List of system indices in the subset, ordered by their new indices.
            The order determines the new indices (0, 1, 2, ..., len(subset)-1).

        Returns
        -------
        dict
            Dictionary containing:
            - 'full_to_subset': mapping from full sys2ind indices to subset indices
            - 'subset_to_full': mapping from subset indices to full sys2ind indices
            - 'subset_sys2ind': new system to index mapping for subset
            - 'subset_ind2sys': new index to system mapping for subset

        Examples
        --------
        >>> subset_systems = ['GO:0008150', 'GO:0006412', 'GO:0003674']
        >>> mapping = tree_parser.create_subset_index_mapping(subset_systems)
        >>> print(mapping['subset_sys2ind'])  # {'GO:0008150': 0, 'GO:0006412': 1, 'GO:0003674': 2}
        """
        # Validate that all subset systems exist in the full system


        # Create new index mappings for subset
        subset_sys2ind = {system: i for i, system in enumerate(subset_system_indices)}
        subset_ind2sys = {i: system for i, system in enumerate(subset_system_indices)}

        # Create bidirectional mapping between full and subset indices
        full_to_subset = {}
        subset_to_full = {}

        for subset_idx, system in enumerate(subset_system_indices):
            full_idx = system
            full_to_subset[full_idx] = subset_idx
            subset_to_full[subset_idx] = full_idx

        return {
            'full_to_subset': full_to_subset,
            'subset_to_full': subset_to_full,
            'subset_sys2ind': subset_sys2ind,
            'subset_ind2sys': subset_ind2sys
        }

    def remap_hierarchical_indices(self, hierarchical_masks, index_mapping, filter_missing=True):
        """
        Remap indices in hierarchical masks from full system indices to subset indices.

        This function takes hierarchical masks that use full system indices and remaps
        them to use subset indices. It handles both 'indices' format and mask format.

        Parameters
        ----------
        hierarchical_masks : list
            List of hierarchical masks as returned by get_hierarchical_interactions().
        index_mapping : dict
            Index mapping dictionary as returned by create_subset_index_mapping().
        filter_missing : bool, optional
            If True, filters out query/key pairs where either index is not in the subset.
            If False, raises an error when encountering missing indices. Default is True.

        Returns
        -------
        list
            Remapped hierarchical masks with subset indices.

        Examples
        --------
        >>> # Get full hierarchical masks
        >>> full_masks = tree_parser.get_hierarchical_interactions(['is_a'], format='indices')
        >>> # Create subset mapping
        >>> mapping = tree_parser.create_subset_index_mapping(subset_systems)
        >>> # Remap to subset indices
        >>> subset_masks = tree_parser.remap_hierarchical_indices(full_masks, mapping)
        """
        full_to_subset = index_mapping['full_to_subset']
        subset_size = len(index_mapping['subset_sys2ind'])

        remapped_masks = []

        for level_masks in hierarchical_masks:
            remapped_level = {}

            for interaction_type, mask_data in level_masks.items():
                # Handle 'indices' format
                original_queries = mask_data['query_indices'].tolist()
                original_keys = mask_data['key_indices'].tolist()

                new_queries = [full_to_subset[q] for q in original_queries if q in full_to_subset.keys()]
                new_keys = [full_to_subset[k] for k in original_keys if k in full_to_subset.keys()]
                if (len(new_queries) == 0) or (len(new_keys) == 0):
                    continue
                new_query_indices = [i for i, q in enumerate(original_queries) if q in full_to_subset.keys()]
                new_key_indices = [i for i, k in enumerate(original_keys) if k in full_to_subset.keys()]
                new_mask = mask_data['mask'][new_query_indices][:, new_key_indices]
                if (new_mask == 0).sum() == 0:
                    continue
                query_padded, key_padded, result_mask = self.pad_query_key_mask(new_queries,
                                                                                new_keys,
                                                                                new_mask,
                                                                                query_padding_index=subset_size,
                                                                                key_padding_index=subset_size)
                remapped_level[interaction_type] = {
                    'query': torch.tensor(query_padded, dtype=torch.long),
                    'key': torch.tensor(key_padded, dtype=torch.long),
                    'query_indices': torch.tensor(new_queries, dtype=torch.long),
                    'key_indices': torch.tensor(new_keys, dtype=torch.long),
                    'mask': result_mask
                }
            if len(remapped_level) != 0:
                remapped_masks.append(remapped_level)

        return remapped_masks

    @staticmethod
    def alias_indices(indices, source_ind2id_dict:dict, target_id2ind_dict: dict):
        return [target_id2ind_dict[source_ind2id_dict[ind]] for ind in indices]