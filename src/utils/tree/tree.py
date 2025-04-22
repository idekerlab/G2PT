import pandas as pd
import numpy as np
import networkx as nx
import torch
import copy
from itertools import product

class TreeParser(object):

    def __init__(self, ontology, dense_attention=False, sys_annot_file=None):
        ontology = pd.read_csv(ontology, sep='\t', names=['parent', 'child', 'interaction'])
        self.dense_attention = dense_attention
        if sys_annot_file:
            sys_descriptions = pd.read_csv(sys_annot_file, header=None, names=['Term', 'Term_Description'], index_col=0, sep='\t')

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
        obj.ontology = ontology_df
        obj.sys_df = ontology_df.loc[ontology_df['interaction'] != 'gene']
        obj.gene2sys_df = ontology_df.loc[ontology_df['interaction'] == 'gene']
        obj.sys_graph = nx.from_pandas_edgelist(obj.sys_df, create_using=nx.DiGraph(),
                                                 source='parent', target='child', edge_attr='interaction')
        obj.interaction_types = obj.sys_df['interaction'].unique()
        obj.interaction_dict = { (row['parent'], row['child']): row['interaction']
                                 for i, row in obj.sys_df.iterrows() }
        genes = obj.gene2sys_df['child'].unique()
        obj.gene2ind = { gene: index for index, gene in enumerate(genes) }
        obj.ind2gene = { index: gene for index, gene in enumerate(genes) }
        systems = np.unique(obj.sys_df[['parent', 'child']].values)
        obj.sys2ind = { system: i for i, system in enumerate(systems) }
        obj.ind2sys = { i: system for system, i in obj.sys2ind.items() }
        obj.n_systems = len(obj.sys2ind)
        obj.n_genes = len(obj.gene2ind)


        sys2gene_grouped_by_sys = obj.gene2sys_df.groupby('parent')
        sys2gene_grouped_by_gene = obj.gene2sys_df.groupby('child')
        sys2gene_dict = { sys: sys2gene_grouped_by_sys.get_group(sys)['child'].values.tolist()
                          for sys in sys2gene_grouped_by_sys.groups.keys() }

        # Delete genes in child system (assumes delete_parent_genes_from_child is defined)
        for sys in list(sys2gene_dict.keys()):
            obj.delete_parent_genes_from_child(obj.sys_graph, sys, sys2gene_dict)

        obj.sys2gene = copy.deepcopy(sys2gene_dict)
        obj.gene2sys = {}
        for sys, genes in obj.sys2gene.items():
            for gene in genes:
                if gene in obj.gene2sys:
                    obj.gene2sys[gene].append(sys)
                else:
                    obj.gene2sys[gene] = [sys]

        obj.system2system_mask = np.zeros((len(obj.sys2ind), len(obj.sys2ind)))
        for parent_system, child_system in zip(obj.sys_df['parent'], obj.sys_df['child']):
            obj.system2system_mask[obj.sys2ind[parent_system], obj.sys2ind[child_system]] = 1

        obj.gene2sys_mask = np.zeros((len(obj.sys2ind), len(obj.gene2ind)))
        obj.sys2gene_dict = { obj.sys2ind[system]: [] for system in systems }
        obj.gene2sys_dict = { gene: [] for gene in range(obj.n_genes) }
        for system, gene in zip(obj.gene2sys_df['parent'], obj.gene2sys_df['child']):
            obj.gene2sys_mask[obj.sys2ind[system], obj.gene2ind[gene]] = 1.
            obj.sys2gene_dict[obj.sys2ind[system]].append(obj.gene2ind[gene])
            obj.gene2sys_dict[obj.gene2ind[gene]].append(obj.sys2ind[system])

        if obj.dense_attention:
            obj.gene2sys_mask = torch.ones_like(torch.tensor(obj.gene2sys_mask))
        obj.sys2gene_mask = obj.gene2sys_mask.T
        obj.subtree_types = obj.sys_df['interaction'].unique()

        obj.node_height_dict = obj.compute_node_heights()

        for sys in systems:
            if sys not in sys2gene_dict:
                sys2gene_dict[sys] = []
        for sys in systems:
            obj.add_child_genes_to_parents(obj.sys_graph, sys, sys2gene_dict)
        obj.sys2gene_full = sys2gene_dict

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
            print("Total %d Gene-System interactions are queried" % obj.gene2sys_mask.sum())
            print("Building descendant dict")
            print("Subtree types: ", obj.subtree_types)
        if not inplace:
            return obj

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

        Parameters
        ----------
        to_keep : list, optional
            Systems to retain. Default is None.
        min_term_size : int, optional
            Minimum term size to retain. Default is 2.
        verbose : bool, optional
            Whether to print progress messages. Default is True.
        """
        # Copy the current sys2gene mapping to preserve the original
        ont_full = copy.deepcopy(self.sys2gene_full)
        term_hash = {term: hash(tuple(genes)) for term, genes in ont_full.items()}

        # Identify terms to collapse based on redundancy and size
        to_collapse = {
            parent
            for parent in self.sys2ind
            for child in self.sys_graph[parent]
            if term_hash[parent] == term_hash[child]
        }

        # Add terms below the minimum size to the collapse set
        if min_term_size is not None:
            small_terms = {
                term for term, size in zip(self.sys2ind, [len(self.sys2gene_full[sys]) for sys in self.sys2ind])
                if size < min_term_size
            }
            to_collapse.update(small_terms)

        # Exclude terms specified in `to_keep`
        if to_keep:
            to_collapse.difference_update(to_keep)

        if verbose:
            print(f'The number of nodes to collapse: {len(to_collapse)}')

        # Sort terms to collapse by node height
        to_collapse = sorted(to_collapse, key=lambda term: self.node_height_dict[term])

        # Copy system graph and sys2gene for modification
        sys_graph_copied = copy.deepcopy(self.sys_graph)
        sys2gene_copied = copy.deepcopy(self.sys2gene)

        # Perform collapse
        sys_graph_collapsed, sys2gene_collapsed = self.delete_systems(to_collapse, sys_graph_copied, sys2gene_copied)

        # Generate the collapsed ontology
        collapsed_ontology = self.sys_graph_to_ontology_table(sys_graph_collapsed, sys2gene_collapsed)

        return collapsed_ontology

        # Reinitialize ontology with the collapsed data
        #if inplace:
        #    self.init_ontology(collapsed_ontology)
        #else:
        #    return self.init_ontology(collapsed_ontology, inplace=False)

    def delete_systems(self, systems, sys_graph, sys2gene):
        """
        Delete specified systems from the graph and update mappings.

        Parameters:
        ----------
        systems : list
            List of systems to delete.
        sys_graph : NetworkX DiGraph
            Graph representing the system relationships.
        sys2gene : dict
            Dictionary mapping systems to their genes.

        Returns:
        -------
        tuple
            Updated graph and system-to-gene dictionary.
        """
        for system in systems:
            # Get parents and children of the current system
            parents = list(sys_graph.predecessors(system))
            children = list(sys_graph.successors(system))

            # Connect every parent of the system to every child
            for parent, child in product(parents, children):
                sys_graph.add_edge(parent, child, interaction=sys_graph.edges[(system, child)]['interaction'])
            #sys_graph.add_edges_from((parent, child) for parent, child in )

            # Update sys2gene mappings for parents
            parent_genes = {parent: sys2gene.get(parent, []) for parent in parents}
            system_genes = sys2gene.get(system, [])
            for parent in parents:
                interaction_type = sys_graph.edges[(parent, system)]['interaction']
                #if (interaction_type == 'is_a') | (interaction_type == 'part_of') | (interaction_type == 'default'):
                parent_genes[parent] = list(parent_genes[parent])
                parent_genes[parent].extend(system_genes)
                sys2gene[parent] = parent_genes[parent]

            # Remove the system node
            sys_graph.remove_node(system)

            # Remove the system from sys2gene if it exists
            sys2gene.pop(system, None)

        # remove duplicated genes
        sys2gene = {sys: list(set(genes)) for sys, genes in sys2gene.items()}
        return sys_graph, sys2gene

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
            masks = {interaction_type:np.zeros((len(self.sys2ind), len(self.sys2ind))) for interaction_type in interaction_types}
            queries = {interaction_type: [] for interaction_type in interaction_types}
            keys = {interaction_type: [] for interaction_type in interaction_types}
            new_parents = set()

            for parent in cur_parents:
                children = list(self.sys_graph.successors(parent))
                for child in children:
                    edge_type = self.interaction_dict[(parent, child)]
                    if direction == 'forward':
                        masks[edge_type][self.sys2ind[parent], self.sys2ind[child]] = 1
                        if format == 'indices':
                            queries[edge_type].append(self.sys2ind[parent])
                            keys[edge_type].append(self.sys2ind[child])
                    elif direction == 'backward':
                        masks[edge_type][self.sys2ind[child], self.sys2ind[parent]] = 1
                        if format == 'indices':
                            queries[edge_type].append(self.sys2ind[child])
                            keys[edge_type].append(self.sys2ind[parent])
                    new_parents.add(child)
            if sum([mask.sum() for mask in masks.values()]) == 0:
                break
            if format == 'indices':
                result_dict = {}

                for interaction_type, mask in masks.items():
                    if len(queries[interaction_type]) == 0:
                        continue
                    result_mask = mask[queries[interaction_type], :]
                    result_mask = result_mask[:, keys[interaction_type]]
                    result_mask = torch.tensor(result_mask, dtype=torch.float32)
                    if self.dense_attention:
                        result_mask = torch.ones_like(result_mask)
                    if sys_ind_alias_dict is not None:
                        result_dict[interaction_type] = {"query": torch.tensor(self.alias_indices(queries[interaction_type], self.ind2sys, sys_ind_alias_dict), dtype=torch.long),
                                                           "key": torch.tensor(self.alias_indices(keys[interaction_type], self.ind2sys, sys_ind_alias_dict), dtype=torch.long),
                                                           'mask': result_mask}
                    else:
                        result_dict[interaction_type] = {"query": torch.tensor(queries[interaction_type], dtype=torch.long),
                                                           "key": torch.tensor(keys[interaction_type], dtype=torch.long),
                                                           'mask': result_mask}
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


    @staticmethod
    def alias_indices(indices, source_ind2id_dict:dict, target_id2ind_dict: dict):
        return [target_id2ind_dict[source_ind2id_dict[ind]] for ind in indices]