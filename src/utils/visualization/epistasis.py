import networkx as nx
import pygraphviz as pgv
import math

class EpistasisVisualizer:
    """
    A class for visualizing epistatic interactions within a hierarchical system.

    Attributes:
    -----------
    tree_parser : object
        An object that provides access to the system graph and annotations.
    sys_graph : networkx.DiGraph
        The hierarchical graph representing system relationships.
    sys_annot_dict : dict
        A dictionary mapping system nodes to their annotations.

    Methods:
    --------
    visualize_epistasis(target_system, epistatic_interactions)
        Generates a visual representation of the epistatic interactions within a subgraph of the system hierarchy.
    """

    def __init__(self, tree_parser):
        """
        Initializes the EpistasisVisualizer.

        Parameters:
        -----------
        tree_parser : object
            An object containing system graph data and annotations.
        """
        self.tree_parser = tree_parser
        self.sys_graph = self.tree_parser.sys_graph
        self.sys_annot_dict = tree_parser.sys_annot_dict

    def visualize_epistasis(self, target_system, epistatic_interactions):
        """
        Generates a directed graph visualization highlighting epistatic interactions.

        Parameters:
        -----------
        target_system : str
            The root system from which to extract the subgraph.
        epistatic_interactions : list of tuples
            A list of gene interactions, where each tuple represents an epistatic interaction (gene1, gene2).

        Returns:
        --------
        sub_G : pygraphviz.AGraph
            A Graphviz-directed graph visualization of the epistatic interactions.
        """

        # Extract relevant subgraph nodes
        descendants = nx.descendants(self.sys_graph, target_system)
        sub_nodes = set(descendants) | {target_system}
        systems = list(descendants) + [target_system]

        # Get sub-hierarchy edges
        sub_edges = self.get_sub_hierarchy_edges(self.sys_graph, target_system)
        sub_edges_for_graphviz = [(child, parent) for parent, child in sub_edges]

        # Identify epistatic genes and their associated systems
        epistatic_genes = sorted(set(sum(epistatic_interactions, [])))
        system_to_highlight = []

        for sys in systems:
            if sys not in self.tree_parser.sys2gene:
                continue
            for epistatic_gene in epistatic_genes:
                if epistatic_gene in self.tree_parser.sys2gene[sys]:
                    sub_edges_for_graphviz.append((epistatic_gene, sys))
                    system_to_highlight.append(sys)

        # Create a Graphviz graph
        sub_G = pgv.AGraph(directed=True)
        sub_G.graph_attr.update(rankdir="BT")
        sub_G.add_edges_from(sub_edges_for_graphviz)

        # Customize node attributes
        for node in sub_G.nodes():
            node_obj = sub_G.get_node(node)
            if node in systems:
                node_obj.attr['shape'] = 'circle'
                node_obj.attr['style'] = 'filled'
                node_obj.attr['label'] = ''
                node_obj.attr['width'] = node_obj.attr['height'] = f"{math.log(len(self.tree_parser.sys2gene_full[node])) / 2}"

                # Assign labels if annotation dictionary exists
                if self.sys_annot_dict:
                    node_obj.attr['xlabel'] = self.sys_annot_dict.get(node, node)
                else:
                    node_obj.attr['label'] = node

                # Highlight systems involved in epistasis
                if str(node) in system_to_highlight:
                    node_obj.attr.update(fillcolor='red', color='black')
                else:
                    node_obj.attr.update(fillcolor='white', color='gray')
            else:
                node_obj.attr['shape'] = 'ellipse'
                node_obj.attr['style'] = 'filled'
                node_obj.attr['fillcolor'] = 'lightgray'

        # Add and style epistatic interaction edges
        for gene1, gene2 in epistatic_interactions:
            sub_G.add_edge(gene1, gene2)
            edge_obj = sub_G.get_edge(gene1, gene2)
            edge_obj.attr.update(color="red", dir="both", style="dashed")

        # Layout and return the graph
        sub_G.layout(prog="dot")
        return sub_G

    def get_sub_hierarchy_edges(self, graph, start):
        """
        Returns all edges in the sub-hierarchy (subtree) from 'start' down to its leaves.

        In a DiGraph, 'successors(node)' are the direct children of 'node'.
        This function traverses all children, grandchildren, etc.,
        collecting edges until it reaches leaves (no children).
        """
        edges = []
        for child in graph.successors(start):
            # Add edge from start to child
            edges.append((start, child))
            # Recursively get child's sub-hierarchy
            edges.extend(self.get_sub_hierarchy_edges(graph, child))
        return edges