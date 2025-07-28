import networkx as nx
import pygraphviz as pgv
import matplotlib.cm as cm
from matplotlib.colors import Normalize, to_rgba, to_hex
import matplotlib as mpl
import numpy as np


class TreeVisualizer:
    """
    A class for visualizing and manipulating hierarchical ontology graphs.

    This class provides methods to extract subgraphs, collapse ontologies,
    and create visualizations with highlighted significant terms using
    NetworkX and Graphviz.

    Attributes:
        tree_parser: An object containing the ontology structure and annotations
        sys_graph: The system graph from the tree_parser
    """

    def __init__(self, tree_parser):
        """
        Initialize the TreeVisualizer with a tree parser object.

        Parameters:
        -----------
        tree_parser : object
            An object containing the ontology structure, system graph,
            and annotation dictionaries
        """
        self.tree_parser = tree_parser
        self.sys_graph = self.tree_parser.sys_graph

    @staticmethod
    def get_one_path_to_root(graph, start):
        """
        Returns a single path (as edges) from a starting node up to a root in the DAG.

        The function performs a traversal upward from the starting node, following
        parent relationships until it reaches a root node (node with no incoming edges).
        When multiple parents exist, it arbitrarily selects the first one.

        Parameters:
        -----------
        graph : networkx.DiGraph
            The directed graph to traverse
        start : node
            The starting node to find path from

        Returns:
        --------
        list of tuples
            List of edges (parent, child) representing the path from start to root
        """
        # We'll do a BFS upward (reverse) starting from 'start'
        path_edges = []

        current = start
        while True:
            # Get all parents of current
            parents = list(graph.predecessors(current))
            if not parents:
                # current is a root or has no parents -> stop
                break
            # Just pick the first parent (or pick the one with the shortest path if you prefer)
            parent = parents[0]

            # Add edge (parent -> current) to the path
            path_edges.append((parent, current))

            # Move up
            current = parent

        return path_edges

    @staticmethod
    def collapse_ontology(G, significant_nodes):
        """
        Collapses an ontology graph while preserving important structural elements.

        This method simplifies the ontology by:
        - Identifying and preserving bridge nodes that connect multiple significant nodes
        - Keeping the root, direct children, and grandchildren of the root
        - Maintaining all significant nodes and their hierarchy
        - Collapsing linear chains of nodes with single children

        Parameters:
        -----------
        G : networkx.DiGraph
            The ontology graph to collapse
        significant_nodes : set
            The set of nodes marked as "significant" that must be preserved

        Returns:
        --------
        networkx.DiGraph
            The simplified ontology graph with collapsed paths

        Raises:
        -------
        ValueError
            If the graph has multiple or no root nodes
        """
        # Identify the root node (a node without incoming edges)
        root_candidates = [node for node in G.nodes if G.in_degree(node) == 0]
        if len(root_candidates) != 1:
            raise ValueError(f"Graph has multiple or no root nodes: {root_candidates}")

        root = root_candidates[0]

        # Identify direct children of the root
        direct_children = set(G.successors(root))

        direct_grandchildren = []
        for child in direct_children:
            direct_grandchildren += G.successors(child)
        direct_grandchildren = set(direct_grandchildren)

        # Identify bridge nodes (nodes that connect multiple significant nodes)
        def is_bridge(node):
            """
            Determines if a node is a bridge by checking if it connects multiple significant nodes.

            Parameters:
            -----------
            node : node
                The node to check

            Returns:
            --------
            bool
                True if the node connects multiple significant nodes, False otherwise
            """
            descendant_significant = [desc for desc in nx.descendants(G, node) if (desc in significant_nodes)]
            return len(set(descendant_significant)) > 1

        bridge_nodes = {node for node in G.nodes if is_bridge(node)}

        # Nodes to keep: root, direct children, significant nodes, and bridge nodes
        nodes_to_keep = {root} | direct_children | significant_nodes | bridge_nodes | direct_grandchildren

        # Create a new graph with only relevant nodes
        collapsed_G = nx.DiGraph()
        collapsed_G.add_nodes_from(nodes_to_keep)

        def collapse_paths():
            """
            Iterates through the graph, collapsing linear paths while maintaining bridges.

            This function removes intermediate nodes that have only one child,
            rewiring their parent to connect directly to their child.

            Returns:
            --------
            bool
                True if any modifications were made, False otherwise
            """
            modified = False
            for node in list(G.nodes):
                if node in (nodes_to_keep - bridge_nodes) or node == root:
                    continue  # Keep significant nodes, root, and bridges

                parents = list(G.predecessors(node))
                children = list(G.successors(node))
                if len(children) == 1:  # Only one child -> Collapse
                    parent = parents[0] if parents else None
                    child = children[0]

                    if parent and child and parent in G and child in G:
                        G.add_edge(parent, child)  # Rewire parent to child
                        G.remove_node(node)  # Remove the intermediate node
                        modified = True

            return modified

        # Keep collapsing until no more changes occur
        while collapse_paths():
            pass
        #print('collapse path done')

        # Rewire edges for the final graph
        for parent, child in G.edges():
            if parent in nodes_to_keep and child in nodes_to_keep:
                collapsed_G.add_edge(parent, child)

        return collapsed_G

    def get_subgraph_with_root_paths(self, G, significant_terms):
        """
        Extract a minimal subgraph containing paths from significant terms to the root.

        This method finds one path from each significant term to the root and
        creates a subgraph containing only the nodes and edges necessary to
        maintain these connections.

        Parameters:
        -----------
        G : networkx.DiGraph
            The original graph
        significant_terms : iterable
            Collection of significant nodes that need paths to root

        Returns:
        --------
        networkx.DiGraph
            A minimal subgraph containing paths from significant terms to root
        """
        edges_in_subgraph = set()
        nodes_in_subgraph = set()

        for term in significant_terms:
            path_edges = self.get_one_path_to_root(G, term)
            edges_in_subgraph.update(path_edges)

        for (src, dst) in edges_in_subgraph:
            nodes_in_subgraph.add(src)
            nodes_in_subgraph.add(dst)

        # Include the significant terms themselves explicitly (in case they have no edges)
        nodes_in_subgraph.update(significant_terms)

        minimal_subG = nx.DiGraph()
        minimal_subG.add_nodes_from(nodes_in_subgraph)
        minimal_subG.add_edges_from(edges_in_subgraph)
        return minimal_subG

    def extract_and_collapse_ontology(self, significant_terms):
        """
        Extract paths to root for significant terms and collapse the resulting ontology.

        This is a convenience method that combines path extraction and ontology
        collapse into a single operation.

        Parameters:
        -----------
        significant_terms : iterable
            Collection of significant nodes to highlight in the ontology

        Returns:
        --------
        networkx.DiGraph
            A collapsed ontology graph containing only essential structure
        """
        minimal_subG = self.get_subgraph_with_root_paths(self.sys_graph, significant_terms)
        collapsed_G = self.collapse_ontology(minimal_subG, significant_nodes=set(significant_terms))
        return collapsed_G

    @staticmethod
    def convert_to_graphviz(collapsed_G):
        """
        Convert a NetworkX graph to a Graphviz graph with basic styling.

        This method creates a Graphviz representation of the graph with
        bottom-to-top layout and basic node/edge styling.

        Parameters:
        -----------
        collapsed_G : networkx.DiGraph
            The NetworkX graph to convert

        Returns:
        --------
        pygraphviz.AGraph
            A Graphviz graph ready for visualization
        """
        G = pgv.AGraph(directed=True)
        edge_list = [(child, parent) for parent, child in list(collapsed_G.edges())]
        G.add_edges_from(edge_list)
        G.graph_attr.update(rankdir="BT")  # bottom-to-top
        G.node_attr.update(shape="ellipse", style="filled", fillcolor="lightblue", fontcolor="black")
        G.edge_attr.update(color="gray")

        bottleneck_nodes = []
        for node in collapsed_G.nodes:
            if collapsed_G.out_degree(node) >= 2:
                bottleneck_nodes.append(node)
        return G

    @staticmethod
    def color_with_alpha(value, is_significant, cmap, norm):
        """
        Generate a hex color with alpha transparency based on significance.

        This method creates colors for nodes based on their importance values,
        with significant nodes having full opacity and non-significant nodes
        having reduced opacity.

        Parameters:
        -----------
        value : float
            The importance value to map to color
        is_significant : bool
            Whether the node is marked as significant
        cmap : matplotlib.colors.Colormap
            The colormap to use for color generation
        norm : matplotlib.colors.Normalize
            The normalization object for mapping values to colors

        Returns:
        --------
        str
            A hex color string with alpha channel (e.g., '#rrggbbaa')
        """
        # Get the RGBA tuple from the colormap
        rgba = cmap(norm(value))  # e.g., (R, G, B, 1.0)

        if not is_significant:
            # Adjust alpha
            rgba = (rgba[0], rgba[1], rgba[2], 0.5)

        # Convert to hex, preserving alpha: e.g. '#rrggbbaa'
        return to_hex(rgba, keep_alpha=True)

    def give_attribute_to_node(self, G, significant_terms, sys_importance_df, vmin=0, vmax=0.8):
        """
        Apply visual attributes to graph nodes based on their significance and importance.

        This method assigns colors, sizes, and labels to nodes based on their
        importance values and significance status. Significant nodes get full
        opacity and special styling, while others get reduced opacity.

        Parameters:
        -----------
        G : pygraphviz.AGraph
            The Graphviz graph to modify
        significant_terms : iterable
            Collection of significant node identifiers
        sys_importance_df : pandas.DataFrame
            DataFrame containing importance metrics with 'System', 'corr_mean_abs',
            and 'Size' columns
        vmin : float, optional
            Minimum value for color normalization (default: 0)
        vmax : float, optional
            Maximum value for color normalization (default: 0.8)

        Returns:
        --------
        pygraphviz.AGraph
            The modified Graphviz graph with applied attributes
        """
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_red', ['#ffffff', '#ff0000'], N=256)
        sys_importance_df = sys_importance_df.set_index('System')
        annot2color_dict = {}

        for idx, row in sys_importance_df.iterrows():
            system_id = idx
            value = row["corr_mean_abs"]
            is_sig = system_id in significant_terms

            # Get color with possible alpha=0.5
            color_hex = self.color_with_alpha(value, is_sig, cmap, norm)
            annot2color_dict[system_id] = color_hex

        for node in G.nodes():
            if str(node) in significant_terms:
                info = sys_importance_df.loc[node]
                G.get_node(node).attr['width'] = "%f" % (np.log(info.Size) / 7.5)
                G.get_node(node).attr['height'] = "%f" % (np.log(info.Size) / 7.5)
                G.get_node(node).attr['shape'] = 'circle'
                G.get_node(node).attr['label'] = ''
                G.get_node(node).attr['xlabel'] = self.tree_parser.sys_annot_dict[
                    node] if node in self.tree_parser.sys_annot_dict.keys() else ''
                G.get_node(node).attr['fillcolor'] = annot2color_dict[node]
                G.get_node(node).attr['color'] = 'black'
                G.get_node(node).attr['style'] = 'filled'
            elif node in sys_importance_df.index.values:
                info = sys_importance_df.loc[node]
                G.get_node(node).attr['width'] = "%f" % (np.log(info.Size) / 7.5)
                G.get_node(node).attr['height'] = "%f" % (np.log(info.Size) / 7.5)
                G.get_node(node).attr['label'] = ''
                G.get_node(node).attr['style'] = 'filled'
                G.get_node(node).attr['color'] = 'lightgray'
                G.get_node(node).attr['fillcolor'] = annot2color_dict[node]
                G.get_node(node).attr['xlabel'] = self.tree_parser.sys_annot_dict[
                    node] if node in self.tree_parser.sys_annot_dict.keys() else ''
            else:
                G.get_node(node).attr['width'] = "%f" % (np.log(len(self.tree_parser.sys2gene_full[node])) / 7.5)
                G.get_node(node).attr['height'] = "%f" % (np.log(len(self.tree_parser.sys2gene_full[node])) / 7.5)
                G.get_node(node).attr['label'] = ''
                G.get_node(node).attr['style'] = 'filled'
                G.get_node(node).attr['color'] = 'lightgray'
                G.get_node(node).attr['fillcolor'] = 'white'

        return G

    def plot_highlighted_ontology(self, significant_terms, sys_importance_df, vmin=0, vmax=0.8,
                                             out_dir=None):
        """
        Create a complete ontology visualization with highlighted significant systems.

        This is the main method that orchestrates the entire visualization process:
        extracting subgraphs, collapsing the ontology, converting to Graphviz,
        applying visual attributes, and optionally saving the result.

        Parameters:
        -----------
        significant_terms : iterable
            Collection of significant node identifiers to highlight
        sys_importance_df : pandas.DataFrame
            DataFrame containing importance metrics for systems
        vmin : float, optional
            Minimum value for color normalization (default: 0)
        vmax : float, optional
            Maximum value for color normalization (default: 0.8)
        out_dir : str, optional
            Output directory path to save the visualization (default: None)

        Returns:
        --------
        pygraphviz.AGraph
            The complete styled Graphviz graph ready for display or export
        """
        collapsed_G = self.extract_and_collapse_ontology(significant_terms)
        graphviz_G = self.convert_to_graphviz(collapsed_G)
        G = self.give_attribute_to_node(graphviz_G, significant_terms, sys_importance_df, vmin=vmin, vmax=vmax)
        G.layout(prog="dot")
        if out_dir is not None:
            G.draw(out_dir)
        return G