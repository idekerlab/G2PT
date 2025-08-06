import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib
from dash import Dash, html, dcc, Output, Input

# --- Utility Functions (formerly in sankey_utils.py) ---

def _softmax(x):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def _get_sankey_color(value, min_val, max_val, cmap_name='viridis'):
    """Maps a value to a color using a specified colormap."""
    cmap = matplotlib.cm.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=min_val, vmax=max_val)
    rgba = cmap(norm(value))
    return f'rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]})'


class SankeyVisualizer:
    """
    Generates interactive and static Sankey diagrams from attention data.

    This class encapsulates the logic for processing attention weights and
    ontology information to create clear, hierarchical Sankey plots that
    visualize the flow of attention from SNPs to genes to biological systems.
    """
    def __init__(self, tree_parser, genotype_annot_dict={}, gene_annot_dict={}, system_annot_dict={}):
        """
        Initializes the SankeyVisualizer.

        Args:
            tree_parser (SNPTreeParser): An initialized SNPTreeParser object
                containing the ontology and SNP-gene relationships.
            genotype_annot_dict (dict, optional): A dictionary mapping SNP IDs
                to descriptive annotations for tooltips. Defaults to {}.
            gene_annot_dict (dict, optional): A dictionary mapping gene IDs to
                descriptive annotations for tooltips. Defaults to {}.
            system_annot_dict (dict, optional): A dictionary mapping system IDs
                to descriptive annotations for tooltips. Defaults to {}.
        """
        self.tree_parser = tree_parser
        self.genotype_annot_dict = genotype_annot_dict
        self.gene_annot_dict = gene_annot_dict
        self.system_annot_dict = system_annot_dict

    def _factorize_genotype_attention(self, attention_df, target_gene, weight, genotypes):
        """Recursively multiplies the attention weights of SNPs connected to a gene."""
        for snp in self.tree_parser.gene2snp.get(target_gene, []):
            for genotype in genotypes:
                try:
                    attention_df.at[(target_gene, snp, genotype), 'Value'] *= weight
                except KeyError:
                    continue
        return attention_df

    def _factorize_attention_recursively(self, attention_df, target, weight, direction, genotypes):
        """
        Recursively traverses the ontology graph to factorize attention weights.

        This method applies a multiplicative weight down the hierarchy, ensuring
        that the attention flow is conserved from parent to child nodes.
        """
        for gene in self.tree_parser.sys2gene.get(target, []):
            module = 'gene2sys' if direction == 'forward' else 'sys2gene'
            try:
                attention_df.at[(target, gene, module), 'Value'] *= weight
                if direction == 'forward':
                    snp_weight = attention_df.loc[(target, gene, module)]['Value']
                    attention_df = self._factorize_genotype_attention(attention_df, gene, snp_weight, genotypes)
            except KeyError:
                continue

        for _, child in self.tree_parser.sys_graph.out_edges(target):
            module = 'sys2env' if direction == 'forward' else 'env2sys'
            try:
                attention_df.at[(target, child, module), 'Value'] *= weight
                sys2env_value = attention_df.loc[(target, child, module)]['Value']
                attention_df = self._factorize_attention_recursively(attention_df, child, sys2env_value, direction, genotypes)
            except KeyError:
                continue
        return attention_df

    def _prepare_sankey_data(self, attention_df, target_go, direction, genotypes, factorize=True, sort_by_chr=True):
        """
        Prepares all necessary data components for plotting the Sankey diagram.

        This is the main data processing method. It takes the raw attention
        DataFrame and orchestrates the factorization, ordering, and positioning
        of all nodes and links.

        Args:
            attention_df (pd.DataFrame): The input DataFrame of attention values.
            target_go (str): The root system (GO term) for the plot.
            direction (str): The flow direction ('forward' or 'backward').
            genotypes (tuple): The genotypes to include.
            factorize (bool): Whether to apply recursive factorization.
            sort_by_chr (bool): Whether to sort genes and SNPs by chromosome.

        Returns:
            dict: A dictionary containing structured data for nodes, links, and layout.
        """
        target_gos, target_genes, target_snps = self.tree_parser.get_target_components(target_go)
        
        # Filter by direction
        attention_df = attention_df.loc[direction]

        if factorize:
            attention_df = self._factorize_attention_recursively(
                self.tree_parser, attention_df.copy(), target_go, 1, direction, genotypes
            )

        # Get sorted lists of components
        gene_sorted, snp_sorted, _ = self._get_component_orders(target_gos, sort_by_chr)
        if direction == 'backward':
            snp_sorted = []

        # Get component indices
        total_inds = self._get_sankey_component_inds(target_gos, gene_sorted, snp_sorted)
        
        # Get node positions
        nested_gos = self._get_nested_systems_by_heights(target_gos)
        node_x, node_y = self._calculate_node_positions(nested_gos, gene_sorted, snp_sorted)

        # Get link values
        sources, targets, values, colors, types = self._get_sankey_link_values(
            attention_df, target_gos, gene_sorted, total_inds, genotypes
        )

        return {
            "nodes": {
                "labels": snp_sorted + gene_sorted + target_gos,
                "colors": ["aqua"] * len(snp_sorted) + ['magenta'] * len(gene_sorted) + ['blue'] * len(target_gos),
                "x": node_x,
                "y": node_y,
                "customdata": [self.genotype_annot_dict.get(s, '') for s in snp_sorted] +
                              [self.gene_annot_dict.get(g, '') for g in gene_sorted] +
                              [self.system_annot_dict.get(s, '') for s in target_gos]
            },
            "links": {
                "sources": sources,
                "targets": targets,
                "values": values,
                "colors": colors,
                "types": types,
                "customdata": [(total_inds.get(s), total_inds.get(t), m) for s, t, m in zip(sources, targets, types)]
            },
            "layout": {
                "height": 20 * len(snp_sorted) if snp_sorted else 50 * len(gene_sorted)
            }
        }

    def _get_component_orders(self, target_gos, gene2chr=True):
        """Sorts SNPs, genes, and systems for stable plot layout."""
        gene2chr_dict = {gene: int(snps[0].split(':')[0]) for gene, snps in self.tree_parser.gene2snp.items()} if gene2chr else {}
        
        gene_sorted = []
        for go in target_gos:
            genes = sorted(self.tree_parser.sys2gene.get(go, []), key=lambda g: gene2chr_dict.get(g, 0))
            for gene in genes:
                if gene not in gene_sorted:
                    gene_sorted.append(gene)
        
        snp_sorted = []
        for gene in gene_sorted:
            snps = sorted(self.tree_parser.gene2snp.get(gene, []), key=lambda a: (int(a.split(":")[0]), int(a.split(":")[1])))
            for snp in snps:
                if snp not in snp_sorted:
                    snp_sorted.append(snp)
        
        total_order = snp_sorted + gene_sorted + target_gos
        return gene_sorted, snp_sorted, total_order

    def _get_sankey_component_inds(self, go_sorted, gene_sorted, snp_sorted):
        """Creates a mapping from component ID to its final index in the plot."""
        snp_ind = {snp: i for i, snp in enumerate(snp_sorted)}
        gene_ind = {gene: i + len(snp_sorted) for i, gene in enumerate(gene_sorted)}
        sys_ind = {sys: i + len(snp_sorted) + len(gene_sorted) for i, sys in enumerate(go_sorted)}
        return {**snp_ind, **gene_ind, **sys_ind}

    def _get_nested_systems_by_heights(self, systems):
        """Groups systems by their height in the ontology graph."""
        levels_dict = {}
        for node in systems:
            level = self.tree_parser.node_height_dict.get(node, 0)
            levels_dict.setdefault(level, []).append(node)
        sorted_levels = sorted(levels_dict.keys())
        return [levels_dict[level] for level in sorted_levels]

    def _calculate_node_positions(self, system_nested, gene_sorted, snp_sorted):
        """Calculates the X and Y coordinates for each node in the Sankey diagram."""
        max_depth = len(system_nested) + 1
        x_gap = (0.90 - 0.10) / max_depth if max_depth > 0 else 0.8

        x = [0.05] * len(snp_sorted) + [0.10 + x_gap] * len(gene_sorted)
        for i, systems in enumerate(system_nested):
            x.extend([0.10 + x_gap * (i + 2)] * len(systems))

        y_snp = np.linspace(0.05, 0.95, len(snp_sorted)) if snp_sorted else []
        y_gene = np.linspace(0.15, 0.85, len(gene_sorted)) if gene_sorted else []
        
        y = list(y_snp) + list(y_gene)
        
        y_sys_min, y_sys_max = 0.25, 0.75
        total_systems = sum(len(s) for s in system_nested)
        y_sys_points = np.linspace(y_sys_min, y_sys_max, total_systems) if total_systems > 1 else [0.5]
        
        y_idx = 0
        for systems in system_nested:
            for _ in systems:
                y.append(y_sys_points[y_idx])
                y_idx += 1
        
        return x, y

    def _get_sankey_link_values(self, attention_df, target_gos, target_genes, total_ind, genotypes):
        """Constructs the lists required for the 'link' property of the Sankey figure."""
        sources, targets, values, colors, types = [], [], [], [], []
        genotype_colors = {genotype: c for genotype, c in zip(genotypes, ['red', 'yellow', 'orange'])}
        module_colors = {'gene2sys': 'orange', 'sys2gene': 'blue', 'sys2env': 'brown', 'env2sys': 'magenta'}

        for key, value in attention_df.iterrows():
            source_node, target_node, module = key
            
            if source_node not in total_ind or target_node not in total_ind:
                continue

            sources.append(total_ind[source_node])
            targets.append(total_ind[target_node])
            values.append(value['Value'] + 1e-9) # Add small value to ensure visibility
            types.append(module)

            color_name = genotype_colors.get(module, module_colors.get(module, 'grey'))
            rgba = list(matplotlib.colors.to_rgba(color_name))
            rgba[-1] = 0.5
            colors.append(f'rgba({",".join(map(str, rgba))})')
            
        return sources, targets, values, colors, types

    def plot(self, attention_df, target_go, direction, genotypes, title="Sankey Diagram"):
        """
        Generates a Plotly Sankey Figure.

        Args:
            attention_df (pd.DataFrame): DataFrame with processed attention values.
            target_go (str): The root GO term for the visualization.
            direction (str): 'forward' or 'backward'.
            genotypes (tuple): Genotypes to include (e.g., ('homozygous', 'heterozygous')).
            title (str): The title of the plot.

        Returns:
            go.Figure: A Plotly Figure object.
        """
        sankey_data = self._prepare_sankey_data(attention_df, target_go, direction, genotypes)
        
        total_order_rev = {i: label for i, label in enumerate(sankey_data["nodes"]["labels"])}
        
        link_customdata = [
            (total_order_rev.get(s, ''), total_order_rev.get(t, ''), m)
            for s, t, m in zip(sankey_data["links"]["sources"], sankey_data["links"]["targets"], sankey_data["links"]["types"])
        ]

        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=sankey_data["nodes"]["labels"],
                color=sankey_data["nodes"]["colors"],
                x=sankey_data["nodes"]["x"],
                y=sankey_data["nodes"]["y"],
                customdata=sankey_data["nodes"]["customdata"],
                hovertemplate='%{customdata}<extra></extra>'
            ),
            link=dict(
                source=sankey_data["links"]["sources"],
                target=sankey_data["links"]["targets"],
                value=sankey_data["links"]["values"],
                color=sankey_data["links"]["colors"],
                customdata=link_customdata,
                hovertemplate='Link from %{customdata[0]}<br />to %{customdata[1]}<br />Value: %{value}<extra></extra>'
            )
        ))

        fig.update_layout(
            title_text=title,
            font_size=10,
            height=sankey_data["layout"]["height"]
        )
        return fig

def create_sankey_app(visualizer, attention_df, target_go_list, initial_go, genotypes=('homozygous', 'heterozygous')):
    """
    Creates a Dash application for interactive Sankey visualization.

    Args:
        visualizer (SankeyVisualizer): An initialized SankeyVisualizer instance.
        attention_df (pd.DataFrame): The full, pre-processed attention data.
        target_go_list (list): A list of GO terms to populate the dropdown.
        initial_go (str): The initial GO term to display.
        genotypes (tuple): Genotypes to include.

    Returns:
        Dash: A Dash application instance.
    """
    app = Dash(__name__)

    app.layout = html.Div([
        html.H2("Attention Flow Sankey Diagram"),
        html.Div([
            dcc.Dropdown(
                id='system-dropdown',
                options=[{'label': go, 'value': go} for go in target_go_list],
                value=initial_go
            ),
            dcc.RadioItems(
                id='direction-radio',
                options=[{'label': d.capitalize(), 'value': d} for d in ['forward', 'backward']],
                value='forward',
                labelStyle={'display': 'inline-block', 'margin-top': '10px'}
            )
        ]),
        dcc.Graph(id="sankey-graph")
    ])

    @app.callback(
        Output("sankey-graph", "figure"),
        [Input('system-dropdown', 'value'),
         Input('direction-radio', 'value')]
    )
    def update_graph(selected_go, direction):
        if not selected_go:
            return go.Figure()
        
        # Process the multi-indexed DataFrame for the selected head
        attention_mean_df_dict = {
            head: pd.DataFrame(attention_df[head].mean(axis=0)).rename(columns={0: 'Value'})
            for head in range(4) # Assuming 4 heads
        }
        
        # For simplicity, this example uses head 0. You could add a dropdown for heads.
        processed_df = attention_mean_df_dict[0]

        return visualizer.plot(processed_df, selected_go, direction, genotypes, title=f"Attention Flow for {selected_go}")

    return app