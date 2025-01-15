from dash import Dash, html, dcc, Output, Input
from jupyter_dash import JupyterDash
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import matplotlib
import pandas as pd
from .sankey_utils import softmax, factorize_attention_recursively
import warnings

warnings.filterwarnings("ignore")

class SankeyVisualizer(object):

    def __init__(self, tree_parser):
        self.tree_parser = tree_parser


    def get_snp_sankey_app_from_g2pt(self, target_go, attention_collector, direction='forward'):

        coords_dict, sankey_dict, target_gos, gene_sorted, snp_sorted = self.get_g2pt_attention_collection(target_go, attention_collector, direction=direction)
        app = Dash(__name__)
        app.layout = html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        options=[{'label': name, 'value': name} for name in target_gos],
                        value=target_gos[-1],
                        id='system',
                    ),
                ], style={'width': '49%', 'display': 'inline-block'}),

                html.Div([
                    dcc.RadioItems(
                        options=[{'label': f'Head {i}', 'value': i} for i in range(4)],
                        value=0,
                        id='head-number',
                        labelStyle={'display': 'inline-block', 'marginTop': '5px'}
                    )
                ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
            ], style={
                'padding': '10px 5px'
            }),

            html.Div([dcc.Graph(id="graph")])
        ])

        @app.callback(
            Output("graph", "figure"),
            Input('head-number', 'value')
        )
        def display_sankey(head):
            i = 0
            # for x, y, e in zip(coords_dict[1][0], coords_dict[1][1], total_order):
            #    print(i, e, x, y)
            #    i += 1
            figure = go.Figure(go.Sankey(
                node=dict(
                    pad=300,
                    thickness=100,
                    line=None,  # dict(color = "white", width = 0.25),
                    label=snp_sorted + gene_sorted + target_gos,  # [""]*len(snp_sorted)
                    color = ["aqua"]*len(snp_sorted) + ['magenta'] * len(gene_sorted) + ['blue'],
                    x=coords_dict[head][0],
                    y=coords_dict[head][1]
                ),
                link=dict(
                    # arrowlen=30,
                    source=sankey_dict[head][0],  # indices correspond to labels, eg A1, A2, A1, B1, ...
                    target=sankey_dict[head][1],
                    value=sankey_dict[head][2],
                    color=sankey_dict[head][3],
                    line=dict(color="black", width=0.25)
                )))
            # print(sankey_dict[head])
            # for value in zip(*sankey_dict[head]):
            #    print(value)
            figure.update_layout(
                title_text=f"Sankey Diagram for Attention Head {head}",
                autosize=True,
                # width=1000,
                height=1000
            )
            return figure
        return app

    def get_g2pt_attention_collection(self, target_go, attention_collector, direction='forward'):
        target_gos, target_genes, target_snps = self.tree_parser.get_target_components(target_go)
        attention_result_df = attention_collector.forward(target_gos, target_genes, target_snps)
        for head in range(4):
            attention_result_df[head] = attention_result_df[head].fillna(-1e9)
            for go in target_gos:
                attention_result_df[head][('forward', go,)] = attention_result_df[head][('forward', go,)].apply(softmax, axis=1)
                attention_result_df[head][('backward', go,)] = attention_result_df[head][('backward', go,)].apply(softmax, axis=1)
            for gene in target_genes:
                attention_result_df[head][('forward', gene,)] = attention_result_df[head][('forward', gene,)].apply(softmax, axis=1)

        attention_mean_df_dict = {i: pd.DataFrame(attention_result_df[i].mean(axis=0)).rename(columns={0: 'Value'}) for i
                                  in range(4)}
        attention_mean_forward_factorized_df_dict = {
            j: factorize_attention_recursively(self.tree_parser, attention_mean_df_dict[j].loc['forward'], target_go, 1) for j in
            range(4)}
        attention_mean_backward_factorized_df_dict = {
            j: factorize_attention_recursively(self.tree_parser, attention_mean_df_dict[j].loc['backward'], target_go, 1,
                                               direction='backward') for j in range(4)}

        if direction == 'forward':
            attention_mean_factorized_df_dict = attention_mean_forward_factorized_df_dict
        else:
            attention_mean_factorized_df_dict = attention_mean_backward_factorized_df_dict

        gene_sorted, snp_sorted, total_order = self.get_component_orders(target_gos)
        total_inds = self.get_sankey_component_inds(target_gos, gene_sorted, snp_sorted)
        all_paths = self.tree_parser.get_paths_from_node_to_leaves(self.tree_parser.system_graph, target_go)
        x_gap = (0.95 - 0.05) / (max([len(path) for path in all_paths]) + 1)
        coords_dict = {
            i: self.get_sankey_coords(attention_mean_factorized_df_dict[i], target_gos, gene_sorted, snp_sorted, x_gap) for i
            in range(4)}
        sankey_dict = {i: self.get_sankey_values(attention_mean_factorized_df_dict[i], target_gos, gene_sorted, total_inds)
                       for i in range(4)}
        return coords_dict, sankey_dict, target_gos, gene_sorted, snp_sorted


    def get_component_orders(self, target_gos, gene2chr=True):
        if gene2chr:
            gene2chr_dict = {gene: int(snps[0].split(':')[0]) for gene, snps in self.tree_parser.gene2snp.items()}
        gene_sorted = []

        for go in target_gos:
            if gene2chr:
                genes = list(sorted(self.tree_parser.sys2gene[go], key=lambda gene: gene2chr_dict[gene]))
            else:
                genes = list(sorted(self.tree_parser.sys2gene[go]))
            gene_sorted += genes
        snp_sorted = []
        for gene in gene_sorted:
            snps = sorted(self.tree_parser.gene2snp[gene], key=lambda a: (int(a.split(":")[0]), int(a.split(":")[1])))
            snp_sorted += snps

        total_order = snp_sorted
        for target_go in target_gos:
            total_order = total_order + [gene for gene in gene_sorted if gene not in self.tree_parser.sys2gene[target_go]]
            total_order = total_order + [target_go]

        return gene_sorted, snp_sorted, total_order


    def get_sankey_component_inds(self, go_sorted, gene_sorted, snp_sorted):
        total_ind = {}
        snp_ind = {snp: i for i, snp in enumerate(snp_sorted)}
        gene_ind = {gene: i + len(snp_sorted) for i, gene in enumerate(gene_sorted)}
        sys_ind = {sys: i + len(snp_sorted) + len(gene_sorted) for i, sys in enumerate(go_sorted)}
        total_ind.update(snp_ind)
        total_ind.update(gene_ind)
        total_ind.update(sys_ind)
        return total_ind

    def get_sankey_coords(self, attention_mean_df_factorized, system_sorted, gene_sorted, snp_sorted, x_gap):
        x = [0.05] * len(snp_sorted) + [0.05 + x_gap] * len(gene_sorted) + [0.05 + x_gap * 2] * (len(system_sorted) - 1) + [
            0.95]
        y = [0.05 + (i * 0.90 / len(snp_sorted)) for i in range(len(snp_sorted))]
        y_gap = 0.025
        gene_y_gap_total = y_gap * (len(gene_sorted) - 1)
        gene_y_min = 0.15
        gene_y_max = 0.85
        y_temp = gene_y_min
        y_normalizing_factor = (gene_y_max - gene_y_min - gene_y_gap_total)
        #snp_y_dict = {snp: snp_y for snp, snp_y in zip(snp_sorted, y)}

        for i, gene in enumerate(gene_sorted):
            y.append(y_temp)
            if i < len(gene_sorted) - 1:
                y_temp += (attention_mean_df_factorized.loc[gene_sorted[i]]['Value'].sum() / 2 +
                           attention_mean_df_factorized.loc[gene_sorted[i + 1]]['Value'].sum() / 2 + y_gap) * y_normalizing_factor

        go_y_min = 0.25
        go_y_max = 0.50
        y_temp = go_y_min
        for i, go in enumerate(system_sorted[:-1]):
            go_y = y_temp
            y.append(go_y)
            if i < len(system_sorted[:-1]) - 1:
                y_temp += (attention_mean_df_factorized.loc[system_sorted[i]]['Value'] + y_gap) * y_normalizing_factor
        y = y + [0.5]
        return x, y


    def get_sankey_values(self, attention_mean_df_factorized, target_gos, target_genes, total_ind):
        sources = []
        targets = []
        values = []
        colors = []
        types = []
        for key, value in attention_mean_df_factorized.iterrows():
            #print(key, value)
            if key[0] in target_genes:
                sources.append(total_ind[key[1]])
                targets.append(total_ind[key[0]])
                values.append(value['Value']+1e-3)
                types.append(key[2])
                if key[2] == 'homozygous':
                    color = list(matplotlib.colors.to_rgba('red'))
                    color[-1] = 0.5
                    color = 'rgba(%s)'%(",".join([str(c) for c in color]))
                    colors.append(color)
                elif key[2] == 'heterozygous':
                    color = list(matplotlib.colors.to_rgba('yellow'))
                    color[-1] = 0.5
                    color = 'rgba(%s)'%(",".join([str(c) for c in color]))
                    colors.append(color)
            elif key[0] in target_gos:
                sources.append(total_ind[key[1]])
                targets.append(total_ind[key[0]])
                values.append(value['Value']+1e-3)
                types.append(key[2])
                if key[2] == 'gene2sys':
                    color = list(matplotlib.colors.to_rgba('orange'))
                    color[-1] = 0.5
                    color = 'rgba(%s)'%(",".join([str(c) for c in color]))
                    colors.append(color)
                elif key[2] == 'sys2gene':
                    color = list(matplotlib.colors.to_rgba('blue'))
                    color[-1] = 0.5
                    color = 'rgba(%s)'%(",".join([str(c) for c in color]))
                    colors.append(color)
                elif key[2] == 'sys2env':
                    color = list(matplotlib.colors.to_rgba('brown'))
                    color[-1] = 0.5
                    color = 'rgba(%s)'%(",".join([str(c) for c in color]))
                    colors.append(color)
                elif key[2] == 'env2sys':
                    color = list(matplotlib.colors.to_rgba('magenta'))
                    color[-1] = 0.5
                    color = 'rgba(%s)'%(",".join([str(c) for c in color]))
                    colors.append(color)
        return sources, targets, values, colors, types



