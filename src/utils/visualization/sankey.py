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

    def __init__(self, tree_parser, genotype_annot_dict={}, gene_annot_dict={}, system_annot_dict={}):
        self.tree_parser = tree_parser
        self.genotype_annot_dict = genotype_annot_dict
        self.gene_annot_dict = gene_annot_dict
        self.system_annot_dict = system_annot_dict


    def get_sankey_app_from_g2pt(self, target_go, attention_collector):
        attention_df = self.get_g2pt_attention_collection(target_go, attention_collector)
        return self.get_sankey_app(target_go, attention_df)

    def get_sankey_app(self, target_go, attention_df, factorize=True, genotypes=('homozygous', 'heterozygous'), sort_by_chr=True):

        genotypes = genotypes
        attention_df = attention_df
        target_gos, target_genes, target_snps = self.tree_parser.get_target_components(target_go)


        app = Dash(__name__)
        app.layout = html.Div([
            html.Div([
                html.Div([
                    dcc.Dropdown(
                        options=[{'label': name, 'value': name} for name in target_gos],
                        value=target_go,
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
                ], style={'width': '24%', 'float': 'right', 'display': 'inline-block'}),

                html.Div([
                    dcc.RadioItems(
                        options=[i for i in ['forward', 'backward']],
                        value='forward',
                        id='direction',
                        labelStyle={'display': 'inline-block', 'marginTop': '5px'}
                    )
                ], style={'width': '24%', 'float': 'right', 'display': 'inline-block'})

            ], style={
                'padding': '10px 5px'
            }),

            html.Div([dcc.Graph(id="graph")])
        ])

        @app.callback(
            Output("graph", "figure"),
            Input('head-number', 'value'),
            Input('system', 'value'),
            Input('direction', 'value')
        )
        def display_sankey(head, selected_go, direction):
            coords_dict, sankey_dict, target_gos, gene_sorted, snp_sorted, total_order = self.get_sankey_components(
                attention_df, selected_go, factorize=factorize, direction=direction, genotypes=genotypes, sort_by_chr=sort_by_chr)
            total_order_reversed = {value: key for key, value in total_order.items()}
            sankey_customdata = [(total_order_reversed[s], total_order_reversed[t], m) for s, t, m in
                                zip(sankey_dict[head][0], sankey_dict[head][1], sankey_dict[head][4])]
            if len(snp_sorted)==0:
                height = 50 * len(gene_sorted)
            else:
                height = 20 * len(snp_sorted)
            figure = go.Figure(go.Sankey(
                node=dict(
                    pad=300,
                    thickness=100,
                    line=None,  # dict(color = "white", width = 0.25),
                    label=snp_sorted + gene_sorted + target_gos,  # [""]*len(snp_sorted)
                    color=["aqua"]*len(snp_sorted) + ['magenta'] * len(gene_sorted) + ['blue'] * len(target_gos),
                    customdata=[self.genotype_annot_dict[snp] if snp in self.genotype_annot_dict.keys() else '' for snp in snp_sorted] +
                               [self.gene_annot_dict[gene] if gene in self.gene_annot_dict.keys() else '' for gene in gene_sorted] +
                               [self.system_annot_dict[sys] if sys in self.system_annot_dict.keys() else '' for sys in target_gos],
                    hovertemplate='%{customdata}<extra></extra>',
                    x=coords_dict[head][0],
                    y=coords_dict[head][1]
                ),
                link=dict(
                    # arrowlen=30,
                    source=sankey_dict[head][0],  # indices correspond to labels, eg A1, A2, A1, B1, ...
                    target=sankey_dict[head][1],
                    value=sankey_dict[head][2],
                    color=sankey_dict[head][3],
                    customdata=sankey_customdata,
                    hovertemplate='Link from %{customdata[0]}<br />' +
                                  'to %{customdata[1]}<br />has value %{value}' +
                                  '<br />and module %{customdata[2]}<extra></extra>',
                    line=None#dict(color="black", width=0.25),
                )))
            # print(sankey_dict[head])
            # for value in zip(*sankey_dict[head]):
            #    print(value)
            figure.update_layout(
                title_text=f"Sankey Diagram for Attention Head {head}",
                autosize=True,
                # width=1000,
                height=height
            )
            return figure
        return app

    def display_sankey(self, coords_dict, sankey_dict, head, target_gos, gene_sorted, snp_sorted, total_order, width=1000, height=1000):
        i = 0
        # for x, y, e in zip(coords_dict[1][0], coords_dict[1][1], total_order):
        #    print(i, e, x, y)
        #    i += 1
        total_order_reversed = {value: key for key, value in total_order.items()}
        sankey_customdata = [(total_order_reversed[s], total_order_reversed[t], m) for s, t, m in
                             zip(sankey_dict[head][0], sankey_dict[head][1], sankey_dict[head][4])]
        if len(snp_sorted) == 0:
            height = 50 * len(gene_sorted)
        else:
            height = 20 * len(snp_sorted)
        figure = go.Figure(go.Sankey(
            node=dict(
                pad=300,
                thickness=100,
                line=None,  # dict(color = "white", width = 0.25),
                label=snp_sorted + gene_sorted + target_gos,  # [""]*len(snp_sorted)
                color=["aqua"] * len(snp_sorted) + ['magenta'] * len(gene_sorted) + ['blue'] * len(target_gos),
                customdata=[self.genotype_annot_dict[snp] if snp in self.genotype_annot_dict.keys() else '' for snp in
                            snp_sorted] +
                           [self.gene_annot_dict[gene] if gene in self.gene_annot_dict.keys() else '' for gene in
                            gene_sorted] +
                           [self.system_annot_dict[sys] if sys in self.system_annot_dict.keys() else '' for sys in
                            target_gos],
                hovertemplate='%{customdata}<extra></extra>',
                x=coords_dict[head][0],
                y=coords_dict[head][1]
            ),
            link=dict(
                # arrowlen=30,
                source=sankey_dict[head][0],  # indices correspond to labels, eg A1, A2, A1, B1, ...
                target=sankey_dict[head][1],
                value=sankey_dict[head][2],
                color=sankey_dict[head][3],
                customdata=sankey_customdata,
                hovertemplate='Link from %{customdata[0]}<br />' +
                              'to %{customdata[1]}<br />has value %{value}' +
                              '<br />and module %{customdata[2]}<extra></extra>',
                line=None  # dict(color="black", width=0.25),
            )))
        # print(sankey_dict[head])
        # for value in zip(*sankey_dict[head]):
        #    print(value)
        figure.update_layout(
            title_text=f"Sankey Diagram for Attention Head {head}",
            autosize=True,
            width=width,
            height=height
        )
        return figure

    def get_g2pt_attention_collection(self, target_go, attention_collector):
        target_gos, target_genes, target_snps = self.tree_parser.get_target_components(target_go)

        attention_result_df = attention_collector.forward(target_gos, target_genes, target_snps)
        return attention_result_df

    def marginalize_attention_collection(self, target_go, attention_result_df, n_softmax=1, normalize=False):
        target_gos, target_genes, target_snps = self.tree_parser.get_target_components(target_go)
        for head in range(4):
            attention_result_df[head] = attention_result_df[head].fillna(-1e9)
            for k in range(n_softmax):
                for go in target_gos:
                    if normalize:
                        attention_result_df[head][('forward', go,)] = attention_result_df[head][('forward', go,)] / attention_result_df[head][('forward', go,)].shape[0]
                        attention_result_df[head][('backward', go,)] = attention_result_df[head][('backward', go,)] / attention_result_df[head][('forward', go,)].shape[0]
                    attention_result_df[head][('forward', go,)] = attention_result_df[head][('forward', go,)].apply(softmax, axis=1)
                    attention_result_df[head][('backward', go,)] = attention_result_df[head][('backward', go,)].apply(softmax, axis=1)
                for gene in target_genes:
                    if normalize:
                        attention_result_df[head][('forward', gene,)] = attention_result_df[head][('forward', gene,)] / attention_result_df[head][('forward', gene,)].shape[0]
                    attention_result_df[head][('forward', gene,)] = attention_result_df[head][('forward', gene,)].apply(softmax, axis=1)
        attention_mean_df_dict = {i: pd.DataFrame(attention_result_df[i].mean(axis=0)).rename(columns={0: 'Value'}) for i
                                  in range(4)}
        return attention_mean_df_dict

    def get_sankey_components(self, attention_mean_df_dict, target_go, factorize=True, direction='forward', genotypes=('homozygous', 'heterozygous'), sort_by_chr=True):
        target_gos, target_genes, target_snps = self.tree_parser.get_target_components(target_go)
        if direction =='forward':
            attention_mean_df_dict = {head:df.loc['forward'] for head, df in attention_mean_df_dict.items()}
        elif direction == 'backward':
            attention_mean_df_dict = {head: df.loc['backward'] for head, df in attention_mean_df_dict.items()}
        else:
            attention_mean_df_dict = {head: pd.concat([df.loc['forward'], df.loc['backward']]) for head, df in attention_mean_df_dict.items()}
        query_list = list(target_gos)+list(target_genes)
        attention_mean_df_dict = {head: df#.loc[df.index.get_level_values(0).isin(query_list)]
                                   for head, df in attention_mean_df_dict.items()}

        if factorize:
            if direction == 'forward':
                attention_mean_df_dict = {j: factorize_attention_recursively(self.tree_parser, attention_mean_df_dict[j], target_go, 1,
                                                   direction='forward', genotypes=genotypes) for j in range(4)}
            elif direction == 'backward':
                attention_mean_df_dict = {j: factorize_attention_recursively(self.tree_parser, attention_mean_df_dict[j], target_go, 1,
                                                   direction='backward', genotypes=genotypes) for j in range(4)}
            else:
                attention_mean_df_dict_forward = {
                    j: factorize_attention_recursively(self.tree_parser, attention_mean_df_dict[j],
                                                       target_go, 1,
                                                       direction='forward', genotypes=genotypes) for j in range(4)}
                attention_mean_df_dict_backward = {
                    j: factorize_attention_recursively(self.tree_parser, attention_mean_df_dict[j],
                                                       target_go, 1,
                                                       direction='backward', genotypes=genotypes) for j in range(4)}
                attention_mean_df_dict = {j: pd.concat([attention_mean_df_dict_forward[j], attention_mean_df_dict_backward[j]]) for j in range(4)}


        gene_sorted, snp_sorted, total_order = self.get_component_orders(target_gos, gene2chr=sort_by_chr)

        nested_gos = self.get_nested_systems_by_heights(target_gos)
        if direction=='backward':
            snp_sorted = []
        total_inds = self.get_sankey_component_inds(target_gos, gene_sorted, snp_sorted)

        coords_dict = {
            i: self.get_sankey_coords(attention_mean_df_dict[i], nested_gos, gene_sorted, snp_sorted) for i
            in range(4)}
        sankey_dict = {i: self.get_sankey_values(attention_mean_df_dict[i], target_gos, gene_sorted, total_inds, genotypes=genotypes)
                       for i in range(4)}
        return coords_dict, sankey_dict, target_gos, gene_sorted, snp_sorted, total_inds


    def get_component_orders(self, target_gos, gene2chr=True):
        if gene2chr:
            gene2chr_dict = {gene: int(snps[0].split(':')[0]) for gene, snps in self.tree_parser.gene2snp.items()}
        gene_sorted = []

        for go in target_gos:
            if gene2chr:
                genes = list(sorted(list(self.tree_parser.sys2gene[go]), key=lambda gene: gene2chr_dict[gene]))
            else:
                genes = list(sorted(list(self.tree_parser.sys2gene[go])))
            for gene in genes:
                if gene not in gene_sorted:
                    gene_sorted.append(gene)
        snp_sorted = []
        for gene in gene_sorted:
            if gene2chr:
                snps = sorted(list(self.tree_parser.gene2snp[gene]), key=lambda a: (int(a.split(":")[0]), int(a.split(":")[1])))
            else:
                snps = sorted(list(self.tree_parser.gene2snp[gene]))
            for snp in snps:
                if snp not in snp_sorted:
                    snp_sorted.append(snp)
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

    def get_nested_systems_by_heights(self, systems):
        levels_dict = {}
        for node in systems:
            level = self.tree_parser.node_height_dict[node]
            if level in levels_dict.keys():
                levels_dict[level].append(node)
            else:
                levels_dict[level] = [node]
        sorted_levels = sorted(levels_dict.keys())
        nested_list = [levels_dict[level] for level in sorted_levels]
        return nested_list

    def get_sankey_coords(self, attention_mean_df_factorized, system_nested, gene_sorted, snp_sorted):
        all_paths = self.tree_parser.get_paths_from_node_to_leaves(self.tree_parser.system_graph, system_nested[-1][-1])
        if len(all_paths)==0:
            x_gap = (0.90 - 0.05)/2
        else:
            x_gap = (0.90 - 0.05) / (max([len(path) for path in all_paths]) + 1)

        x = [0.05] * len(snp_sorted) + [0.1 + x_gap] * len(gene_sorted)

        for i, systems in enumerate(system_nested[:-1]):
            x = x + [0.05 + x_gap * (i+2)] * (len(systems))
        x = x + [0.90]
        y = [0.05 + (i * 0.90 / len(snp_sorted)) for i in range(len(snp_sorted))]
        y_gap = 0.1 / (len(gene_sorted) - 1)
        gene_y_gap_total = 0.1
        gene_y_min = 0.15
        gene_y_max = 0.85
        y_temp = gene_y_min
        y_normalizing_factor = (gene_y_max - gene_y_min - gene_y_gap_total)

        #snp_y_dict = {snp: snp_y for snp, snp_y in zip(snp_sorted, y)}

        #for i, gene in enumerate(gene_sorted):
        #    y.append(y_temp)
        #    if i < len(gene_sorted) - 1:
        #        y_temp += (attention_mean_df_factorized.loc[gene_sorted[i]]['Value'].sum() / 2 +
        #                   attention_mean_df_factorized.loc[gene_sorted[i + 1]]['Value'].sum() / 2 + y_gap) * y_normalizing_factor
        y += [gene_y_min + (i * (gene_y_max-gene_y_min) / len(gene_sorted)) for i, gene in enumerate(gene_sorted)]

        go_y_min = 0.25
        go_y_max = 0.70
        y_normalizing_factor = (go_y_max - go_y_min - gene_y_gap_total)
        y_temp = go_y_min
        n_sys = len(sum(system_nested, []))
        for i, systems in enumerate(system_nested[:-1]):
            for system in systems:
                go_y = y_temp
                y.append(go_y)

                y_temp += (go_y_max-go_y_min) / n_sys #(attention_mean_df_factorized.loc[system]['Value'] + y_gap) * y_normalizing_factor
                #print(system, y_temp)
        y = y + [0.5]
        return x, y


    def get_sankey_values(self, attention_mean_df_factorized, target_gos, target_genes, total_ind, genotypes=('homozygous', 'heterozygous')):
        sources = []
        targets = []
        values = []
        colors = []
        types = []
        genotype2color_dict = {genotype: c for genotype, c in zip(genotypes, ['red', 'yellow', 'orange'])}
        for key, value in attention_mean_df_factorized.iterrows():
            #print(key, value)
            if key[0] in target_genes:
                #print(key, value)
                sources.append(total_ind[key[1]])
                targets.append(total_ind[key[0]])
                values.append(value['Value'])
                types.append(key[2])
                color = list(matplotlib.colors.to_rgba(genotype2color_dict[key[2]]))
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



