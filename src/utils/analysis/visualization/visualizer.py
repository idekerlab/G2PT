import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA



class EmbeddingVisualizer(object):

    def __init__(self, drug_response_model, tree_parser, auto_fit=True, n_components=2, subtree_order=['default']):

        self.model = drug_response_model.to("cpu")
        self.tree_parser = tree_parser
        self.gene2gene_mask = torch.tensor(self.tree_parser.gene2gene_mask, dtype=torch.float32)

        self.nested_subtrees_forward = self.tree_parser.get_nested_subtree_mask(subtree_order, direction='forward')
        self.nested_subtrees_backward = self.tree_parser.get_nested_subtree_mask(subtree_order, direction='backward')
        self.system_embedding_tensor = self.model.system_embedding.weight.unsqueeze(0)
        self.system_embedding = self.model.system_embedding.weight.detach().cpu().numpy()
        self.pca = None
        if auto_fit:
            self.fit(self.system_embedding)


    def fit(self, embeddings, n_components=5):
        self.pca = PCA(n_components=n_components)
        self.pca.fit(embeddings)

    def get_gene_loc(self, gene):
        system_onehot = np.zeros(shape=len(self.tree_parser.system2ind))
        gene_inds = [self.tree_parser.system2ind[sys] for sys in self.tree_parser.get_parent_system_of_gene(gene)]
        system_onehot[gene_inds] = 1
        updated_loc = system_onehot == 1
        not_updated_loc = system_onehot == 0
        return updated_loc, not_updated_loc

    def draw_change_embedding(self, axe, prev, update, updated_loc, not_updated_loc, name=None, draw_unhchanged=True,
                              xaxis=None, yaxis=None, x=0, y=1, arrow_color='blue'):
        if (sum(not_updated_loc)!=0)&draw_unhchanged:
            unchanged_transformed = self.pca.transform(prev[not_updated_loc])
            axe.scatter(unchanged_transformed[:, x], unchanged_transformed[:, y], alpha=0.1, c='grey')
        changed_system_embedding = (prev + update)[updated_loc]
        transformed_previous = self.pca.transform(prev[updated_loc])
        transformed_after = self.pca.transform(changed_system_embedding)
        changed_system_ids = self.tree_parser.get_ind2system(np.where(updated_loc)[0].tolist())
        for transformed_previous_i, transformed_after_i, system_name in zip(transformed_previous, transformed_after, changed_system_ids):
            axe.text(transformed_previous_i[x], transformed_previous_i[y], system_name)
            axe.text(transformed_after_i[x], transformed_after_i[y], system_name)
        axe.scatter(transformed_previous[:, x], transformed_previous[:, y], c='yellow', edgecolors='black')
        axe.scatter(transformed_after[:, x], transformed_after[:, y], c='red', edgecolors='black', marker="^")
        for m, n in zip(transformed_previous, transformed_after):
            arrow = axe.arrow(m[x], m[y], n[x] - m[x], n[y] - m[y], head_width=0.03, linestyle='--', linewidth=0.5, color=arrow_color)
        if name is not None:
            axe.set_title(name)
        if xaxis is not None:
            axe.set_xlim(xaxis)
        if yaxis is not None:
            axe.set_ylim(yaxis)
        return arrow

    def draw_mutation_effect(self, genotypes, system=None, gene=None, ax=None,  fig_size=7, output_path=None, x=0, y=1):
        mut_embedding = self.model.gene_embedding.weight.unsqueeze(0)
        mutation_updated, mutation_updates_dict = self.model.get_mut2system(self.system_embedding_tensor,
                                                                                                mut_embedding, genotypes)
        if ax is not None:
            mutation_total = \
            torch.sum(torch.stack([mutation_updates_dict[genotype] for genotype in ["mutation", "cna", "cnd"]], dim=0),
                      dim=0)[0].detach().numpy()
            update = mutation_total  # .detach().cpu().numpy()
            if gene is not None:
                if system is not None:
                    updated_loc = np.zeros(len(self.tree_parser.system2ind))
                    updated_loc[self.tree_parser.system2ind[system]] = 1
                    not_updated_loc = updated_loc==0
                    updated_loc = updated_loc == 1
                else:
                    updated_loc, not_updated_loc = self.get_gene_loc(gene)
            else:
                updated_loc = np.sum(update, axis=-1) != 0
                not_updated_loc = np.sum(update, axis=-1) == 0
            self.draw_change_embedding(ax, self.system_embedding, update, updated_loc, not_updated_loc,
                                       name="Mutation Total", x=x, y=y)
        else:
            fig, axes = plt.subplots(1, len(genotypes)+1, figsize=(fig_size * (len(genotypes)+1), fig_size),
                                     sharex=True, sharey=True)
            axes = axes.ravel()
            for i, genotype in enumerate(genotypes.keys()):
                update = mutation_updates_dict[genotype][0].detach().cpu().numpy()
                if gene is not None:
                    updated_loc, not_updated_loc = self.get_gene_loc(gene)
                else:
                    updated_loc = np.sum(update, axis=-1) != 0
                    not_updated_loc = np.sum(update, axis=-1) == 0
                self.draw_change_embedding(axes[i], self.system_embedding, update, updated_loc, not_updated_loc, name=genotype, x=x, y=y)
            mutation_total = \
            torch.sum(torch.stack([mutation_updates_dict[genotype] for genotype in ["mutation", "cna", "cnd"]], dim=0),
                      dim=0)[0].detach().numpy()
            update = mutation_total  # .detach().cpu().numpy()
            if gene is not None:
                updated_loc, not_updated_loc = self.get_gene_loc(gene)
            else:
                updated_loc = np.sum(update, axis=-1) != 0
                not_updated_loc = np.sum(update, axis=-1) == 0
            if sum(not_updated_loc):
                unchanged_transformed = self.pca.transform(self.system_embedding[not_updated_loc])
                axes[i + 1].scatter(unchanged_transformed[:, x], unchanged_transformed[:, y], alpha=0.1, c='grey')
            self.draw_change_embedding(axes[i+1], self.system_embedding, update, updated_loc, not_updated_loc, name="Mutation Total", x=x, y=y)
            if output_path is not None:
                plt.savefig(output_path)
            plt.show()


    def draw_total_embedding_change(self, genotypes, figsize=7, xaxis=None, yaxis=None, x=0, y=1):
        mut_embedding = self.model.gene_embedding.weight.unsqueeze(0)
        mutation_updated, mutation_updates_dict = self.model.get_mut2system(self.system_embedding_tensor,
                                                                                                mut_embedding, genotypes)
        tree_updated_forward, tree_updates_forward = self.model.get_system2system(mutation_updated,
                                                                                self.nested_subtrees_forward,
                                                                                direction='forward')
        tree_updated_backward, tree_updates_backward = self.model.get_system2system(tree_updated_forward[-1][-1],
                                                                                self.nested_subtrees_forward,
                                                                                direction='backward')

        fig, axes = plt.subplots(1, 4, figsize=(figsize * 4, figsize), sharex=True, sharey=True)
        axes = axes.ravel()
        system_embedding_transformed = self.pca.transform(self.system_embedding)
        axes[0].scatter(system_embedding_transformed[:, x], system_embedding_transformed[:, y], c='yellow', edgecolors='black')
        axes[0].set_title("original")

        mutation_updated_transformed = self.pca.transform(mutation_updated[0].detach().cpu().numpy())

        axes[1].scatter(mutation_updated_transformed[:, x], mutation_updated_transformed[:, y], c='red', edgecolors='black', marker="^")
        axes[1].set_title("Mut2Systems")

        tree_updated_forward_transformed = self.pca.transform(tree_updated_forward[-1][-1][0].detach().cpu().numpy())
        axes[2].scatter(tree_updated_forward_transformed[:, x], tree_updated_forward_transformed[:, y], c='blue', edgecolors='black',
                        marker="^")
        axes[2].set_title("Sys2Env")

        tree_updated_backward_transformed = self.pca.transform(tree_updated_backward[-1][-1][0].detach().cpu().numpy())
        axes[3].scatter(tree_updated_backward_transformed[:, x], tree_updated_backward_transformed[:, y], c='blue', edgecolors='black', marker="^")
        axes[3].set_title("Env2Sys")

        plt.show()

    def draw_system_change_decompose(self, genotypes, system, figsize=7, xaxis=None, yaxis=None, x=0, y=1):
        mut_embedding = self.model.gene_embedding.weight.unsqueeze(0)
        mutation_updated, mutation_updates_dict = self.model.get_mut2system(self.system_embedding_tensor,
                                                                                                mut_embedding, genotypes)
        tree_updated_forward, tree_updates_forward = self.model.get_system2system(mutation_updated,
                                                                                self.nested_subtrees_forward,
                                                                                direction='forward')
        tree_updated_backward, tree_updates_backward = self.model.get_system2system(tree_updated_forward[-1][-1],
                                                                                self.nested_subtrees_backward,
                                                                                direction='backward')


        system_indices = np.zeros(shape=len(self.tree_parser.system2ind))
        system_ind = self.tree_parser.system2ind[system]
        system_indices[system_ind] = 1
        updated_loc = system_indices == 1
        # print((prev+update)[parent_ind]==tree_updated[m][n][0].detach().cpu().numpy()[parent_ind])
        not_updated_loc = system_indices == 0

        fig, axes = plt.subplots(1, 1, figsize=(figsize, figsize), sharex=True, sharey=True)

        mut2sys_arrow = self.draw_change_embedding(axes, self.system_embedding, mutation_updated[0].detach().cpu().numpy() - self.system_embedding,
                                                   updated_loc, not_updated_loc,
                                   name="Mut2Sys", draw_unhchanged=False, xaxis=xaxis, yaxis=yaxis,
                                   x=x, y=y, arrow_color='red')
        sys2env_arrow = self.draw_change_embedding(axes, mutation_updated[0].detach().cpu().numpy(),
                                                   tree_updated_forward[-1][-1][0].detach().cpu().numpy() - mutation_updated[0].detach().cpu().numpy(),
                                   updated_loc, not_updated_loc,
                                   name="Sys2Env", draw_unhchanged=False, xaxis=xaxis, yaxis=yaxis,
                                   x=x, y=y, arrow_color='blue')
        env2sys_arrow = self.draw_change_embedding(axes, tree_updated_forward[-1][-1][0].detach().cpu().numpy(),
                                                   tree_updated_backward[-1][-1][0].detach().cpu().numpy() - tree_updated_forward[-1][-1][0].detach().cpu().numpy(),
                                   updated_loc, not_updated_loc, name="Env2Sys", draw_unhchanged=True, xaxis=xaxis, yaxis=yaxis,
                                   x=x, y=y, arrow_color='yellow')

        plt.legend([mut2sys_arrow, sys2env_arrow, env2sys_arrow], ["Mut2Sys", "Sys2Env", "Env2Sys"])


    def draw_embedding_changes(self, genotypes, figsize=7, n_cols=1, n_rows=1, xaxis=None, yaxis=None, names=None, x=0, y=1):
        mut_embedding = self.model.gene_embedding.weight.unsqueeze(0)
        mutation_updated, mutation_updates_dict = self.model.get_mut2system(self.system_embedding_tensor,
                                                                                                mut_embedding, genotypes)
        tree_updated, tree_updates = self.model.get_system2system(mutation_updated,
                                                                                self.nested_subtrees_forward,
                                                                                direction='forward')
        tree_updated, tree_updates = self.model.get_system2system(tree_updated[-1][-1],
                                                                                self.nested_subtrees_backward,
                                                                                direction='backward')
        tree_updated = tree_updated[-1][-1].detach().numpy()
        fig, axes = plt.subplots(n_cols, n_rows, figsize=(figsize * n_cols, figsize * n_rows), sharex=True, sharey=True)
        axes = axes.ravel()
        for i, system_change in enumerate(tree_updated):
            tree_updated_transformed = self.pca.transform(system_change)
            axes[i].scatter(tree_updated_transformed[:, x], tree_updated_transformed[:, y], c='blue', edgecolors='black')
            if names is not None:
                axes[i].set_title(names[i])
        plt.show()

    def draw_hierarchical_convolution_for_gene(self, genotypes, gene, interaction_type, system=None, figsize=7, xaxis=None,
                                               yaxis=None, by_level=False, output_path=None, x=0, y=1):
        mut_embedding = self.model.gene_embedding.weight.unsqueeze(0)
        mutation_updated, mutation_updates_dict = self.model.get_mut2system(self.system_embedding_tensor,
                                                                                                mut_embedding, genotypes)
        tree_updated_forward, tree_updates_forward = self.model.get_system2system(mutation_updated,
                                                                                self.nested_subtrees_forward,
                                                                                direction='forward')
        tree_updated_backward, tree_updates_backward = self.model.get_system2system(tree_updated_forward[-1][-1],
                                                                                self.nested_subtrees_backward,
                                                                                direction='backward')
        tree_updated_backward.reverse()
        parent_systems = self.tree_parser.get_parent_system_of_gene(gene)
        print("Gene:", gene)
        print("Interaction type:", interaction_type)

        parent_system_paths = {system:self.tree_parser.get_system_hierarchies(system, interaction_type) for system in parent_systems}
        print(parent_system_paths)

        #axes.ravel()
        print(parent_systems)
        for parent_system in parent_systems:
            if system is not None:
                if parent_system!=system:
                    continue
            z = 0
            parent_system_path_i = parent_system_paths[parent_system]
            if len(parent_system_path_i)==0:
                print(gene, parent_system)
                continue
            if not by_level:
                fig, axes = plt.subplots(1, len(parent_system_path_i),
                                         figsize=(figsize * len(parent_system_path_i), figsize)
                                         , sharex=True, sharey=True)
                if len(parent_system_path_i)!=1:
                    axes = axes.ravel()
                else:
                    axes = [axes]

            print(parent_system_path_i)

            for parent_system_path in parent_system_path_i:

                parent_system_path_inds = self.tree_parser.get_system2ind(parent_system_path)
                parent_system_path_edges = [(parent_system_path[i], parent_system_path[i + 1]) for i in
                                                range(len(parent_system_path_inds) - 1)]
                parent_system_path_edge_inds = [(parent_system_path_inds[i], parent_system_path_inds[i+1]) for i in range(len(parent_system_path_inds)-1)]
                if by_level:
                    fig, axes = plt.subplots(1, len(parent_system_path_edges),
                                             figsize=(figsize * len(parent_system_path_edges), figsize)
                                             , sharex=True, sharey=True)
                else:
                    if gene:
                        self.draw_mutation_effect(genotypes, system=parent_system, gene=gene, ax=axes[z])
                        #self.draw_change_embedding(axes[z], mutation_updated, update, updated_loc, not_updated_loc,
                        #                           name=name, draw_unhchanged=draw_unhchanged, xaxis=xaxis, yaxis=yaxis,
                        #                           x=x, y=y)

                for i, parent_system_path_edge in enumerate(zip(parent_system_path_edges, parent_system_path_edge_inds)):
                    edge_ids, edge_inds = parent_system_path_edge
                    child_id, parent_id = edge_ids
                    child_ind, parent_ind = edge_inds
                    for m, nested_subtree_m in enumerate(self.nested_subtrees_forward):
                        for n, nested_subtree_n in enumerate(nested_subtree_m):
                            if nested_subtree_n[parent_ind, child_ind]==1:
                                if n==0:
                                    if m==0:
                                        prev = mutation_updated
                                    else:
                                        prev = tree_updated_backward[m-1][-1]
                                else:
                                    prev = tree_updated_backward[m][n-1]

                                prev = prev[0].detach().cpu().numpy()
                                update = tree_updated_backward[m][n][0].detach().cpu().numpy()
                                system_indices = np.zeros(shape=len(self.tree_parser.system2ind))
                                system_indices[parent_ind] = 1
                                updated_loc = system_indices==1
                                #print((prev+update)[parent_ind]==tree_updated[m][n][0].detach().cpu().numpy()[parent_ind])
                                not_updated_loc = system_indices==0
                                if by_level:
                                    axes_ind = i
                                    name = child_id + " -> " + parent_id
                                    draw_unhchanged = True
                                else:
                                    axes_ind = z
                                    name = " -> ".join([gene] + parent_system_path)
                                    if i == (len(parent_system_path_edges) - 1):
                                        draw_unhchanged = True
                                    else:
                                        draw_unhchanged = False
                                self.draw_change_embedding(axes[axes_ind], prev, update, updated_loc, not_updated_loc,
                                                           name=name, draw_unhchanged=draw_unhchanged, xaxis=xaxis, yaxis=yaxis,
                                                           x=x, y=y)
                                prev_parent = prev[[parent_ind]]
                                prev_child = prev[[child_ind]]
                                self.draw_parent_child(axes[axes_ind], prev_parent, prev_child, parent_id, child_id, x=x, y=y)

                z += 1
        if output_path is not None:
            plt.savefig(output_path)
        plt.show()


    def draw_parent_child(self, axe, parent, child, parent_id, child_id, x=0, y=1, parent_color='yellow', child_color='blue'):
        transformed_parent = self.pca.transform(parent)
        transformed_child = self.pca.transform(child)
        axe.scatter(transformed_parent[:, x], transformed_parent[:, y], c=parent_color, edgecolors='black')
        axe.scatter(transformed_child[:, x], transformed_child[:, y], c=child_color, edgecolors='black')
        axe.text(transformed_parent[0, x], transformed_parent[0, y], parent_id)
        axe.text(transformed_child[0, x], transformed_child[0, y], child_id)
        axe.arrow(transformed_child[0, x],
                  transformed_child[0, y],
                  transformed_parent[0, x] - transformed_child[0, x],
                  transformed_parent[0, y] - transformed_child[0, y],
                  head_width=0.03, linewidth=0.5)

    def draw_attention(self, genotypes, compound, figsize=7, top_k=10):
        compound_embedding = self.model.get_compound_embedding(compound, unsqueeze=True)
        mut_embedding = self.model.gene_embedding.weight.unsqueeze(0)
        #mut_embedding = self.model.get_gene2gene(mut_embedding, self.gene2gene_mask)
        #print(self.system_embedding_tensor.shape, mut_embedding.shape)
        mutation_updated, mutation_updates_dict = self.model.get_mut2system(self.system_embedding_tensor,
                                                                                                mut_embedding, genotypes)
        tree_updated, tree_updates = self.model.get_system2system(mutation_updated,
                                                                                self.nested_subtrees_forward,
                                                                                direction='forward')
        tree_updated, tree_updates = self.model.get_system2system(tree_updated[-1][-1],
                                                                                self.nested_subtrees_backward,
                                                                                direction='backward')
        result, compound_attention2system = self.model.get_system2comp(compound_embedding,
                                                                                              tree_updated[-1][-1],
                                                                                attention=True)
        compound_attention2system = compound_attention2system[0, 0, 0, :].detach().cpu().numpy()
        #print(compound_attention2system)
        updated_total = self.pca.transform(tree_updated[-1][-1][0].detach().cpu().numpy())
        plt.figure(figsize=(figsize, figsize))
        plt.scatter(updated_total[:, 0], updated_total[:, 1], marker="^", s=100, c=compound_attention2system,
                    cmap='inferno')
        threshold = np.flip(np.sort(compound_attention2system))[top_k-1]
        print("Threshold: ", threshold)
        result_indices = []
        for system, ind in self.tree_parser.system2ind.items():
            if compound_attention2system[ind]>=threshold:
                plt.text(updated_total[ind, 0], updated_total[ind, 1], system)
                result_indices.append(ind)
        result_indices = np.array(result_indices)[np.flip(np.argsort(compound_attention2system[result_indices]))].tolist()
        plt.colorbar()
        plt.show()
        return compound_attention2system[result_indices], result_indices

    def draw_G2P_space(self, genotypes, phenotype, figsize=7, top_k=10, botoom_k=10, xaxis=0, yaxis=1, model='compound', annot_dict=None):

        mut_embedding = self.model.gene_embedding.weight.unsqueeze(0)
        mutation_updated, mutation_updates_dict = self.model.get_mut2system(self.system_embedding_tensor,
                                                                                                mut_embedding, genotypes)
        tree_updated, tree_updates = self.model.get_system2system(mutation_updated,
                                                                                self.nested_subtrees_forward,
                                                                                direction='forward')
        tree_updated, tree_updates = self.model.get_system2system(tree_updated[-1][-1],
                                                                                self.nested_subtrees_backward,
                                                                                direction='backward')
        result, phenotype_attention2system = self.model.get_system2comp(phenotype,
                                                                                              tree_updated[-1][-1],
                                                                                score=True)
        phenotype_attention2system = phenotype_attention2system[0, 0, 0, :].detach().cpu().numpy()
        #print(compound_attention2system)
        if model=='compound':
            phenotype_transformed = self.model.sys2comp.attention.linear_layers[0](phenotype).detach().cpu().numpy()[0]
            system_transformed = self.model.sys2comp.attention.linear_layers[1](tree_updated[-1][-1][0]).detach().cpu().numpy()
        else:
            phenotype_transformed = self.model.system2phenotype.attention.linear_layers[0](phenotype).detach().cpu().numpy()
            system_transformed = self.model.system2phenotype.attention.linear_layers[1](tree_updated[-1][-1][0]).detach().cpu().numpy()
        print(system_transformed.shape, phenotype_transformed.shape)
        pca = PCA()
        pca = pca.fit(np.concatenate([phenotype_transformed, system_transformed], axis=0))

        phenotype_pca_transformed = pca.transform(phenotype_transformed)
        system_pca_transformed = pca.transform(system_transformed)
        plt.figure(figsize=(figsize, figsize))
        plt.scatter(system_pca_transformed[:, xaxis], system_pca_transformed[:, yaxis], marker="^", s=100, c=phenotype_attention2system,
                    cmap='inferno')
        plt.scatter(phenotype_pca_transformed[:, xaxis], phenotype_pca_transformed[:, yaxis], marker='o', s=100, c='green')

        threshold = np.sort(phenotype_attention2system)[top_k-1]#np.flip(np.sort(phenotype_attention2system))[top_k-1]
        print("Threshold: ", threshold)
        result_indices = []
        for system, ind in self.tree_parser.system2ind.items(): # need to get low attended
            if phenotype_attention2system[ind]<=threshold:
                if annot_dict is not None:
                    system = annot_dict[system]
                plt.text(system_pca_transformed[ind, xaxis], system_pca_transformed[ind, yaxis], system)
                result_indices.append(ind)
        result_indices = np.array(result_indices)[np.flip(np.argsort(phenotype_attention2system[result_indices]))].tolist()
        plt.colorbar()
        plt.show()
        return phenotype_attention2system[result_indices], result_indices

    def draw_change_hierarchy_for_gene(self, genotypes, gene, interaction_type, system=None, figsize=7, xaxis=None,
                                               yaxis=None, by_level=False, output_path=None, x=0, y=1):
        
        mut_embedding = self.model.gene_embedding.weight.unsqueeze(0)
        mutation_updated, mutation_updates_dict = self.model.get_mut2system(self.system_embedding_tensor,
                                                                                                mut_embedding, genotypes)
        tree_updated, tree_updates = self.model.get_system2system(mutation_updated,
                                                                                self.nested_subtrees_forward,
                                                                                direction='forward')
        tree_updated, tree_updates = self.model.get_system2system(tree_updated[-1][-1],
                                                                                self.nested_subtrees_backward,
                                                                                direction='backward')
        tree_updated.reverse()
        parent_systems = self.tree_parser.get_parent_system_of_gene(gene)
        print("Gene:", gene)
        print("Interaction type:", interaction_type)

        parent_system_paths = {system:self.tree_parser.get_system_hierarchies(system, interaction_type) for system in parent_systems}
        print(parent_system_paths)

        #axes.ravel()
        print(parent_systems)
        fig, axes = plt.subplots(1, 2, figsize=(figsize*2, figsize), sharex=True, sharey=True)
        transformed_orig = self.pca.transform(self.system_embedding)  # axes[z].sc
        axes[0].scatter(transformed_orig[:, x], transformed_orig[:, y], c='lightgray')
        total_change = tree_updated[-1][-1][0].detach().cpu().numpy()
        transformed_change = self.pca.transform(total_change)  # axes[z].sc
        axes[1].scatter(transformed_orig[:, x], transformed_orig[:, y], c='lightgray', alpha=0.5)
        axes[1].scatter(transformed_change[:, x], transformed_change[:, y], c='lightblue')
        for parent_system in parent_systems:
            if system is not None:
                if parent_system!=system:
                    continue
            z = 0
            parent_system_path_i = parent_system_paths[parent_system]
            if len(parent_system_path_i)==0:
                print(gene, parent_system)
                continue


            print(parent_system_path_i)

            for parent_system_path in parent_system_path_i:

                parent_system_path_inds = self.tree_parser.get_system2ind(parent_system_path)
                parent_system_path_edges = [(parent_system_path[i], parent_system_path[i + 1]) for i in
                                                range(len(parent_system_path_inds) - 1)]
                parent_system_path_edge_inds = [(parent_system_path_inds[i], parent_system_path_inds[i+1]) for i in range(len(parent_system_path_inds)-1)]
                for i, parent_system_path_edge in enumerate(zip(parent_system_path_edges, parent_system_path_edge_inds)):
                    edge_ids, edge_inds = parent_system_path_edge
                    child_id, parent_id = edge_ids
                    child_ind, parent_ind = edge_inds
                    for m, nested_subtree_m in enumerate(self.nested_subtrees_forward):
                        for n, nested_subtree_n in enumerate(nested_subtree_m):
                            if nested_subtree_n[parent_ind, child_ind]==1:
                                orig_parent = self.system_embedding[[parent_ind]]
                                orig_child = self.system_embedding[[child_ind]]
                                self.draw_parent_child(axes[0], orig_parent, orig_child, parent_id, child_id, x=x, y=y,
                                                       parent_color='gray', child_color='gray')
                                changed_parent = total_change[[parent_ind]]
                                changed_child = total_change[[child_ind]]
                                self.draw_parent_child(axes[1], changed_parent, changed_child, parent_id, child_id, x=x,
                                                       y=y, parent_color='blue', child_color='blue')
                z += 1
        if output_path is not None:
            plt.savefig(output_path)
        plt.show()