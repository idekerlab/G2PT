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
        self.n_systems = len(self.system2ind)
        self.n_genes = len(self.gene2ind)
        print("%d Systems are queried"%self.n_systems)
        print(self.system2ind)
        print("%d Genes are queried"%self.n_genes)
        print(self.gene2ind)
        self.system2system_mask = np.zeros((len(self.system2ind), len(self.system2ind)))

        for parent_system, child_system in zip(self.system_df['parent'], self.system_df['child']):
            self.system2system_mask[self.system2ind[parent_system], self.system2ind[child_system]] = 1

        self.gene2system_mask = np.zeros((len(self.system2ind), len(self.gene2ind)))
        self.system2gene_dict = {self.system2ind[system]: [] for system in systems}
        for system, gene in zip(self.gene2system_df['parent'], self.gene2system_df['child']):
            self.gene2system_mask[self.system2ind[system], self.gene2ind[gene]] = 1
            self.system2gene_dict[self.system2ind[system]].append(self.gene2ind[gene])
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


class SNPTreeParser(TreeParser):

    def __init__(self, ontology, snp2gene, gene2ind, snp2id, by_chr=False):
        super(SNPTreeParser, self).__init__(ontology, gene2ind)
        snp2ind = pd.read_csv(snp2id, sep='\t', names=['index', 'snp'])
        self.snp2ind = {snp: index for index, snp in zip(snp2ind['index'], snp2ind['snp'])}
        self.ind2snp = {index: snp for index, snp in zip(snp2ind['index'], snp2ind['snp'])}
        self.n_snps = len(self.snp2ind)
        self.snp_pad_index = self.n_snps
        self.snp2gene_df = pd.read_csv(snp2gene, sep='\t', names=['snp', 'gene', 'chr'])
        self.chromosomes = sorted(self.snp2gene_df.chr.unique())
        self.gene2chr = {gene:CHR for gene, CHR in zip(self.snp2gene_df['gene'], self.snp2gene_df['chr'])}
        self.snp2chr = {snp: CHR for snp, CHR in zip(self.snp2gene_df['snp'], self.snp2gene_df['chr'])}
        self.chr2gene = {CHR: [self.gene2ind[gene] for gene in
                               self.snp2gene_df.loc[self.snp2gene_df['chr'] == CHR]['gene'].values.tolist()] for CHR in
                         self.chromosomes}
        self.chr2snp = {CHR: [self.snp2ind[snp] for snp in
                              self.snp2gene_df.loc[self.snp2gene_df['chr'] == CHR]['snp'].values.tolist()] for CHR in
                        self.chromosomes}

        print("%d SNPs are queried" % self.n_snps)
        print(self.snp2ind)
        if by_chr:
            print("Embedding will feed by chromosome")
        self.by_chr = by_chr
        self.snp2gene_mask = np.zeros((self.n_genes, self.n_snps))
        self.gene2snp_dict = {ind:[] for gene, ind in self.gene2ind.items()}
        self.snp2gene_dict = {ind:[] for ind in range(self.n_snps)}
        for snp, gene in zip(self.snp2gene_df['snp'], self.snp2gene_df['gene']):
            self.snp2gene_mask[self.gene2ind[gene], self.snp2ind[snp]] = 1
            self.gene2snp_dict[self.gene2ind[gene]].append(self.snp2ind[snp])
            self.snp2gene_dict[self.snp2ind[snp]].append(self.gene2ind[gene])
        self.gene2snp_mask = self.snp2gene_mask.T

    def get_system2genotype_mask(self):
        return self.gene2system_mask

    def get_snp2gene_mask(self, CHR, gene_indices, snp_indices, type_indices=None):

        if len(gene_indices)==0:
            return torch.zeros((len(self.chr2gene[CHR]), len(self.chr2snp[CHR])))
        else:
            if type_indices is not None:
                snp2gene_mask = self.snp2gene_mask
                for key, value in type_indices.items():
                    type_mask = np.zeros_like(self.snp2gene_mask)
                    type_mask[:, value] = key
                    snp2gene_mask *= type_mask
            else:
                snp2gene_mask = self.snp2gene_mask
            snp2gene_mask =  torch.tensor(snp2gene_mask)[gene_indices, :]
            snp2gene_mask = snp2gene_mask[:, snp_indices]
            if self.by_chr:
                result_mask = torch.zeros((len(self.chr2gene[CHR]), len(self.chr2snp[CHR])))
            else:
                result_mask = torch.zeros((self.n_genes, self.n_snps))
            result_mask[:snp2gene_mask.size(0), :snp2gene_mask.size(1)] = snp2gene_mask
            return result_mask

    def get_snps_indices_from_genes(self, gene_indices):
        return list(set(sum([self.gene2snp_dict[gene] for gene in gene_indices], [])))

    def get_gene_snps_indices_from_genes_grouped_by_chromosome(self, gene_indices):
        return list(set(sum([self.gene2snp_dict[gene] for gene in gene_indices], [])))

    def get_snp2gene_embeddings(self, snp_indices):
        snp_embedding_indices = sorted(list(set(sum([[snp]*len(self.snp2gene_dict[snp]) for snp in snp_indices], []))))
        gene_embedding_indices = sorted(list(set(sum([self.snp2gene_dict[snp] for snp in snp_indices], []))))
        return {"snp":torch.tensor(snp_embedding_indices), "gene":torch.tensor(gene_embedding_indices)}

    def get_snp2gene_indices(self, snp_indices):
        return sorted(list(set(sum([self.snp2gene_dict[snp] for snp in snp_indices], []))))

    def get_snp2gene(self, snp_indices, type_indices=None):
        if self.by_chr:
            return self.get_snp2gene_by_chromosome(snp_indices, type_indices=type_indices)
        else:
            embeddings = self.get_snp2gene_embeddings(snp_indices)
            mask = self.get_snp2gene_mask(0, embeddings['gene'], embeddings['snp'], type_indices=type_indices)
            return {"snp":embeddings['snp'], 'gene':embeddings['gene'], 'mask':mask}

    def get_snp2gene_by_chromosome(self, snp_indices, type_indices=None):
        embeddings = {CHR: self.get_snp2gene_embeddings([snp for snp in snp_indices if snp in self.chr2snp[CHR]])  for CHR in self.chromosomes}
        #print(snp_indices, embeddings)
        mask = {CHR: self.get_snp2gene_mask(CHR, embeddings[CHR]['gene'], embeddings[CHR]['snp'], type_indices=type_indices) for CHR in self.chromosomes}
        return {CHR: {"snp":embeddings[CHR]['snp'], "gene":embeddings[CHR]['gene'], 'mask':mask[CHR] } for CHR in self.chromosomes}





