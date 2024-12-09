import pandas as pd
import numpy as np
import torch
from . import TreeParser


class SNPTreeParser(TreeParser):

    def __init__(self, ontology, snp2gene, sys_annot_file=None, by_chr=False):
        super(SNPTreeParser, self).__init__(ontology, sys_annot_file=sys_annot_file)
        print("%d in gene2sys mask" % self.gene2sys_mask.sum())
        #self.snp2gene_df = self.ontology.loc[self.ontology['interaction'] == 'snp']
        self.snp2gene_df = pd.read_csv(snp2gene, sep='\t', names=['snp', 'gene', 'chr'])
        genes = self.snp2gene_df.gene.unique()
        genes_not_in_sys2gene = [gene for gene in genes if gene not in self.gene2ind.keys()]
        gene_max_ind = max(self.gene2ind.values())
        # Adding Genes not in Ontology
        for gene in genes_not_in_sys2gene:
            gene_max_ind += 1
            self.gene2ind[gene] = gene_max_ind
            self.ind2gene[gene_max_ind] = gene
        self.n_genes = len(self.gene2ind.keys())
        self.gene2sys_mask = np.zeros((len(self.sys2ind), len(self.gene2ind)))
        self.sys2gene_dict = {self.sys2ind[system]: [] for system in self.sys2ind.keys()}
        self.gene2sys_dict = {gene: [] for gene in range(self.n_genes)}
        for system, gene in zip(self.gene2sys_df['parent'], self.gene2sys_df['child']):
            #print(system, gene)
            self.gene2sys_mask[self.sys2ind[system], self.gene2ind[gene]] = 1.
            self.sys2gene_dict[self.sys2ind[system]].append(self.gene2ind[gene])
            self.gene2sys_dict[self.gene2ind[gene]].append(self.sys2ind[system])
        print("Total %d Gene-System interactions are queried"%self.gene2sys_mask.sum())
        self.sys2gene_mask = self.gene2sys_mask.T
        self.subtree_types = self.system_df['interaction'].unique()


        snps2gene_df_group_by_snps = self.snp2gene_df.groupby('snp')
        snps2gene_df_group_by_genes = self.snp2gene_df.groupby('gene')

        self.gene2snp= {gene: snps2gene_df_group_by_genes.get_group(gene)['snp'].values.tolist() for gene in
                         snps2gene_df_group_by_genes.groups.keys()}
        self.snp2gene= {snp: snps2gene_df_group_by_snps.get_group(snp)['gene'].values.tolist() for snp in
                         snps2gene_df_group_by_snps.groups.keys()}

        snps = self.snp2gene_df['snp'].unique()
        self.snp2ind = {snp: index for index, snp in enumerate(snps)}
        self.ind2snp = {index: snp for index, snp in enumerate(snps)}
        self.n_snps = len(self.snp2ind)
        self.snp_pad_index = self.n_snps
        self.chromosomes = sorted(self.snp2gene_df.chr.unique())
        self.gene2chr = {gene:CHR for gene, CHR in zip(self.snp2gene_df['gene'], self.snp2gene_df['chr'])}
        self.snp2chr = {snp: CHR for snp, CHR in zip(self.snp2gene_df['snp'], self.snp2gene_df['chr'])}
        self.chr2gene = {CHR: [self.gene2ind[gene] for gene in
                               self.snp2gene_df.loc[self.snp2gene_df['chr'] == CHR]['gene'].values.tolist()] for CHR in
                         self.chromosomes}
        self.chr2snp = {CHR: [self.snp2ind[snp] for snp in
                              self.snp2gene_df.loc[self.snp2gene_df['chr'] == CHR]['snp'].values.tolist()] for CHR in
                        self.chromosomes}

        self.sys2snp = {sys:self.get_sys2snp(sys) for sys in self.sys2ind.keys()}
        self.snp2sys = {}
        for sys, snps in self.sys2snp.items():
            for snp in snps:
                if snp in self.snp2sys.keys():
                    self.snp2sys[snp].append(sys)
                else:
                    self.snp2sys[snp] = [sys]

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
        print("%d in snp2gene mask" % self.snp2gene_mask.sum())

    def get_sys2snp(self, sys):
        genes = self.sys2gene_full[sys]
        snps = [self.gene2snp[gene] for gene in genes if gene in self.gene2snp.keys()]
        snps = sum(snps, [])
        return snps

    def get_snp2gene_mask(self, gene_indices, snp_indices, type_indices=None, CHR=None):

        if len(gene_indices)==0:
            return torch.zeros((self.n_genes, self.n_snps))
        else:
            if type_indices is not None:
                snp2gene_mask = np.copy(self.snp2gene_mask)
                for key, value in type_indices.items():
                    type_mask = np.zeros_like(self.snp2gene_mask)
                    type_mask[:, value] = key
                    snp2gene_mask *= type_mask
            else:
                snp2gene_mask = np.copy(self.snp2gene_mask)
            snp2gene_mask =  torch.tensor(snp2gene_mask)[gene_indices, :]
            snp2gene_mask = snp2gene_mask[:, snp_indices]
            if CHR:
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
        return {"snp":torch.tensor(snp_embedding_indices, dtype=torch.int), "gene":torch.tensor(gene_embedding_indices, dtype=torch.int)}

    def get_snp2gene_indices(self, snp_indices):
        return sorted(list(set(sum([self.snp2gene_dict[snp] for snp in snp_indices], []))))

    def get_snp2gene(self, snp_indices, type_indices=None):
        if self.by_chr:
            return self.get_snp2gene_by_chromosome(snp_indices, type_indices=type_indices)
        else:
            embeddings = self.get_snp2gene_embeddings(snp_indices)
            mask = self.get_snp2gene_mask(embeddings['gene'], embeddings['snp'], type_indices=type_indices)
            return {"snp":embeddings['snp'], 'gene':embeddings['gene'], 'mask':mask}

    def get_snp2gene_by_chromosome(self, snp_indices, type_indices=None):
        embeddings = {CHR: self.get_snp2gene_embeddings([snp for snp in snp_indices if snp in self.chr2snp[CHR]])  for CHR in self.chromosomes}
        #print(snp_indices, embeddings)
        mask = {CHR: self.get_snp2gene_mask(embeddings[CHR]['gene'], embeddings[CHR]['snp'], type_indices=type_indices, CHR=CHR) for CHR in self.chromosomes}
        return {CHR: {"snp":embeddings[CHR]['snp'], "gene":embeddings[CHR]['gene'], 'mask':mask[CHR] } for CHR in self.chromosomes}

