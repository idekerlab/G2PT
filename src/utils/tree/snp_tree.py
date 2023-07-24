import pandas as pd
import numpy as np
import torch
from . import TreeParser


class SNPTreeParser(TreeParser):

    def __init__(self, ontology, snp2gene, gene2ind, snp2id, by_chr=False):
        super(SNPTreeParser, self).__init__(ontology, gene2ind)
        print("%d in gene2sys mask" % self.gene2sys_mask.sum())
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
        print("%d in snp2gene mask" % self.snp2gene_mask.sum())

    def get_snp2gene_mask(self, gene_indices, snp_indices, type_indices=None, CHR=None):

        if len(gene_indices)==0:
            return torch.zeros((len(self.chr2gene[CHR]), len(self.chr2snp[CHR])))
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
        return {"snp":torch.tensor(snp_embedding_indices), "gene":torch.tensor(gene_embedding_indices)}

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

