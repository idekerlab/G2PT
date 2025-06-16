import pandas as pd
import numpy as np
import torch
from . import TreeParser


class SNPTreeParser(TreeParser):

    def __init__(self,
                 ontology,        # path or DataFrame for parent–child ontology
                 snp2gene,        # path or DataFrame for SNP→gene mapping
                 dense_attention=False,
                 sys_annot_file=None,
                 by_chr=False,
                 multiple_phenotypes=False):
        # 1. Initialize all system–level structures
        #super().__init__(ontology,
        #                 dense_attention=dense_attention,
        #                 sys_annot_file=sys_annot_file)
        ontology = pd.read_csv(ontology, sep='\t', names=['parent', 'child', 'interaction'])
        self.dense_attention = dense_attention
        if sys_annot_file:
            sys_descriptions = pd.read_csv(sys_annot_file, header=None, names=['Term', 'Term_Description'], index_col=0, sep='\t')

            self.sys_annot_dict = sys_descriptions.to_dict()["Term_Description"]
        else:
            self.sys_annot_dict = None
        super().init_ontology(ontology,
                              inplace=True)
        # 2. Now build both the system AND SNP structures in one shot:
        #    pass `snp2gene` through to init_ontology
        self.init_ontology_with_snp(self.ontology,
                           snp2gene,
                           inplace=True,
                           multiple_phenotypes=multiple_phenotypes)

        self.by_chr = by_chr


    def init_ontology_with_snp(self,
                      ontology_df,
                      snp2gene,
                      inplace=True,
                      multiple_phenotypes=False,
                      verbose=True):
        """
        Extend TreeParser.init_ontology by also loading and wiring
        the SNP→gene table (snp2gene).
        """
        # ———— 1) do all the gene–system work in the parent ————
        parent_result = super().init_ontology(ontology_df,
                                              inplace=inplace,
                                              verbose=verbose)
        parser = parent_result if parent_result is not None else self

        # ———— 2) load snp2gene (either file‐path or DataFrame) ————
        '''
        if isinstance(snp2gene, str):
            # assume tab‑delimited with headers ['snp','gene','chr'] or auto‑detect
            if multiple_phenotypes:
                parser.snp2gene_df = pd.read_csv(snp2gene, sep='\t')
                #print(parser.snp2gene_df.head())
            else:
        
            parser.snp2gene_df = pd.read_csv(snp2gene, names=['snp', 'gene', 'chr'],
                                            sep='\t',
                                            dtype={'snp':str,'gene':str,'chr':str})
        '''
        parser.snp2gene_df = pd.read_csv(snp2gene, sep='\t')
        #else:
        #    parser.snp2gene_df = snp2gene.copy()
        print(parser.snp2gene_df.head())
        if 'block' in parser.snp2gene_df.columns:
            parser.snp2gene_df =  parser.snp2gene_df.sort_values(by=["chr", "block"]).reset_index(drop=True)
            parser.blocks = sorted(list(
                parser.snp2gene_df[["chr", "block"]].drop_duplicates().sort_values(["chr", "block"]).itertuples(
                    index=False, name=None)))
            #print(parser.blocks)
            parser.n_blocks = len(parser.blocks)
            parser.block2ind = {block: i for i, block in enumerate(parser.blocks)}
            parser.ind2block = {i:block for i, block in enumerate(parser.blocks)}
            snps = parser.snp2gene_df.drop_duplicates(subset=['snp'])['snp'].tolist()

            parser.snp2ind_all = {s: i for i, s in enumerate(snps)}
            parser.ind2snp_all = {i: s for s, i in parser.snp2ind_all.items()}

            parser.snp2block = {}
            parser.block2snp = {block:[] for block in parser.blocks}
            parser.block2sig_ind = {block: [] for block in parser.blocks}
            for i, row in parser.snp2gene_df.drop_duplicates(subset=['snp']).iterrows():
                parser.snp2block[row.snp] = (row.chr, row.block)
                parser.block2snp[(row.chr, row.block)].append(row.snp)
                if 'block_ind' in parser.snp2gene_df.columns:
                    parser.block2sig_ind[(row.chr, row.block)].append(row.block_ind)
                else:
                    #print((row.chr, row.block))
                    parser.block2sig_ind[(row.chr, row.block)].append(-1)
                    #print(parser.block2sig_ind[(row.chr, row.block)])
            #parser.block2sig_ind = {block: sorted(list(set(sig_inds))) for block, sig_inds in parser.block2sig_ind.items()}
        #else:
        #    parser.snp2gene_df = parser.snp2gene_df.sort_values(by=["chr", "snp"]).reset_index(drop=True)
        parser.snp2gene_df = parser.snp2gene_df.loc[parser.snp2gene_df['gene'].isin(parser.gene2ind.keys())]
        #print(parser.snp2gene_df.head())
        '''
        print(parser.snp2gene_df.head())
        # optional multiple‐phenotype branch
        genes_in_ont = list(parser.gene2ind.keys())
        genes_in_snp2gene = parser.snp2gene_df.gene.tolist()
        print([gene for gene in genes_in_snp2gene if gene not in genes_in_ont])
        genes = list(set(genes_in_ont + genes_in_snp2gene))
        #print(len(genes_in_ont), len(set(genes_in_snp2gene)), len(genes))
        parser.n_genes = len(genes)
        parser.gene2ind = {gene: i for i, gene in enumerate(genes)}
        parser.ind2gene = {i: gene for i, gene in enumerate(genes)}

        parser.gene2sys_mask = np.full((int(np.ceil(parser.n_systems/8)*8), (int(np.ceil(parser.n_genes/8)*8))), -10**4)#np.zeros((len(parser.sys2ind), len(parser.gene2ind)))
        parser.sys2gene_dict = { parser.sys2ind[system]: [] for system in parser.sys2ind.keys() }
        parser.gene2sys_dict = { gene: [] for gene in range(parser.n_genes) }
        for system, gene in zip(parser.gene2sys_df['parent'], parser.gene2sys_df['child']):
            #print(system, gene, parser.sys2ind[system], parser.gene2ind[gene])
            parser.gene2sys_mask[parser.sys2ind[system], parser.gene2ind[gene]] = 0.
            parser.sys2gene_dict[parser.sys2ind[system]].append(parser.gene2ind[gene])
            parser.gene2sys_dict[parser.gene2ind[gene]].append(parser.sys2ind[system])

        if parser.dense_attention:
            parser.gene2sys_mask = torch.ones_like(torch.tensor(parser.gene2sys_mask))
        parser.sys2gene_mask = parser.gene2sys_mask.T
        '''

        # ———— 3) filter to known genes ————
        #parser.snp2gene_df = parser.snp2gene_df.loc[
        #    parser.snp2gene_df['gene'].isin(parser.gene2ind)
        #]
        #print(parser.snp2gene_df.shape)
        snps = parser.snp2gene_df.drop_duplicates(subset=['snp'])['snp'].values.tolist()
        if multiple_phenotypes:
            if 'block' in parser.snp2gene_df.columns:
                if 'sig_in' in parser.snp2gene_df.columns:
                    parser.phenotypes = list(parser.snp2gene_df.columns[5:])
                else:
                    parser.phenotypes = list(parser.snp2gene_df.columns[4:])
            else:
                parser.phenotypes = list(parser.snp2gene_df.columns[3:])
            parser.pheno2snp = {
                ph: parser.snp2gene_df.loc[parser.snp2gene_df[ph], 'snp']
                     .unique()
                     .tolist()
                for ph in parser.phenotypes
            }
            parser.pheno2gene = {
                ph: parser.snp2gene_df.loc[parser.snp2gene_df[ph], 'gene']
                     .unique()
                     .tolist()
                for ph in parser.phenotypes
            }
            parser.snp2pheno = {snp:[] for snp in snps}
            for ph in parser.phenotypes:
                # mask of rows where this phenotype is "True"/non-zero
                mask = parser.snp2gene_df[ph].astype(bool)
                # unique SNPs for this phenotype
                for snp in parser.snp2gene_df.loc[mask, 'snp'].unique():
                    parser.snp2pheno[snp].append(ph)

        # ———— 4) build SNP↔gene dicts & masks ————
        by_snp  = parser.snp2gene_df.groupby('snp')
        by_gene = parser.snp2gene_df.groupby('gene')

        parser.gene2snp = {
            gene: grp['snp'].tolist()
            for gene, grp in by_gene
        }
        parser.snp2gene = {
            snp: grp['gene'].tolist()
            for snp, grp in by_snp
        }


        # integer indices


        parser.snp2ind = { s:i for i,s in enumerate(snps) }
        parser.ind2snp = { i:s for s,i in parser.snp2ind.items() }
        ordered_genes = self._compute_gene_order_from_snps()
        parser.sys2ind, parser.ind2sys, parser.gene2ind, parser.ind2gene, parser.gene2sys_mask = parser.build_mask(list(parser.sys2ind.keys()), ordered_genes,
                                                                                                                   parser.sys2gene)
        parser.sys2gene_mask = parser.gene2sys_mask.T
        parser.n_snps = len(snps)
        print("The number of SNPs:", parser.n_snps)
        parser.snp_pad_index = parser.n_snps

        # chromosomes
        parser.chromosomes = sorted(parser.snp2gene_df['chr'].unique())
        parser.gene2chr = dict(zip(parser.snp2gene_df['gene'],
                                   parser.snp2gene_df['chr']))
        parser.snp2chr  = dict(zip(parser.snp2gene_df['snp'],
                                   parser.snp2gene_df['chr']))
        parser.chr2gene = {
            c: [parser.gene2ind[g]
                for g in parser.snp2gene_df
                              .loc[parser.snp2gene_df['chr']==c, 'gene']]
            for c in parser.chromosomes
        }
        parser.chr2snp = {
            c: [parser.snp2ind[s]
                for s in parser.snp2gene_df
                              .loc[parser.snp2gene_df['chr']==c, 'snp']]
            for c in parser.chromosomes
        }

        # system→SNP mapping (uses your existing get_sys2snp)
        parser.sys2snp = {
            sys_idx: parser.get_sys2snp(sys_idx)
            for sys_idx in parser.sys2ind
        }
        parser.snp2sys = {}
        for sys_idx, snp_list in parser.sys2snp.items():
            for snp in snp_list:
                parser.snp2sys.setdefault(snp, []).append(sys_idx)

        # finally masks and dicts
        parser.snp2gene_mask   = np.full((int(np.ceil((parser.n_genes+1)/8)*8), (int(np.ceil(parser.n_snps/8)*8))), -10**4)
        parser.gene2snp_dict    = {gi:[] for gi in range(parser.n_genes)}
        parser.snp2gene_dict    = {si:[] for si in range(parser.n_snps)}


        for snp, gene in zip(parser.snp2gene_df['snp'],
                              parser.snp2gene_df['gene']):
            #if gene in genes_in_ont:
            gi = parser.gene2ind[gene]
            si = parser.snp2ind[snp]
            parser.snp2gene_mask[gi, si] = 0
            parser.gene2snp_dict[gi].append(si)
            parser.snp2gene_dict[si].append(gi)

        parser.gene2snp_mask = parser.snp2gene_mask.T

        if multiple_phenotypes:
            print("Build Partial Embeddings and Mask")
            parser.snp2gene_by_phenotype_dict = {}
            for pheno in self.phenotypes:
                pheno_dict = {}

                pheno_dict['snp'] = [parser.snp2ind[snp] for snp in parser.pheno2snp[pheno]]
                pheno_dict['gene'] = [parser.gene2ind[gene] for gene in parser.pheno2gene[pheno]]
                snp2gene_pheno_mask = np.full((int(np.ceil(len(pheno_dict['gene']) / 8) * 8),
                                               (int(np.ceil(len(pheno_dict['snp']) / 8) * 8))),
                                        -10 ** 4)
                #print(pheno_dict)
                for i, gene_ind in enumerate(pheno_dict['gene']):
                    for snp_ind in parser.gene2snp_dict[gene_ind]:
                        if snp_ind in pheno_dict['snp']:
                            snp2gene_pheno_mask[i, pheno_dict['snp'].index(snp_ind)] = 0
                pheno_dict['mask'] = torch.tensor(snp2gene_pheno_mask, dtype=torch.float32)


                parser.snp2gene_by_phenotype_dict[pheno] = pheno_dict

        # ———— 5) return only if non‑inplace ————
        if parent_result is not None:
            return parser


    def summary(self, system=True, gene=True, snp=True):
        super(SNPTreeParser, self).summary(system=system, gene=gene)
        if snp:
            print("SNP Index: ")
            if self.by_chr:
                print("Embedding will feed by chromosome")
            print("The number of SNP-Gene connections: %d" % self.snp2gene_mask.sum())
            for i in range(len(self.ind2snp)):
                print(i, ":", self.ind2snp[i], " -> ", ",".join(self.snp2gene))
            print(" ")

    def _compute_gene_order_from_snps(self):
        """
        Compute new gene order based on SNP chromosome/block structure
        """
        # SNPs are already ordered by chromosome and block
        # Group genes by their first appearing SNP
        gene_to_first_snp = {}

        for snp_idx, snp_id in enumerate(self.ind2snp.values()):
            # Get genes connected to this SNP
            connected_genes = self.snp2gene[snp_id]

            for gene in connected_genes:
                if gene not in gene_to_first_snp:
                    gene_to_first_snp[gene] = snp_idx

        # Sort genes by their first appearing SNP index
        ordered_genes = sorted(gene_to_first_snp.keys(),
                               key=lambda g: gene_to_first_snp[g])

        # Add any genes not connected to SNPs at the end
        all_genes = set(self.ind2gene.values())
        unconnected_genes = all_genes - set(ordered_genes)
        ordered_genes.extend(sorted(unconnected_genes))

        return ordered_genes


    def get_sys2snp(self, sys):
        genes = self.sys2gene_full[sys]
        snps = [self.gene2snp[gene] for gene in genes if gene in self.gene2snp.keys()]
        snps = list(set(sum(snps, [])))
        return snps

    def get_snp2gene_mask(self, type_indices=None):
        if type_indices is not None:
            snp2gene_mask = np.copy(self.snp2gene_mask)
            for key, value in type_indices.items():
                type_mask = np.zeros_like(self.snp2gene_mask)
                type_mask[:, value] = key
                snp2gene_mask *= type_mask
        else:
            snp2gene_mask = np.copy(self.snp2gene_mask)

        return snp2gene_mask

    def get_snp2gene_sub_mask(self, gene_indices, snp_indices, type_indices=None, CHR=None):

        if len(gene_indices)==0:
            return torch.zeros((self.n_genes, self.n_snps))
        else:
            snp2gene_mask = self.get_snp2gene_mask(type_indices)
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
        return {"snp":snp_embedding_indices, "gene":gene_embedding_indices}

    def get_snp2gene_indices(self, snp_indices):
        return sorted(list(set(sum([self.snp2gene_dict[snp] for snp in snp_indices], []))))

    def get_snp2gene(self, snp_indices, type_indices=None, snp_ind_alias_dict=None, gene_ind_alias_dict=None):
        if self.by_chr:
            return self.get_snp2gene_by_chromosome(snp_indices, type_indices=type_indices)
        else:
            embeddings = self.get_snp2gene_embeddings(snp_indices)
            mask = self.get_snp2gene_sub_mask(embeddings['gene'], embeddings['snp'], type_indices=type_indices)
            if snp_ind_alias_dict is not None:
                snp_indices = torch.tensor(self.alias_indices(embeddings['snp'], self.ind2snp, snp_ind_alias_dict), dtype=torch.int)
            else:
                snp_indices = torch.tensor(embeddings['snp'], dtype=torch.int)
            if gene_ind_alias_dict is not None:
                gene_indices = torch.tensor(self.alias_indices(embeddings['gene'], self.ind2gene, gene_ind_alias_dict), dtype=torch.int)
            else:
                gene_indices = torch.tensor(embeddings['gene'], dtype=torch.int)
            return {"snp": snp_indices, 'gene': gene_indices, 'mask': mask}

    def get_snp2gene_by_chromosome(self, snp_indices, type_indices=None):
        embeddings = {CHR: self.get_snp2gene_embeddings([snp for snp in snp_indices if snp in self.chr2snp[CHR]])  for CHR in self.chromosomes}
        #print(snp_indices, embeddings)
        mask = {CHR: self.get_snp2gene_mask(embeddings[CHR]['gene'], embeddings[CHR]['snp'], type_indices=type_indices, CHR=CHR) for CHR in self.chromosomes}
        return {CHR: {"snp":embeddings[CHR]['snp'], "gene":embeddings[CHR]['gene'], 'mask':mask[CHR] } for CHR in self.chromosomes}

    def get_target_indices(self, target_gos, target_genes, target_snps):
        """Fetch integer indices from parser for the target GO, gene, and SNP lists."""
        target_go_inds = [self.sys2ind[go] for go in target_gos]
        target_gene_inds = [self.gene2ind[g] for g in target_genes]
        target_snp_inds = [self.snp2ind[s] for s in target_snps]
        return target_go_inds, target_gene_inds, target_snp_inds

    def get_target_components(self, target_go):
        if self.node_height_dict[target_go] != 0:
            target_gos = self.get_descendants_sorted_by_height(target_go) + [target_go]
        else:
            target_gos = [target_go]

        target_genes = self.sys2gene_full[target_go]
        target_snps = self.sys2snp[target_go]
        return target_gos, target_genes, target_snps

    def retain_snps(self, snp_list, inplace=False, verbose=True):
        """
        retain genes in input gene list and rebuild ontology

        Parameters:
        ----------
        gene_list : list, tuple
            list of genes to retain
        """
        new_snp2gene_df = self.snp2gene_df.loc[self.snp2gene_df.snp.isin(snp_list)]
        gene2keep = new_snp2gene_df.gene.unique()
        gene2sys_to_keep = self.gene2sys_df.loc[self.gene2sys_df.child.isin(gene2keep)]
        ontology_df_new = pd.concat([self.sys_df, gene2sys_to_keep])
        if inplace:
            self.init_ontology_with_snp(ontology_df_new, new_snp2gene_df, inplace=inplace, verbose=verbose)
        else:
            return self.init_ontology_with_snp(ontology_df_new, new_snp2gene_df, inplace=inplace, verbose=verbose)




