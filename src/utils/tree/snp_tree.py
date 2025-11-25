import pandas as pd
import numpy as np
import torch
import math
from . import TreeParser
import re
from collections import deque



class SNPTreeParser(TreeParser):

    def __init__(self,
                 ontology,        # path or DataFrame for parent–child ontology
                 snp2gene,        # path or DataFrame for SNP→gene mapping
                 dense_attention=False,
                 sys_annot_file=None,
                 by_chr=False,
                 multiple_phenotypes=False,
                 block_bias=False):
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
        self.block_bias = block_bias
        self.by_chr = by_chr
        self.init_ontology_with_snp(self.ontology,
                           snp2gene,
                           inplace=True,
                           multiple_phenotypes=multiple_phenotypes)





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
        if isinstance(snp2gene, str):
            parser.snp2gene_df = pd.read_csv(snp2gene, sep='\t')
        else:
            parser.snp2gene_df = snp2gene.copy()
        print(parser.snp2gene_df.head())
        
        # Conditionally sort by position if available
        sort_columns = ["chr"]
        if 'pos' in parser.snp2gene_df.columns:
            sort_columns.append('pos')
        if 'block' in parser.snp2gene_df.columns:
            sort_columns.append('block')
        
        if len(sort_columns) > 1:
            parser.snp2gene_df = parser.snp2gene_df.sort_values(by=sort_columns).reset_index(drop=True)

        if 'block' in parser.snp2gene_df.columns:
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
            parser.gene2block = {}
            parser.block2snp = {block:[] for block in parser.blocks}
            parser.block2gene = {block: [] for block in parser.blocks}
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
            for i, row in parser.snp2gene_df.drop_duplicates(subset=['gene']).iterrows():
                parser.gene2block[row.gene] = (row.chr, row.block)
                parser.block2gene[(row.chr, row.block)].append(row.gene)
            #parser.block2sig_ind = {block: sorted(list(set(sig_inds))) for block, sig_inds in parser.block2sig_ind.items()}
        #else:
        #    parser.snp2gene_df = parser.snp2gene_df.sort_values(by=["chr", "snp"]).reset_index(drop=True)
        #parser.snp2gene_df = parser.snp2gene_df.loc[parser.snp2gene_df['gene'].isin(parser.gene2ind.keys())]
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
        parser.snp2gene_df = parser.snp2gene_df.loc[
            parser.snp2gene_df['gene'].isin(parser.gene2ind)
        ]

        #print(parser.snp2gene_df.shape)
        snps = sorted(parser.snp2gene_df.drop_duplicates(subset=['snp'])['snp'].values.tolist())
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
        parser.n_genes = len(parser.gene2ind)
        parser.n_snps = len(snps)
        print("The number of SNPs:", parser.n_snps)
        parser.snp_pad_index = parser.n_snps

        # chromosomes
        parser.chromosomes = sorted(parser.snp2gene_df['chr'].unique())
        parser.gene2chr = dict(zip(parser.snp2gene_df['gene'],
                                   parser.snp2gene_df['chr']))
        parser.snp2chr  = dict(zip(parser.snp2gene_df['snp'],
                                   parser.snp2gene_df['chr']))
        if 'pos' in parser.snp2gene_df.columns:
            parser.snp2pos = dict(zip(parser.snp2gene_df['snp'],
                                      parser.snp2gene_df['pos']))
        else:
            parser.snp2pos = self.snp_pos_dict(parser.snp2gene_df['snp'].values.tolist())

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
        parser.snp2gene_mask   = np.full((int(np.ceil((parser.n_genes+1)/8)*8), (int(np.ceil((parser.n_snps+1)/8)*8))), -10**4)
        parser.gene2snp_dict    = {gi:[] for gi in range(parser.n_genes)}
        parser.snp2gene_dict    = {si:[] for si in range(parser.n_snps)}


        for snp, gene in zip(parser.snp2gene_df['snp'],
                              parser.snp2gene_df['gene']):
            #if gene in genes_in_ont:
            gi = parser.gene2ind[gene]
            si = parser.snp2ind[snp]
            if parser.block_bias:
                snp_block = self.snp2block[snp]
                parser.snp2gene_mask[gi, si] = 0
                block2genes = self.block2gene[snp_block]
                for gene_in_block in block2genes:
                    if gene_in_block in parser.gene2ind.keys():
                        gi_block = parser.gene2ind[gene_in_block]
                        parser.snp2gene_mask[gi_block, si] = 0
            else:
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
        #if parent_result is not None:
        #    return parser
        if not inplace:
            return parser

    @staticmethod
    def snp_pos_dict(ids):
        """
        Build {snp_id: pos} for items like CHR:POS:REF:ALT (rsIDs are skipped).
        Accepts optional 'chr' prefix and common chrom labels.
        """
        pat = re.compile(
            r'^(?:chr)?(?:[0-9]{1,2}|X|Y|MT|M):'  # CHR
            r'(\d+):'  # POS (capture)
            r'[ACGTN\-]+:'  # REF (allow N / indels)
            r'[ACGTN,\-]+$',  # ALT (allow multiple alts, indels)
            re.IGNORECASE
        )
        out = {}
        for s in ids:
            s = str(s).strip()
            m = pat.match(s)
            if m:
                out[s] = int(m.group(1))
        return out

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
        all_genes_in_parser = set(self.gene2ind.keys())
        genes_in_ordered_list = set(ordered_genes)
        unconnected_genes = all_genes_in_parser - genes_in_ordered_list
        ordered_genes.extend(sorted(list(unconnected_genes)))

        return ordered_genes


    def get_sys2snp(self, sys):
        genes = self.sys2gene_full[sys]
        snps = [self.gene2snp[gene] for gene in genes if gene in self.gene2snp.keys()]
        snps = list(set(sum(snps, [])))
        return snps
    '''
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
    '''
    def get_snp2gene_mask_with_interation(self, snp_indices):
        snp2gene_mask = torch.tensor(self.snp2gene_mask, dtype=torch.float32)
        snp2gene_mask_interaction = torch.full_like(snp2gene_mask, fill_value=-10000)
        snp2gene_mask_interaction[:, snp_indices] = snp2gene_mask[:, snp_indices]
        return snp2gene_mask_interaction

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

    def collect_systems_to_root(self, target_systems):
        """
        Collect all systems from the specified target systems up to the root node(s).

        This function takes a list of target systems and collects all ancestor systems
        (including the target systems themselves) up to the root nodes. It traverses
        the system graph in the reverse direction (from children to parents) and
        returns all unique systems encountered in the path to any root node.

        Parameters
        ----------
        target_systems : list
            List of system names to start collection from.

        Returns
        -------
        set
            Set of all systems from target systems to root nodes (including target systems).

        Examples
        --------
        >>> # Collect all systems from specific targets to root
        >>> target_systems = ['GO:0006412', 'GO:0006413']
        >>> collected = tree_parser.collect_systems_to_root(target_systems)
        >>> print(f"Collected {len(collected)} systems from targets to root")
        """
        if not target_systems:
            return set()

        # Validate that all target systems exist in the graph
        invalid_systems = [sys for sys in target_systems if sys not in self.sys_graph.nodes]
        if invalid_systems:
            raise ValueError(f"The following systems are not found in the system graph: {invalid_systems}")

        # Set to store all collected systems
        collected_systems = set()

        # Queue for breadth-first traversal (systems to process)
        systems_to_process = set(target_systems)

        # Add target systems to the collected set
        collected_systems.update(target_systems)

        # Traverse upward to collect all ancestors
        while systems_to_process:
            current_system = systems_to_process.pop()

            # Get all parent systems (predecessors in the directed graph)
            parent_systems = list(self.sys_graph.predecessors(current_system))

            # Add new parent systems to collection and processing queue
            for parent in parent_systems:
                if parent not in collected_systems:
                    collected_systems.add(parent)
                    systems_to_process.add(parent)

        return collected_systems

    def collapse_by_gene_similarity(self, similarity_threshold=0.7, to_keep=None, verbose=True, inplace=False):
        # Call the parent method to get the collapsed ontology
        collapsed_parser = super().collapse_by_gene_similarity(
            similarity_threshold=similarity_threshold,
            to_keep=to_keep,
            verbose=verbose,
            inplace=False  # Always work on a copy
        )

        # Save the collapsed ontology to a temporary file
        temp_ontology_path = "/cellar/projects/G2PT_T2D/tests/temp_collapsed_ontology.tsv"
        collapsed_parser.ontology.to_csv(temp_ontology_path, sep='	', index=False, header=False)

        # Re-initialize with the SNP data
        if inplace:
            self.init_ontology_with_snp(
                collapsed_parser.ontology,
                self.snp2gene_df,
                inplace=True,
                verbose=verbose
            )
        else:
            # Create a new SNPTreeParser instance
            new_parser = SNPTreeParser(
                ontology=temp_ontology_path,
                snp2gene=self.snp2gene_df,
                dense_attention=self.dense_attention,
                sys_annot_file=None, # Annotations are already in the collapsed parser
                by_chr=self.by_chr,
                multiple_phenotypes=hasattr(self, 'phenotypes'),
                block_bias=self.block_bias
            )
            new_parser.sys_annot_dict = collapsed_parser.sys_annot_dict
            return new_parser

    def resnik_similarity(self, snp_a, snp_b, smoothing=1e-9):
        cache = self._ensure_resnik_cache(smoothing)
        terms_a = self._resnik_terms_for_snp(snp_a, cache)
        terms_b = self._resnik_terms_for_snp(snp_b, cache)
        if not terms_a or not terms_b:
            return 0.0
        return self._resnik_pairwise(terms_a, terms_b, cache)

    def resnik_similarity_matrix(self, snps=None, smoothing=1e-9, as_dataframe=True):
        cache = self._ensure_resnik_cache(smoothing)
        if snps is None:
            snp_names = self._resnik_all_snp_names()
        else:
            snp_names = [self._resnik_normalise_snp(s) for s in snps]

        n = len(snp_names)
        matrix = np.zeros((n, n), dtype=float)

        for i, snp_i in enumerate(snp_names):
            terms_i = self._resnik_terms_for_snp(snp_i, cache)
            if not terms_i:
                continue
            for j in range(i, n):
                terms_j = self._resnik_terms_for_snp(snp_names[j], cache)
                if not terms_j:
                    continue
                score = self._resnik_pairwise(terms_i, terms_j, cache)
                matrix[i, j] = matrix[j, i] = score

        if as_dataframe:
            return pd.DataFrame(matrix, index=snp_names, columns=snp_names)
        return matrix

    def _ensure_resnik_cache(self, smoothing):
        if not hasattr(self, '_resnik_cache'):
            self._resnik_cache = {}

        key = float(smoothing)
        cache = self._resnik_cache.get(key)
        if cache is None:
            self._validate_resnik_requirements()
            cache = {
                'smoothing': smoothing,
                'term_ic': self._compute_resnik_information_content(smoothing),
                'term_ancestors': self._compute_resnik_term_ancestors(),
                'snp_terms': {}
            }
            self._resnik_cache[key] = cache
        return cache

    def _validate_resnik_requirements(self):
        required_attrs = ['sys_graph', 'sys2gene_full', 'gene2ind', 'snp2sys']
        missing = [attr for attr in required_attrs if not hasattr(self, attr)]
        if missing:
            raise AttributeError(f"SNPTreeParser is missing required attributes for Resnik similarity: {missing}")

    def _resnik_all_snp_names(self):
        if hasattr(self, 'snp2ind'):
            return [snp for snp, _ in sorted(self.snp2ind.items(), key=lambda item: item[1])]
        return sorted(self.snp2sys.keys())

    def _resnik_normalise_snp(self, snp):
        if isinstance(snp, str):
            if hasattr(self, 'snp2ind') and snp in self.snp2ind:
                return snp
            if snp in self.snp2sys:
                return snp
        elif isinstance(snp, int) and hasattr(self, 'ind2snp') and snp in self.ind2snp:
            return self.ind2snp[snp]
        raise KeyError(f"Unknown SNP identifier: {snp}")

    def _resnik_terms_for_snp(self, snp, cache):
        snp_name = self._resnik_normalise_snp(snp)
        snp_terms = cache['snp_terms']
        if snp_name not in snp_terms:
            terms = self.snp2sys.get(snp_name, [])
            snp_terms[snp_name] = {term for term in terms if term in cache['term_ic']}
        return snp_terms[snp_name]

    def _compute_resnik_information_content(self, smoothing):
        total_genes = len(self.gene2ind)
        if total_genes == 0:
            raise ValueError("SNPTreeParser does not contain any genes")

        ic = {}
        for term, genes in self.sys2gene_full.items():
            gene_count = len(set(genes))
            probability = (gene_count + smoothing) / (total_genes + smoothing)
            ic[term] = -math.log(probability)
        return ic

    def _compute_resnik_term_ancestors(self):
        ancestors = {}
        for term in self.sys_graph.nodes:
            visited = set()
            stack = list(self.sys_graph.predecessors(term))
            while stack:
                parent = stack.pop()
                if parent in visited:
                    continue
                visited.add(parent)
                stack.extend(self.sys_graph.predecessors(parent))
            visited.add(term)
            ancestors[term] = visited
        return ancestors

    def _resnik_pairwise(self, terms_a, terms_b, cache):
        best = 0.0
        term_ic = cache['term_ic']
        term_ancestors = cache['term_ancestors']
        for term_a in terms_a:
            ancestors_a = term_ancestors.get(term_a)
            if not ancestors_a:
                continue
            for term_b in terms_b:
                ancestors_b = term_ancestors.get(term_b)
                if not ancestors_b:
                    continue
                common = ancestors_a & ancestors_b
                if not common:
                    continue
                score = max(term_ic[term] for term in common)
                if score > best:
                    best = score
        return best

    def wu_palmer_similarity(self, snp_a, snp_b):
        """Wu & Palmer style similarity derived from LCA depth."""
        cache = self._ensure_structural_cache()
        terms_a = self._structural_terms_for_snp(snp_a, cache)
        terms_b = self._structural_terms_for_snp(snp_b, cache)
        if not terms_a or not terms_b:
            return 0.0

        def score_terms(term_a, term_b):
            return self._wu_palmer_terms(term_a, term_b, cache)

        score = self._pairwise_term_score(terms_a, terms_b, score_terms, prefer='max')
        return float(score) if score is not None else 0.0

    def wu_palmer_similarity_matrix(self, snps=None, as_dataframe=True):
        return self._structural_matrix(
            snps,
            scorer=self._wu_palmer_terms,
            prefer='max',
            default=0.0,
            as_dataframe=as_dataframe,
        )

    def tree_distance(self, snp_a, snp_b):
        """Shortest undirected path length between SNP leaf systems."""
        cache = self._ensure_structural_cache()
        terms_a = self._structural_terms_for_snp(snp_a, cache)
        terms_b = self._structural_terms_for_snp(snp_b, cache)
        if not terms_a or not terms_b:
            return math.inf

        def distance_terms(term_a, term_b):
            return self._tree_distance_terms(term_a, term_b, cache)

        distance = self._pairwise_term_score(terms_a, terms_b, distance_terms, prefer='min')
        if distance is None:
            return math.inf
        return float(distance)

    def tree_distance_matrix(self, snps=None, as_dataframe=True):
        return self._structural_matrix(
            snps,
            scorer=self._tree_distance_terms,
            prefer='min',
            default=math.inf,
            as_dataframe=as_dataframe,
        )

    def ancestor_jaccard_similarity(self, snp_a, snp_b):
        """Jaccard similarity of ancestor sets for SNP leaf systems."""
        cache = self._ensure_structural_cache()
        terms_a = self._structural_terms_for_snp(snp_a, cache)
        terms_b = self._structural_terms_for_snp(snp_b, cache)
        if not terms_a or not terms_b:
            return 0.0

        def jaccard_terms(term_a, term_b):
            return self._ancestor_jaccard_terms(term_a, term_b, cache)

        score = self._pairwise_term_score(terms_a, terms_b, jaccard_terms, prefer='max')
        return float(score) if score is not None else 0.0

    def ancestor_jaccard_similarity_matrix(self, snps=None, as_dataframe=True):
        return self._structural_matrix(
            snps,
            scorer=self._ancestor_jaccard_terms,
            prefer='max',
            default=0.0,
            as_dataframe=as_dataframe,
        )

    def _ensure_structural_cache(self):
        if not hasattr(self, '_structural_cache'):
            self._structural_cache = {}

        cache = self._structural_cache
        if 'term_ancestors' not in cache:
            if getattr(self, '_resnik_cache', None):
                any_cache = next(iter(self._resnik_cache.values()), None)
                if any_cache and 'term_ancestors' in any_cache:
                    cache['term_ancestors'] = {
                        term: set(ancestors)
                        for term, ancestors in any_cache['term_ancestors'].items()
                    }
            if 'term_ancestors' not in cache:
                self._validate_resnik_requirements()
                cache['term_ancestors'] = self._compute_resnik_term_ancestors()

        if 'term_depth' not in cache:
            cache['term_depth'] = self._compute_structural_term_depths()

        if 'leaf_systems' not in cache:
            cache['leaf_systems'] = {
                node for node in self.sys_graph.nodes
                if self.sys_graph.out_degree(node) == 0
            }

        cache.setdefault('snp_leaf_terms', {})
        return cache

    def _compute_structural_term_depths(self):
        roots = [node for node, indegree in self.sys_graph.in_degree() if indegree == 0]
        depths = {}
        queue = deque()

        if roots:
            for root in roots:
                depths[root] = 0
                queue.append((root, 0))
        else:
            # Fallback to treat every node as a root-level node
            for node in self.sys_graph.nodes:
                depths[node] = 0

        while queue:
            node, depth = queue.popleft()
            for child in self.sys_graph.successors(node):
                child_depth = depth + 1
                if child not in depths or child_depth < depths[child]:
                    depths[child] = child_depth
                    queue.append((child, child_depth))

        for node in self.sys_graph.nodes:
            depths.setdefault(node, 0)

        return depths

    def _structural_terms_for_snp(self, snp, cache):
        snp_name = self._resnik_normalise_snp(snp)
        snp_terms = cache['snp_leaf_terms']
        if snp_name not in snp_terms:
            systems = self.snp2sys.get(snp_name, [])
            leaf_terms = [term for term in systems if term in cache['leaf_systems']]
            if not leaf_terms:
                leaf_terms = [term for term in systems if term in cache['term_depth']]
            snp_terms[snp_name] = set(leaf_terms)
        return snp_terms[snp_name]

    def _pairwise_term_score(self, terms_a, terms_b, scorer, prefer='max'):
        best = None
        for term_a in terms_a:
            for term_b in terms_b:
                value = scorer(term_a, term_b)
                if value is None:
                    continue
                if best is None:
                    best = value
                elif prefer == 'max':
                    if value > best:
                        best = value
                elif prefer == 'min':
                    if value < best:
                        best = value
                else:
                    raise ValueError(f"Unknown preference '{prefer}' for pairwise scoring")
        return best

    def _structural_matrix(self, snps, scorer, prefer, default, as_dataframe):
        cache = self._ensure_structural_cache()
        if snps is None:
            snp_names = self._resnik_all_snp_names()
        else:
            snp_names = [self._resnik_normalise_snp(s) for s in snps]

        n = len(snp_names)
        matrix = np.zeros((n, n), dtype=float)
        snp_terms = {
            snp_name: self._structural_terms_for_snp(snp_name, cache)
            for snp_name in snp_names
        }

        def score_terms(term_a, term_b):
            return scorer(term_a, term_b, cache)

        for i, snp_i in enumerate(snp_names):
            terms_i = snp_terms[snp_i]
            for j in range(i, n):
                terms_j = snp_terms[snp_names[j]]
                if not terms_i or not terms_j:
                    value = default
                else:
                    value = self._pairwise_term_score(terms_i, terms_j, score_terms, prefer=prefer)
                    if value is None:
                        value = default
                matrix[i, j] = matrix[j, i] = value

        if as_dataframe:
            return pd.DataFrame(matrix, index=snp_names, columns=snp_names)
        return matrix

    def _lca_info(self, term_a, term_b, cache):
        term_ancestors = cache['term_ancestors']
        ancestors_a = term_ancestors.get(term_a)
        ancestors_b = term_ancestors.get(term_b)
        if not ancestors_a or not ancestors_b:
            return None, None
        common = ancestors_a & ancestors_b
        if not common:
            return None, None
        term_depth = cache['term_depth']
        lca = max(common, key=lambda term: term_depth.get(term, 0))
        return lca, term_depth.get(lca, 0)

    def _wu_palmer_terms(self, term_a, term_b, cache):
        lca, depth_lca = self._lca_info(term_a, term_b, cache)
        if lca is None:
            return None
        term_depth = cache['term_depth']
        depth_a = term_depth.get(term_a, depth_lca)
        depth_b = term_depth.get(term_b, depth_lca)
        denominator = depth_a + depth_b
        if denominator == 0:
            return 1.0 if term_a == term_b else 0.0
        return (2.0 * depth_lca) / denominator

    def _tree_distance_terms(self, term_a, term_b, cache):
        lca, depth_lca = self._lca_info(term_a, term_b, cache)
        if lca is None:
            return None
        term_depth = cache['term_depth']
        depth_a = term_depth.get(term_a, depth_lca)
        depth_b = term_depth.get(term_b, depth_lca)
        return max(0.0, (depth_a + depth_b) - (2.0 * depth_lca))

    def _ancestor_jaccard_terms(self, term_a, term_b, cache):
        term_ancestors = cache['term_ancestors']
        ancestors_a = term_ancestors.get(term_a)
        ancestors_b = term_ancestors.get(term_b)
        if not ancestors_a or not ancestors_b:
            return None
        union = ancestors_a | ancestors_b
        if not union:
            return None
        return len(ancestors_a & ancestors_b) / len(union)

