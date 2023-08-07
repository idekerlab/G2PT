import torch
import torch.nn as nn

from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import Genotype2Phenotype
from src.model.hierarchical_transformer import HierarchicalTransformer

class SNP2PhenotypeModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, genotypes, hidden_dims, effective_allele='heterozygous', n_covariates=13, dropout=0.2, binary=False):
        super(SNP2PhenotypeModel, self).__init__(tree_parser, genotypes, hidden_dims, dropout=dropout, activation='softmax')
        self.effective_allele = effective_allele
        self.n_snps = self.tree_parser.n_snps
        self.by_chr = self.tree_parser.by_chr
        self.gene_padding_ind = self.n_genes
        self.snp_padding_ind = self.n_snps
        self.chromosomes = tree_parser.chromosomes
        #phenotype_vector = torch.empty(1, hidden_dims)
        #nn.init.xavier_normal(phenotype_vector)
        self.snp_embedding = nn.Embedding(self.n_snps + 1, hidden_dims, padding_idx=self.n_snps)
        self.gene_embedding = nn.Embedding(self.n_genes + 1, hidden_dims, padding_idx=self.n_genes)

        self.snp_linear = nn.Linear(int(hidden_dims/4), hidden_dims)
        self.snp_norm = nn.LayerNorm(hidden_dims)

        self.snp2gene_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.snp2gene_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)
        if self.effective_allele=='heterozygous':
            n_type = 2
        else:
            n_type = 1
        self.snp2gene = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                           self.snp2gene_update_norm_inner,
                                                           self.snp2gene_update_norm_outer,
                                                           dropout, norm_channel_first=self.norm_channel_first,
                                                           conv_type='genotype', n_type=n_type, activation='softmax')
        '''
        self.snp2gene_heterozygous = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                self.snp2gene_update_norm_inner,
                                                self.snp2gene_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='genotype', n_type=1)
        self.snp2gene_homozygous = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                self.snp2gene_update_norm_inner,
                                                self.snp2gene_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='genotype', n_type=1)
        '''

        self.gene2sys_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2sys_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2sys = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                self.gene2sys_update_norm_inner,
                                                self.gene2sys_update_norm_outer,
                                                dropout, norm_channel_first=self.norm_channel_first,
                                                conv_type='genotype', activation='softmax')
        self.n_covariates = n_covariates
        #self.phenotype_vector = nn.Embedding(1, hidden_dims)
        self.covariate_linear_1 = nn.Linear(self.n_covariates, hidden_dims)
        self.covariate_norm_1 = nn.LayerNorm(hidden_dims)
        self.covariate_linear_2 = nn.Linear(hidden_dims, hidden_dims)
        self.covariate_norm_2 = nn.LayerNorm(hidden_dims)

        #self.genotype_norm = nn.LayerNorm(hidden_dims)
        self.sys2pheno_norm = nn.LayerNorm(hidden_dims)
        self.gene2pheno_norm = nn.LayerNorm(hidden_dims)
        #self.geno2pheno_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        #self.geno2pheno_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2pheno_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2pheno_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2pheno_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2pheno_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)


        self.system2phenotype = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4,
                                                   inner_norm=self.sys2pheno_update_norm_inner,
                                                   outer_norm=self.sys2pheno_update_norm_outer, dropout=dropout,
                                                   transform=True, activation='softmax')
        self.gene2phenotype = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4,
                                                 inner_norm=self.gene2pheno_update_norm_inner,
                                                 outer_norm=self.gene2pheno_update_norm_outer, dropout=dropout,
                                                 transform=True, activation='softmax')

        self.last_activation = nn.Tanh()

        #self.prediction_norm = nn.LayerNorm(hidden_dims * 2, elementwise_affine=False)
        self.phenotype_predictor_1 = nn.Linear(hidden_dims * 2, hidden_dims)
        self.phenotype_predictor_2 = nn.Linear(hidden_dims, 1)
        self.sigmoid = nn.Sigmoid()
        self.binary = binary
        if self.binary:
            print("Model will predict binary")
        else:
            print("Model will do regression")

    def forward(self, genotype_dict, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                gene2sys_mask, sys2gene_mask, gene_weight=None, sys2cell=True, cell2sys=True, sys2gene=True):
        if self.by_chr:
            batch_size, _ = genotype_dict['embedding']['homozygous'][1]['snp'].size()
        else:
            #batch_size, _ = genotype_dict['embedding']['homozygous']['snp'].size()
            batch_size, _ = genotype_dict['embedding']['snp'].size()
        gene_embedding = self.get_snp2gene(batch_size, genotype=genotype_dict)[:, :-1, :]
        system_embedding = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
        system_embedding = self.get_gene2sys(system_embedding, gene_embedding, genotype_dict)
        if sys2cell:
            system_embedding = self.get_sys2sys(system_embedding, nested_hierarchical_masks_forward, direction='forward', return_updates=False)
        if cell2sys:
            system_embedding = self.get_sys2sys(system_embedding, nested_hierarchical_masks_backward, direction='backward', return_updates=False)
        if sys2gene:
            gene_embedding = self.get_sys2gene(system_embedding, gene_embedding, sys2gene_mask)
        phenotype_vector = self.get_phenotype_vector(genotype_dict['covariates'])
        prediction = self.prediction(phenotype_vector, system_embedding, gene_embedding)
        return prediction

    def get_snp2gene(self, batch_size, genotype):
        if self.by_chr:
            heterozygous_result = [self.get_snp_effects(genotype['embedding']['heterozygous'][CHR], self.snp2gene_heterozygous) for CHR in self.chromosomes]
            heterozygous_gene_indices = torch.cat(
                [gene_indices for gene_indices, chr_snp_effect in heterozygous_result if gene_indices is not None],
                dim=1)
            heterozygous_snp_effect_from_embedding = torch.cat(
                [chr_snp_effect for gene_indices, chr_snp_effect in heterozygous_result if chr_snp_effect is not None],
                dim=1)
            homozygous_result = [self.get_snp_effects(genotype['embedding']['homozygous'][CHR], self.snp2gene_homozygous) for
                                 CHR in self.chromosomes]
            homozygous_gene_indices = torch.cat(
                [gene_indices for gene_indices, chr_snp_effect in homozygous_result if gene_indices is not None],
                dim=1)
            homozygous_snp_effect_from_embedding = torch.cat(
                [chr_snp_effect for gene_indices, chr_snp_effect in homozygous_result if chr_snp_effect is not None],
                dim=1)
            gene_indices = torch.cat([heterozygous_gene_indices, homozygous_gene_indices], dim=-1)
            snp_effect_from_embedding = torch.cat([heterozygous_snp_effect_from_embedding, homozygous_snp_effect_from_embedding], dim=1)

        else:
            '''
            heterozygous_gene_indices, heterozygous_snp_effect_from_embedding = self.get_snp_effects(
                genotype['embedding']['heterozygous'], self.snp2gene_heterozygous)
            homozygous_gene_indices, homozygous_snp_effect_from_embedding = self.get_snp_effects(
                genotype['embedding']['homozygous'], self.snp2gene_homozygous)
            gene_indices = torch.cat([heterozygous_gene_indices, homozygous_gene_indices], dim=-1)
            snp_effect_from_embedding = torch.cat(
                [heterozygous_snp_effect_from_embedding, homozygous_snp_effect_from_embedding], dim=1)
            '''
            gene_indices, snp_effect_from_embedding = self.get_snp_effects(genotype['embedding'], self.snp2gene)
        batch_size = gene_indices.size(0)
        snp_effect = torch.zeros_like(self.gene_embedding.weight).unsqueeze(0).expand(batch_size, -1, -1)
        results = []
        for b, value in enumerate(snp_effect):
            results.append(
                snp_effect[b].index_add(0, gene_indices[b], snp_effect_from_embedding[b]))
        snp_effect = torch.stack(results, dim=0)
        gene_embedding = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return gene_embedding + snp_effect

    def get_snp_effects(self, genotype, transformer):
        snp_indices = genotype['snp']
        batch_size, snp_length = snp_indices.size()
        gene_indices = genotype['gene']
        if len(gene_indices) == 0:
            return None, None
        batch_size, gene_length = gene_indices.size()
        #snp_embedding = self.snp_norm(self.snp_linear(self.snp_embedding(snp_indices)))
        snp_embedding = self.snp_norm(self.snp_embedding(snp_indices))
        gene_embedding = self.gene_norm(self.gene_embedding(gene_indices))
        mask = genotype['mask']
        if self.effective_allele=='heterozygous':
            snp_effect_from_embedding = transformer(gene_embedding, snp_embedding, [(1, (mask==1).to(torch.int32)), (2, (mask==2).to(torch.int32))])
        else:
            snp_effect_from_embedding = transformer(gene_embedding, snp_embedding, mask)
        return gene_indices, snp_effect_from_embedding


    def get_gene2sys(self, system_embedding, gene_embedding, genotype):
        system_embedding_input = self.sys_norm(system_embedding)
        gene_embedding = self.gene_norm(gene_embedding)
        gene_effect = self.gene2sys.forward(system_embedding_input, gene_embedding, genotype['gene2sys_mask'])
        return system_embedding + gene_effect


    def get_phenotype_vector(self, covariates, batch_size=1):
        #covariate_vector = self.covariate_norm(self.covariate_linear(covariates))
        covariate_vector = self.covariate_norm_1(self.activation(self.covariate_linear_1(covariates)))
        covariate_vector = self.covariate_norm_2(self.covariate_linear_2(covariate_vector))
        covariate_vector = covariate_vector.unsqueeze(1)
        #phenotype_vector = self.phenotype_norm(self.phenotype_vector.weight.unsqueeze(0).expand(batch_size, -1, -1))
        return covariate_vector

    def prediction(self, phenotype_vector, system_embedding, gene_embedding):
        phenotype_weighted_by_systems = self.get_sys2pheno(phenotype_vector, system_embedding, system_mask=None)
        phenotype_weighted_by_genes = self.get_gene2pheno(phenotype_vector, gene_embedding, gene_mask=None)
        phenotype_feature = torch.cat([phenotype_weighted_by_systems, phenotype_weighted_by_genes], dim=-1)
        phenotype_prediction = self.phenotype_predictor_2(self.last_activation(self.phenotype_predictor_1(phenotype_feature)))
        if self.binary:
            phenotype_prediction = self.sigmoid(phenotype_prediction)
        return phenotype_prediction

    def get_sys2pheno(self, phenotype_vector, system_embedding, system_mask=None, attention=False, score=False):
        system_embedding = self.sys2pheno_norm(system_embedding)
        sys2phenotype_result = self.system2phenotype.forward(phenotype_vector, system_embedding, system_embedding, mask=system_mask)
        if attention:
            sys2phenotype_attention = self.system2phenotype.get_attention(phenotype_vector, system_embedding, system_embedding)
            sys2phenotype_result = [sys2phenotype_result, sys2phenotype_attention]
            if score:
                sys2phenotype_score = self.system2phenotype.get_score(phenotype_vector, system_embedding, system_embedding)
                sys2phenotype_result += [sys2phenotype_score]
            return sys2phenotype_result
        else:
            if score:
                sys2phenotype_score = self.system2phenotype.get_score(phenotype_vector, system_embedding,
                                                                      system_embedding)
                sys2phenotype_result = [sys2phenotype_result, sys2phenotype_score]
            return sys2phenotype_result


    def get_gene2pheno(self, phenotype_vector, gene_embedding, gene_mask=None, attention=False, score=False):
        gene_embedding = self.gene2pheno_norm(gene_embedding)
        gene2phenotype_result = self.gene2phenotype.forward(phenotype_vector, gene_embedding, gene_embedding, mask=gene_mask)
        if attention:
            gene2phenotype_attention = self.gene2phenotype.get_attention(phenotype_vector, gene_embedding, gene_embedding)
            gene2phenotype_result = [gene2phenotype_result, gene2phenotype_attention]
            if score:
                gene2phenotype_score = self.gene2phenotype.get_score(phenotype_vector, gene_embedding, gene_embedding)
                gene2phenotype_result += [gene2phenotype_score]
            return gene2phenotype_result
        else:
            if score:
                gene2phenotype_score = self.gene2phenotype.get_score(phenotype_vector, gene_embedding,
                                                                      gene_embedding)
                gene2phenotype_result = [gene2phenotype_result, gene2phenotype_score]
            return gene2phenotype_result
