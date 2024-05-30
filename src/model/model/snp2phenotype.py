import torch
import torch.nn as nn

from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import Genotype2Phenotype
from src.model.hierarchical_transformer import HierarchicalTransformer

class SNP2PhenotypeModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, hidden_dims, subtree_order=('default',), n_covariates=13, dropout=0.2, binary=False,
                 activation='softmax'):
        super(SNP2PhenotypeModel, self).__init__(tree_parser, hidden_dims, subtree_order=subtree_order, dropout=dropout)
        self.n_snps = self.tree_parser.n_snps
        self.by_chr = self.tree_parser.by_chr
        self.gene_padding_ind = self.n_genes
        self.snp_padding_ind = self.n_snps
        self.chromosomes = tree_parser.chromosomes
        self.snp_embedding = nn.Embedding(self.n_snps + 1, hidden_dims, padding_idx=self.n_snps)
        self.gene_embedding = nn.Embedding(self.n_genes + 1, hidden_dims, padding_idx=self.n_genes)

        self.snp_linear = nn.Linear(int(hidden_dims/4), hidden_dims)
        self.snp_norm = nn.LayerNorm(hidden_dims)

        self.snp2gene_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.snp2gene_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)
        self.snp2gene_heterozygous = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                self.snp2gene_update_norm_inner,
                                                self.snp2gene_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='genotype',
                                                             activation=activation, n_type=1)
        self.snp2gene_homozygous = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                self.snp2gene_update_norm_inner,
                                                self.snp2gene_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='genotype',
                                                           activation=activation, n_type=1)


        self.gene2sys_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2sys_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2sys = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                self.gene2sys_update_norm_inner,
                                                self.gene2sys_update_norm_outer,
                                                dropout, norm_channel_first=self.norm_channel_first,
                                                conv_type='genotype',
                                                activation='softmax')
        self.n_covariates = n_covariates
        self.covariate_linear_1 = nn.Linear(self.n_covariates, hidden_dims)
        self.covariate_norm_1 = nn.LayerNorm(hidden_dims)
        self.covariate_linear_2 = nn.Linear(hidden_dims, hidden_dims)
        self.covariate_norm_2 = nn.LayerNorm(hidden_dims)

        self.sys2pheno_norm = nn.LayerNorm(hidden_dims)
        self.gene2pheno_norm = nn.LayerNorm(hidden_dims)
        self.sys2pheno_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2pheno_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2pheno_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2pheno_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)


        self.sys2pheno = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4,
                                                   inner_norm=self.sys2pheno_update_norm_inner,
                                                   outer_norm=self.sys2pheno_update_norm_outer, dropout=dropout,
                                                   transform=True, activation='softmax')#'softmax')
        self.gene2pheno = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4,
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

    def forward(self, genotype_dict, covariates, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                gene2sys_mask, sys2gene_mask, sys2env=True, env2sys=True, sys2gene=True):
        batch_size = covariates.size(0)
        gene_embedding = self.get_snp2gene(genotype=genotype_dict)[:, :-1, :]
        system_embedding = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
        system_embedding = self.get_gene2sys(system_embedding, gene_embedding, gene2sys_mask)
        update_tensor = torch.zeros_like(system_embedding)
        if sys2env:
            system_embedding, update_tensor = self.get_sys2sys(system_embedding, nested_hierarchical_masks_forward, direction='forward', return_updates=True, with_indices=True, update_tensor=update_tensor)
        if env2sys:
            system_embedding = self.get_sys2sys(system_embedding, nested_hierarchical_masks_backward, direction='backward', return_updates=False, with_indices=True, update_tensor=update_tensor)
        if sys2gene:
            gene_embedding = self.get_sys2gene(system_embedding, gene_embedding, sys2gene_mask)
        phenotype_vector = self.get_phenotype_vector(covariates)
        prediction = self.prediction(phenotype_vector, system_embedding, gene_embedding)
        return prediction

    def get_snp2gene(self, genotype):
        heterozygous_gene_indices, heterozygous_snp_effect_from_embedding = self.get_snp_effects(
            genotype['embedding']['heterozygous'], self.snp2gene_heterozygous)
        homozygous_gene_indices, homozygous_snp_effect_from_embedding = self.get_snp_effects(
            genotype['embedding']['homozygous_a1'], self.snp2gene_homozygous)
        gene_indices = torch.cat([heterozygous_gene_indices, homozygous_gene_indices], dim=-1)
        snp_effect_from_embedding = torch.cat(
            [heterozygous_snp_effect_from_embedding, homozygous_snp_effect_from_embedding], dim=1)

        batch_size = gene_indices.size(0)
        snp_effect = torch.zeros_like(self.gene_embedding.weight).unsqueeze(0).expand(batch_size, -1, -1)
        results = []
        for b, value in enumerate(snp_effect):
            results.append(
                snp_effect[b].index_add(0, gene_indices[b], snp_effect_from_embedding[b]))
        snp_effect = torch.stack(results, dim=0)
        gene_embedding = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        return gene_embedding + snp_effect

    def get_snp_effects(self, genotype, transformer, attention=False, score=False):
        snp_indices = genotype['snp']
        gene_indices = genotype['gene']
        if len(gene_indices) == 0:
            return None, None
        snp_embedding = self.snp_norm(self.snp_embedding(snp_indices))
        gene_embedding = self.gene_norm(self.gene_embedding(gene_indices))
        mask = genotype['mask']
        snp_effect_from_embedding = transformer(gene_embedding, snp_embedding, mask)
        if attention:
            attention_result = transformer.get_attention(gene_embedding, snp_embedding, mask)
            if score:
                score_result = transformer.get_score(gene_embedding, snp_embedding, mask)
                return gene_indices, snp_effect_from_embedding, attention_result, score_result
            else:
                return gene_indices, snp_effect_from_embedding, attention_result
        elif score:
            score_result = transformer.get_score(gene_embedding, snp_embedding, mask)
            return gene_indices, snp_effect_from_embedding, score_result
        else:
            return gene_indices, snp_effect_from_embedding


    def get_gene2sys(self, system_embedding, gene_embedding, gene2sys_mask):
        system_embedding_input = self.sys_norm(system_embedding)
        gene_embedding = self.gene_norm(gene_embedding)
        gene_effect = self.gene2sys.forward(system_embedding_input, gene_embedding, gene2sys_mask)
        return system_embedding + gene_effect


    def get_phenotype_vector(self, covariates):
        covariates_vector = self.covariate_norm_1(self.activation(self.covariate_linear_1(covariates)))
        covariates_vector = self.covariate_norm_2(self.covariate_linear_2(covariates_vector))
        covariates_vector = covariates_vector.unsqueeze(1)
        return covariates_vector

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
        sys2pheno_result = self.sys2pheno.forward(phenotype_vector, system_embedding, system_embedding, mask=system_mask)
        if attention:
            sys2pheno_attention = self.sys2pheno.get_attention(phenotype_vector, system_embedding, system_embedding)
            sys2pheno_result = [sys2pheno_result, sys2pheno_attention]
            if score:
                sys2pheno_score = self.sys2pheno.get_score(phenotype_vector, system_embedding, system_embedding)
                sys2pheno_result += [sys2pheno_score]
            return sys2pheno_result
        else:
            if score:
                sys2pheno_score = self.sys2pheno.get_score(phenotype_vector, system_embedding,
                                                                      system_embedding)
                sys2pheno_result = [sys2pheno_result, sys2pheno_score]
            return sys2pheno_result


    def get_gene2pheno(self, phenotype_vector, gene_embedding, gene_mask=None, attention=False, score=False):
        gene_embedding = self.gene2pheno_norm(gene_embedding)
        gene2pheno_result = self.gene2pheno.forward(phenotype_vector, gene_embedding, gene_embedding, mask=gene_mask)
        if attention:
            gene2pheno_attention = self.gene2pheno.get_attention(phenotype_vector, gene_embedding, gene_embedding)
            gene2pheno_result = [gene2pheno_result, gene2pheno_attention]
            if score:
                gene2pheno_score = self.gene2pheno.get_score(phenotype_vector, gene_embedding, gene_embedding)
                gene2pheno_result += [gene2pheno_score]
            return gene2pheno_result
        else:
            if score:
                gene2pheno_score = self.gene2pheno.get_score(phenotype_vector, gene_embedding,
                                                                      gene_embedding)
                gene2pheno_result = [gene2pheno_result, gene2pheno_score]
            return gene2pheno_result
