import torch
import torch.nn as nn

from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import Genotype2Phenotype
from src.model.hierarchical_transformer import HierarchicalTransformer
from src.model.utils import PoincareNorm

class SNP2PhenotypeModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, hidden_dims, snp2pheno=False, gene2pheno=True, sys2pheno=True,
                 subtree_order=['default'], n_covariates=13, dropout=0.2, binary=False,
                 activation='softmax', input_format='indices', poincare=False):
        super(SNP2PhenotypeModel, self).__init__(tree_parser, hidden_dims, subtree_order=subtree_order, dropout=dropout,
                                                 input_format=input_format, poincare=poincare)
        self.n_snps = self.tree_parser.n_snps
        self.by_chr = self.tree_parser.by_chr
        self.gene_padding_ind = self.n_genes
        self.snp_padding_ind = self.n_snps
        self.chromosomes = tree_parser.chromosomes
        self.snp_embedding = nn.Embedding(self.n_snps + 1, hidden_dims, padding_idx=self.n_snps)
        self.gene_embedding = nn.Embedding(self.n_genes + 1, hidden_dims, padding_idx=self.n_genes)
        self.snp_norm = nn.LayerNorm(hidden_dims)


        self.snp2gene_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.snp2gene_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)
        self.snp2gene_heterozygous = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                self.snp2gene_update_norm_inner,
                                                self.snp2gene_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='genotype',
                                                             activation=activation, n_type=1, poincare=poincare)
        self.snp2gene_homozygous = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                self.snp2gene_update_norm_inner,
                                                self.snp2gene_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='genotype',
                                                           activation=activation, n_type=1, poincare=poincare)

        self.gene2sys_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2sys_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2sys = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                self.gene2sys_update_norm_inner,
                                                self.gene2sys_update_norm_outer,
                                                dropout, norm_channel_first=self.norm_channel_first,
                                                conv_type='system',
                                                activation='softmax', poincare=poincare)
        '''
        self.gene2sys = HierarchicalTransformer(hidden_dims, 4, hidden_dims * 4,
                                                self.gene2sys_update_norm_inner,
                                                self.gene2sys_update_norm_outer,
                                                dropout, norm_channel_first=self.norm_channel_first,
                                                conv_type='genotype',
                                                activation='softmax')
        '''
        self.n_covariates = n_covariates
        self.covariate_linear_1 = nn.Linear(self.n_covariates, hidden_dims)
        self.covariate_linear_2 = nn.Linear(hidden_dims, hidden_dims)
        self.covariate_norm_1 = nn.LayerNorm(hidden_dims, eps=0.1)
        self.covariate_norm_2 = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2pheno_norm = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2pheno_norm = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2pheno_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.sys2pheno_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2pheno_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.gene2pheno_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)

        if sys2pheno:
            self.sys2pheno = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4,
                                                       inner_norm=self.sys2pheno_update_norm_inner,
                                                       outer_norm=self.sys2pheno_update_norm_outer, dropout=0.0,
                                                       transform=True, activation='softmax', poincare=poincare)#'softmax')
        else:
            self.sys2pheno = None
        if gene2pheno:
            self.gene2pheno = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4,
                                                 inner_norm=self.gene2pheno_update_norm_inner,
                                                 outer_norm=self.gene2pheno_update_norm_outer, dropout=0.0,
                                                 transform=True, activation='softmax', poincare=poincare)
        else:
            self.gene2pheno = None

        if snp2pheno:
            self.hetero2pheno = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4,
                                                 inner_norm=self.gene2pheno_update_norm_inner,
                                                 outer_norm=self.gene2pheno_update_norm_outer, dropout=0.0,
                                                 transform=True, activation='softmax', poincare=poincare)
            self.homo2pheno = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4,
                                                 inner_norm=self.gene2pheno_update_norm_inner,
                                                 outer_norm=self.gene2pheno_update_norm_outer, dropout=0.0,
                                                 transform=True, activation='softmax', poincare=poincare)
        else:
            self.hetero2pheno = None
            self.homo2pheno = None

        self.last_activation = nn.Tanh()
        n_geno2pheno = sum([(self.sys2pheno is not None), (self.gene2pheno is not None), (self.hetero2pheno is not None),
                            (self.homo2pheno is not None)])
        #self.prediction_norm = nn.LayerNorm(hidden_dims, elementwise_affine=True)

        self.phenotype_predictor_1 = nn.Linear(hidden_dims * n_geno2pheno, hidden_dims)
        self.phenotype_predictor_2 = nn.Linear(hidden_dims, 1)
        self.sigmoid = nn.Sigmoid()
        self.binary = binary
        if self.binary:
            print("Model will predict binary")
        else:
            print("Model will do regression")

    def forward(self, genotype_dict, covariates, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                gene2sys_mask, sys2gene_mask, sys2env=True, env2sys=True, sys2gene=True, score=False, attention=False):
        batch_size = covariates.size(0)
        gene_embedding, snp_effect_on_gene = self.get_snp2gene(genotype=genotype_dict)
        if self.input_format == 'indices':
            gene_embedding = gene_embedding[:, :-1, :] + self.effect_norm(snp_effect_on_gene[:, :-1, :])
        else:
            gene_embedding = gene_embedding + self.effect_norm(snp_effect_on_gene)
        #gene_embedding = gene_embedding[:, :-1, :] + snp_effect_on_gene[:, :-1, :]
        system_embedding = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
        #print(gene2sys_mask.size())
        system_embedding, gene_effect_on_system = self.get_gene2sys(system_embedding, gene_embedding, gene2sys_mask)
        total_update = self.effect_norm(gene_effect_on_system)
        #total_update = gene_effect_on_system
        if sys2env:
            system_embedding, system_effect_forward = self.get_sys2sys(system_embedding,
                                                                       nested_hierarchical_masks_forward,
                                                                       direction='forward', return_updates=True,
                                                                       input_format=self.input_format,
                                                                       update_tensor=total_update)
            total_update = total_update + system_effect_forward
        if env2sys:
            system_embedding, system_effect_backward = self.get_sys2sys(system_embedding,
                                                                        nested_hierarchical_masks_backward,
                                                                        direction='backward', return_updates=True,
                                                                        input_format=self.input_format,
                                                                        update_tensor=total_update)
            total_update = total_update + system_effect_backward
            system_embedding = system_embedding + total_update
        if sys2gene:
            gene_embedding, system_effect_on_gene = self.get_sys2gene(gene_embedding, system_embedding, sys2gene_mask)
            gene_embedding = gene_embedding + self.effect_norm(system_effect_on_gene)
            #gene_embedding = gene_embedding + system_effect_on_gene
        phenotype_vector = self.get_phenotype_vector(covariates)
        prediction = self.prediction(phenotype_vector, system_embedding, gene_embedding, genotype=genotype_dict['embedding'])
        if attention:
            if score:
                system_embedding, system_attention, system_score = self.get_sys2pheno(phenotype_vector,
                                                                                      system_embedding, attention=True,
                                                                                      score=True)
                gene_embedding, gene_attention, gene_score = self.get_gene2pheno(phenotype_vector,
                                                                                      gene_embedding, attention=True,
                                                                                      score=True)
                return prediction, system_attention, gene_attention, system_score, gene_score
            else:
                system_embedding, system_attention = self.get_sys2pheno(phenotype_vector, system_embedding,
                                                                        attention=True, score=False)
                gene_embedding, gene_attention = self.get_gene2pheno(phenotype_vector, gene_embedding, attention=True,
                                                                     score=False)
                return prediction, system_attention, gene_attention
        else:
            if score:
                system_embedding, system_score = self.get_sys2pheno(phenotype_vector, system_embedding,
                                                                        attention=False, score=True)
                gene_embedding, gene_score = self.get_gene2pheno(phenotype_vector, gene_embedding, attention=False,
                                                                     score=True)
                return prediction, system_score, gene_score
            else:
                return prediction

    def get_snp2gene(self, genotype):
        if self.input_format=='indices':
            heterozygous_gene_indices, heterozygous_snp_effect_from_embedding = self.get_snp_effects(
                genotype['embedding']['heterozygous'], self.snp2gene_heterozygous)
            homozygous_gene_indices, homozygous_snp_effect_from_embedding = self.get_snp_effects(
                genotype['embedding']['homozygous_a1'], self.snp2gene_homozygous)
            batch_size = homozygous_gene_indices.size(0)
            snp_effect = torch.zeros_like(self.gene_embedding.weight).unsqueeze(0).expand(batch_size, -1, -1)
            '''
            gene_indices = torch.cat([heterozygous_gene_indices, homozygous_gene_indices], dim=-1)
            snp_effect_from_embedding = torch.cat(
                [heterozygous_snp_effect_from_embedding, homozygous_snp_effect_from_embedding], dim=1)
            for b, value in enumerate(snp_effect):
                results.append(
                    snp_effect[b].index_add(0, gene_indices[b], snp_effect_from_embedding[b]))
            snp_effect = torch.stack(results, dim=0)
            '''
            snp_effect = snp_effect.scatter_add(1, heterozygous_gene_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dims), heterozygous_snp_effect_from_embedding)
            snp_effect = snp_effect.scatter_add(1, homozygous_gene_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dims), homozygous_snp_effect_from_embedding)
            gene_embedding = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
            return gene_embedding,  snp_effect
        else:
            batch_size = genotype['embedding']['homozygous_a1'].size(0)
            snp_embedding = self.snp_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
            gene_embedding = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
            hetero_effect = self.snp2gene_heterozygous.forward(self.gene_norm(gene_embedding),
                                                               self.snp_norm(snp_embedding),
                                                             genotype['embedding']['heterozygous'], dropout=False)
            homo_effect = self.snp2gene_homozygous.forward(self.gene_norm(gene_embedding),
                                                               self.snp_norm(snp_embedding),
                                                               genotype['embedding']['homozygous_a1'], dropout=False)
            return gene_embedding, hetero_effect + homo_effect

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
    '''
    def get_gene2sys(self, system_embedding, gene_embedding, gene2sys_mask):
        system_embedding_input = self.sys_norm(system_embedding)
        gene_embedding = self.gene_norm(gene_embedding)
        gene_effect = self.gene2sys.forward(system_embedding_input, gene_embedding, gene2sys_mask)
        return system_embedding, gene_effect
    '''
    def get_phenotype_vector(self, covariates):
        covariates_vector = self.covariate_norm_1(self.activation(self.covariate_linear_1(self.dropout(covariates))))
        #covariates_vector = self.covariate_norm_2(self.covariate_linear_2(self.dropout(covariates_vector)))
        covariates_vector = self.dropout(covariates_vector.unsqueeze(1))
        return covariates_vector

    def prediction(self, phenotype_vector, system_embedding, gene_embedding, genotype=None):
        if self.sys2pheno is not None:
            phenotype_weighted_by_systems = self.get_sys2pheno(phenotype_vector, system_embedding, system_mask=None)
        if self.gene2pheno is not None:
            phenotype_weighted_by_genes = self.get_gene2pheno(phenotype_vector, gene_embedding, gene_mask=None)
        if self.homo2pheno is not None:
            phenotype_weighted_by_hetero, phenotype_weighted_by_homo = self.get_snp2pheno(phenotype_vector, genotype)

        if (self.sys2pheno is not None) & (self.gene2pheno is not None):
            phenotype_feature = torch.cat([phenotype_weighted_by_systems, phenotype_weighted_by_genes], dim=-1)
        elif self.sys2pheno is not None:
            phenotype_feature = phenotype_weighted_by_systems
        elif self.gene2pheno is not None:
            phenotype_feature = phenotype_weighted_by_genes
        if (self.homo2pheno is not None):
            phenotype_feature = torch.cat([phenotype_feature, phenotype_weighted_by_hetero, phenotype_weighted_by_homo], dim=-1)

        phenotype_prediction = self.phenotype_predictor_2(self.last_activation(self.phenotype_predictor_1(phenotype_feature)))
        if self.binary:
            phenotype_prediction = self.sigmoid(phenotype_prediction)
        return phenotype_prediction

    def get_sys2pheno(self, phenotype_vector, system_embedding, system_mask=None, attention=False, score=False):
        system_embedding_input = self.sys2pheno_norm(system_embedding)
        sys2phenotype_result = self.sys2pheno.forward(phenotype_vector, system_embedding_input, system_embedding_input, mask=system_mask)
        if attention:
            sys2phenotype_attention = self.sys2pheno.get_attention(phenotype_vector, system_embedding_input, system_embedding_input)
            sys2phenotype_result = [sys2phenotype_result, sys2phenotype_attention]
            if score:
                sys2phenotype_score = self.sys2pheno.get_score(phenotype_vector, system_embedding_input, system_embedding_input)
                sys2phenotype_result += [sys2phenotype_score]
            return sys2phenotype_result
        else:
            if score:
                sys2phenotype_score = self.sys2pheno.get_score(phenotype_vector, system_embedding_input,
                                                                      system_embedding_input)
                sys2phenotype_result = [sys2phenotype_result, sys2phenotype_score]
            return sys2phenotype_result


    def get_gene2pheno(self, phenotype_vector, gene_embedding, gene_mask=None, attention=False, score=False):
        gene_embedding_input = self.gene2pheno_norm(gene_embedding)
        gene2phenotype_result = self.gene2pheno.forward(phenotype_vector, gene_embedding_input, gene_embedding_input, mask=gene_mask)
        if attention:
            gene2phenotype_attention = self.gene2pheno.get_attention(phenotype_vector, gene_embedding_input, gene_embedding_input)
            gene2phenotype_result = [gene2phenotype_result, gene2phenotype_attention]
            if score:
                gene2phenotype_score = self.gene2pheno.get_score(phenotype_vector, gene_embedding_input, gene_embedding_input)
                gene2phenotype_result += [gene2phenotype_score]
            return gene2phenotype_result
        else:
            if score:
                gene2phenotype_score = self.gene2pheno.get_score(phenotype_vector, gene_embedding_input,
                                                                      gene_embedding_input)
                gene2phenotype_result = [gene2phenotype_result, gene2phenotype_score]
            return gene2phenotype_result

    def get_snp2pheno(self, phenotype_vector, genotype, attention=False, score=False):
        hetero_indices = genotype['heterozygous']['snp']
        homo_indices = genotype['homozygous_a1']['snp']
        hetero_snp_embedding = self.snp_norm(self.snp_embedding(hetero_indices))
        homo_snp_embedding = self.snp_norm(self.snp_embedding(homo_indices))
        hetero2phenotype_result = self.hetero2pheno.forward(phenotype_vector,
                                                            hetero_snp_embedding,
                                                            hetero_snp_embedding)
        homo2phenotype_result = self.hetero2pheno.forward(phenotype_vector,
                                                          homo_snp_embedding,
                                                          homo_snp_embedding)
        return hetero2phenotype_result, homo2phenotype_result