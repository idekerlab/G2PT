import torch
import torch.nn as nn
import numpy as np


from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import Genotype2Phenotype
from src.model.hierarchical_transformer import HierarchicalTransformer
from src.model.utils import MoEHeadPrediction, FiLM
from src.model.LD_infuser.LDRoBERTa import RoBERTa, TransformerLayer, RoBERTaConfig, BlockAdapter

class SNP2PhenotypeModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, hidden_dims, snp2pheno=False, gene2pheno=True, sys2pheno=True,
                 interaction_types=['default'], n_covariates=13, n_phenotypes=1, dropout=0.2,
                 activation='softmax', input_format='indices', poincare=False, cov_effect='pre',
                 pretrained_transformer=None, freeze_pretrained=True,
                 phenotypes=('PHENOTYPE')):
        super(SNP2PhenotypeModel, self).__init__(tree_parser, hidden_dims, interaction_types=interaction_types, dropout=dropout,
                                                 input_format=input_format, poincare=poincare)
        self.n_snps = self.tree_parser.n_snps
        self.by_chr = self.tree_parser.by_chr
        self.gene_padding_ind = self.n_genes
        self.snp_padding_ind = self.n_snps * 3
        self.chromosomes = tree_parser.chromosomes
        snp_embedding_length = int(np.ceil(self.n_snps/8)*8)
        self.snp_adapters = nn.Parameter(torch.randn(snp_embedding_length, self.hidden_dims, self.hidden_dims))
        self.snp_embedding = nn.Embedding(self.n_snps * 3 + 1, hidden_dims, padding_idx=self.n_snps * 3)
        self.gene_embedding = nn.Embedding(self.n_genes + 1, hidden_dims, padding_idx=self.n_genes)
        if self.input_format == 'embedding':
            self.n_blocks = self.tree_parser.n_blocks
            self.block_embedding = nn.Embedding(self.n_blocks + 1, hidden_dims, padding_idx=self.n_blocks)
        elif self.input_format == 'block':
            self.blocks = pretrained_transformer.keys()
            self.pretrained_transformer = nn.ModuleDict(pretrained_transformer)
            #print(self.pretrained_transformer)
            self.freeze_pretrained = freeze_pretrained
            self.n_snps2pad = int(np.ceil(self.tree_parser.n_snps/8 * 8) - self.tree_parser.n_snps)
            if freeze_pretrained:
                for block_id, model in self.pretrained_transformer.items():
                    for param in model.parameters():
                        param.requires_grad = False
            config = RoBERTaConfig(hidden_size=self.hidden_dims, num_attention_heads=4,intermediate_size=self.hidden_dims, dropout=dropout)
            self.pretrained_transformer_adapter = nn.ModuleDict(
                {block_id: BlockAdapter(self.hidden_dims, self.hidden_dims) for block_id in pretrained_transformer.keys()})
            self.pretrained_transformer_tuner = nn.ModuleDict({block_id: TransformerLayer(config, 1) for block_id in pretrained_transformer.keys()})


        self.snp_norm = nn.LayerNorm(hidden_dims)
        '''
        snp2gene_dict = {}
        for pheno in phenotypes:
            snp2gene_update_norm_inner = nn.LayerNorm(hidden_dims)
            snp2gene_update_norm_outer = nn.LayerNorm(hidden_dims)
            higt = HierarchicalTransformer(hidden_dims, 4, hidden_dims,
                                                snp2gene_update_norm_inner,
                                                snp2gene_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='genotype',
                                                             activation=activation, n_type=1, poincare=poincare)
            snp2gene_dict[pheno] = higt
        self.snp2gene = nn.ModuleDict(snp2gene_dict)
        '''
        self.snp2gene_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.snp2gene_update_norm_outer = nn.LayerNorm(hidden_dims)
        self.snp2gene = HierarchicalTransformer(hidden_dims, 4, hidden_dims,
                                                self.snp2gene_update_norm_inner,
                                                self.snp2gene_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='system',
                                                             activation=activation, n_type=1, poincare=poincare)

        #self.snp2gene_homozygous = HierarchicalTransformer(hidden_dims, 4, hidden_dims,
        #                                        self.snp2gene_update_norm_inner,
        #                                        self.snp2gene_update_norm_outer,
        #                                       dropout, norm_channel_first=self.norm_channel_first, conv_type='genotype',
        #                                                   activation=activation, n_type=1, poincare=poincare)

        self.gene2sys_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.gene2sys_update_norm_outer = nn.LayerNorm(hidden_dims)
        self.gene2sys = HierarchicalTransformer(hidden_dims, 4, hidden_dims,
                                                self.gene2sys_update_norm_inner,
                                                self.gene2sys_update_norm_outer,
                                                dropout, norm_channel_first=self.norm_channel_first,
                                                conv_type='system',
                                                activation='softmax', poincare=poincare)
        self.cov_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.cov_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)

        self.cov2gene = FiLM(n_covariates, hidden_dims)
        self.cov2sys = FiLM(n_covariates, hidden_dims)
        self.cov2snp = FiLM(n_covariates, hidden_dims)
        '''
        self.cov2gene = HierarchicalTransformer(hidden_dims, 4, hidden_dims,
                                                self.cov_update_norm_inner,
                                                self.cov_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='genotype',
                                                             activation=None, n_type=1, poincare=poincare)
        self.cov2sys = HierarchicalTransformer(hidden_dims, 4, hidden_dims,
                                                self.cov_update_norm_inner,
                                                self.cov_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='genotype',
                                                             activation=None, n_type=1, poincare=poincare)
        '''

        self.n_covariates = n_covariates

        self.covariate_linear_1 = nn.Linear(self.n_covariates, hidden_dims)
        self.covariate_linear_2 = nn.Linear(hidden_dims, hidden_dims)
        self.covariate_norm_1 = nn.LayerNorm(hidden_dims)
        self.covariate_norm_2 = nn.LayerNorm(hidden_dims)


        self.n_phenotypes = n_phenotypes
        self.phenotype_embeddings = nn.Embedding(self.n_phenotypes, hidden_dims)

        self.geno2pheno_norm = nn.LayerNorm(hidden_dims)
        self.geno2pheno_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.geno2pheno_update_norm_outer = nn.LayerNorm(hidden_dims)
        self.geno2pheno = Genotype2Phenotype(hidden_dims, 1, hidden_dims,
                                            inner_norm=self.geno2pheno_update_norm_inner,
                                            outer_norm=self.geno2pheno_update_norm_outer, dropout=0.0,
                                            transform=True, activation='softmax', poincare=poincare)  # 'softmax')
        '''
        if sys2pheno:
            self.sys2pheno_norm = nn.LayerNorm(hidden_dims)
            self.sys2pheno_update_norm_inner = nn.LayerNorm(hidden_dims)
            self.sys2pheno_update_norm_outer = nn.LayerNorm(hidden_dims)
            self.sys2pheno = Genotype2Phenotype(hidden_dims, 1, hidden_dims,
                                                       inner_norm=self.sys2pheno_update_norm_inner,
                                                       outer_norm=self.sys2pheno_update_norm_outer, dropout=0.0,
                                                       transform=True, activation='softmax', poincare=poincare)#'softmax')
        else:
            self.sys2pheno = None

        if gene2pheno:
            self.gene2pheno_norm = nn.LayerNorm(hidden_dims, eps=0.1)
            self.gene2pheno_update_norm_inner = nn.LayerNorm(hidden_dims)
            self.gene2pheno_update_norm_outer = nn.LayerNorm(hidden_dims)
            self.gene2pheno = Genotype2Phenotype(hidden_dims, 1, hidden_dims,
                                                 inner_norm=self.gene2pheno_update_norm_inner,
                                                 outer_norm=self.gene2pheno_update_norm_outer, dropout=0.0,
                                                 transform=True, activation='softmax', poincare=poincare)
        else:
            self.gene2pheno = None
        '''


        self.last_activation = nn.Tanh()
        self.n_geno2pheno = sum([sys2pheno, gene2pheno])
        self.cov_effect = cov_effect
        print("Cov effect: ", self.cov_effect)
        print("Input format: ", self.input_format)
        self.predictor = MoEHeadPrediction(hidden_dims * self.n_geno2pheno, k_experts=8, top_k=2)




    def forward(self, genotype_dict, covariates, phenotype_ids, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                snp2gene_mask, gene2sys_mask, sys2gene_mask, sys2env=True, env2sys=True, sys2gene=True, score=False, attention=False, snp_only=False):
        batch_size = covariates.size(0)

        #covariate_embedding = self.get_covariate_embedding(covariates)
        if self.input_format == 'embedding':
            snp_embedding = genotype_dict['embedding'] + self.snp_embedding(genotype_dict['snp'])
            #snp_embedding = torch.einsum('blh,lhm->blm', snp_embedding, self.snp_adapters)
            block_embedding = self.block_embedding(genotype_dict['block'])
            snp_embedding = snp_embedding + block_embedding
        elif self.input_format == 'block':
            embedding_results = []
            snp_embedding = self.snp_embedding(genotype_dict['snp'])
            offset = 0
            for block_id, block_value in genotype_dict['block'].items():
                chromosome, block = block_id
                block_model_id = f'chr{chromosome}_block{block}'
                block_embedding_result = self.pretrained_transformer[block_model_id](block_value['snp'])
                #if self.freeze_pretrained:
                block_embedding_result = self.pretrained_transformer_adapter[block_model_id](block_embedding_result)
                snp_embedding_partial = snp_embedding[:, offset:offset+len(block_value['sig_ind']), :]
                block_embedding_result = self.pretrained_transformer_tuner[block_model_id](snp_embedding_partial, block_embedding_result, block_embedding_result)
                #print(block_embedding_result.size(), block_embedding_result[:, block_value['sig_ind'], :])
                embedding_results.append(block_embedding_result)#[:, block_value['sig_ind'], :])
                offset += len(block_value['sig_ind'])
            embedding_results = torch.cat(embedding_results, dim=1)
            #print(embedding_results.size())
            #print(embedding_results)
            snp_embedding = torch.nn.functional.pad(embedding_results, (0, 0, 0, self.n_snps2pad, 0, 0), value=0)

        else:
            snp_embedding = self.snp_embedding(genotype_dict['snp'])

        if self.cov_effect == 'snp':
            cov_effect_on_snp = self.get_cov2snp(snp_embedding, covariates)
            snp_embedding = snp_embedding + cov_effect_on_snp
        phenotype_embedding = self.phenotype_embeddings(phenotype_ids)

        if snp_only:
            return self.prediction_with_snp(phenotype_embedding, snp_embedding)

        #gene_embedding = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        gene_embedding = self.gene_embedding(genotype_dict['gene'])
        system_embedding = self.system_embedding(genotype_dict['sys'])
        #system_embedding = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        #gene_embedding = gene_embedding[:, :-1, :] + snp_effect_on_gene[:, :-1, :]
        #self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)[:, :-1, :]
        #print(snp_embedding, snp_embedding.size())
        if self.cov_effect == 'pre':
            cov_effect_on_gene = self.get_cov2gene(gene_embedding, covariates)
            gene_embedding = gene_embedding + self.effect_norm(cov_effect_on_gene)

        gene_embedding, snp_effect_on_gene = self.get_snp2gene(gene_embedding, snp_embedding, snp2gene_mask)
        gene_embedding = gene_embedding + self.effect_norm(snp_effect_on_gene)

        system_embedding, gene_effect_on_system = self.get_gene2sys(system_embedding, gene_embedding, gene2sys_mask)
        total_update = self.effect_norm(gene_effect_on_system)

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



        if self.cov_effect=='post':
            cov_effect_on_gene = self.get_cov2gene(gene_embedding, covariates)
            gene_embedding = gene_embedding + self.effect_norm(cov_effect_on_gene)
            cov_effect_on_sys = self.get_cov2gene(system_embedding, covariates)
            system_embedding = system_embedding + self.effect_norm(cov_effect_on_sys)


        prediction = self.prediction(phenotype_embedding, system_embedding, gene_embedding, genotype=None)#genotype_dict['embedding'])
        if attention:
            if score:
                system_embedding, system_attention, system_score = self.get_sys2pheno(phenotype_embedding,
                                                                                      system_embedding, attention=True,
                                                                                      score=True)
                gene_embedding, gene_attention, gene_score = self.get_gene2pheno(phenotype_embedding,
                                                                                      gene_embedding, attention=True,
                                                                                      score=True)
                return prediction, system_attention, gene_attention, system_score, gene_score
            else:
                system_embedding, system_attention = self.get_sys2pheno(phenotype_embedding, system_embedding,
                                                                        attention=True, score=False)
                gene_embedding, gene_attention = self.get_gene2pheno(phenotype_embedding, gene_embedding, attention=True,
                                                                     score=False)
                return prediction, system_attention, gene_attention
        else:
            if score:
                system_embedding, system_score = self.get_sys2pheno(phenotype_embedding, system_embedding,
                                                                        attention=False, score=True)
                gene_embedding, gene_score = self.get_gene2pheno(phenotype_embedding, gene_embedding, attention=False,
                                                                     score=True)
                return prediction, system_score, gene_score
            else:
                return prediction

    def get_snp2gene(self, gene_embedding, snp_embedding, snp2gene_mask):
        gene_embedding_input = self.gene_norm(gene_embedding)
        snp_embedding_input = self.snp_norm(snp_embedding)
        snp_effect_on_gene = self.snp2gene.forward(gene_embedding_input, snp_embedding_input, snp2gene_mask)
        return gene_embedding, snp_effect_on_gene
    '''
    def get_snp2gene(self, genotype, gene_embedding):
        snp_effect = torch.zeros_like(gene_embedding)
        for pheno, indice_dict in genotype.items():
            #print(indice_dict)
            snp_indices = indice_dict['snp']
            gene_indices = indice_dict['gene']

            if len(gene_indices) == 0:
                return None, None
            snp_embedding_input = self.snp_norm(self.snp_embedding(snp_indices))
            gene_embedding_input = self.gene_norm(self.gene_embedding(gene_indices))
            mask = indice_dict['mask']
            transformer = self.snp2gene[pheno]
            snp_effect_from_embedding = transformer(gene_embedding_input, snp_embedding_input, mask)
            snp_effect = snp_effect.scatter_add(1, gene_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dims), snp_effect_from_embedding)
        return gene_embedding, snp_effect
    
    
    def get_snp2gene(self, genotype, gene_embedding):
        if self.input_format=='indices':
            heterozygous_gene_indices, heterozygous_snp_effect_from_embedding = self.get_snp_effects(
                genotype['embedding']['heterozygous'], self.snp2gene_heterozygous)
            homozygous_gene_indices, homozygous_snp_effect_from_embedding = self.get_snp_effects(
                genotype['embedding']['homozygous_a1'], self.snp2gene_homozygous)
            batch_size = homozygous_gene_indices.size(0)
            snp_effect = torch.zeros_like(self.gene_embedding.weight).unsqueeze(0).expand(batch_size, -1, -1)

            snp_effect = snp_effect.scatter_add(1, heterozygous_gene_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dims), heterozygous_snp_effect_from_embedding)
            snp_effect = snp_effect.scatter_add(1, homozygous_gene_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dims), homozygous_snp_effect_from_embedding)

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

    def get_covariate_embedding(self, covariates):
        covariates_vector = self.covariate_norm_1(self.activation(self.covariate_linear_1(self.dropout(covariates))))
        #covariates_vector = self.covariate_norm_2(self.covariate_linear_2(self.dropout(covariates_vector)))
        covariates_vector = self.dropout(covariates_vector.unsqueeze(1))
        return covariates_vector

    def prediction(self, phenotype_vector, system_embedding, gene_embedding, genotype=None):
        phenotype_weighted_by_systems = self.get_geno2pheno(phenotype_vector, system_embedding, mask=None)
        phenotype_weighted_by_genes = self.get_geno2pheno(phenotype_vector, gene_embedding, mask=None)
        '''
        if self.sys2pheno is not None:
            
        if self.gene2pheno is not None:
            phenotype_weighted_by_genes = self.get_geno2pheno(phenotype_vector, gene_embedding, mask=None)
        if self.homo2pheno is not None:
            phenotype_weighted_by_hetero, phenotype_weighted_by_homo = self.get_snp2pheno(phenotype_vector, genotype)

        if (self.sys2pheno is not None) & (self.gene2pheno is not None):
            phenotype_feature = torch.cat([phenotype_weighted_by_systems, phenotype_weighted_by_genes], dim=-1)
        elif self.sys2pheno is not None:
            phenotype_feature = phenotype_weighted_by_systems
        elif self.gene2pheno is not None:
            phenotype_feature = phenotype_weighted_by_genes
        if (self.homo2pheno is not None):
            
        '''
        phenotype_feature = torch.cat([phenotype_weighted_by_systems, phenotype_weighted_by_genes], dim=-1)
        #phenotype_feature = self.pheno2pheno.forward(phenotype_feature, phenotype_feature, mask=None)
        #phenotype_prediction = self.phenotype_predictor_2(self.last_activation(self.phenotype_predictor_1(phenotype_feature)))

        phenotype_prediction = self.predictor(phenotype_feature)
        #print(phenotype_feature.size())
        return phenotype_prediction

    def prediction_with_snp(self, phenotype_embedding, snp_embedding):
        phenotype_weighted_by_snp = self.get_geno2pheno(phenotype_embedding, snp_embedding, mask=None)
        phenotype_feature = torch.cat([phenotype_weighted_by_snp]*self.n_geno2pheno, dim=-1)
        phenotype_prediction = self.predictor(phenotype_feature)
        return phenotype_prediction

    def get_geno2pheno(self, phenotype_embedding, genotype_embedding, mask=None, attention=False, score=False):
        genotype_embedding_input = self.geno2pheno_norm(genotype_embedding)
        sys2phenotype_result = self.geno2pheno.forward(phenotype_embedding, genotype_embedding_input, genotype_embedding_input,
                                                      mask=mask)
        if attention:
            sys2phenotype_attention = self.geno2pheno.get_attention(phenotype_embedding, genotype_embedding_input,
                                                                   genotype_embedding_input)
            sys2phenotype_result = [sys2phenotype_result, sys2phenotype_attention]
            if score:
                sys2phenotype_score = self.geno2pheno.get_score(phenotype_embedding, genotype_embedding_input,
                                                               genotype_embedding_input)
                sys2phenotype_result += [sys2phenotype_score]
            return sys2phenotype_result
        else:
            if score:
                sys2phenotype_score = self.geno2pheno.get_score(phenotype_embedding, genotype_embedding_input,
                                                               genotype_embedding_input)
                sys2phenotype_result = [sys2phenotype_result, sys2phenotype_score]
            return sys2phenotype_result

    '''
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
    '''

    def get_snp2pheno(self, phenotype_vector, genotype, attention=False, score=False):
        hetero_indices = genotype['heterozygous']['snp']

        homo_indices = genotype['homozygous_a1']['snp']
        hetero_snp_embedding = self.snp2pheno_norm(self.snp_embedding(hetero_indices))

        homo_snp_embedding = self.snp2pheno_norm(self.snp_embedding(homo_indices))
        hetero2phenotype_result = self.hetero2pheno.forward(phenotype_vector,
                                                            hetero_snp_embedding,
                                                            hetero_snp_embedding)
        homo2phenotype_result = self.homo2pheno.forward(phenotype_vector,
                                                          homo_snp_embedding,
                                                          homo_snp_embedding)
        return hetero2phenotype_result, homo2phenotype_result

    def get_cov2snp(self, snp_embedding, cov_embedding):
        snp_embedding = self.snp_norm(snp_embedding)
        cov_effect_on_snp = self.cov2snp(snp_embedding, cov_embedding)#, None)
        return cov_effect_on_snp

    def get_cov2gene(self, gene_embedding, cov_embedding):

        gene_embedding = self.gene_norm(gene_embedding)
        cov_effect_on_gene = self.cov2gene(gene_embedding, cov_embedding)
        return cov_effect_on_gene

    def get_cov2sys(self, sys_embedding, cov_embedding):

        sys_embedding = self.sys_norm(sys_embedding)
        cov_effect_on_sys = self.cov2gene(sys_embedding, cov_embedding)
        return cov_effect_on_sys

    #def set_temperature(self, temperature):
    #    self.gene2pheno.set_temperature(temperature)
    #    self.sys2pheno.set_temperature(temperature)
