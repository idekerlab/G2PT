import torch
import torch.nn as nn
import numpy as np


from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import Genotype2Phenotype
from src.model.hierarchical_transformer import HierarchicalTransformer
from src.model.utils import MoEHeadPrediction, FiLM, BatchNorm1d_BatchOnly_NLC
from src.model.LD_infuser.LDRoBERTa import RoBERTa, TransformerLayer, RoBERTaConfig, TransformerBlockGumbel

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
        self.snp_embedding = nn.Embedding(self.n_snps * 3 + 2, hidden_dims, padding_idx=self.n_snps * 3)
        self.gene_embedding = nn.Embedding(self.n_genes + 1, hidden_dims, padding_idx=self.n_genes)
        self.n_snps2pad = int(np.ceil(self.tree_parser.n_snps / 8) * 8) - self.tree_parser.n_snps
        self.n_blocks = self.tree_parser.n_blocks
        self.block_embedding = nn.Embedding(self.n_blocks + 1, hidden_dims, padding_idx=self.n_blocks)
        self.snp_batch_norm = BatchNorm1d_BatchOnly_NLC(hidden_dims, length=(self.n_snps + self.n_snps2pad))
        self.gate_mlp = nn.Sequential(nn.Linear(self.hidden_dims * 3, self.hidden_dims), nn.ReLU(), nn.Linear(self.hidden_dims, 3))

        if self.input_format == 'block':
            self.blocks = tree_parser.blocks#pretrained_transformer.keys()
            self.pretrained_transformer = nn.ModuleDict(pretrained_transformer)
            #print(self.pretrained_transformer)
            config = RoBERTaConfig(hidden_size=self.hidden_dims, num_attention_heads=4,intermediate_size=self.hidden_dims, dropout=dropout)
            #self.pretrained_transformer_adapter = nn.ModuleDict(
            #    {block_id: BlockAdapter(self.hidden_dims, self.hidden_dims) for block_id in pretrained_transformer.keys()})
            #self.pretrained_transformer_tuner = nn.ModuleDict({block_id: nn.Linear(hidden_dims, hidden_dims) for block_id in pretrained_transformer.keys()})
            self.pretrained_transformer_tuner = nn.ModuleDict(
                {block_id: TransformerLayer(config=config, lora=False, temperature=1) for block_id in pretrained_transformer.keys()})
            self.gumbel_sampler = nn.ModuleDict(
                {block_id: TransformerLayer(self.hidden_dims, lora=False, temperature=1) # 4, mlp_ratio=1.0, dropout=0.1, temp=1.0)
                 for block_id in pretrained_transformer.keys()})
            #self.gate_mlp = nn.ModuleDict(
            #    {block_id: nn.Sequential(nn.Linear(self.hidden_dims * 3, self.hidden_dims), nn.ReLU(), nn.Linear(self.hidden_dims, 3))
            #     for block_id in pretrained_transformer.keys()})



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
        self.gene2snp = HierarchicalTransformer(hidden_dims, 4, hidden_dims,
                                                self.snp2gene_update_norm_inner,
                                                self.snp2gene_update_norm_outer,
                                                dropout, norm_channel_first=self.norm_channel_first, conv_type='system',
                                                activation=activation, n_type=1, poincare=poincare)

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
        #self.cov2sys = FiLM(n_covariates, hidden_dims)
        #self.cov2snp = FiLM(n_covariates, hidden_dims)
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

        self.last_activation = nn.Tanh()
        self.n_geno2pheno = sum([sys2pheno, gene2pheno])
        self.cov_effect = cov_effect
        print("Cov effect: ", self.cov_effect)
        print("Input format: ", self.input_format)
        self.predictor = MoEHeadPrediction(hidden_dims * self.n_geno2pheno, k_experts=8, top_k=2)
        #self.snp_predictor = nn.Linear(hidden_dims, self.n_snps * 3 + 2)
        self.block_sampling_prob = 0.1



    def forward(self, genotype_dict, covariates, phenotype_ids, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                snp2gene_mask, gene2sys_mask, sys2gene_mask, sys_temp=None, sys2env=True, env2sys=True, sys2gene=True, score=False, attention=False, snp_only=False,
                predict_snp=False, chunk=False):

        '''
        #covariate_embedding = self.get_covariate_embedding(covariates)


        #snp_embedding = self.snp_batch_norm(snp_embedding)

        if self.cov_effect == 'snp':
            cov_effect_on_snp = self.get_cov2snp(snp_embedding, covariates)
            snp_embedding = snp_embedding + cov_effect_on_snp
        

        if snp_only:
            return self.prediction_with_snp(phenotype_embedding, snp_embedding)
        '''


        #for i in range(2):
        if not chunk:
            gene_embedding = self.gene_embedding(genotype_dict['gene'])
            system_embedding = self.system_embedding(genotype_dict['sys'])

            if (self.cov_effect == 'pre') or (self.cov_effect == 'both'):
                cov_effect_on_gene = self.get_cov2gene(gene_embedding, covariates)
                gene_embedding = gene_embedding + self.effect_norm(cov_effect_on_gene)

            snp_embedding, snp_prediction = self.get_snp_embedding(genotype_dict)
            gene_embedding, snp_effect_on_gene = self.get_snp2gene(gene_embedding, snp_embedding, snp2gene_mask)
            gene_embedding = gene_embedding + self.effect_norm(snp_effect_on_gene)
            gene_embedding, system_embedding = self.propagate(gene_embedding, system_embedding, gene2sys_mask, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward, sys2gene_mask)
        else:
            ## chunk-wise propagation
            gene_embedding, system_embedding = self.chunk_wise_propagate(genotype_dict, covariates, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward)

        if (self.cov_effect=='post') or (self.cov_effect=='both'):
            cov_effect_on_gene = self.get_cov2gene(gene_embedding, covariates)
            gene_embedding = gene_embedding + self.effect_norm(cov_effect_on_gene)
            cov_effect_on_sys = self.get_cov2gene(system_embedding, covariates)
            system_embedding = system_embedding + self.effect_norm(cov_effect_on_sys)

        phenotype_embedding = self.phenotype_embeddings(phenotype_ids)
        prediction = self.prediction(phenotype_embedding, system_embedding, gene_embedding, sys_temp=sys_temp)#genotype_dict['embedding'])


        #if predict_snp:
        #    return prediction, snp_prediction
        if attention:
            if score:
                system_embedding, system_attention, system_score = self.get_geno2pheno(phenotype_embedding,
                                                                                      system_embedding, mask=sys_temp, attention=True,
                                                                                      score=True)
                gene_embedding, gene_attention, gene_score = self.get_geno2pheno(phenotype_embedding,
                                                                                      gene_embedding, attention=True,
                                                                                      score=True)
                return prediction, system_attention, gene_attention, system_score, gene_score
            else:
                system_embedding, system_attention = self.get_geno2pheno(phenotype_embedding, system_embedding, mask=sys_temp,
                                                                        attention=True, score=False)
                gene_embedding, gene_attention = self.get_geno2pheno(phenotype_embedding, gene_embedding, attention=True,
                                                                     score=False)
                return prediction, system_attention, gene_attention
        else:
            if score:
                system_embedding, system_score = self.get_geno2pheno(phenotype_embedding, system_embedding, mask=sys_temp,
                                                                        attention=False, score=True)
                gene_embedding, gene_score = self.get_geno2pheno(phenotype_embedding, gene_embedding, attention=False,
                                                                     score=True)
                return prediction, system_score, gene_score
            else:
                return prediction



    def propagate(self, gene_embedding, system_embedding, gene2sys_mask, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward, sys2gene_mask):
        system_embedding, gene_effect_on_system = self.get_gene2sys(self.dropout(system_embedding),
                                                                    self.dropout(gene_embedding), gene2sys_mask)
        total_update = self.effect_norm(gene_effect_on_system)

        system_embedding, system_effect_forward = self.get_sys2sys(system_embedding,
                                                                   nested_hierarchical_masks_forward,
                                                                   direction='forward', return_updates=True,
                                                                   input_format=self.input_format,
                                                                   update_tensor=total_update)
        total_update = total_update + system_effect_forward

        system_embedding, system_effect_backward = self.get_sys2sys(system_embedding,
                                                                    nested_hierarchical_masks_backward,
                                                                    direction='backward', return_updates=True,
                                                                    input_format=self.input_format,
                                                                    update_tensor=total_update)
        total_update = total_update + system_effect_backward
        system_embedding = system_embedding + total_update

        gene_embedding, system_effect_on_gene = self.get_sys2gene(gene_embedding, system_embedding, sys2gene_mask)
        gene_embedding = gene_embedding + self.effect_norm(system_effect_on_gene)
        return gene_embedding, system_embedding

    def chunk_wise_propagate(self, genotype_dict, covariates, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward):
        batch_size = covariates.size(0)
        gene_embedding_results = torch.zeros_like(self.gene_embedding.weight)
        gene_embedding_results = gene_embedding_results.unsqueeze(0).expand(batch_size, -1, -1)

        sys_embedding_results = self.system_embedding.weight
        sys_embedding_results = sys_embedding_results.unsqueeze(0).expand(batch_size, -1, -1)

        sys_effect_results = torch.zeros_like(self.system_embedding.weight)#.unsqueeze(0).expand(batch_size, -1, -1)
        sys_effect_results = sys_effect_results.unsqueeze(0).expand(batch_size, -1, -1)

        for chunk_dict in genotype_dict:
            snp_embedding, snp_prediction = self.get_snp_embedding(chunk_dict)
            gene_embedding = self.gene_embedding(chunk_dict['gene'])
            if (self.cov_effect == 'pre') or (self.cov_effect == 'both'):
                cov_effect_on_gene = self.get_cov2gene(gene_embedding, covariates)
                gene_embedding = gene_embedding + self.effect_norm(cov_effect_on_gene)
            gene_embedding, snp_effect_on_gene = self.get_snp2gene(gene_embedding, snp_embedding, chunk_dict['snp2gene_mask'])
            gene_embedding = gene_embedding + self.effect_norm(snp_effect_on_gene)
            B, N, H = gene_embedding.size()
            idx_exp = chunk_dict['gene'].view(B, N, 1).expand(B, N, H)
            #print(idx_exp.size())
            #print(snp_effect_on_gene.size())
            #print(gene_embedding.size())
            #gene_embedding_results[:, chunk_dict['gene']] = gene_embedding#gene_embedding_results[:, chunk_dict['gene']] + self.effect_norm(snp_effect_on_gene)
            gene_embedding_results = gene_embedding_results.scatter_add(dim=1, index=idx_exp, src=gene_embedding)

            system_embedding = self.system_embedding(chunk_dict['sys'])
            system_embedding, gene_effect_on_system = self.get_gene2sys(system_embedding,
                                                                        gene_embedding, chunk_dict['gene2sys_mask'])
            B, N, H = system_embedding.size()
            idx_exp = chunk_dict['sys'].view(B, N, 1).expand(B, N, H)
            #sys_embedding_results = sys_embedding_results.scatter_add(dim=1, index=idx_exp, src=system_embedding)
            #sys_effect_results[:, chunk_dict['sys']] = sys_effect_results[:, chunk_dict['sys']] + self.effect_norm(gene_effect_on_system)
            sys_effect_results = sys_effect_results.scatter_add(dim=1, index=idx_exp, src=gene_effect_on_system)

        total_update = self.effect_norm(sys_effect_results)
        sys_embedding_results, system_effect_forward = self.get_sys2sys(sys_embedding_results,
                                                                   nested_hierarchical_masks_forward,
                                                                   direction='forward', return_updates=True,
                                                                   input_format=self.input_format,
                                                                   update_tensor=total_update)
        total_update = total_update + system_effect_forward

        sys_embedding_results, system_effect_backward = self.get_sys2sys(sys_embedding_results,
                                                                    nested_hierarchical_masks_backward,
                                                                    direction='backward', return_updates=True,
                                                                    input_format=self.input_format,
                                                                    update_tensor=total_update)
        total_update = total_update + system_effect_backward
        sys_embedding_results = sys_embedding_results + total_update
        H = self.hidden_dims
        for chunk_dict in genotype_dict:
            gene_idx = chunk_dict['gene']
            B, N = gene_idx.size()
            gene_idx_exp = gene_idx.unsqueeze(-1).expand(B, N, H)
            sys_idx = chunk_dict['sys']
            B, N = sys_idx.size()
            sys_idx_exp = sys_idx.unsqueeze(-1).expand(B, N, H)
            gene_embedding = gene_embedding_results.gather(dim=1, index=gene_idx_exp)
            system_embedding = sys_embedding_results.gather(dim=1, index=sys_idx_exp)
            #gene_embedding = gene_embedding_results[:, gene_idx]#torch.index_select(gene_embedding_results, 1, gene_idx.flatten()).view(B, N, H)
            #system_embedding = sys_embedding_results[:, sys_idx]
            gene_embedding, system_effect_on_gene = self.get_sys2gene(gene_embedding, system_embedding, chunk_dict['gene2sys_mask'].T)
            #idx_exp = chunk_dict['gene'].view(B, N, 1).expand(B, N, H)
            #gene_embedding_results[:, chunk_dict['gene']] = gene_embedding_results[:, chunk_dict['gene']] + self.effect_norm(system_effect_on_gene)
            gene_embedding_results = gene_embedding_results.scatter_add(dim=1, index=gene_idx_exp, src=system_effect_on_gene)

        return gene_embedding_results, sys_embedding_results

    def get_snp_embedding(self, genotype_dict):

        snp_embedding = self.snp_embedding(genotype_dict['snp'])
        block_embedding = self.block_embedding(genotype_dict['block_ind'])


        #block_specific_results = torch.zeros_like(snp_embedding)
        '''
        embedding_concatenated = torch.cat([snp_embedding, block_embedding, block_specific_results], dim=-1)
        logits = self.gate_mlp(embedding_concatenated)
        gates = torch.nn.functional.softmax(logits, dim=-1)
        embedding_results = gates[..., 0:1] * snp_embedding \
                            + gates[..., 1:2] * block_embedding \
                            + gates[..., 2:3] * block_specific_results
        '''
        embedding_results = (snp_embedding + block_embedding) / 2 # change to embedding_results
        snp_prediction_results = {}
        if self.input_format == 'block':
            offset = 0
            for block_id in self.blocks:
                chromosome, block = block_id
                if block_id in genotype_dict['block'].keys():
                    block_value = genotype_dict['block'][block_id]
                    block_model_id = f'chr{chromosome}_block{block}'
                    #print(block_id, block_value['sig_ind'])
                    snp_embedding_in_block_result = self.pretrained_transformer[block_model_id](block_value['snp'])
                    snp_embedding_in_block_result = self.pretrain_mapper[block_model_id](snp_embedding_in_block_result)
                    indices = torch.tensor(list(range(offset, offset + int(len(block_value['sig_ind'])))), dtype=torch.long, device=snp_embedding.device)
                    snp_embedding_partial = snp_embedding[:, indices, :]
                    block_embedding_partial = block_embedding[:, indices, :]
                    snp_embedding_in_block_result = self.pretrained_transformer_tuner[block_model_id](
                        snp_embedding_partial, snp_embedding_in_block_result, snp_embedding_in_block_result, residual=True)
                    #snp_embedding_in_block_result = snp_embedding_partial
                    #snp_embedding_in_block_result = self.gumbel_sampler[block_model_id](snp_embedding_in_block_result, snp_embedding_in_block_result, snp_embedding_in_block_result)
                    '''
                    embedding_concatenated = torch.cat([snp_embedding_partial, block_embedding_partial, snp_embedding_in_block_result], dim=-1)
                    logits = self.gate_mlp_block[block_model_id](embedding_concatenated)
                    gates = torch.nn.functional.softmax(logits, dim=-1)
                    snp_embedding_in_block_result = gates[..., 0:1] * snp_embedding_partial \
                            + gates[..., 1:2] * block_embedding_partial \
                            + gates[..., 2:3] * snp_embedding_in_block_result
                    '''
                    embedding_results[:, indices, :] = (embedding_results[:, indices, :] + snp_embedding_in_block_result) * len(block_value['sig_ind'])
                    snp_embedding_in_block_marginalized = torch.mean(embedding_results[:, indices, :], dim=1)
                    snp_predicted = self.snp_predictor[block_model_id](snp_embedding_in_block_marginalized)
                    snp_prediction_results[block_id] = snp_predicted
                    offset += int(len(block_value['sig_ind']))
                else:
                    offset += int(len(self.tree_parser.block2sig_ind[block_id]))
        '''
        embedding_concatenated = torch.cat([snp_embedding, block_embedding, block_specific_results], dim=-1) # remove this
        logits = self.gate_mlp(embedding_concatenated)
        gates = torch.nn.functional.softmax(logits, dim=-1)
        embedding_results = gates[..., 0:1] * snp_embedding \
                            + gates[..., 1:2] * block_embedding \
                            + gates[..., 2:3] * block_specific_results
        '''
        return embedding_results, snp_prediction_results

    def get_snp2gene(self, gene_embedding, snp_embedding, snp2gene_mask):
        gene_embedding_input = self.dropout(self.gene_norm(gene_embedding))
        snp_embedding_input = self.dropout(snp_embedding)
        snp_effect_on_gene = self.snp2gene.forward(gene_embedding_input, snp_embedding_input, snp2gene_mask)
        return gene_embedding, snp_effect_on_gene

    def get_gene2sys(self, system_embedding, gene_embedding, gene2sys_mask):
        system_embedding_input = self.dropout(self.sys_norm(system_embedding))
        gene_embedding = self.dropout(self.gene_norm(gene_embedding))
        gene_effect = self.gene2sys.forward(system_embedding_input, gene_embedding, gene2sys_mask)
        return system_embedding, gene_effect

    def get_covariate_embedding(self, covariates):
        covariates_vector = self.covariate_norm_1(self.activation(self.covariate_linear_1(self.dropout(covariates))))
        #covariates_vector = self.covariate_norm_2(self.covariate_linear_2(self.dropout(covariates_vector)))
        covariates_vector = self.dropout(covariates_vector.unsqueeze(1))
        return covariates_vector

    def prediction(self, phenotype_vector, system_embedding, gene_embedding, sys_temp=None):
        phenotype_weighted_by_systems = self.get_geno2pheno(phenotype_vector, system_embedding, mask=sys_temp)
        phenotype_weighted_by_genes = self.get_geno2pheno(phenotype_vector, gene_embedding, mask=None)
        phenotype_feature = torch.cat([phenotype_weighted_by_systems, phenotype_weighted_by_genes], dim=-1)

        phenotype_prediction = self.predictor(phenotype_feature)
        #print(phenotype_feature.size())
        return phenotype_prediction

    def prediction_with_snp(self, phenotype_embedding, snp_embedding):
        phenotype_weighted_by_snp = self.get_geno2pheno(phenotype_embedding, snp_embedding, mask=None)
        phenotype_feature = torch.cat([phenotype_weighted_by_snp]*self.n_geno2pheno, dim=-1)
        phenotype_prediction = self.predictor(phenotype_feature)
        return phenotype_prediction

    def get_geno2pheno(self, phenotype_embedding, genotype_embedding, mask=None, attention=False, score=False):
        genotype_embedding_input = self.dropout(self.geno2pheno_norm(genotype_embedding))
        sys2phenotype_result = self.geno2pheno.forward(self.dropout(phenotype_embedding), genotype_embedding_input, genotype_embedding_input,
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




    def get_cov2snp(self, snp_embedding, cov_embedding):
        snp_embedding = self.dropout(self.snp_norm(snp_embedding))
        cov_effect_on_snp = self.cov2snp(snp_embedding, cov_embedding)#, None)
        return cov_effect_on_snp

    def get_cov2gene(self, gene_embedding, cov_embedding):
        gene_embedding = self.dropout(self.gene_norm(gene_embedding))
        cov_effect_on_gene = self.cov2gene(gene_embedding, cov_embedding)
        return cov_effect_on_gene

    def get_cov2sys(self, sys_embedding, cov_embedding):
        sys_embedding = self.dropout(self.sys_norm(sys_embedding))
        cov_effect_on_sys = self.cov2gene(sys_embedding, cov_embedding)
        return cov_effect_on_sys

    def set_temperature(self, temperature):
        #self.gene2pheno.set_temperature(temperature)
        #self.sys2pheno.set_temperature(temperature)
        for block_id, module in self.pretrained_transformer_tuner.items():
            module.set_temperature(temperature)
