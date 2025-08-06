import torch
import torch.nn as nn
import numpy as np


from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import Genotype2Phenotype
from src.model.hierarchical_transformer import HierarchicalTransformer
from src.model.utils import MoEHeadPrediction, FiLM, BatchNorm1d_BatchOnly_NLC
from src.model.LD_infuser.LDRoBERTa import RoBERTa, TransformerLayer, RoBERTaConfig, TransformerBlockGumbel
import torch.utils.checkpoint as cp

class SNP2PhenotypeModel(Genotype2PhenotypeTransformer):
    """
    A hierarchical transformer model to predict phenotypes from genotypes, guided by a biological ontology.

    This model translates SNP-level genetic information up through a biological hierarchy
    (SNPs -> Genes -> Biological Systems) to predict one or more phenotypes. It uses a series
    of transformer-based modules to propagate information and learn context-aware embeddings
    at each level of the hierarchy.

    The core workflow is as follows:
    1.  **Embedding:** SNPs, genes, systems, and phenotypes are embedded into a high-dimensional space.
    2.  **Propagation:** Information flows up the hierarchy. SNP effects are propagated to genes,
        gene effects are propagated to systems, and system-system interactions are resolved.
    3.  **Prediction:** The final embeddings for genes and/or systems are used to predict the
        phenotype, modulated by covariate information.

    Args:
        tree_parser (SNPTreeParser): An object that provides the hierarchical structure
            (SNP-gene-system mappings) and corresponding masks for the model.
        hidden_dims (int): The dimensionality of the embeddings and hidden layers.
        snp2pheno (bool, optional): Unused parameter for future extension. Defaults to False.
        gene2pheno (bool, optional): If True, use the final gene embeddings for phenotype prediction.
            Defaults to True.
        sys2pheno (bool, optional): If True, use the final system embeddings for phenotype prediction.
            Defaults to True.
        interaction_types (list, optional): The types of interactions to use for system-to-system
            propagation. Defaults to ['default'].
        n_covariates (int, optional): The number of covariate features to include in the model.
            Defaults to 13.
        n_phenotypes (int, optional): The number of distinct phenotypes the model can predict.
            Defaults to 1.
        dropout (float, optional): The dropout rate for regularization. Defaults to 0.2.
        activation (str, optional): The activation function for attention mechanisms.
            Defaults to 'softmax'.
        input_format (str, optional): The format of the genotype input ('indices' or 'block').
            Defaults to 'indices'.
        poincare (bool, optional): Unused parameter for future extension. Defaults to False.
        cov_effect (str, optional): Specifies how covariates affect the model ('pre', 'post',
            'direct', or 'both'). Defaults to 'pre'.
        pretrained_transformer (dict, optional): A dictionary of pretrained transformer models
            for block-based input. Defaults to None.
        freeze_pretrained (bool, optional): Unused parameter. Defaults to True.
        phenotypes (tuple, optional): Unused parameter. Defaults to ('PHENOTYPE',).
        use_hierarchical_transformer (bool, optional): If True, uses a hierarchical transformer
            for the final prediction heads. Defaults to False.
    """

    # --- Initialization ---
    def __init__(self, tree_parser, hidden_dims, snp2pheno=False, gene2pheno=True, sys2pheno=True,
                 interaction_types=['default'], n_covariates=13, n_phenotypes=1, dropout=0.2,
                 activation='softmax', input_format='indices', poincare=False, cov_effect='pre',
                 pretrained_transformer=None, freeze_pretrained=True,
                 phenotypes=('PHENOTYPE'), use_hierarchical_transformer=False):
        super(SNP2PhenotypeModel, self).__init__(tree_parser, hidden_dims, interaction_types=interaction_types, dropout=dropout,
                                                 input_format=input_format, poincare=poincare)
        self.use_gene2pheno = gene2pheno
        self.use_sys2pheno = sys2pheno
        self.n_snps = self.tree_parser.n_snps
        self.by_chr = self.tree_parser.by_chr
        self.gene_padding_ind = self.n_genes
        self.snp_padding_ind = self.n_snps * 3
        self.chromosomes = tree_parser.chromosomes
        snp_embedding_length = int(np.ceil(self.n_snps/8)*8)
        #self.snp_adapters = nn.Parameter(torch.randn(snp_embedding_length, self.hidden_dims, self.hidden_dims))
        self.snp_embedding = nn.Embedding(self.n_snps * 3 + 2, hidden_dims, padding_idx=self.n_snps * 3)
        self.gene_embedding = nn.Embedding(self.n_genes + 1, hidden_dims, padding_idx=self.n_genes)
        self.n_snp2pad = int(np.ceil((self.tree_parser.n_snps+1) / 8) * 8) - self.tree_parser.n_snps
        self.n_gene2pad = int(np.ceil((self.tree_parser.n_genes + 1) / 8) * 8) - self.tree_parser.n_genes
        self.n_sys2pad = int(np.ceil((self.tree_parser.n_systems + 1) / 8) * 8) - self.tree_parser.n_systems
        if self.input_format == 'block':
            self.n_blocks = self.tree_parser.n_blocks
            self.block_embedding = nn.Embedding(self.n_blocks + 1, hidden_dims, padding_idx=self.n_blocks)
        self.snp_batch_norm = BatchNorm1d_BatchOnly_NLC(hidden_dims, length=(self.n_snps + self.n_snp2pad))
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
        self.cov2sys = FiLM(n_covariates, hidden_dims)
        self.cov2snp = FiLM(n_covariates, hidden_dims)
        self.cov2pheno = FiLM(n_covariates, hidden_dims)
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

        self.gene2pheno_norm = nn.LayerNorm(hidden_dims)
        self.gene2pheno_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.gene2pheno_update_norm_outer = nn.LayerNorm(hidden_dims)
        self.gene2pheno = Genotype2Phenotype(hidden_dims, 4, hidden_dims,
                                             inner_norm=self.gene2pheno_update_norm_inner,
                                             outer_norm=self.gene2pheno_update_norm_outer, dropout=dropout,
                                             attention_dropout=0.0,
                                             transform=True, activation='softmax', poincare=poincare,
                                             use_hierarchical_transformer=use_hierarchical_transformer)

        self.sys2pheno_norm = nn.LayerNorm(hidden_dims)
        self.sys2pheno_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.sys2pheno_update_norm_outer = nn.LayerNorm(hidden_dims)
        self.sys2pheno = Genotype2Phenotype(hidden_dims, 4, hidden_dims,
                                            inner_norm=self.sys2pheno_update_norm_inner,
                                            outer_norm=self.sys2pheno_update_norm_outer, dropout=dropout,
                                            attention_dropout=0.0,
                                            transform=True, activation='softmax', poincare=poincare,
                                            use_hierarchical_transformer=use_hierarchical_transformer)

        self.last_activation = nn.Tanh()
        self.n_geno2pheno = sum([sys2pheno, gene2pheno])
        self.cov_effect = cov_effect
        print("Cov effect: ", self.cov_effect)
        print("Input format: ", self.input_format)
        print("The number of geno2pheno: ", self.n_geno2pheno)
        self.predictor = MoEHeadPrediction(hidden_dims * self.n_geno2pheno, k_experts=8, top_k=2)
        self.system_value_scale = nn.Parameter(torch.ones(self.n_systems + 1))
        self.gene_value_scale = nn.Parameter(torch.ones(self.n_genes + 1))
        #self.snp_predictor = nn.Linear(hidden_dims, self.n_snps * 3 + 2)
        self.block_sampling_prob = 0.1


    # --- Core Forward Pass ---
    def forward(self, genotype_dict, covariates, phenotype_ids, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                snp2gene_mask, gene2sys_mask, sys2gene_mask, sys_temp=None, sys2env=True, env2sys=True, sys2gene=True, score=False, attention=False, snp_only=False,
                predict_snp=False, chunk=False):
        """
        Defines the main forward pass of the model.

        Args:
            genotype_dict (dict): A dictionary containing genotype information (e.g., SNP indices).
            covariates (torch.Tensor): A tensor of covariate data for the batch.
            phenotype_ids (torch.Tensor): A tensor of phenotype IDs for the batch.
            nested_hierarchical_masks_forward (list): Masks for forward system-system propagation.
            nested_hierarchical_masks_backward (list): Masks for backward system-system propagation.
            snp2gene_mask (torch.Tensor): The attention mask for SNP-to-gene propagation.
            gene2sys_mask (torch.Tensor): The attention mask for gene-to-system propagation.
            sys2gene_mask (torch.Tensor): The attention mask for system-to-gene propagation.
            sys_temp (torch.Tensor, optional): A temperature mask for system attention. Defaults to None.
            score (bool, optional): If True, return attention scores. Defaults to False.
            attention (bool, optional): If True, return attention weights. Defaults to False.
            chunk (bool, optional): If True, use chunk-wise propagation. Defaults to False.

        Returns:
            torch.Tensor or tuple: The phenotype prediction tensor. If `attention` or `score` is True,
            returns a tuple containing the prediction and the requested attention/score tensors.
        """
        # 1. Propagate effects up the hierarchy from SNPs to Systems
        if not chunk:
            gene_embedding, system_embedding = self.propagate(genotype_dict, covariates, snp2gene_mask, gene2sys_mask, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward, sys2gene_mask)
        else:
            gene_embedding, system_embedding = self.chunk_wise_propagate(genotype_dict, covariates, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward)

        # 2. Get phenotype-specific embeddings
        phenotype_embedding = self.phenotype_embeddings(phenotype_ids)

        # 3. Predict phenotype from final embeddings
        prediction = self.prediction(phenotype_embedding, system_embedding, gene_embedding,
                                     genotype_dict, chunk,
                                     sys_temp=sys_temp, covariates=covariates)

        # 4. Optionally, return attention scores for interpretability
        if attention or score:
            return self._get_attention_and_scores(
                prediction, phenotype_embedding, system_embedding, gene_embedding,
                genotype_dict, chunk, sys_temp, attention, score
            )
        else:
            return prediction

    def _get_attention_and_scores(self, prediction, phenotype_embedding, system_embedding, gene_embedding,
                                  genotype_dict, chunk, sys_temp, attention, score):
        """Helper to compute and return attention and/or scores for interpretability."""
        system_embedding_value = system_embedding * self.system_value_scale[genotype_dict['sys']].unsqueeze(-1) if not chunk else system_embedding * self.system_value_scale.view(1, -1, 1)
        gene_embedding_value = gene_embedding * self.gene_value_scale[genotype_dict['gene']].unsqueeze(-1) if not chunk else gene_embedding * self.gene_value_scale.view(1, -1, 1)

        system_outputs = self.get_sys2pheno(phenotype_embedding, system_embedding, system_embedding_value, mask=sys_temp, attention=attention, score=score)
        gene_outputs = self.get_gene2pheno(phenotype_embedding, gene_embedding, gene_embedding_value, attention=attention, score=score)

        if attention and score:
            _, system_attention, system_score = system_outputs
            _, gene_attention, gene_score = gene_outputs
            return prediction, system_attention, gene_attention, system_score, gene_score
        elif attention:
            _, system_attention = system_outputs
            _, gene_attention = gene_outputs
            return prediction, system_attention, gene_attention
        elif score:
            _, system_score = system_outputs
            _, gene_score = gene_outputs
            return prediction, system_score, gene_score
        return prediction # Should not be reached

    # --- Propagation Mechanisms ---
    def propagate(self, genotype_dict, covariates, snp2gene_mask, gene2sys_mask, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward, sys2gene_mask):
        """
        Performs the main information propagation up the biological hierarchy for a single batch.
        """
        gene_embedding = self.gene_embedding(genotype_dict['gene'])
        system_embedding = self.system_embedding(genotype_dict['sys'])

        if (self.cov_effect == 'pre') or (self.cov_effect == 'both'):
            gene_embedding = self.get_cov2gene(gene_embedding, covariates)

        snp_embedding, _ = self.get_snp_embedding(genotype_dict)
        snp_effect_on_gene = self.get_snp2gene(gene_embedding, snp_embedding, snp2gene_mask)
        gene_embedding = gene_embedding + self.effect_norm(snp_effect_on_gene)

        gene_effect_on_system = self.get_gene2sys(self.dropout(system_embedding), self.dropout(gene_embedding), gene2sys_mask)

        batch_size = covariates.size(0)
        system_embedding_total = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
        system_embedding_total[:, genotype_dict['sys_indices']] += self.effect_norm(gene_effect_on_system)

        system_effect_forward = self.get_sys2sys(system_embedding_total, nested_hierarchical_masks_forward, direction='forward')
        system_embedding_total = system_embedding_total + self.effect_norm(system_effect_forward)

        system_effect_backward = self.get_sys2sys(system_embedding_total, nested_hierarchical_masks_backward, direction='backward')
        system_embedding_total = system_embedding_total + self.effect_norm(system_effect_backward)
        system_embedding = system_embedding_total[:, genotype_dict['sys_indices']]

        system_effect_on_gene = self.get_sys2gene(gene_embedding, system_embedding, sys2gene_mask)
        gene_embedding = gene_embedding + self.effect_norm(system_effect_on_gene)

        if (self.cov_effect=='post') or (self.cov_effect=='both'):
            gene_embedding = self.get_cov2gene(gene_embedding, covariates)
            system_embedding = self.get_cov2sys(system_embedding, covariates)

        return gene_embedding, system_embedding

    def chunk_wise_propagate(self, genotype_dict, covariates, masks_fwd, masks_bwd):
        """
        Performs information propagation in chunks to conserve memory, suitable for large datasets.
        """
        B, H = covariates.size(0), self.hidden_dims
        G, S = self.n_genes + 1, self.n_systems + 1

        gene_results = torch.zeros((B * G, H), dtype=torch.float32, device=covariates.device)
        sys_results = torch.zeros((B * S, H), dtype=torch.float32, device=covariates.device)

        for chunk in genotype_dict:
            snp_emb, _ = self.get_snp_embedding(chunk)
            gene_emb = self.gene_embedding(chunk['gene'])
            if self.cov_effect in ('pre', 'both'):
                gene_emb = self.get_cov2gene(gene_emb, covariates)

            gene_emb += self.effect_norm(self.get_snp2gene(gene_emb, snp_emb, chunk['snp2gene_mask']))

            flat_gene = gene_emb.view(-1, H)
            flat_gidx = (chunk['gene'] + torch.arange(B, device=gene_emb.device)[:, None] * G).view(-1)
            gene_results.index_add_(0, flat_gidx, flat_gene)

            sys_emb = self.system_embedding(chunk['sys'])
            sys_emb += self.effect_norm(self.get_gene2sys(sys_emb, gene_emb, chunk['gene2sys_mask']))

            flat_sys = sys_emb.view(-1, H)
            flat_sidx = (chunk['sys'] + torch.arange(B, device=sys_emb.device)[:, None] * S).view(-1)
            sys_results.index_add_(0, flat_sidx, flat_sys)

        sys_results = sys_results.view(B, S, H)
        sys_results = cp.checkpoint(self.get_sys2sys, sys_results, masks_fwd, 'forward')
        sys_results = cp.checkpoint(self.get_sys2sys, sys_results, masks_bwd, 'backward')

        gene_results = gene_results.view(B, G, H)
        for chunk in genotype_dict:
            gene_idx, sys_idx = chunk['gene'], chunk['sys']
            gene_emb = gene_results.gather(1, gene_idx.unsqueeze(-1).expand(-1, -1, H))
            sys_emb = sys_results.gather(1, sys_idx.unsqueeze(-1).expand(-1, -1, H))

            delta = self.get_sys2gene(gene_emb, sys_emb, chunk['gene2sys_mask'].T)
            flat_gidx = (gene_idx + torch.arange(B, device=gene_idx.device)[:, None] * G).view(-1)
            gene_results.view(-1, H).index_add_(0, flat_gidx, delta.view(-1, H))

        gene_embedding, system_embedding = gene_results.view(B, G, H), sys_results.view(B, S, H)

        if (self.cov_effect == 'post') or (self.cov_effect == 'both'):
            gene_embedding = self.get_cov2gene(gene_embedding, covariates)
            system_embedding = self.get_cov2sys(system_embedding, covariates)

        return gene_embedding, system_embedding

    # --- Embedding Layers ---
    def get_snp_embedding(self, genotype_dict):
        """Combines raw SNP embeddings with genomic block embeddings."""
        snp_embedding = self.snp_embedding(genotype_dict['snp'])
        if self.input_format == 'block':
            block_embedding = self.block_embedding(genotype_dict['block_ind'])
            embedding_results = (snp_embedding + block_embedding) / 2
        else:
            embedding_results = snp_embedding
        snp_prediction_results = {}
        if self.input_format == 'block':
            offset = 0
            for block_id in self.blocks:
                chromosome, block = block_id
                if block_id in genotype_dict['block'].keys():
                    block_value = genotype_dict['block'][block_id]
                    block_model_id = f'chr{chromosome}_block{block}'
                    snp_embedding_in_block_result = self.pretrained_transformer[block_model_id](block_value['snp'])
                    snp_embedding_in_block_result = self.pretrain_mapper[block_model_id](snp_embedding_in_block_result)
                    indices = torch.tensor(list(range(offset, offset + int(len(block_value['sig_ind'])))), dtype=torch.long, device=snp_embedding.device)
                    snp_embedding_partial = snp_embedding[:, indices, :]
                    block_embedding_partial = block_embedding[:, indices, :]
                    snp_embedding_in_block_result = self.pretrained_transformer_tuner[block_model_id](
                        snp_embedding_partial, snp_embedding_in_block_result, snp_embedding_in_block_result, residual=True)
                    embedding_results[:, indices, :] = (embedding_results[:, indices, :] + snp_embedding_in_block_result) * len(block_value['sig_ind'])
                    snp_embedding_in_block_marginalized = torch.mean(embedding_results[:, indices, :], dim=1)
                    snp_predicted = self.snp_predictor[block_model_id](snp_embedding_in_block_marginalized)
                    snp_prediction_results[block_id] = snp_predicted
                    offset += int(len(block_value['sig_ind']))
                else:
                    offset += int(len(self.tree_parser.block2sig_ind[block_id]))
        return embedding_results, snp_prediction_results

    def get_snp2gene(self, gene_embedding, snp_embedding, snp2gene_mask):
        """Propagates information from SNP embeddings to gene embeddings."""
        gene_embedding_input = self.dropout(self.gene_norm(gene_embedding))
        snp_embedding_input = self.dropout(snp_embedding)
        snp_effect_on_gene = self.snp2gene.forward(gene_embedding_input, snp_embedding_input, snp2gene_mask)
        return snp_effect_on_gene

    def get_gene2sys(self, system_embedding, gene_embedding, gene2sys_mask):
        """Propagates information from gene embeddings to system embeddings."""
        system_embedding_input = self.dropout(self.sys_norm(system_embedding))
        gene_embedding = self.dropout(self.gene_norm(gene_embedding))
        gene_effect = self.gene2sys.forward(system_embedding_input, gene_embedding, gene2sys_mask)
        return gene_effect

    def get_covariate_embedding(self, covariates):
        """Computes a single vector embedding from the covariate data."""
        covariates_vector = self.covariate_norm_1(self.activation(self.covariate_linear_1(self.dropout(covariates))))
        covariates_vector = self.dropout(covariates_vector.unsqueeze(1))
        return covariates_vector

    # --- Attention & Prediction Heads ---
    def prediction(self, phenotype_vector, system_embedding, gene_embedding, genotype_dict, chunk, sys_temp=None, covariates=None):
        """
        Generates the final phenotype prediction from the propagated embeddings.

        Args:
            phenotype_vector (torch.Tensor): The embedding for the target phenotype.
            system_embedding (torch.Tensor): The final system embeddings.
            gene_embedding (torch.Tensor): The final gene embeddings.
            genotype_dict (dict): Dictionary with genotype data for the batch.
            chunk (bool): Flag indicating if chunk-wise propagation was used.
            sys_temp (torch.Tensor, optional): Temperature mask for system attention. Defaults to None.
            covariates (torch.Tensor, optional): Covariate data. Defaults to None.

        Returns:
            torch.Tensor: The final prediction tensor.
        """
        if chunk:
            system_embedding_value = system_embedding
            gene_embedding_value = gene_embedding
        else:
            system_scales = self.system_value_scale[genotype_dict['sys']]
            system_embedding_value = system_embedding * system_scales.unsqueeze(-1)
            gene_scales = self.gene_value_scale[genotype_dict['gene']]
            gene_embedding_value = gene_embedding * gene_scales.unsqueeze(-1)

        phenotype_features = []
        if self.use_sys2pheno:
            phenotype_weighted_by_systems = self.get_sys2pheno(phenotype_vector, system_embedding, system_embedding_value, mask=sys_temp)
            if self.cov_effect == 'direct':
                phenotype_weighted_by_systems = self.cov2pheno(phenotype_weighted_by_systems, covariates)
            phenotype_features.append(phenotype_weighted_by_systems)

        if self.use_gene2pheno:
            phenotype_weighted_by_genes = self.get_gene2pheno(phenotype_vector, gene_embedding, gene_embedding_value, mask=None)
            if self.cov_effect == 'direct':
                phenotype_weighted_by_genes = self.cov2pheno(phenotype_weighted_by_genes, covariates)
            phenotype_features.append(phenotype_weighted_by_genes)

        phenotype_feature = torch.cat(phenotype_features, dim=-1)
        phenotype_prediction = self.predictor(phenotype_feature)
        return phenotype_prediction

    def prediction_with_snp(self, phenotype_embedding, snp_embedding):
        """An alternative prediction head that uses SNP embeddings directly (unused)."""
        phenotype_weighted_by_snp = self.get_gene2pheno(phenotype_embedding, snp_embedding, snp_embedding, mask=None)
        phenotype_feature = torch.cat([phenotype_weighted_by_snp]*self.n_geno2pheno, dim=-1)
        phenotype_prediction = self.predictor(phenotype_feature)
        return phenotype_prediction

    def get_gene2pheno(self, phenotype_embedding, genotype_embedding_key, genotype_embedding_value, mask=None, attention=False, score=False):
        """Performs attention from phenotype to genes to get a phenotype-specific gene representation."""
        genotype_embedding_key_input = self.dropout(self.gene2pheno_norm(genotype_embedding_key))
        genotype_embedding_value_input = self.dropout(self.gene2pheno_norm(genotype_embedding_value))
        if mask is not None and self.training:
            mask = (1-self.dropout.p) * mask
        sys2phenotype_result = self.gene2pheno.forward(self.dropout(phenotype_embedding), genotype_embedding_key_input, genotype_embedding_value_input,
                                                      mask=mask)
        if attention or score:
            outputs = [sys2phenotype_result]
            if attention:
                outputs.append(self.gene2pheno.get_attention(phenotype_embedding, genotype_embedding_key_input, genotype_embedding_value_input))
            if score:
                outputs.append(self.gene2pheno.get_score(phenotype_embedding, genotype_embedding_key_input, genotype_embedding_value_input))
            return outputs
        return sys2phenotype_result

    def get_sys2pheno(self, phenotype_embedding, genotype_embedding_key, genotype_embedding_value, mask=None, attention=False, score=False):
        """Performs attention from phenotype to systems to get a phenotype-specific system representation."""
        genotype_embedding_key_input = self.dropout(self.sys2pheno_norm(genotype_embedding_key))
        genotype_embedding_value_input = self.dropout(self.sys2pheno_norm(genotype_embedding_value))
        if mask is not None and self.training:
            mask = (1-self.dropout.p) * mask
        sys2phenotype_result = self.sys2pheno.forward(self.dropout(phenotype_embedding), genotype_embedding_key_input, genotype_embedding_value_input,
                                                      mask=mask)
        if attention or score:
            outputs = [sys2phenotype_result]
            if attention:
                outputs.append(self.sys2pheno.get_attention(phenotype_embedding, genotype_embedding_key_input, genotype_embedding_value_input))
            if score:
                outputs.append(self.sys2pheno.get_score(phenotype_embedding, genotype_embedding_key_input, genotype_embedding_value_input))
            return outputs
        return sys2phenotype_result

    # --- Covariate Effects ---
    def get_cov2snp(self, snp_embedding, cov_embedding):
        """Applies covariate effects to SNP embeddings using a FiLM layer."""
        snp_embedding = self.dropout(self.snp_norm(snp_embedding))
        cov_effect_on_snp = self.cov2snp(snp_embedding, cov_embedding)
        return cov_effect_on_snp

    def get_cov2gene(self, gene_embedding, cov_embedding):
        """Applies covariate effects to gene embeddings using a FiLM layer."""
        gene_embedding = self.dropout(self.gene_norm(gene_embedding))
        cov_effect_on_gene = self.cov2gene(gene_embedding, cov_embedding)
        return cov_effect_on_gene

    def get_cov2sys(self, sys_embedding, cov_embedding):
        """Applies covariate effects to system embeddings using a FiLM layer."""
        sys_embedding = self.dropout(self.sys_norm(sys_embedding))
        cov_effect_on_sys = self.cov2sys(sys_embedding, cov_embedding)
        return cov_effect_on_sys

    # --- Utility Methods ---
    def set_temperature(self, temperature):
        """Sets the temperature for Gumbel-Softmax sampling in block-based transformers."""
        for block_id, module in self.pretrained_transformer_tuner.items():
            module.set_temperature(temperature)