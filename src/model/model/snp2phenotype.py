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
        dropout (float, optional): The dropout rate for regularization. Defaults to 0.2.
        activation (str, optional): The activation function for attention mechanisms.
            Defaults to 'softmax'.
        input_format (str, optional): The format of the genotype input ('indices' or 'block').
            Defaults to 'indices'.
        cov_effect (str, optional): Specifies how covariates affect the model ('pre', 'post',
            'direct', or 'both'). Defaults to 'pre'.
        pretrained_transformer (dict, optional): A dictionary of pretrained transformer models
            for block-based input. Defaults to None.
        freeze_pretrained (bool, optional): Unused parameter. Defaults to True.
        phenotypes (list, optional): A list of phenotype names for which the model will create heads.
        ind2pheno (dict, optional): A mapping from phenotype index to phenotype name.
        use_hierarchical_transformer (bool, optional): If True, uses a hierarchical transformer
            for the final prediction heads. Defaults to False.
        use_moe (bool, optional): If True, uses a Mixture-of-Experts layer for the predictor.
            Otherwise, uses a simple MLP. Defaults to True.
        use_independent_predictors (bool, optional): If True, creates a separate prediction head
            for each phenotype. Defaults to False.
    """

    # --- Initialization ---
    def __init__(self, tree_parser, hidden_dims, snp2pheno=False, gene2pheno=True, sys2pheno=True,
                 interaction_types=['default'], n_covariates=13, dropout=0.2,
                 activation='softmax', input_format='indices', cov_effect='pre',
                 pretrained_transformer=None, freeze_pretrained=True,
                 phenotypes=('PHENOTYPE',), ind2pheno=None, use_hierarchical_transformer=False, use_moe=False,
                 use_independent_predictors=False, prediction_head=1, n_heads=4,
                 use_sparse_attention=True,  # NEW: enable sparse attention
                 use_snp2gene_gating=False,
                 use_snp2gene_null_gate=False,
                 snp2gene_attention_dropout=None,
                 snp2gene_null_logit_init=-10.0,
                 use_snp2gene_source_prior=False,
                 snp2gene_source_prior_init=0.0,
                 use_snp_shallow_predictor=False,
                 snp_shallow_feed_g2pt=False,
                 snp_shallow_layers=1,
                 snp_shallow_alpha_init=1.0,
                 g2pt_residual_alpha_init=1.0,
                 prediction_activation='tanh',
                 use_block_embedding=None):
        super(SNP2PhenotypeModel, self).__init__(tree_parser, hidden_dims, interaction_types=interaction_types, dropout=dropout,
                                                 n_heads=4, input_format=input_format)
        self.use_sparse_attention = use_sparse_attention  # NEW: store flag
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
        if use_block_embedding is None:
            self.use_block_embedding = bool(getattr(self.tree_parser, "block", False))
        else:
            self.use_block_embedding = bool(use_block_embedding)
        if self.use_block_embedding:
            self.n_blocks = self.tree_parser.n_blocks
            self.block_embedding = nn.Embedding(self.n_blocks + 1, hidden_dims, padding_idx=self.n_blocks)
        else:
            self.n_blocks = None
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
        self.snp2gene_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.snp2gene_update_norm_outer = nn.LayerNorm(hidden_dims)
        self.snp2gene = HierarchicalTransformer(hidden_dims, n_heads, hidden_dims,
                                                self.snp2gene_update_norm_inner,
                                                self.snp2gene_update_norm_outer,
                                               dropout, norm_channel_first=self.norm_channel_first, conv_type='system',
                                                             activation=activation, n_type=1)
        self.snp2gene_attention_dropout = snp2gene_attention_dropout
        if self.snp2gene_attention_dropout is not None:
            self.snp2gene.hierarchical_transformer_update.attention_dropout = float(self.snp2gene_attention_dropout)
        self.use_snp2gene_gating = use_snp2gene_gating
        self.use_snp2gene_null_gate = use_snp2gene_gating and use_snp2gene_null_gate
        self.snp2gene_null_logit_init = float(snp2gene_null_logit_init)
        self.prediction_activation = prediction_activation
        self.use_snp_shallow_predictor = bool(use_snp_shallow_predictor)
        self.snp_shallow_feed_g2pt = bool(use_snp_shallow_predictor and snp_shallow_feed_g2pt)
        self.use_snp2gene_source_prior = bool(use_snp2gene_source_prior)
        self.snp2gene_source_names = list(getattr(tree_parser, "snp2gene_source_names", []))
        source_matrix = getattr(tree_parser, "snp2gene_edge_source_matrix", None)
        if self.use_snp2gene_source_prior and source_matrix is not None and self.snp2gene_source_names:
            self.register_buffer("snp2gene_edge_source_matrix", source_matrix.float(), persistent=False)
            self.snp2gene_source_bias = nn.Parameter(
                torch.full(
                    (len(self.snp2gene_source_names),),
                    float(snp2gene_source_prior_init),
                    dtype=torch.float32,
                )
            )
            print("Using SNP2gene source prior:", ", ".join(self.snp2gene_source_names))
        else:
            self.use_snp2gene_source_prior = False
        if self.use_snp2gene_gating:
            self.snp2gene_gate = nn.Sequential(
                nn.Linear(hidden_dims * 2, hidden_dims),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims, 1),
            )
            nn.init.zeros_(self.snp2gene_gate[-1].weight)
            nn.init.constant_(self.snp2gene_gate[-1].bias, 4.0)
            if self.use_snp2gene_null_gate:
                self.snp2gene_null_logit = nn.Parameter(torch.tensor([self.snp2gene_null_logit_init]))
        self.gene2snp = HierarchicalTransformer(hidden_dims, n_heads, hidden_dims,
                                                self.snp2gene_update_norm_inner,
                                                self.snp2gene_update_norm_outer,
                                                dropout, norm_channel_first=self.norm_channel_first, conv_type='system',
                                                activation=activation, n_type=1)

        self.gene2sys_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.gene2sys_update_norm_outer = nn.LayerNorm(hidden_dims)
        self.gene2sys = HierarchicalTransformer(hidden_dims, n_heads, hidden_dims,
                                                self.gene2sys_update_norm_inner,
                                                self.gene2sys_update_norm_outer,
                                                dropout, norm_channel_first=self.norm_channel_first,
                                                conv_type='system',
                                                activation='softmax')

        # NEW: Enable sparse mode for transformers if using sparse attention
        if self.use_sparse_attention:
            self.snp2gene.hierarchical_transformer_update.use_sparse = True
            self.gene2sys.hierarchical_transformer_update.use_sparse = True
            self.gene2snp.hierarchical_transformer_update.use_sparse = True
            self.sys2gene.hierarchical_transformer_update.use_sparse = True

            # Enable sparse mode for hierarchical sys2sys transformers
            for inter_type in self.interaction_types:
                if inter_type in self.sys2env:
                    self.sys2env[inter_type].hierarchical_transformer_update.use_sparse = True
                if inter_type in self.env2sys:
                    self.env2sys[inter_type].hierarchical_transformer_update.use_sparse = True

            # Store references to precomputed edge indices if available
            if hasattr(tree_parser, 'snp2gene_edge_index'):
                self.snp2gene_edge_index = tree_parser.snp2gene_edge_index
                self.snp2gene_edge_attr = tree_parser.snp2gene_edge_attr

                # Compute gene2snp edge index (transpose of snp2gene)
                # snp2gene_edge_index = [gene_idx (query), snp_idx (key)]
                # gene2snp reverses: query=snps, key=genes
                # gene2snp_edge_index = [snp_idx (query), gene_idx (key)]
                self.gene2snp_edge_index = torch.stack([
                    self.snp2gene_edge_index[1],  # snp indices → queries  (row 1 of snp2gene = snp keys)
                    self.snp2gene_edge_index[0]   # gene indices → keys    (row 0 of snp2gene = gene queries)
                ], dim=0)
                self.gene2snp_edge_attr = self.snp2gene_edge_attr
            else:
                print("WARNING: tree_parser missing snp2gene_edge_index - sparse attention will be disabled")
                self.use_sparse_attention = False

            if hasattr(tree_parser, 'gene2sys_edge_index'):
                self.gene2sys_edge_index = tree_parser.gene2sys_edge_index
                self.gene2sys_edge_attr = tree_parser.gene2sys_edge_attr

                # Compute sys2gene edge index (transpose of gene2sys)
                # gene2sys_edge_index = [system_idx (query), gene_idx (key)]
                # sys2gene reverses: query=genes, key=systems
                # sys2gene_edge_index = [gene_idx (query), system_idx (key)]
                self.sys2gene_edge_index = torch.stack([
                    self.gene2sys_edge_index[1],  # gene indices → queries  (row 1 of gene2sys = gene keys)
                    self.gene2sys_edge_index[0]   # system indices → keys   (row 0 of gene2sys = system queries)
                ], dim=0)
                self.sys2gene_edge_attr = self.gene2sys_edge_attr
            else:
                print("WARNING: tree_parser missing gene2sys_edge_index")

            # Store hierarchical edges for sys2sys transformers
            if hasattr(tree_parser, 'hierarchical_edges_forward'):
                self.hierarchical_edges_forward = tree_parser.hierarchical_edges_forward
                self.hierarchical_edges_backward = tree_parser.hierarchical_edges_backward
                print("✓ Hierarchical edges loaded for sys2sys sparse attention")
            else:
                print("WARNING: tree_parser missing hierarchical_edges - sys2sys will use dense attention")

        self.cov_update_norm_inner = nn.LayerNorm(hidden_dims, eps=0.1)
        self.cov_update_norm_outer = nn.LayerNorm(hidden_dims, eps=0.1)

        self.cov2gene = FiLM(n_covariates, hidden_dims)
        self.cov2sys = FiLM(n_covariates, hidden_dims)
        self.cov2snp = FiLM(n_covariates, hidden_dims)
        self.cov2pheno = FiLM(n_covariates, hidden_dims)

        self.n_covariates = n_covariates

        self.covariate_linear_1 = nn.Linear(self.n_covariates, hidden_dims)
        self.covariate_linear_2 = nn.Linear(hidden_dims, hidden_dims)
        self.covariate_norm_1 = nn.LayerNorm(hidden_dims)
        self.covariate_norm_2 = nn.LayerNorm(hidden_dims)


        self.phenotypes = phenotypes
        self.ind2pheno = ind2pheno
        self.n_phenotypes = len(phenotypes)
        self.phenotype_embeddings = nn.Embedding(self.n_phenotypes, hidden_dims)

        if self.use_snp_shallow_predictor:
            self.snp_shallow_input_norm = nn.LayerNorm(hidden_dims)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dims,
                nhead=n_heads,
                dim_feedforward=hidden_dims * 2,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.snp_shallow_transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=int(snp_shallow_layers),
            )
            self.snp_shallow_output_norm = nn.LayerNorm(hidden_dims)
            self.snp_shallow_head = nn.Sequential(
                nn.Linear(hidden_dims, hidden_dims),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims, self.n_phenotypes),
            )
            self.snp_shallow_alpha = nn.Parameter(torch.tensor(float(snp_shallow_alpha_init)))
            self.g2pt_residual_alpha = nn.Parameter(torch.tensor(float(g2pt_residual_alpha_init)))

        self.gene2pheno_norm = nn.LayerNorm(hidden_dims)
        self.gene2pheno_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.gene2pheno_update_norm_outer = nn.LayerNorm(hidden_dims)
        self.gene2pheno = Genotype2Phenotype(hidden_dims, prediction_head, hidden_dims,
                                             inner_norm=self.gene2pheno_update_norm_inner,
                                             outer_norm=self.gene2pheno_update_norm_outer, dropout=dropout,
                                             attention_dropout=0.0,
                                             transform=True, activation='softmax',
                                             use_hierarchical_transformer=use_hierarchical_transformer)

        self.sys2pheno_norm = nn.LayerNorm(hidden_dims)
        self.sys2pheno_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.sys2pheno_update_norm_outer = nn.LayerNorm(hidden_dims)
        self.sys2pheno = Genotype2Phenotype(hidden_dims, prediction_head, hidden_dims,
                                            inner_norm=self.sys2pheno_update_norm_inner,
                                            outer_norm=self.sys2pheno_update_norm_outer, dropout=dropout,
                                            attention_dropout=0.0,
                                            transform=True, activation='softmax',
                                            use_hierarchical_transformer=use_hierarchical_transformer)



        self.last_activation = nn.Tanh()
        self.n_geno2pheno = sum([sys2pheno, gene2pheno])
        self.cov_effect = cov_effect
        self.use_independent_predictors = use_independent_predictors
        print("Cov effect: ", self.cov_effect)
        print("Input format: ", self.input_format)
        print("Using block embedding: ", self.use_block_embedding)
        print("Using SNP2gene gating: ", self.use_snp2gene_gating)
        print("Using SNP2gene null gate: ", self.use_snp2gene_null_gate)
        if self.use_snp2gene_null_gate:
            print("SNP2gene null logit init: ", self.snp2gene_null_logit_init)
        print("Using SNP shallow predictor: ", self.use_snp_shallow_predictor)
        if self.use_snp_shallow_predictor:
            print("SNP shallow feeds G2PT: ", self.snp_shallow_feed_g2pt)
        print("SNP2gene attention dropout: ", self.snp2gene.hierarchical_transformer_update.attention_dropout)
        print("Prediction activation: ", self.prediction_activation)
        print("The number of geno2pheno: ", self.n_geno2pheno)
        print(f"Using independent predictors: {self.use_independent_predictors}")

        def create_prediction_activation():
            if self.prediction_activation == 'identity':
                return nn.Identity()
            if self.prediction_activation == 'gelu':
                return nn.GELU()
            if self.prediction_activation == 'relu':
                return nn.ReLU()
            if self.prediction_activation == 'tanh':
                return nn.Tanh()
            raise ValueError(f"Unknown prediction_activation: {self.prediction_activation}")

        def create_predictor():
            if use_moe:
                return MoEHeadPrediction(hidden_dims * self.n_geno2pheno, k_experts=8, top_k=2)
            else:
                return nn.Sequential(
                    nn.Linear(hidden_dims * self.n_geno2pheno, hidden_dims),
                    create_prediction_activation(),
                    nn.Linear(hidden_dims, 1)
                )

        if self.use_independent_predictors:
            print(f"Creating independent predictors for {self.n_phenotypes} phenotypes.")
            self.predictors = nn.ModuleDict({pheno: create_predictor() for pheno in self.phenotypes})
        else:
            print("Using a single shared predictor.")
            self.predictor = create_predictor()

        self.system_value_scale = nn.Parameter(torch.ones(self.n_systems + 1))
        self.gene_value_scale = nn.Parameter(torch.ones(self.n_genes + 1))
        self.snp_predictor = nn.Linear(hidden_dims, self.n_snps * 3 + 2)
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
            snp_embedding, gene_embedding, system_embedding = self.propagate(genotype_dict, covariates, snp2gene_mask, gene2sys_mask, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward, sys2gene_mask)
        else:
            snp_embedding, gene_embedding, system_embedding = self.chunk_wise_propagate(genotype_dict, covariates, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward)

        # 2. Get phenotype-specific embeddings
        phenotype_embedding = self.phenotype_embeddings(phenotype_ids)

        # 3. Predict phenotype from final embeddings
        prediction = self.prediction(phenotype_embedding, system_embedding, gene_embedding,
                                     genotype_dict, chunk, phenotype_ids,
                                     sys_temp=sys_temp, covariates=covariates)
        if self.use_snp_shallow_predictor:
            snp_prediction_branch = self.get_snp_shallow_prediction(snp_embedding, phenotype_ids)
            prediction = self.g2pt_residual_alpha * prediction + self.snp_shallow_alpha * snp_prediction_branch
        if predict_snp:
            # Pass gene2snp edge indices if using sparse attention
            gene2snp_edge_idx = self.gene2snp_edge_index.to(gene_embedding.device) if (self.use_sparse_attention and hasattr(self, 'gene2snp_edge_index')) else None
            gene2snp_edge_att = self.gene2snp_edge_attr.to(gene_embedding.device) if (gene2snp_edge_idx is not None and hasattr(self, 'gene2snp_edge_attr') and self.gene2snp_edge_attr is not None) else None

            snp_embedding_updated = snp_embedding + self.get_gene2snps(snp_embedding, gene_embedding, snp2gene_mask.T, gene2snp_edge_idx, gene2snp_edge_att)
            snp_prediction = self.snp_predictor(snp_embedding_updated)
            return prediction, snp_prediction
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
            gene_embedding = gene_embedding + self.get_cov2gene(gene_embedding, covariates)

        snp_embedding, _ = self.get_snp_embedding(genotype_dict)
        if self.snp_shallow_feed_g2pt:
            snp_embedding = self.get_snp_shallow_embedding(snp_embedding)
        snp_effect_on_gene = self.get_snp2gene(gene_embedding, snp_embedding, snp2gene_mask)
        snp_update = self.apply_snp2gene_null_mass(self.effect_norm(snp_effect_on_gene))
        gene_embedding = gene_embedding + snp_update

        gene_effect_on_system = self.get_gene2sys(self.dropout(system_embedding), self.dropout(gene_embedding), gene2sys_mask)

        batch_size = covariates.size(0)
        system_embedding_total = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1).clone()
        system_embedding_total[:, genotype_dict['sys_indices']] = system_embedding_total[:, genotype_dict['sys_indices']] + self.effect_norm(gene_effect_on_system)

        # Pass hierarchical edges if using sparse attention
        hierarchical_edges_fwd = self.hierarchical_edges_forward if (self.use_sparse_attention and hasattr(self, 'hierarchical_edges_forward')) else None
        hierarchical_edges_bwd = self.hierarchical_edges_backward if (self.use_sparse_attention and hasattr(self, 'hierarchical_edges_backward')) else None

        system_effect_forward = self.get_sys2sys(system_embedding_total, nested_hierarchical_masks_forward, direction='forward', hierarchical_edges=hierarchical_edges_fwd)
        system_embedding_total = system_embedding_total + self.effect_norm(system_effect_forward)

        system_effect_backward = self.get_sys2sys(system_embedding_total, nested_hierarchical_masks_backward, direction='backward', hierarchical_edges=hierarchical_edges_bwd)
        system_embedding_total = system_embedding_total + self.effect_norm(system_effect_backward)
        system_embedding = system_embedding_total[:, genotype_dict['sys_indices']]

        # Pass sys2gene edge indices only when embeddings cover all global indices.
        # sys2gene_edge_index has global gene/system indices; if embeddings are a
        # phenotype-subset, their dim is smaller and those indices would be OOB.
        _use_sys2gene_sparse = (
            self.use_sparse_attention
            and hasattr(self, 'sys2gene_edge_index')
            and gene_embedding.size(1) >= self.n_genes
            and system_embedding.size(1) >= self.n_systems
        )
        sys2gene_edge_idx = self.sys2gene_edge_index.to(gene_embedding.device) if _use_sys2gene_sparse else None
        sys2gene_edge_att = self.sys2gene_edge_attr.to(gene_embedding.device) if (_use_sys2gene_sparse and self.sys2gene_edge_attr is not None) else None

        system_effect_on_gene = self.get_sys2gene(gene_embedding, system_embedding, sys2gene_mask, sys2gene_edge_idx, sys2gene_edge_att)
        gene_embedding = gene_embedding + self.effect_norm(system_effect_on_gene)

        if (self.cov_effect=='post') or (self.cov_effect=='both'):
            gene_embedding = gene_embedding + self.get_cov2gene(gene_embedding, covariates)
            system_embedding = system_embedding + self.get_cov2sys(system_embedding, covariates)

        return snp_embedding, gene_embedding, system_embedding

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

            snp_effect_on_gene = self.get_snp2gene(gene_emb, snp_emb, chunk['snp2gene_mask'])
            snp_update = self.apply_snp2gene_null_mass(self.effect_norm(snp_effect_on_gene))
            gene_emb += snp_update

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
        if self.use_block_embedding and 'block_ind' in genotype_dict:
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

    def get_snp_shallow_embedding(self, snp_embedding):
        """One-layer SNP-level transformer over SNP + block embeddings."""
        shallow_input = self.snp_shallow_input_norm(snp_embedding)
        shallow_embedding = self.snp_shallow_transformer(shallow_input)
        return self.snp_shallow_output_norm(shallow_embedding)

    def get_snp_shallow_prediction(self, snp_embedding, phenotype_ids):
        """Predict phenotype logits directly from shallow SNP representations."""
        if not self.snp_shallow_feed_g2pt:
            snp_embedding = self.get_snp_shallow_embedding(snp_embedding)
        pooled = snp_embedding.mean(dim=1)
        all_scores = self.snp_shallow_head(pooled)

        if phenotype_ids.dim() == 0:
            pheno_ids = phenotype_ids.view(1, 1).expand(all_scores.size(0), -1)
        elif phenotype_ids.dim() == 1:
            if phenotype_ids.numel() == all_scores.size(0):
                pheno_ids = phenotype_ids.view(-1, 1)
            else:
                pheno_ids = phenotype_ids.view(1, -1).expand(all_scores.size(0), -1)
        else:
            pheno_ids = phenotype_ids

        pheno_ids = pheno_ids.to(device=all_scores.device, dtype=torch.long)
        return all_scores.gather(1, pheno_ids).unsqueeze(-1)

    def get_snp2gene(self, gene_embedding, snp_embedding, snp2gene_mask):
        """Propagates information from SNP embeddings to gene embeddings."""
        gene_embedding_input = self.dropout(self.gene_norm(gene_embedding))
        snp_embedding_input = self.dropout(snp_embedding)

        if self.use_snp2gene_gating:
            return self.get_gated_snp2gene(gene_embedding_input, snp_embedding_input, snp2gene_mask)

        # Use sparse edges only when embedding covers all global indices.
        # When select_phenotypes() creates a subset batch, embedding dim is smaller
        # than n_genes/n_snps and the precomputed global edge_index would be OOB.
        if (self.use_sparse_attention and hasattr(self, 'snp2gene_edge_index')
                and gene_embedding.size(1) >= self.n_genes
                and snp_embedding.size(1) >= self.n_snps):
            edge_index = self.snp2gene_edge_index.to(gene_embedding.device)
            edge_attr = self.snp2gene_edge_attr.to(gene_embedding.device) if self.snp2gene_edge_attr is not None else None
            source_edge_bias = self.get_snp2gene_source_edge_bias(gene_embedding.device, gene_embedding.dtype)
            if source_edge_bias is not None:
                edge_attr = source_edge_bias if edge_attr is None else edge_attr + source_edge_bias
            snp_effect_on_gene = self.snp2gene.forward(
                gene_embedding_input, snp_embedding_input, mask=None,
                edge_index=edge_index, edge_attr=edge_attr
            )
            snp_effect_on_gene = self.zero_noedge_snp2gene_queries(snp_effect_on_gene, edge_index)
        else:
            snp_effect_on_gene = self.snp2gene.forward(gene_embedding_input, snp_embedding_input, snp2gene_mask)

        return snp_effect_on_gene

    def get_snp2gene_source_edge_bias(self, device, dtype):
        """Return learnable source bias for each SNP→gene edge."""
        if not self.use_snp2gene_source_prior:
            return None
        source_matrix = self.snp2gene_edge_source_matrix.to(device=device, dtype=dtype)
        return source_matrix.matmul(self.snp2gene_source_bias.to(device=device, dtype=dtype))

    def zero_noedge_snp2gene_queries(self, snp_effect_on_gene, edge_index):
        """Keep sparse SNP→gene behavior aligned with dense masks for no-edge genes."""
        has_edge = torch.zeros(
            snp_effect_on_gene.size(1),
            dtype=torch.bool,
            device=snp_effect_on_gene.device,
        )
        if edge_index.numel() > 0:
            query_idx = edge_index[0]
            query_idx = query_idx[query_idx < snp_effect_on_gene.size(1)]
            has_edge[query_idx] = True
        return snp_effect_on_gene * has_edge.view(1, -1, 1).to(dtype=snp_effect_on_gene.dtype)

    def get_gated_snp2gene(self, gene_embedding_input, snp_embedding_input, snp2gene_mask):
        """SNP→gene attention with learned edge gates normalized within each gene.

        The learned gate is an additive edge logit before the per-gene softmax. If
        null gating is enabled, each gene also gets a no-propagation option, so a
        gene with one weak SNP can assign most mass to null instead of forcing the
        lone SNP to weight 1.
        """
        if (self.use_sparse_attention and hasattr(self, 'snp2gene_edge_index')
                and gene_embedding_input.size(1) >= self.n_genes
                and snp_embedding_input.size(1) >= self.n_snps):
            edge_index = self.snp2gene_edge_index.to(gene_embedding_input.device)
        else:
            edge_index = self.snp2gene_mask_to_edges(
                snp2gene_mask,
                gene_embedding_input.size(1),
                snp_embedding_input.size(1),
                gene_embedding_input.device,
            )
        return self.gated_snp2gene_transformer(gene_embedding_input, snp_embedding_input, edge_index)

    def snp2gene_mask_to_edges(self, snp2gene_mask, n_genes, n_snps, device):
        """Convert a dense SNP→gene additive mask to local sparse edges."""
        mask = snp2gene_mask.to(device) if isinstance(snp2gene_mask, torch.Tensor) else torch.tensor(snp2gene_mask, device=device)
        valid_edges = torch.nonzero(mask > -9999, as_tuple=False)
        if valid_edges.numel() == 0:
            return torch.empty((2, 0), dtype=torch.long, device=device)
        valid_edges = valid_edges[(valid_edges[:, 0] < n_genes) & (valid_edges[:, 1] < n_snps)]
        return valid_edges.t().contiguous().long()

    def gated_snp2gene_transformer(self, gene_embedding_input, snp_embedding_input, edge_index):
        """Run the SNP→gene transformer using gated sparse attention."""
        update = self.snp2gene.hierarchical_transformer_update
        q = gene_embedding_input
        k = snp_embedding_input
        attn_output, snp_mass = self.gated_snp2gene_attention(update.norm_attn(q), k, edge_index)
        x = q + update.drop(attn_output)
        x = x + update.drop(update.ffn(update.norm_ffn(x)))
        result = self.snp2gene.norm(x)
        result = self.zero_noedge_snp2gene_queries(result, edge_index)
        self.last_snp2gene_snp_mass = snp_mass.detach() if snp_mass is not None else None
        return result

    def apply_snp2gene_null_mass(self, snp_update):
        """Apply no-propagation mass after LayerNorm so null gating can suppress updates."""
        snp_mass = getattr(self, 'last_snp2gene_snp_mass', None)
        if not self.use_snp2gene_null_gate or snp_mass is None:
            return snp_update
        return snp_update * snp_mass.mean(dim=-1, keepdim=True).to(dtype=snp_update.dtype)

    def gated_snp2gene_attention(self, q, k, edge_index):
        """Sparse cross-attention where edge gate logits compete within each gene."""
        update = self.snp2gene.hierarchical_transformer_update
        batch_size, n_genes, hidden_dims = q.shape
        n_snps = k.size(1)
        n_edges = edge_index.size(1)
        if n_edges == 0:
            empty = torch.zeros_like(q)
            snp_mass = torch.zeros(batch_size, n_genes, update.h, device=q.device, dtype=q.dtype)
            return empty, snp_mass

        query_idx = edge_index[0]
        key_idx = edge_index[1]
        self.last_snp2gene_gate_edge_index = edge_index.detach()
        q_proj = update.q_proj(q).view(batch_size, n_genes, update.h, update.d_h)
        k_proj = update.k_proj(k).view(batch_size, n_snps, update.h, update.d_h)
        v_proj = update.v_proj(k).view(batch_size, n_snps, update.h, update.d_h)

        q_edges = q_proj[:, query_idx, :, :]
        k_edges = k_proj[:, key_idx, :, :]
        v_edges = v_proj[:, key_idx, :, :]
        scale = 1.0 / (update.d_h ** 0.5)
        attention_scores = (q_edges * k_edges).sum(dim=-1) * scale

        gate_input = torch.cat([q[:, query_idx, :], k[:, key_idx, :]], dim=-1)
        gate_logits = self.snp2gene_gate(gate_input).squeeze(-1)
        scores = attention_scores + gate_logits.unsqueeze(-1)

        attention_weights, snp_mass = self.snp2gene_sparse_softmax(scores, query_idx, n_genes)
        if self.training and update.attention_dropout > 0:
            attention_weights = torch.nn.functional.dropout(
                attention_weights,
                p=update.attention_dropout,
                training=True,
            )

        weighted_values = attention_weights.unsqueeze(-1) * v_edges
        output = torch.zeros(
            batch_size, n_genes, update.h, update.d_h,
            device=q.device, dtype=q.dtype,
        )
        scatter_idx = query_idx.view(1, n_edges, 1, 1).expand(batch_size, -1, update.h, update.d_h)
        output.scatter_add_(1, scatter_idx, weighted_values)
        output = output.reshape(batch_size, n_genes, hidden_dims)
        return update.o_proj(output), snp_mass

    def snp2gene_sparse_softmax(self, scores, query_idx, n_genes):
        """Softmax SNP edge scores by target gene, optionally including a null edge."""
        batch_size, n_edges, n_heads = scores.shape
        max_scores = torch.full(
            (batch_size, n_genes, n_heads),
            float('-inf'),
            dtype=scores.dtype,
            device=scores.device,
        )
        scatter_idx = query_idx.view(1, n_edges, 1).expand(batch_size, -1, n_heads)
        max_scores.scatter_reduce_(1, scatter_idx, scores, reduce='amax', include_self=False)

        null_scores = None
        if self.use_snp2gene_null_gate:
            null_scores = self.snp2gene_null_logit.to(dtype=scores.dtype).view(1, 1, 1)
            max_scores = torch.maximum(max_scores, null_scores.expand_as(max_scores))

        max_per_edge = max_scores[:, query_idx, :]
        exp_scores = torch.exp(scores - max_per_edge)
        sum_exp = torch.zeros_like(max_scores)
        sum_exp.scatter_add_(1, scatter_idx, exp_scores)

        if self.use_snp2gene_null_gate:
            null_exp = torch.exp(null_scores.expand_as(max_scores) - max_scores)
            denominator = sum_exp + null_exp
            snp_mass = sum_exp / (denominator + 1e-8)
        else:
            denominator = sum_exp
            snp_mass = None

        attention_weights = exp_scores / (denominator[:, query_idx, :] + 1e-8)
        self.last_snp2gene_gate = attention_weights.mean(dim=-1).detach()
        if self.use_snp2gene_null_gate:
            self.last_snp2gene_null_weight = (1.0 - snp_mass.mean(dim=-1)).detach()
        return attention_weights, snp_mass

    def get_gene2sys(self, system_embedding, gene_embedding, gene2sys_mask):
        """Propagates information from gene embeddings to system embeddings."""
        system_embedding_input = self.dropout(self.sys_norm(system_embedding))
        gene_embedding_input = self.dropout(self.gene_norm(gene_embedding))

        # Use sparse edges only when embedding covers all global indices.
        if (self.use_sparse_attention and hasattr(self, 'gene2sys_edge_index')
                and system_embedding.size(1) >= self.n_systems
                and gene_embedding.size(1) >= self.n_genes):
            edge_index = self.gene2sys_edge_index.to(system_embedding.device)
            edge_attr = self.gene2sys_edge_attr.to(system_embedding.device) if self.gene2sys_edge_attr is not None else None
            gene_effect = self.gene2sys.forward(
                system_embedding_input, gene_embedding_input, mask=None,
                edge_index=edge_index, edge_attr=edge_attr
            )
        else:
            gene_effect = self.gene2sys.forward(system_embedding_input, gene_embedding_input, gene2sys_mask)

        return gene_effect

    def get_gene2snps(self, snp_embedding, gene_embedding, gene2snp_mask, gene2snp_edge_index=None, gene2snp_edge_attr=None):
        """Propagates information from gene embeddings to SNP embeddings."""
        gene_embedding_input = self.dropout(self.gene_norm(gene_embedding))
        snp_embedding_input = self.dropout(snp_embedding)
        gene_effect_on_snp = self.gene2snp.forward(snp_embedding_input, gene_embedding_input, gene2snp_mask,
                                                    edge_index=gene2snp_edge_index, edge_attr=gene2snp_edge_attr)
        return gene_effect_on_snp


    def get_covariate_embedding(self, covariates):
        """Computes a single vector embedding from the covariate data."""
        covariates_vector = self.covariate_norm_1(self.activation(self.covariate_linear_1(self.dropout(covariates))))
        covariates_vector = self.dropout(covariates_vector.unsqueeze(1))
        return covariates_vector

    # --- Attention & Prediction Heads ---
    def prediction(self, phenotype_vector, system_embedding, gene_embedding, genotype_dict, chunk, phenotype_ids, sys_temp=None, covariates=None):
        """
        Generates the final phenotype prediction from the propagated embeddings.

        Args:
            phenotype_vector (torch.Tensor): The embedding for the target phenotype.
            system_embedding (torch.Tensor): The final system embeddings.
            gene_embedding (torch.Tensor): The final gene embeddings.
            genotype_dict (dict): Dictionary with genotype data for the batch.
            chunk (bool): Flag indicating if chunk-wise propagation was used.
            phenotype_ids (torch.Tensor): Tensor of phenotype IDs for the current batch.
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
                phenotype_weighted_by_systems = phenotype_weighted_by_systems + self.cov2pheno(phenotype_weighted_by_systems, covariates)
            phenotype_features.append(phenotype_weighted_by_systems)

        if self.use_gene2pheno:
            phenotype_weighted_by_genes = self.get_gene2pheno(phenotype_vector, gene_embedding, gene_embedding_value, mask=None)
            if self.cov_effect == 'direct':
                phenotype_weighted_by_genes = phenotype_weighted_by_genes + self.cov2pheno(phenotype_weighted_by_genes, covariates)
            phenotype_features.append(phenotype_weighted_by_genes)

        phenotype_feature = torch.cat(phenotype_features, dim=-1)

        if self.use_independent_predictors:
            # phenotype_ids may be 1-D (n_phenos,) or 2-D (B, n_phenos).
            # All samples in a batch share the same phenotype IDs, so collapse
            # to a 1-D view to get the canonical per-phenotype ID list.
            if phenotype_ids.dim() > 1:
                pheno_ids_1d = phenotype_ids[0]   # first row: (n_phenos,)
            else:
                pheno_ids_1d = phenotype_ids      # already (n_phenos,)

            if pheno_ids_1d.numel() == 1:
                # Single-phenotype batch — original path
                pheno_name = self.ind2pheno[pheno_ids_1d.item()]
                phenotype_prediction = self.predictors[pheno_name](phenotype_feature)
            else:
                # Multi-phenotype batch.
                # phenotype_feature is (B, n_phenos, feature_dim); apply each
                # phenotype's independent predictor to its own slice.
                predictions = []
                for i in range(pheno_ids_1d.numel()):
                    pid        = int(pheno_ids_1d[i].item())
                    pheno_name = self.ind2pheno[pid]
                    feat_i     = phenotype_feature[:, i, :]   # (B, feature_dim)
                    predictions.append(self.predictors[pheno_name](feat_i))
                # Stack → (B, n_phenos, out_dim)
                phenotype_prediction = torch.stack(predictions, dim=1)
        else:
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

    def get_gumbel_gates(self):
        """Retrieves the Gumbel gates from block-based transformers."""
        gumbel_gates = {}
        for block_id, module in self.pretrained_transformer_tuner.items():
            gumbel_gates[block_id] = module.gumbel_gates
        return gumbel_gates
