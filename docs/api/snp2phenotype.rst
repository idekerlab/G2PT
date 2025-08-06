SNP2Phenotype
=============

.. class:: SNP2PhenotypeModel

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

   :param tree_parser: An object that provides the hierarchical structure (SNP-gene-system mappings) and corresponding masks for the model.
   :type tree_parser: SNPTreeParser
   :param hidden_dims: The dimensionality of the embeddings and hidden layers.
   :type hidden_dims: int
   :param snp2pheno: Unused parameter for future extension.
   :type snp2pheno: bool, optional
   :param gene2pheno: If True, use the final gene embeddings for phenotype prediction.
   :type gene2pheno: bool, optional
   :param sys2pheno: If True, use the final system embeddings for phenotype prediction.
   :type sys2pheno: bool, optional
   :param interaction_types: The types of interactions to use for system-to-system propagation.
   :type interaction_types: list, optional
   :param n_covariates: The number of covariate features to include in the model.
   :type n_covariates: int, optional
   :param n_phenotypes: The number of distinct phenotypes the model can predict.
   :type n_phenotypes: int, optional
   :param dropout: The dropout rate for regularization.
   :type dropout: float, optional
   :param activation: The activation function for attention mechanisms.
   :type activation: str, optional
   :param input_format: The format of the genotype input ('indices' or 'block').
   :type input_format: str, optional
   :param poincare: Unused parameter for future extension.
   :type poincare: bool, optional
   :param cov_effect: Specifies how covariates affect the model ('pre', 'post', 'direct', or 'both').
   :type cov_effect: str, optional
   :param pretrained_transformer: A dictionary of pretrained transformer models for block-based input.
   :type pretrained_transformer: dict, optional
   :param freeze_pretrained: Unused parameter.
   :type freeze_pretrained: bool, optional
   :param phenotypes: Unused parameter.
   :type phenotypes: tuple, optional
   :param use_hierarchical_transformer: If True, uses a hierarchical transformer for the final prediction heads.
   :type use_hierarchical_transformer: bool, optional

   .. method:: forward(genotype_dict, covariates, phenotype_ids, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward, snp2gene_mask, gene2sys_mask, sys2gene_mask, sys_temp=None, sys2env=True, env2sys=True, sys2gene=True, score=False, attention=False, snp_only=False, predict_snp=False, chunk=False)

      Defines the main forward pass of the model.

      :param genotype_dict: A dictionary containing genotype information (e.g., SNP indices).
      :type genotype_dict: dict
      :param covariates: A tensor of covariate data for the batch.
      :type covariates: torch.Tensor
      :param phenotype_ids: A tensor of phenotype IDs for the batch.
      :type phenotype_ids: torch.Tensor
      :param nested_hierarchical_masks_forward: Masks for forward system-system propagation.
      :type nested_hierarchical_masks_forward: list
      :param nested_hierarchical_masks_backward: Masks for backward system-system propagation.
      :type nested_hierarchical_masks_backward: list
      :param snp2gene_mask: The attention mask for SNP-to-gene propagation.
      :type snp2gene_mask: torch.Tensor
      :param gene2sys_mask: The attention mask for gene-to-system propagation.
      :type gene2sys_mask: torch.Tensor
      :param sys2gene_mask: The attention mask for system-to-gene propagation.
      :type sys2gene_mask: torch.Tensor
      :param sys_temp: A temperature mask for system attention.
      :type sys_temp: torch.Tensor, optional
      :param score: If True, return attention scores.
      :type score: bool, optional
      :param attention: If True, return attention weights.
      :type attention: bool, optional
      :param chunk: If True, use chunk-wise propagation.
      :type chunk: bool, optional
      :return: The phenotype prediction tensor. If `attention` or `score` is True, returns a tuple containing the prediction and the requested attention/score tensors.
      :rtype: torch.Tensor or tuple
