SNP2P Dataset
=============

This page summarizes the dataset and data-collation utilities used by SNP2P
training. The datasets build genotype indices that align with the
``SNPTreeParser`` catalogs and emit dictionaries that the model consumes via
the collators. Use these classes to load genotype sources (TSV or PLINK),
attach covariates/phenotypes, and optionally enable block/chunk processing.

Example: load a PLINK dataset
-----------------------------

.. code-block:: python

   from src.utils.tree import SNPTreeParser
   from src.utils.data.dataset.SNP2PDataset import PLINKDataset

   tree_parser = SNPTreeParser(ontology="ontology.tsv", snp2gene="snp2gene.tsv")
   dataset = PLINKDataset(
       tree_parser=tree_parser,
       bfile="data/geno/plink_prefix",
       cov="data/covariates.tsv",
       pheno="data/phenotypes.tsv",
       cov_ids=("AGE", "SEX"),
       pheno_ids=("BMI",),
   )

Example: create a collated batch
--------------------------------

.. code-block:: python

   from torch.utils.data import DataLoader
   from src.utils.data.dataset.SNP2PDataset import SNP2PCollator

   collator = SNP2PCollator(tree_parser=tree_parser, input_format="indices")
   loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collator)
   batch = next(iter(loader))
   # batch["genotype"]["snp"], batch["covariates"], batch["phenotype"]

.. class:: GenotypeDataset

   Base dataset for SNP2P training that prepares covariates and phenotype targets.

   :param tree_parser: Parsed SNP ontology and masks.
   :type tree_parser: SNPTreeParser
   :param cov: Path to the covariates TSV.
   :type cov: str
   :param pheno: Optional phenotype TSV.
   :type pheno: str, optional
   :param cov_mean_dict: Optional covariate mean overrides.
   :type cov_mean_dict: dict, optional
   :param cov_std_dict: Optional covariate standard deviation overrides.
   :type cov_std_dict: dict, optional
   :param cov_ids: Subset of covariate column names to load.
   :type cov_ids: tuple, optional
   :param pheno_ids: Subset of phenotype column names to load.
   :type pheno_ids: tuple, optional
   :param bt: Binary phenotype IDs.
   :type bt: tuple, optional
   :param qt: Quantitative phenotype IDs.
   :type qt: tuple, optional
   :param dynamic_phenotype_sampling: Whether phenotype sampling changes per batch.
   :type dynamic_phenotype_sampling: bool, optional

   .. method:: __getitem__(index)

      Returns a dictionary with covariate tensors and phenotype targets for the sample.

      :param index: Sample index.
      :type index: int
      :return: Sample payload with covariates/phenotype tensors.
      :rtype: dict


.. class:: TSVDataset

   Loads genotype data from a TSV and returns SNP, gene, and system indices.

   :param tree_parser: Parsed SNP ontology and masks.
   :type tree_parser: SNPTreeParser
   :param genotype_path: Path to genotype TSV.
   :type genotype_path: str
   :param cov: Path to covariates TSV.
   :type cov: str
   :param pheno: Optional phenotype TSV.
   :type pheno: str, optional
   :param cov_mean_dict: Optional covariate mean overrides.
   :type cov_mean_dict: dict, optional
   :param cov_std_dict: Optional covariate standard deviation overrides.
   :type cov_std_dict: dict, optional
   :param flip: Whether to flip reference/alternate allele encodings.
   :type flip: bool, optional
   :param input_format: Input format (``indices`` by default).
   :type input_format: str, optional
   :param cov_ids: Subset of covariate column names to load.
   :type cov_ids: tuple, optional
   :param pheno_ids: Subset of phenotype column names to load.
   :type pheno_ids: tuple, optional
   :param bt: Binary phenotype IDs.
   :type bt: tuple, optional
   :param qt: Quantitative phenotype IDs.
   :type qt: tuple, optional


.. class:: PLINKDataset

   Loads genotype data from PLINK binaries and aligns covariates/phenotypes.

   :param tree_parser: Parsed SNP ontology and masks.
   :type tree_parser: SNPTreeParser
   :param bfile: PLINK file prefix.
   :type bfile: str
   :param cov: Path to covariates TSV (optional).
   :type cov: str, optional
   :param pheno: Path to phenotype TSV (optional).
   :type pheno: str, optional
   :param cov_mean_dict: Optional covariate mean overrides.
   :type cov_mean_dict: dict, optional
   :param cov_std_dict: Optional covariate standard deviation overrides.
   :type cov_std_dict: dict, optional
   :param flip: Whether to flip reference/alternate allele encodings.
   :type flip: bool, optional
   :param block: Whether to include block indices in outputs.
   :type block: bool, optional
   :param input_format: Input format (``indices`` by default).
   :type input_format: str, optional
   :param cov_ids: Subset of covariate column names to load.
   :type cov_ids: tuple, optional
   :param pheno_ids: Subset of phenotype column names to load.
   :type pheno_ids: tuple, optional
   :param bt: Binary phenotype IDs.
   :type bt: tuple, optional
   :param qt: Quantitative phenotype IDs.
   :type qt: tuple, optional

   .. method:: summary()

      Print a short dataset summary.

   .. method:: sample_population(n=100)

      Subsample individuals for quick experiments.

      :param n: Number of individuals to keep.
      :type n: int, optional

   .. method:: sample_phenotypes(n, seed=None)

      Sample a subset of phenotypes and update the dataset ranges.

      :param n: Number of phenotypes to sample.
      :type n: int
      :param seed: Optional random seed.
      :type seed: int, optional

   .. method:: select_phenotypes(phenotypes)

      Restrict the dataset to specific phenotype names.

      :param phenotypes: Phenotype IDs to keep.
      :type phenotypes: list


.. class:: EmbeddingDataset

   PLINK-backed dataset that augments samples with pretrained SNP embeddings.

   :param tree_parser: Parsed SNP ontology and masks.
   :type tree_parser: SNPTreeParser
   :param bfile: PLINK file prefix.
   :type bfile: str
   :param embedding: Directory with per-sample embedding tensors.
   :type embedding: str
   :param iid2ind: Mapping from IID to embedding index.
   :type iid2ind: dict
   :param cov: Path to covariates TSV (optional).
   :type cov: str, optional
   :param pheno: Path to phenotype TSV (optional).
   :type pheno: str, optional
   :param cov_mean_dict: Optional covariate mean overrides.
   :type cov_mean_dict: dict, optional
   :param cov_std_dict: Optional covariate standard deviation overrides.
   :type cov_std_dict: dict, optional
   :param cov_ids: Subset of covariate column names to load.
   :type cov_ids: tuple, optional
   :param pheno_ids: Subset of phenotype column names to load.
   :type pheno_ids: tuple, optional
   :param bt: Binary phenotype IDs.
   :type bt: tuple, optional
   :param qt: Quantitative phenotype IDs.
   :type qt: tuple, optional


.. class:: SNPTokenizer

   Simple SNP tokenizer for masked language modeling over SNP blocks.

   :param vocab: Mapping from token string to integer ID.
   :type vocab: dict
   :param max_len: Optional maximum sequence length.
   :type max_len: int, optional


.. class:: BlockDataset

   Dataset that returns SNP indices for block-level pretraining.

   :param bfile: PLINK file prefix.
   :type bfile: str
   :param flip: Whether to flip reference/alternate allele encodings.
   :type flip: bool, optional

   .. method:: get_individual_block_genotype(iid)

      Return SNP indices for a given individual.

      :param iid: Individual ID.
      :type iid: str
      :return: SNP indices for the individual.
      :rtype: torch.Tensor


.. class:: BlockQueryDataset

   Dataset that assembles block-level genotypes from multiple block sources.

   :param tree_parser: Parsed SNP ontology and masks.
   :type tree_parser: SNPTreeParser
   :param bfile: PLINK file prefix.
   :type bfile: str
   :param blocks: Mapping of block identifiers to :class:`BlockDataset` instances.
   :type blocks: dict
   :param cov: Path to covariates TSV (optional).
   :type cov: str, optional
   :param pheno: Path to phenotype TSV (optional).
   :type pheno: str, optional
   :param cov_mean_dict: Optional covariate mean overrides.
   :type cov_mean_dict: dict, optional
   :param cov_std_dict: Optional covariate standard deviation overrides.
   :type cov_std_dict: dict, optional
   :param cov_ids: Subset of covariate column names to load.
   :type cov_ids: tuple, optional
   :param pheno_ids: Subset of phenotype column names to load.
   :type pheno_ids: tuple, optional
   :param bt: Binary phenotype IDs.
   :type bt: tuple, optional
   :param qt: Quantitative phenotype IDs.
   :type qt: tuple, optional
   :param flip: Whether to flip reference/alternate allele encodings.
   :type flip: bool, optional


.. class:: SNP2PCollator

   Collator that assembles batched SNP2P inputs and labels.

   :param tree_parser: Parsed SNP ontology and masks.
   :type tree_parser: SNPTreeParser
   :param input_format: Input format (``indices``, ``embedding``, or ``block``).
   :type input_format: str, optional
   :param pheno_ids: Phenotype IDs used for ordering labels.
   :type pheno_ids: tuple, optional
   :param mlm: Whether to apply SNP masked language modeling.
   :type mlm: bool, optional
   :param mlm_collator_dict: Per-block MLM collators for block input.
   :type mlm_collator_dict: dict, optional


.. class:: ChunkSNP2PCollator

   Collator that breaks SNP2P inputs into chunks for memory efficiency.

   :param tree_parser: Parsed SNP ontology and masks.
   :type tree_parser: SNPTreeParser
   :param chunker: Chunker with ``create_chunks`` output.
   :type chunker: object
   :param input_format: Input format (``indices``, ``embedding``, or ``block``).
   :type input_format: str, optional
   :param pheno_ids: Phenotype IDs used for ordering labels.
   :type pheno_ids: tuple, optional
   :param mlm: Whether to apply SNP masked language modeling.
   :type mlm: bool, optional
   :param mlm_collator_dict: Per-block MLM collators for block input.
   :type mlm_collator_dict: dict, optional


.. class:: DynamicPhenotypeBatchSampler

   Batch sampler that randomly samples phenotypes per batch.

   :param dataset: Dataset supporting ``sample_phenotypes`` and phenotype ranges.
   :type dataset: Dataset
   :param batch_size: Batch size.
   :type batch_size: int
   :param drop_last: Whether to drop the last incomplete batch.
   :type drop_last: bool, optional


.. class:: CohortSampler

   Weighted sampler for continuous phenotypes using skew-normal weights.

   :param dataset: Dataset containing ``cov_df``.
   :type dataset: Dataset
   :param n_samples: Optional number of samples to draw.
   :type n_samples: int, optional
   :param phenotype_col: Column name for the phenotype.
   :type phenotype_col: str, optional
   :param z_weight: Weight multiplier for skew-normal density.
   :type z_weight: float, optional
   :param sex_col: Column name or index for the sex covariate.
   :type sex_col: int or str, optional


.. class:: BinaryCohortSampler

   Weighted sampler for binary phenotypes.

   :param dataset: Dataset containing ``cov_df``.
   :type dataset: Dataset
   :param phenotype_col: Column name for the phenotype.
   :type phenotype_col: str, optional


.. class:: DistributedCohortSampler

   Distributed version of :class:`CohortSampler`.

   :param dataset: Dataset containing ``cov_df``.
   :type dataset: Dataset
   :param num_replicas: Number of distributed replicas.
   :type num_replicas: int, optional
   :param rank: Rank of the current replica.
   :type rank: int, optional
   :param shuffle: Whether to shuffle indices.
   :type shuffle: bool, optional
   :param seed: Random seed.
   :type seed: int, optional
   :param phenotype_col: Column name for the phenotype.
   :type phenotype_col: str, optional
   :param z_weight: Weight multiplier for skew-normal density.
   :type z_weight: float, optional
   :param sex_col: Column name or index for the sex covariate.
   :type sex_col: int or str, optional


.. class:: DistributedBinaryCohortSampler

   Distributed version of :class:`BinaryCohortSampler`.

   :param dataset: Dataset containing ``cov_df``.
   :type dataset: Dataset
   :param num_replicas: Number of distributed replicas.
   :type num_replicas: int, optional
   :param rank: Rank of the current replica.
   :type rank: int, optional
   :param shuffle: Whether to shuffle indices.
   :type shuffle: bool, optional
   :param seed: Random seed.
   :type seed: int, optional
   :param phenotype_col: Column name for the phenotype.
   :type phenotype_col: str, optional
