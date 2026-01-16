Epistasis Simulation
====================

Overview
--------

The epistasis simulation utilities generate synthetic genotype and phenotype
arrays with configurable additive and pairwise interaction effects. A companion
helper builds a hierarchical SNP→Gene→System ontology aligned to the simulated
causal structure, which can be exported as TSV files for downstream pipelines.

Usage and examples
------------------

Simulate epistatic data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils.analysis.epistasis_simulation import simulate_epistasis

   sim = simulate_epistasis(
       n_samples=2000,
       n_snps=50000,
       n_additive=200,
       n_pairs=75,
       h2_additive=0.4,
       h2_epistatic=0.2,
       n_ld_blocks=200,
       ld_rho=0.8,
   )

Build a hierarchy aligned to the simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from src.utils.analysis.epistasis_simulation import build_hierarchical_ontology

   snp_df, gene_df, system_df = build_hierarchical_ontology(
       sim,
       n_genes=800,
       n_systems=60,
       ontology_coherence=0.5,
       overlap_prob=0.25,
       n_causal_systems=20,
   )

API documentation
-----------------

.. function:: simulate_epistasis(n_samples=1000, n_snps=20000, n_additive=100, n_pairs=50, h2_additive=0.5, h2_epistatic=0.1, min_additive_p_value=1e-4, min_epistatic_p_value=1e-3, maf_range=(0.05, 0.5), n_ld_blocks=100, ld_rho=0.8, epistasis_bias=10.0, seed=42)

   Simulate genotype data with additive and epistatic effects.

   :param n_samples: Number of individuals.
   :type n_samples: int
   :param n_snps: Total number of SNPs.
   :type n_snps: int
   :param n_additive: Number of causal additive SNPs.
   :type n_additive: int
   :param n_pairs: Number of causal epistatic SNP pairs.
   :type n_pairs: int
   :param h2_additive: Additive heritability budget.
   :type h2_additive: float
   :param h2_epistatic: Epistatic heritability budget.
   :type h2_epistatic: float
   :param min_additive_p_value: Minimum marginal p-value enforced for causal additive SNPs.
   :type min_additive_p_value: float, optional
   :param min_epistatic_p_value: Minimum interaction p-value enforced for causal pairs.
   :type min_epistatic_p_value: float, optional
   :param maf_range: Minor-allele frequency range for SNPs.
   :type maf_range: tuple[float, float]
   :param n_ld_blocks: Number of LD blocks to simulate.
   :type n_ld_blocks: int
   :param ld_rho: Correlation coefficient for adjacent SNPs in a block.
   :type ld_rho: float
   :param epistasis_bias: Bias factor to steer epistatic pairs toward SNPs with smaller additive effects.
   :type epistasis_bias: float
   :param seed: Random seed.
   :type seed: int
   :return: Dictionary with genotype matrix, phenotype vector, causal SNP indices, epistatic pairs, effect sizes, and LD block assignments.
   :rtype: dict

.. function:: build_hierarchical_ontology(sim, n_genes=800, n_systems=60, ontology_coherence=0.5, overlap_prob=0.25, n_causal_systems=20, causal_system_enrichment=5.0, seed=123)

   Build a hierarchical SNP→Gene→System ontology aligned to the simulation.

   :param sim: Output dictionary from :func:`simulate_epistasis`.
   :type sim: dict
   :param n_genes: Number of genes to simulate.
   :type n_genes: int
   :param n_systems: Number of leaf systems in the ontology.
   :type n_systems: int
   :param ontology_coherence: Controls whether epistatic pairs map to the same, related, or distant systems.
   :type ontology_coherence: float
   :param overlap_prob: Probability of overlapping genes across systems.
   :type overlap_prob: float
   :param n_causal_systems: Number of systems to enrich for causal genes.
   :type n_causal_systems: int
   :param causal_system_enrichment: Weight applied when sampling enriched systems.
   :type causal_system_enrichment: float
   :param seed: Random seed.
   :type seed: int
   :return: Tuple of DataFrames: SNP-to-gene, gene-to-system, and system hierarchy.
   :rtype: tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
