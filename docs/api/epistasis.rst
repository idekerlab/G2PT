Epistasis
=========

.. class:: EpistasisFinder

   Finds and analyzes epistatic interactions between SNPs within biological systems.

   This class integrates attention scores from a trained model with genotype data
   to perform a multi-stage statistical analysis. It first identifies candidate
   SNPs based on their relevance to a system (using attention scores), then
   filters SNP pairs by physical distance, tests for pairwise interaction using
   Fisher's Exact Test, and finally validates significant pairs using a
   regression model to confirm the statistical interaction effect.

   :param tree_parser: An instance of the parser containing SNP, gene, and system relationships.
   :type tree_parser: SNPTreeParser
   :param genotype: A DataFrame of genotype data, with samples as rows and SNPs as columns.
   :type genotype: pd.DataFrame
   :param cov_df: A DataFrame containing covariate and phenotype data.
   :type cov_df: pd.DataFrame
   :param attention_results: A DataFrame of attention scores for each sample and system.
   :type attention_results: pd.DataFrame

   .. method:: __init__(tree_parser, attention_results, tsv_path, cov, pheno=None, flip=False)

      Initializes the EpistasisFinder and loads all necessary data from TSV files.

      :param tree_parser: An initialized SNPTreeParser object.
      :type tree_parser: SNPTreeParser
      :param attention_results: Path to a CSV file or a DataFrame of attention scores.
      :type attention_results: str or pd.DataFrame
      :param tsv_path: Path to the directory containing genotypes.tsv and snp2gene.tsv.
      :type tsv_path: str
      :param cov: Path to a tab-separated covariate file.
      :type cov: str
      :param pheno: Path to a tab-separated phenotype file.
      :type pheno: str, optional
      :param flip: If True, swaps reference and alternate alleles.
      :type flip: bool, optional

   .. method:: search_epistasis_on_system(system, sex=0, quantile=0.9, fisher=True, return_significant_only=True, check_inheritance=True, verbose=0, snp_inheritance_dict={}, binary=False, target='PHENOTYPE')

      Searches for epistatic interactions for a given biological system.

      This method executes a multi-step pipeline:
      1.  Determines the optimal inheritance model for each SNP (optional).
      2.  Filters SNPs using a Chi-Square test based on attention scores to
          identify those prevalent in a high-risk cohort.
      3.  Generates all pairs of the filtered SNPs.
      4.  Filters out SNP pairs that are physically close on the chromosome.
      5.  Performs Fisher's Exact Test on the distant pairs to find
          statistically significant co-occurrences (optional).
      6.  Uses a regression model to test for a statistical interaction
          effect for the remaining pairs, correcting for multiple testing.

      :param system: The name of the system (e.g., GO term) to analyze.
      :type system: str
      :param sex: The sex to include in the analysis (0, 1, or 2 for all).
      :type sex: int, optional
      :param quantile: The attention score quantile to define the high-risk cohort.
      :type quantile: float, optional
      :param fisher: Whether to perform the Fisher's Exact Test step.
      :type fisher: bool, optional
      :param return_significant_only: If True, returns only the pairs that are statistically significant after all tests. If False, returns results for all tested pairs.
      :type return_significant_only: bool, optional
      :param check_inheritance: If True, determines the best-fit inheritance model for each SNP before testing.
      :type check_inheritance: bool, optional
      :param verbose: Verbosity level (0 or 1).
      :type verbose: int, optional
      :param snp_inheritance_dict: A pre-computed dictionary of SNP inheritance models.
      :type snp_inheritance_dict: dict, optional
      :param binary: Whether the target phenotype is binary (for logistic regression) or continuous (for linear regression).
      :type binary: bool, optional
      :param target: The name of the phenotype column in the covariate data.
      :type target: str, optional
      :return: A tuple containing a list of significant epistatic pairs and an updated dictionary of determined SNP inheritance models.
      :rtype: tuple
