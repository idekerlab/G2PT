# Epistasis Analysis

The `epistasis.py` module provides the `EpistasisFinder` class, a powerful tool designed to identify and statistically validate epistatic (SNP-SNP) interactions within specific biological systems or pathways.

## Overview

The core challenge in finding epistatic interactions is the vast combinatorial search space. Testing all possible pairs of SNPs in a genome-wide study is computationally prohibitive. This module addresses the challenge by implementing a knowledge-driven, multi-stage filtering and testing pipeline that integrates machine learning outputs (attention scores) with rigorous statistical methods.

The workflow is designed to progressively narrow down the search space, ensuring that only the most promising candidate pairs are subjected to the final, computationally intensive regression analysis.

## Key Features

-   **Attention-Guided Filtering**: Uses attention scores from a trained model to prioritize SNPs that are most relevant to a specific biological system.
-   **Flexible Data Handling**: Seamlessly loads genotype, covariate, and phenotype data from PLINK files and standard text formats.
-   **Inheritance Model Selection**: Automatically determines the best-fit genetic inheritance model (e.g., additive, dominant, recessive) for each SNP to increase statistical power.
-   **Efficient Statistical Pipeline**: Employs a series of statistical tests in an optimized order to efficiently filter candidate SNP pairs.
-   **Robust Interaction Testing**: Uses a regression framework to test for statistical interactions while controlling for covariates.
-   **Multiple Testing Correction**: Implements False Discovery Rate (FDR) correction at multiple stages to control for false positives.
-   **Visualization**: Includes tools to generate interaction plots for visualizing significant epistatic effects.

## The Analysis Workflow

The main analysis is orchestrated by the `search_epistasis_on_system` method. The workflow proceeds as follows:

1.  **Candidate SNP Selection (Chi-Square Test)**:
    *   First, a "high-risk" cohort of individuals is identified based on high attention scores for the biological system of interest.
    *   A Chi-Square test is performed for each SNP in the system to identify which ones are significantly more frequent in the high-risk cohort compared to the general population. This creates a reduced set of candidate SNPs.

2.  **Pair Generation and Distance Filtering**:
    *   All unique pairs of the candidate SNPs are generated.
    *   To minimize confounding by Linkage Disequilibrium (LD), pairs where the two SNPs are physically close on the same chromosome (default: < 500kb) are filtered out and removed from further consideration. This is a critical step for efficiency, as it avoids wasting computation on pairs that are likely just co-inherited.

3.  **Pairwise Association Test (Fisher's Exact Test)**:
    *   For the remaining, distant pairs, a Fisher's Exact Test is performed. This test assesses whether the two SNPs in a pair co-occur in the high-risk cohort more often than would be expected by chance.
    *   The p-values from this step are adjusted using the Benjamini-Hochberg FDR correction, and only the pairs that remain significant proceed to the final validation step.

4.  **Statistical Interaction Test (Regression)**:
    *   This is the final and most rigorous test. For each significant pair, a regression model is fitted:
        ```
        Phenotype ~ Covariates + SNP_A + SNP_B + SNP_A:SNP_B
        ```
    *   The key term is the interaction `SNP_A:SNP_B`. A statistically significant p-value for this term indicates that the combined effect of the two SNPs on the phenotype is different from the sum of their individual effects, which is the definition of epistasis.
    *   P-values for the interaction terms are again FDR-corrected to produce the final list of significant epistatic pairs.

## Usage Example

```python
from src.utils.tree import SNPTreeParser
from src.utils.analysis.epistasis import EpistasisFinder

# 1. Initialize the SNPTreeParser with ontology and SNP-gene mappings
tree_parser = SNPTreeParser(
    ontology='path/to/ontology.tsv',
    snp2gene='path/to/snp2gene.tsv',
    sys_annot_file='path/to/annotations.tsv'
)

# 2. Initialize the EpistasisFinder
epistasis_finder = EpistasisFinder(
    tree_parser=tree_parser,
    bfile='path/to/genotypes',  # Path to PLINK bfile prefix
    attention_results='path/to/attention_scores.csv',
    cov='path/to/covariates.tsv',
    pheno='path/to/phenotype.tsv'
)

# 3. Define the system to analyze
target_system = 'GO:0006915' # Apoptotic Process

# 4. Run the epistasis search
# This will return a list of significant pairs and the inheritance models used
significant_pairs, snp_models = epistasis_finder.search_epistasis_on_system(
    system=target_system,
    sex=0, # Analyze females
    quantile=0.9, # Define top 10% of attention scores as high-risk
    verbose=1
)

# 5. Print the results
print(f"Found {len(significant_pairs)} significant epistatic interactions in {target_system}.")
for snp1, snp2, raw_p, adj_p in significant_pairs:
    print(f"  - Pair: {snp1} - {snp2}, FDR-adjusted p-value: {adj_p:.4g}")

# 6. Visualize the top hit
if significant_pairs:
    top_hit = significant_pairs[0]
    epistasis_finder.draw_epistasis(
        target_snp_0=top_hit[0],
        target_snp_1=top_hit[1],
        phenotype='PHENOTYPE', # Name of the phenotype column
        out_dir='epistasis_plot.svg'
    )
```
