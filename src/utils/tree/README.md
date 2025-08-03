# `TreeParser` Documentation

The `TreeParser` class, located in `src/utils/tree/tree.py`, is a comprehensive tool for parsing, manipulating, and analyzing hierarchical biological ontologies, such as the Gene Ontology (GO). It is designed to handle complex relationships between systems (e.g., GO terms) and genes, providing a suite of methods for structural analysis, data transformation, and ontology simplification.

The class uses `pandas` for data manipulation and `networkx` to represent the ontology as a directed acyclic graph (DAG), enabling powerful graph-based operations.

## Initialization

The `TreeParser` is initialized with a path to an ontology file and an optional system annotation file.

```python
from tree import TreeParser

# Path to the ontology file (tab-separated: parent, child, interaction_type)
ontology_file = 'path/to/ontology.tsv'
# Optional: Path to system annotations (tab-separated: system_id, description)
annotation_file = 'path/to/annotations.tsv'

# Initialize the parser
tree = TreeParser(ontology_file, sys_annot_file=annotation_file)
```

During initialization (`init_ontology`), the class performs several key setup steps:
1.  **Loads Data**: Reads the ontology and annotations into pandas DataFrames.
2.  **Builds Graph**: Constructs a `networkx.DiGraph` representing the system hierarchy.
3.  **Creates Mappings**: Builds dictionaries to map systems and genes to unique integer indices (e.g., `sys2ind`, `gene2ind`) and vice-versa.
4.  **Propagates Genes**: Calculates `sys2gene_full`, a dictionary where each system is mapped to a list of all genes contained within it and all its descendant systems. This provides a complete view of gene membership up the hierarchy.
5.  **Calculates Heights**: Computes the height of each node in the system graph, which is used in structural similarity calculations. The height is the length of the longest path from a node to a leaf.

---

## Ontology Simplification (Collapsing)

A key feature of `TreeParser` is its ability to simplify the ontology by collapsing, or removing, terms that are redundant or uninformative. This is crucial for reducing complexity and focusing on the most relevant parts of the ontology. When a term is collapsed, its children are re-parented to its parents, preserving the overall hierarchy.

There are three methods for collapsing terms, each with a different strategy.

### 1. `collapse()`

This is a general-purpose collapsing method that removes terms based on two main criteria:

-   **Redundancy**: Systems that contain the exact same set of genes (after gene propagation) are considered redundant. The method keeps one representative term (the first one alphabetically) and collapses the others.
-   **Size**: Systems with fewer genes than a specified `min_term_size` (default is 2) are considered too small to be informative and are collapsed.

**Algorithm:**
1.  The method first identifies groups of redundant terms by hashing their full gene sets.
2.  It then identifies small terms by checking the size of their full gene sets.
3.  A final list of terms to collapse is compiled from both criteria.
4.  The graph is rewired by connecting the parents and children of each collapsed node.
5.  Genes from collapsed nodes are re-assigned to their parent nodes.
6.  The ontology is rebuilt from the modified graph.

### 2. `collapse_by_gene_similarity()`

This method collapses terms based on the similarity of their gene sets, measured by the **Jaccard index**. It is useful for grouping together systems that have a significant overlap in their member genes, even if they are not identical.

**Algorithm:**
1.  **Pairwise Similarity**: The method calculates the Jaccard similarity for every pair of systems based on their `sys2gene_full` gene sets. The Jaccard index is computed as:
    ```
    J(A, B) = |A ∩ B| / |A ∪ B|
    ```
2.  **Clustering**: The resulting similarity matrix is converted into a distance matrix (`1 - similarity`). This matrix is then used with `sklearn.cluster.AgglomerativeClustering` to group systems. The clustering algorithm merges systems into clusters until no two systems in a cluster have a similarity below the given `similarity_threshold`.
3.  **Representative Selection**: For each cluster with more than one term, a single representative is chosen. The representative is the term with the most genes. Ties are broken alphabetically.
4.  **Collapsing**: All non-representative terms in each cluster are collapsed. The graph rewiring and ontology reconstruction process is the same as in the `collapse` method.

### 3. `collapse_by_structural_similarity()`

This method uses the structure of the ontology graph itself to determine similarity, rather than gene sets. It is based on the concept of the **Most Informative Common Ancestor (MICA)** and uses a variation of Lin's similarity metric.

**Algorithm:**
1.  **Pairwise Similarity**: The structural similarity between two terms is calculated based on the height of their MICA. The height of a node is the length of the longest path to a leaf node. The similarity formula is:
    ```
    Sim(A, B) = (2 * Height(MICA(A, B))) / (Height(A) + Height(B))
    ```
    This value will be high if two terms share a "close" common ancestor relative to their own distance from the leaves.
2.  **Clustering**: Unlike the gene similarity method, this approach uses a simpler, iterative clustering logic. It finds all terms with a similarity greater than the `similarity_threshold` to form clusters.
3.  **Representative Selection**: For each cluster, the term with the greatest height is chosen as the representative. This prioritizes terms that are higher up in the ontology. Ties are broken alphabetically.
4.  **Collapsing**: All other terms in the cluster are collapsed, and the ontology is rebuilt.

---

## Other Notable Methods

-   `_get_annotated_name(term)`: A helper to return a formatted string of a term's name and its annotation if available, used for verbose logging.
-   `rename_systems_with_llm(...)`: An advanced feature that uses a Large Language Model (OpenAI's GPT or Google's Gemini) to suggest new, more informative names for systems based on their gene content and a target phenotype. This can greatly improve the interpretability of a simplified ontology.

## Usage Example

```python
# Initialize the parser
tree = TreeParser('data/ontology.tsv', sys_annot_file='data/annotations.tsv')

print(f"Original number of systems: {tree.n_systems}")

# Collapse using gene similarity
tree.collapse_by_gene_similarity(similarity_threshold=0.8, verbose=True, inplace=True)

print(f"Number of systems after collapsing: {tree.n_systems}")

# Save the simplified ontology
tree.save_ontology('output/collapsed_ontology.tsv')
```

---

## `SNPTreeParser`

The `SNPTreeParser` class, located in `src/utils/tree/snp_tree.py`, extends `TreeParser` to integrate Single Nucleotide Polymorphism (SNP) data. It is designed to build a unified hierarchy that connects SNPs to genes and genes to systems, making it a powerful tool for genomic analyses that require an ontology-based approach (e.g., gene set enrichment analysis for GWAS results).

### Initialization

In addition to the `ontology` and `sys_annot_file` arguments from `TreeParser`, `SNPTreeParser` requires a `snp2gene` file.

```python
from snp_tree import SNPTreeParser

# Path to the ontology file
ontology_file = 'path/to/ontology.tsv'
# Path to the SNP-to-gene mapping file (tab-separated: snp, gene, chr)
snp2gene_file = 'path/to/snp2gene.tsv'

# Initialize the parser
snp_tree = SNPTreeParser(ontology_file, snp2gene_file)
```

The `init_ontology_with_snp` method orchestrates the initialization:
1.  **Initializes Parent**: It first calls the parent `TreeParser.init_ontology` method to build the gene-system hierarchy.
2.  **Loads SNP Data**: It reads the `snp2gene` mapping file.
3.  **Creates SNP Mappings**: It builds dictionaries to map SNPs to unique integer indices (`snp2ind`) and vice-versa.
4.  **Builds Masks**: It creates a `snp2gene_mask`, a matrix that represents the connections between all SNPs and all genes in the ontology.

### Key SNP-related Attributes

-   `snp2gene_df`: A pandas DataFrame containing the SNP-to-gene mappings.
-   `snp2ind` / `ind2snp`: Dictionaries for mapping SNP IDs to/from integer indices.
-   `n_snps`: The total number of unique SNPs.
-   `snp2gene`: A dictionary mapping each SNP to a list of genes it's associated with.
-   `gene2snp`: A dictionary mapping each gene to a list of associated SNPs.
-   `sys2snp`: A dictionary mapping each system to a list of all SNPs associated with it (via its genes).

### Usage Example

```python
# Initialize the SNP-aware parser
snp_tree = SNPTreeParser(
    'data/ontology.tsv',
    'data/snp2gene.tsv',
    sys_annot_file='data/annotations.tsv'
)

print(f"Initialized with {snp_tree.n_snps} SNPs.")

# Get all SNPs associated with a specific system
target_system = 'GO:0008150' # biological_process
associated_snps = snp_tree.sys2snp.get(target_system, [])

print(f"Found {len(associated_snps)} SNPs linked to '{target_system}'.")

# Retain only a specific list of SNPs and rebuild the ontology
snps_to_keep = ['rs123', 'rs456']
snp_tree.retain_snps(snps_to_keep, inplace=True)

print(f"Ontology now contains {snp_tree.n_snps} SNPs and {snp_tree.n_genes} genes.")
```

