Tree
====

Overview
--------

The Tree utilities parse hierarchical ontologies of systems (terms) and genes,
and optionally link SNPs to genes. They are used throughout the project to
build attention masks and define structured inputs for epistasis analysis and
SNP2P datasets.

Usage and examples
------------------

TreeParser
~~~~~~~~~~

The ontology input should provide parent/child relationships between systems
(terms) and genes. It can be supplied as a pandas DataFrame or as a path to a
tabular file with ``parent``, ``child``, and ``interaction`` columns (for
example ``is_a`` or ``gene``). Once loaded, you can inspect the structure or
collapse small terms.

.. code-block:: python

   import pandas as pd
   from g2pt.tree import TreeParser

   # Minimal parent/child ontology with interaction types.
   ontology_df = pd.DataFrame(
       {
           "parent": ["immune_system", "immune_system", "adaptive_immunity"],
           "child": ["innate_immunity", "adaptive_immunity", "IL7R"],
           "interaction": ["is_a", "is_a", "gene"],
       }
   )

   tree = TreeParser(ontology_df)
   tree.summary()
   collapsed_tree = tree.collapse(min_term_size=2)

SNPTreeParser
~~~~~~~~~~~~~

The SNP mapping file is expected to include at least ``snp`` and ``gene``
columns (optionally ``chr`` if you plan to use ``by_chr=True``). You can pass
either file paths or pandas DataFrames.

.. code-block:: python

   from g2pt.tree import SNPTreeParser

   tree_parser = SNPTreeParser(
       ontology="ontology.tsv",
       snp2gene="snp2gene.tsv",
       by_chr=True,
   )
   tree_parser.summary()

API documentation
-----------------

.. class:: TreeParser

   Parses and represents a hierarchical ontology of systems and genes.

   This class loads an ontology from a file or DataFrame, builds a graph
   representation, and provides methods for manipulating and analyzing the
   ontology.

   .. method:: __init__(ontology, dense_attention=False, sys_annot_file=None)

      Initializes the TreeParser.

      :param ontology: A pandas DataFrame or path to a file containing the ontology.
      :type ontology: pandas.DataFrame or str
      :param dense_attention: Whether to use dense attention.
      :type dense_attention: bool, optional
      :param sys_annot_file: Path to a file containing system annotations.
      :type sys_annot_file: str, optional

   .. method:: from_obo(obo_path, dense_attention=False)

      Create a TreeParser instance from an OBO file.

      :param obo_path: Path to the OBO file.
      :type obo_path: str
      :param dense_attention: Whether to use dense attention.
      :type dense_attention: bool, optional

   .. method:: init_ontology(ontology_df, inplace=True, verbose=True)

      Initializes the ontology from a DataFrame.

      :param ontology_df: A pandas DataFrame containing the ontology.
      :type ontology_df: pandas.DataFrame
      :param inplace: Whether to modify the object in place.
      :type inplace: bool, optional
      :param verbose: Whether to print progress messages.
      :type verbose: bool, optional

   .. method:: build_mask(ordered_query, ordered_key, query2key_dict, interaction_value=0, mask_value=-10**4)

      Builds a mask for attention.

      :param ordered_query: A list of query items.
      :type ordered_query: list
      :param ordered_key: A list of key items.
      :type ordered_key: list
      :param query2key_dict: A dictionary mapping query items to key items.
      :type query2key_dict: dict
      :param interaction_value: The value to use for interactions.
      :type interaction_value: int, optional
      :param mask_value: The value to use for non-interactions.
      :type mask_value: int, optional
      :return: A tuple containing the query-to-index mapping, the index-to-query mapping, the key-to-index mapping, the index-to-key mapping, and the mask.
      :rtype: tuple

   .. method:: summary(system=True, gene=True)

      Print a summary of the systems and genes in the ontology.

      :param system: Whether to include system summary.
      :type system: bool, optional
      :param gene: Whether to include gene summary.
      :type gene: bool, optional

   .. method:: collapse(to_keep=None, min_term_size=2, verbose=True, inplace=False)

      Collapses the ontology by removing small terms.

      :param to_keep: A list of terms to keep, even if they are small.
      :type to_keep: list, optional
      :param min_term_size: The minimum number of genes a term must have to be kept.
      :type min_term_size: int, optional
      :param verbose: Whether to print progress messages.
      :type verbose: bool, optional
      :param inplace: Whether to modify the object in place.
      :type inplace: bool, optional


.. class:: SNPTreeParser

   Parses SNP→gene mappings alongside the system ontology.

   This class extends ``TreeParser`` by wiring SNPs into the ontology so downstream
   datasets can emit SNP, gene, and system indices. Provide the same parent/child
   ontology used for ``TreeParser`` plus a SNP→gene mapping table.

   .. method:: __init__(ontology, snp2gene, dense_attention=False, sys_annot_file=None, by_chr=False, multiple_phenotypes=False, block_bias=False)

      :param ontology: path or DataFrame for parent–child ontology
      :type ontology: str or pandas.DataFrame
      :param snp2gene: path or DataFrame for SNP→gene mapping
      :type snp2gene: str or pandas.DataFrame
      :param dense_attention: Whether to use dense attention.
      :type dense_attention: bool, optional
      :param sys_annot_file: Path to a file containing system annotations.
      :type sys_annot_file: str, optional
      :param by_chr: Whether to process by chromosome.
      :type by_chr: bool, optional
      :param multiple_phenotypes: Whether to handle multiple phenotypes.
      :type multiple_phenotypes: bool, optional
      :param block_bias: Whether to use block bias.
      :type block_bias: bool, optional

   .. method:: init_ontology_with_snp(ontology_df, snp2gene, inplace=True, multiple_phenotypes=False, verbose=True)

      Extend TreeParser.init_ontology by also loading and wiring the SNP→gene table (snp2gene).

      :param ontology_df: A pandas DataFrame containing the ontology.
      :type ontology_df: pandas.DataFrame
      :param snp2gene: path or DataFrame for SNP→gene mapping
      :type snp2gene: str or pandas.DataFrame
      :param inplace: Whether to modify the object in place.
      :type inplace: bool, optional
      :param multiple_phenotypes: Whether to handle multiple phenotypes.
      :type multiple_phenotypes: bool, optional
      :param verbose: Whether to print progress messages.
      :type verbose: bool, optional
