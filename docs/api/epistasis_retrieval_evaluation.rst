Epistasis Retrieval Evaluation
==============================

.. class:: EvaluationConfig

   Configuration container for epistasis retrieval evaluation inputs and settings.

   :param causal_info: Path to the JSON file containing causal SNP and epistasis information.
   :type causal_info: str
   :param system_importance: Path to the system importance CSV file.
   :type system_importance: str
   :param attention_results: Path to the attention results CSV file.
   :type attention_results: str
   :param tsv: Path to the genotype TSV file.
   :type tsv: str
   :param pheno: Path to the phenotype file.
   :type pheno: str
   :param cov: Path to the covariate file.
   :type cov: str
   :param onto: Path to the ontology file.
   :type onto: str
   :param snp2gene: Path to the SNP-to-gene mapping file.
   :type snp2gene: str
   :param top_n_systems: Number of top-ranked systems to evaluate.
   :type top_n_systems: int
   :param output_prefix: Prefix used when writing summary and p-value outputs.
   :type output_prefix: str
   :param num_workers: Number of workers for parallel processing.
   :type num_workers: int
   :param executor_type: Execution backend (``process`` or ``thread``).
   :type executor_type: str
   :param quantiles: Quantiles used when filtering attention scores.
   :type quantiles: Sequence[float]
   :param snp_threshold: Optional SNP count threshold for skipping large systems.
   :type snp_threshold: int

.. class:: EpistasisRetrievalEvaluator

   Coordinates loading inputs, running diagnostic checks, parallel epistasis searches,
   and exporting evaluation metrics.

   .. method:: __init__(config)

      Stores the evaluation configuration.

      :param config: Parsed evaluation configuration.
      :type config: EvaluationConfig

   .. method:: evaluate()

      Executes the end-to-end evaluation, including metrics and curve output.

.. function:: build_arg_parser()

   Builds the CLI argument parser for the evaluation workflow.

.. function:: build_config_from_args(args)

   Converts CLI arguments into an :class:`EvaluationConfig`.

   :param args: Parsed command-line arguments.
   :type args: argparse.Namespace

.. function:: main()

   CLI entry point that constructs the configuration and runs the evaluator.
