SNP2P Trainer
=============

This page documents the SNP2P training utilities, including helper losses and
the main trainer class. The trainer orchestrates optimization, evaluation, and
metric reporting for mixed phenotype types (quantitative and binary).

Example: initialize and run training
------------------------------------

.. code-block:: python

   from torch.utils.data import DataLoader
   from src.utils.data.dataset.SNP2PDataset import SNP2PCollator, PLINKDataset
   from src.utils.trainer.snp2p_trainer import SNP2PTrainer

   dataset = PLINKDataset(tree_parser, bfile="data/geno/plink_prefix", cov="cov.tsv", pheno="pheno.tsv")
   loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=SNP2PCollator(tree_parser))

   trainer = SNP2PTrainer(
       snp2p_model=model,
       tree_parser=tree_parser,
       snp2p_dataloader=loader,
       device=device,
       args=args,
       target_phenotype="BMI",
   )
   trainer.train(epochs=10, output_path="checkpoints/snp2p")

Utilities
---------

.. toctree::
   :maxdepth: 1

   snp2p_trainer_utils
   snp2p_model_utils

.. function:: correlation_matching_loss(pred, target, lam=0.05)

   Computes a correlation-structure matching loss between predictions and labels.

   :param pred: Model predictions (``[B, P]``).
   :type pred: torch.Tensor
   :param target: Target labels (``[B, P]``).
   :type target: torch.Tensor
   :param lam: Scaling factor for the loss.
   :type lam: float, optional
   :return: Scalar loss value.
   :rtype: torch.Tensor


.. function:: linear_temperature_schedule(epoch, total_epochs, T_init=1.0, T_final=0.1)

   Linear temperature schedule used for annealing.

   :param epoch: Current epoch.
   :type epoch: int
   :param total_epochs: Total training epochs.
   :type total_epochs: int
   :param T_init: Initial temperature.
   :type T_init: float, optional
   :param T_final: Final temperature.
   :type T_final: float, optional
   :return: Temperature for the epoch.
   :rtype: float


.. function:: get_param_groups(model, base_lr)

   Split parameters into base and LoRA groups with different learning rates.

   :param model: Model with named parameters.
   :type model: torch.nn.Module
   :param base_lr: Base learning rate.
   :type base_lr: float
   :return: Parameter groups for an optimizer.
   :rtype: list


.. class:: SNP2PTrainer

   Trainer for SNP2P models with optional validation and MLflow logging.

   :param snp2p_model: Model instance to train.
   :type snp2p_model: torch.nn.Module
   :param tree_parser: Parsed SNP ontology and masks.
   :type tree_parser: SNPTreeParser
   :param snp2p_dataloader: Training dataloader.
   :type snp2p_dataloader: torch.utils.data.DataLoader
   :param device: Device for model and tensors.
   :type device: torch.device
   :param args: Training configuration namespace.
   :type args: argparse.Namespace
   :param target_phenotype: Target phenotype name or ID for logging.
   :type target_phenotype: str
   :param validation_dataloader: Optional validation dataloader.
   :type validation_dataloader: torch.utils.data.DataLoader, optional
   :param fix_system: Whether to freeze system embeddings.
   :type fix_system: bool, optional
   :param pretrain_dataloader: Optional pretraining dataloader.
   :type pretrain_dataloader: torch.utils.data.DataLoader, optional
   :param label_smoothing: Label smoothing for phenotype loss.
   :type label_smoothing: float, optional
   :param use_mlflow: Whether to log artifacts to MLflow.
   :type use_mlflow: bool, optional

   .. method:: train(epochs, output_path=None)

      Run the training loop for the given number of epochs.

      :param epochs: Number of epochs to train.
      :type epochs: int
      :param output_path: Optional checkpoint path prefix.
      :type output_path: str, optional

   .. method:: evaluate(model, dataloader, epoch, phenotypes, name="Validation", print_importance=False, snp_only=False)

      Evaluate a model on a dataloader and compute phenotype metrics.

      :param model: Model to evaluate.
      :type model: torch.nn.Module
      :param dataloader: Dataloader to iterate over.
      :type dataloader: torch.utils.data.DataLoader
      :param epoch: Epoch index for logging.
      :type epoch: int
      :param phenotypes: Phenotype IDs to evaluate.
      :type phenotypes: list
      :param name: Label used for logging output.
      :type name: str, optional
      :param print_importance: Whether to print attention importance scores.
      :type print_importance: bool, optional
      :param snp_only: Whether to evaluate SNP-only prediction.
      :type snp_only: bool, optional
      :return: Aggregate performance score.
      :rtype: float

   .. method:: evaluate_continuous_phenotype(trues, results, covariates=None, phenotype_name="", epoch=0, rank=0)

      Compute regression metrics for continuous phenotypes.

      :param trues: Ground-truth values.
      :type trues: numpy.ndarray
      :param results: Model predictions.
      :type results: numpy.ndarray
      :param covariates: Optional covariates for logging.
      :type covariates: numpy.ndarray, optional
      :param phenotype_name: Phenotype name for logging.
      :type phenotype_name: str, optional
      :param epoch: Epoch index for logging.
      :type epoch: int, optional
      :param rank: Distributed rank for gating output.
      :type rank: int, optional

   .. method:: evaluate_binary_phenotype(trues, results, covariates=None, phenotype_name="", epoch=0, rank=0)

      Compute classification metrics for binary phenotypes.

      :param trues: Ground-truth labels.
      :type trues: numpy.ndarray
      :param results: Model predictions.
      :type results: numpy.ndarray
      :param covariates: Optional covariates for logging.
      :type covariates: numpy.ndarray, optional
      :param phenotype_name: Phenotype name for logging.
      :type phenotype_name: str, optional
      :param epoch: Epoch index for logging.
      :type epoch: int, optional
      :param rank: Distributed rank for gating output.
      :type rank: int, optional

   .. method:: train_epoch(epoch, ccc=False, sex=False)

      Train for a single epoch over the training dataloader.

      :param epoch: Epoch index.
      :type epoch: int
      :param ccc: Whether to compute concordance correlation coefficient loss.
      :type ccc: bool, optional
      :param sex: Whether to include sex-specific logic.
      :type sex: bool, optional

   .. method:: iter_minibatches(model, dataloader, optimizer, epoch, name="", snp_only=False, sex=False)

      Iterate over minibatches and update model parameters.

      :param model: Model to train.
      :type model: torch.nn.Module
      :param dataloader: Data loader to iterate.
      :type dataloader: torch.utils.data.DataLoader
      :param optimizer: Optimizer for parameter updates.
      :type optimizer: torch.optim.Optimizer
      :param epoch: Epoch index.
      :type epoch: int
      :param name: Label for progress logging.
      :type name: str, optional
      :param snp_only: Whether to train on SNP-only inputs.
      :type snp_only: bool, optional
      :param sex: Whether to include sex-specific logic.
      :type sex: bool, optional
