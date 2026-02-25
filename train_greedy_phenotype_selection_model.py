import argparse
import os
import subprocess
import socket
import warnings

import pandas as pd
import mlflow
import re

import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

def extract_info_from_bfile_path(bfile_path):
    p_value = "N/A"
    fold = "N/A"

    # Regex to find pval_X and fold_Y
    p_value_match = re.search(r'pval_([0-9eE.-]+)', bfile_path)
    if p_value_match:
        p_value = p_value_match.group(1)

    fold_match = re.search(r'fold_([0-9]+|total)', bfile_path)
    if fold_match:
        fold = fold_match.group(1)

    return p_value, fold


from prettytable import PrettyTable
from src.utils.chunker import MaskBasedChunker, AttentionAwareChunker

from src.model.model.snp2phenotype import SNP2PhenotypeModel

from torch.utils.data.distributed import DistributedSampler
from src.utils.data.dataset import SNP2PCollator, PLINKDataset, TSVDataset, PhenotypeSelectionDataset, PhenotypeSelectionDatasetDDP
from src.utils.tree import SNPTreeParser
from src.utils.trainer import GreedyMultiplePhenotypeTrainer
from src.utils.config.data_config import create_dataset_config
from src.utils.config.model_config import ModelConfig
from datetime import timedelta

from torch.utils.data.dataloader import DataLoader

_DATASET = None


class SNP2PDatasetFactory:
    @staticmethod
    def create_dataset(
        tree_parser,
        dataset_path,
        dataset_kind,
        dataset_config,
        cov_path=None,
        pheno_path=None,
        cov_mean_dict=None,
        cov_std_dict=None,
    ):
        if not dataset_path:
            return None

        if dataset_kind == "tsv":
            dataset_cls = TSVDataset
        else:
            dataset_cls = PLINKDataset

        return dataset_cls(
            tree_parser,
            dataset_path,
            cov_path,
            pheno_path,
            cov_mean_dict=cov_mean_dict,
            cov_std_dict=cov_std_dict,
            flip=dataset_config.flip,
            input_format=dataset_config.input_format,
            cov_ids=dataset_config.cov_ids,
            pheno_ids=dataset_config.pheno_ids,
            bt=dataset_config.bt,
            qt=dataset_config.qt
        )


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters", "Trainable"])
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params, parameter.requires_grad])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def rank0_print(*args, rank=0, **kwargs):
    """Print only on rank 0 to avoid cluttered logs in distributed training."""
    if rank == 0:
        print(*args, **kwargs)


def ddp_setup():
    """
    Return (distributed, rank, world_size, local_rank, device).
    Works for:
        • plain `python my_train.py`           → distributed=False
        • 1-GPU: `python my_train.py --cuda 0` → distributed=False
        • multi-GPU / single-node:
                      torchrun --nproc_per_node=N my_train.py
          → distributed=True  (DDP across N GPUs)
    """
    # ❶ Detect whether torchrun (or mpirun) has set these
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank        = int(os.environ.get("RANK",        0))
    local_rank  = int(os.environ.get("LOCAL_RANK",  0))

    distributed = world_size > 1
    # ❷ Choose the device for this rank
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    print("DDP setup done")
    return distributed, rank, world_size, local_rank, device


def main():
    parser = argparse.ArgumentParser(description='Some beautiful description')
    # Participant Genotype file
    parser.add_argument('--genotype-csv', help='Personal genotype file', type=str, default=None)
    parser.add_argument('--n_cov', help='The number of covariates', type=int, default=4)
    # Hierarchy files
    parser.add_argument('--onto', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--snp2gene', help='SNP to gene mapping file', type=str)
    parser.add_argument('--interaction-types', help='Subtree cascading order', nargs='+', default=['default'])

    # Train bfile/tsv format
    parser.add_argument('--train-bfile', help='Training genotype dataset', type=str, default=None)
    parser.add_argument('--train-tsv', help='Path to directory with training genotype, covariate, and phenotype TSV files', type=str, default=None)
    parser.add_argument('--train-cov', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--train-pheno', help='Training phenotype dataset', type=str, default=None)
    parser.add_argument('--val-bfile', help='Validation dataset', type=str, default=None)
    parser.add_argument('--val-tsv', help='Path to validation genotype TSV file', type=str, default=None)
    parser.add_argument('--val-cov', help='Validation covariates dataset', type=str, default=None)
    parser.add_argument('--val-pheno', help='Validation phenotype dataset', type=str, default=None)
    parser.add_argument('--test-bfile', help='Test dataset', type=str, default=None)
    parser.add_argument('--test-cov', help='Test covariates dataset', type=str, default=None)
    parser.add_argument('--test-pheno', help='Test phenotype dataset', type=str, default=None)
    parser.add_argument('--input-format', help='input format', type=str, default='plink')

    parser.add_argument('--cov-ids', nargs='*', default=[])
    parser.add_argument('--flip', action='store_true', default=False)
    parser.add_argument('--pheno-ids', nargs='*', default=[])
    parser.add_argument('--bt', nargs='*', default=[])
    parser.add_argument('--qt', nargs='*', default=[])
    parser.add_argument('--target-phenotype', type=str, )
    parser.add_argument('--block-bias', action='store_true')
    parser.add_argument('--subsample', help='Number of individuals to subsample for training', type=int, default=None)
    # Propagation option
    parser.add_argument('--cov-effect', default='pre')
    parser.add_argument('--sys2env', action='store_true', default=False)
    parser.add_argument('--env2sys', action='store_true', default=False)
    parser.add_argument('--sys2gene', action='store_true', default=False)
    parser.add_argument('--sys2pheno', action='store_true', default=True)
    parser.add_argument('--gene2pheno', action='store_true', default=False)
    parser.add_argument('--snp2pheno', action='store_true', default=False)
    parser.add_argument('--use-ld-block-bias', action='store_true', default=False, help='Use LD block information to create an attention bias.')

    parser.add_argument('--dynamic-phenotype-sampling', action='store_true', default=False)

    parser.add_argument('--poincare', action='store_true', default=False)

    parser.add_argument('--dense-attention', action='store_true', default=False)
    parser.add_argument('--use-sparse-attention', type=lambda x: x.lower() != 'false', default=True,
                        help='Use sparse edge-based attention (100-1000× memory savings). Default: True. Pass "False" to disable.')
    parser.add_argument('--mlm', action='store_true', default=False, help="Enable Masked Language Model training for SNP prediction")
    parser.add_argument('--regression', action='store_true', default=False)
    # Model parameters
    parser.add_argument('--hidden-dims', help='hidden dimension for model', default=256, type=int)
    parser.add_argument('--n-heads', help='the number of head in genetic propagation', default=4, type=int)
    parser.add_argument('--prediction-head', help='Number of prediction heads', type=int, default=1)
    # Training parameters
    parser.add_argument('--epochs', help='Training epochs for training', type=int, default=300)
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--wd', help='Weight decay', type=float, default=0.001)
    parser.add_argument('--z-weight', help='Sampling weight', type=float, default=1.0)
    parser.add_argument('--dropout', help='dropout ratio', type=float, default=0.2)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=128)
    parser.add_argument('--val-step', help='Validation step', type=int, default=20)
    parser.add_argument('--patience', help='Patience for early stopping', type=int, default=10)
    parser.add_argument('--jobs', help="The number of threads", type=int, default=0)

    parser.add_argument('--loss', help='loss function', type=str, default='default')
    parser.add_argument('--focal-loss-alpha', help='alpha for focal loss', type=float, default=0.25)
    parser.add_argument('--focal-loss-gamma', help='gamma for focal loss', type=float, default=2.0)
    parser.add_argument('--label_smoothing', help='Label smoothing for BCE loss', type=float, default=0.0)

    parser.add_argument('--use_hierarchical_transformer', action='store_true', default=False)
    parser.add_argument('--use_moe', action='store_true', default=False, help='Use Mixture-of-Experts predictor head')
    parser.add_argument('--independent_predictors', action='store_true', default=False, help='Use independent predictor heads for each phenotype')

    # Gating arguments
    parser.add_argument('--use-gating', type=lambda x: x.lower() == 'true', default=False,
                        help='Enable phenotype-conditioned gating (default: False)')
    parser.add_argument('--gate-hidden-dim', type=int, default=64,
                        help='Hidden dimension for gate MLPs (default: 64)')
    parser.add_argument('--gate-temperature', type=float, default=1.0,
                        help='Temperature for gate softmax (default: 1.0)')
    parser.add_argument('--gate-sparsity-weight', type=float, default=0.01,
                        help='Weight for gate sparsity loss (default: 0.01)')
    parser.add_argument('--gate-entropy-weight', type=float, default=0.001,
                        help='Weight for gate entropy loss (default: 0.001)')
    parser.add_argument('--gate-coherence-weight', type=float, default=0.0,
                        help='Weight for pathway coherence loss (default: 0.0)')

    parser.add_argument('--pretrained', type=str, default=None)

    # GPU option
    parser.add_argument('--cuda', help='Specify GPU', type=int, default=None)

    # Model input and output
    parser.add_argument('--model', help='path to trained model', default=None)
    parser.add_argument('--out', help="output model path")
    parser.add_argument('--use_mlflow', action='store_true', default=False)

    args = parser.parse_args()
    print('Start Process')
    if args.cuda is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node

    distributed, rank, world_size, local_rank, device = ddp_setup()

    args.distributed = distributed
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank

    torch.cuda.set_device(args.local_rank)
    if args.use_mlflow:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    main_worker(args)  # no mp.spawn!


def main_worker(args):
    gpu = args.local_rank

    global _DATASET

    node_name = socket.gethostname()

    rank0_print(f"[{args.rank}/{args.world_size}] running on {node_name} GPU {gpu}, rank: {args.rank}, local_rank: {args.local_rank}", rank=args.rank, flush=True)
    if args.distributed and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(hours=3),
        )
    rank0_print("Finish setup main worker", args.rank, rank=args.rank)

    if args.rank == 0 and args.use_mlflow:
        p_value, fold = extract_info_from_bfile_path(args.train_bfile or "")
        run_name = f"GreedySelection_{args.target_phenotype}_Pval_{p_value}_Fold_{fold}"
        mlflow.start_run(run_name=run_name)
        mlflow.log_params(vars(args))

    if args.distributed:
        device = torch.device("cuda:%d" % gpu)
    elif args.cuda is not None:
        device = torch.device("cuda:%d" % args.cuda)
    elif args.world_size == 1 and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if (len(args.qt) + len(args.bt) > 1):
        multiple_phenotypes = True
    else:
        multiple_phenotypes = False

    model_config = ModelConfig.from_namespace(args)
    dataset_config = create_dataset_config(args)

    tree_parser = SNPTreeParser(model_config.onto, model_config.snp2gene,
                                dense_attention=model_config.dense_attention,
                                multiple_phenotypes=multiple_phenotypes,
                                block_bias=model_config.block_bias,
                                precompute_edges=model_config.use_sparse_attention)
    args.block = hasattr(tree_parser, 'blocks')
    fix_system = False

    if args.train_tsv:
        rank0_print("Loading TSV data from %s" % args.train_tsv, rank=args.rank)
    else:
        rank0_print("Loading PLINK bfile... at %s" % args.train_bfile, rank=args.rank)

    train_dataset_path = args.train_tsv or args.train_bfile
    train_dataset_kind = "tsv" if args.train_tsv else "plink"
    snp2p_dataset = SNP2PDatasetFactory.create_dataset(
        tree_parser,
        train_dataset_path,
        train_dataset_kind,
        dataset_config,
        cov_path=dataset_config.train_cov,
        pheno_path=dataset_config.train_pheno,
    )
    if dataset_config.subsample:
        snp2p_dataset.sample_population(n=dataset_config.subsample)

    args.bt_inds = snp2p_dataset.bt_inds
    args.qt_inds = snp2p_dataset.qt_inds
    args.bt = snp2p_dataset.bt
    args.qt = snp2p_dataset.qt
    args.pheno_ids = snp2p_dataset.pheno_ids
    args.pheno2ind = snp2p_dataset.pheno2ind
    args.ind2pheno = snp2p_dataset.ind2pheno
    args.pheno2type = snp2p_dataset.pheno2type

    args.cov_mean_dict = snp2p_dataset.cov_mean_dict
    args.cov_std_dict = snp2p_dataset.cov_std_dict
    rank0_print("Loading done...", rank=args.rank)

    snp2p_collator = SNP2PCollator(tree_parser, input_format=dataset_config.input_format, mlm=args.mlm)

    rank0_print("Summary of trainable parameters", rank=args.rank)
    if model_config.sys2env:
        rank0_print("Model will use Sys2Env", rank=args.rank)
    if model_config.env2sys:
        rank0_print("Model will use Env2Sys", rank=args.rank)
    if model_config.sys2gene:
        rank0_print("Model will use Sys2Gene", rank=args.rank)

    if args.model is not None:
        snp2p_model_dict = torch.load(args.model, map_location=device)
        rank0_print(args.model, 'loaded', rank=args.rank)
        snp2p_model = SNP2PhenotypeModel(tree_parser, model_config.hidden_dims,
                                         sys2pheno=model_config.sys2pheno, gene2pheno=model_config.gene2pheno, snp2pheno=model_config.snp2pheno,
                                         interaction_types=model_config.interaction_types,
                                         dropout=model_config.dropout, n_covariates=snp2p_dataset.n_cov,
                                         phenotypes=snp2p_dataset.pheno_ids,
                                         ind2pheno=snp2p_dataset.ind2pheno,
                                         activation='softmax', input_format=dataset_config.input_format,
                                         cov_effect=model_config.cov_effect,
                                         use_hierarchical_transformer=model_config.use_hierarchical_transformer,
                                         n_heads=model_config.n_heads,
                                         use_moe=model_config.use_moe, use_independent_predictors=model_config.independent_predictors,
                                         prediction_head=model_config.prediction_head,
                                         use_sparse_attention=model_config.use_sparse_attention)
        rank0_print(args.model, 'initialized', rank=args.rank)
        snp2p_model.load_state_dict(snp2p_model_dict['state_dict'])
        if args.model.split('.')[-1].isdigit():
            args.start_epoch = int(args.model.split('.')[-1])
        else:
            args.start_epoch = 0

    else:
        snp2p_model = SNP2PhenotypeModel(tree_parser, model_config.hidden_dims,
                                         sys2pheno=model_config.sys2pheno, gene2pheno=model_config.gene2pheno, snp2pheno=model_config.snp2pheno,
                                         interaction_types=model_config.interaction_types,
                                         dropout=model_config.dropout, n_covariates=snp2p_dataset.n_cov,
                                         activation='softmax', input_format=dataset_config.input_format,
                                         phenotypes=snp2p_dataset.pheno_ids,
                                         ind2pheno=snp2p_dataset.ind2pheno,
                                         n_heads=model_config.n_heads,
                                         cov_effect=model_config.cov_effect,
                                         use_hierarchical_transformer=model_config.use_hierarchical_transformer,
                                         use_moe=model_config.use_moe, use_independent_predictors=model_config.independent_predictors,
                                         prediction_head=model_config.prediction_head,
                                         use_sparse_attention=model_config.use_sparse_attention)
        args.start_epoch = 0

    if args.distributed:
        rank0_print("Distributed trainings are set up", rank=args.rank)
        args.jobs = int((args.jobs) / args.world_size)
        snp2p_model = snp2p_model.to(device)
        snp2p_model = torch.nn.parallel.DistributedDataParallel(snp2p_model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
    elif args.cuda is not None:
        snp2p_model = snp2p_model.to(device)
        rank0_print("Model is loaded at GPU(%d)" % args.cuda, rank=args.rank)
    else:
        rank0_print("Model is on cpu (not recommended)", rank=args.rank)

    if not args.distributed or (args.distributed and args.rank == 0):
        rank0_print("Summary of trainable parameters", rank=args.rank)
        count_parameters(snp2p_model)

    if args.distributed:
        dataset = PhenotypeSelectionDatasetDDP(tree_parser, snp2p_dataset, snp2p_collator, args.batch_size,
                                                          rank=args.rank,
                                                          world_size=args.world_size,
                                                          shuffle=True)
        snp2p_dataloader = DataLoader(dataset, batch_size=None,
                                      num_workers=args.jobs,
                                      prefetch_factor=2,
                                      persistent_workers=True,
                                      pin_memory=True)

    else:
        dataset = PhenotypeSelectionDataset(tree_parser, snp2p_dataset, snp2p_collator, args.batch_size, shuffle=True)
        snp2p_dataloader = DataLoader(dataset, batch_size=None,
                                      num_workers=args.jobs,
                                      prefetch_factor=2,
                                      persistent_workers=True,
                                      pin_memory=True)

    if args.val_tsv:
        rank0_print("Loading validation TSV data from %s" % args.val_tsv, rank=args.rank)
    elif args.val_bfile is not None:
        rank0_print("Loading validation PLINK bfile... at %s" % args.val_bfile, rank=args.rank)

    val_dataset_path = args.val_tsv or args.val_bfile
    val_dataset_kind = "tsv" if args.val_tsv else "plink"
    val_snp2p_dataset = SNP2PDatasetFactory.create_dataset(
        tree_parser,
        val_dataset_path,
        val_dataset_kind,
        dataset_config,
        cov_path=dataset_config.val_cov,
        pheno_path=dataset_config.val_pheno,
        cov_mean_dict=args.cov_mean_dict,
        cov_std_dict=args.cov_std_dict,
    )

    if val_snp2p_dataset is None:
        val_snp2p_dataloader = None
    else:
        val_snp2p_dataset = PhenotypeSelectionDataset(tree_parser, val_snp2p_dataset, snp2p_collator,
                                                      batch_size=args.batch_size, shuffle=False)
        val_snp2p_dataloader = DataLoader(val_snp2p_dataset, batch_size=None,
                                          num_workers=args.jobs,
                                          prefetch_factor=2,
                                          persistent_workers=True,
                                          pin_memory=True)

    snp2p_trainer = GreedyMultiplePhenotypeTrainer(snp2p_model, tree_parser, snp2p_dataloader, device, args, args.target_phenotype,
                                 validation_dataloader=val_snp2p_dataloader, fix_system=fix_system, pretrained_checkpoint=args.pretrained, use_mlflow=args.use_mlflow)
    skip_initial_training = args.pretrained is not None
    snp2p_trainer.greedy_phenotype_selection(skip_initial_training=skip_initial_training)

    if args.rank == 0 and args.use_mlflow:
        mlflow.end_run()

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    print("Python __main__", flush=True)
    main()
