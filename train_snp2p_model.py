import argparse
import os
import subprocess
import socket
import warnings

import pandas as pd

import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from prettytable import PrettyTable

from src.model.model.snp2phenotype import SNP2PhenotypeModel

from src.utils.data.dataset import SNP2PCollator, CohortSampler, DistributedCohortSampler, BinaryCohortSampler, DistributedBinaryCohortSampler, PLINKDataset
from src.utils.tree import SNPTreeParser
from src.utils.trainer import SNP2PTrainer
from datetime import timedelta

from torch.utils.data.dataloader import DataLoader

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters", "Trainable"])
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        #print(name, params, parameter.requires_grad)
        table.add_row([name, params, parameter.requires_grad])
        if parameter.requires_grad:
            total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def main():
    parser = argparse.ArgumentParser(description='Some beautiful description')

    # Hierarchy files
    parser.add_argument('--onto', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--snp2gene', help='SNP to gene mapping file', type=str)
    parser.add_argument('--subtree-order', help='Subtree cascading order', nargs='+', default=['default'])

    # Participant Genotype file
    parser.add_argument('--train-bfile', help='Training genotype dataset', type=str, default=None)
    parser.add_argument('--train-cov', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--train-pheno', help='Training phenotype dataset', type=str, default=None)
    parser.add_argument('--val-bfile', help='Validation dataset', type=str, default=None)
    parser.add_argument('--val-cov', help='Validation covariates dataset', type=str, default=None)
    parser.add_argument('--val-pheno', help='Training phenotype dataset', type=str, default=None)
    parser.add_argument('--cov-ids', nargs='*', default=[])
    parser.add_argument('--flip', action='store_true', default=False)
    parser.add_argument('--regression', action='store_true', default=False)
    parser.add_argument('--target-phenotype', type=str, default='PHENOTYPE')

    # Propagation option
    parser.add_argument('--sys2env', action='store_true', default=False)
    parser.add_argument('--env2sys', action='store_true', default=False)
    parser.add_argument('--sys2gene', action='store_true', default=False)

    parser.add_argument('--sys2pheno', action='store_true', default=False)
    parser.add_argument('--gene2pheno', action='store_true', default=False)
    parser.add_argument('--snp2pheno', action='store_true', default=False)

    # Additional argument for model training
    parser.add_argument('--poincare', action='store_true', default=False)
    parser.add_argument('--dense-attention', action='store_true', default=False)
    parser.add_argument('--input-format', default='indices', choices=["indices", "binary"])


    # Model parameters
    parser.add_argument('--hidden-dims', help='hidden dimension for model', default=256, type=int)
    # Training parameters
    parser.add_argument('--epochs', help='Training epochs for training', type=int, default=300)
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--wd', help='Weight decay', type=float, default=0.001)
    parser.add_argument('--z-weight', help='Sampling weight', type=float, default=1.0)
    parser.add_argument('--dropout', help='dropout ratio', type=float, default=0.2)
    parser.add_argument('--batch-size', help='Batch size', type=int, default=128)
    parser.add_argument('--val-step', help='Validation step', type=int, default=20)
    parser.add_argument('--jobs', help="The number of threads", type=int, default=0)
    # GPU option
    parser.add_argument('--cuda', help='Specify GPU', type=int, default=None)
    # Multi-GPU option
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local-rank', default=1)
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    # Model input and output
    parser.add_argument('--model', help='path to trained model', default=None)
    parser.add_argument('--out', help="output model path")

    args = parser.parse_args()
    if args.cuda is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        #if 'SLURM_PROCID' in os.environ:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        #    if args.dist_url == "env://" and args.world_size == -1:
        #        args.world_size = int(os.environ["WORLD_SIZE"])
        #        addr = os.environ["MASTER_ADDR"]
        #        port = os.environ["MASTER_PORT"]
        #        print("Address: %s:%s" % (addr, port))
        #else:
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("The world size is %d"%args.world_size)
        mp.spawn(main_worker, nprocs=args.world_size, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.cuda, ngpus_per_node, args)


def main_worker(rank, ngpus_per_node, args):
    global best_acc1
    node_name = socket.gethostname()
    print(f"Initialize main worker {rank} at node {node_name}")

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = rank
            gpu = rank % ngpus_per_node
        print("GPU %d rank is %d" % (gpu, rank))
        timeout = timedelta(hours=5)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank, timeout=timeout)
        print("GPU %d process initialized" % (gpu))
        torch.cuda.empty_cache()
    print("Finish setup main worker", rank)

    if args.distributed:
        device = torch.device("cuda:%d" % gpu)
    elif args.cuda is not None:
        device = torch.device("cuda:%d" % args.cuda)
    else:
        device = torch.device("cpu")

    tree_parser = SNPTreeParser(args.onto, args.snp2gene, dense_attention=args.dense_attention)

    fix_system = False

    print("Loading PLINK bfile... at %s" % args.train_bfile)
    snp2p_dataset = PLINKDataset(tree_parser, args.train_bfile, args.train_cov, args.train_pheno, flip=args.flip, input_format=args.input_format,
                                 cov_ids=args.cov_ids, target_phenotype=args.target_phenotype)
    args.cov_mean_dict = snp2p_dataset.cov_mean_dict
    args.cov_std_dict = snp2p_dataset.cov_std_dict
    print("Loading done...")

    snp2p_collator = SNP2PCollator(tree_parser, input_format=args.input_format)

    print("Summary of trainable parameters")
    if args.sys2env:
        print("Model will use Sys2Env")
    if args.env2sys:
        print("Model will use Env2Sys")
    if args.sys2gene:
        print("Model will use Sys2Gene")
    if args.model is not None:
        snp2p_model_dict = torch.load(args.model, map_location=device)
        print(args.model, 'loaded')
        snp2p_model = SNP2PhenotypeModel(tree_parser, args.hidden_dims,
                                         sys2pheno=args.sys2pheno, gene2pheno=args.gene2pheno, snp2pheno=args.snp2pheno,
                                         subtree_order=args.subtree_order,
                                         dropout=args.dropout, n_covariates=snp2p_dataset.n_cov,
                                         binary=(not args.regression), activation='softmax', input_format=args.input_format)
        print(args.model, 'initialized')
        snp2p_model.load_state_dict(snp2p_model_dict['state_dict'])
        if args.model.split('.')[-1].isdigit():
            args.start_epoch = int(args.model.split('.')[-1])
        else:
            args.start_epoch = 0

    else:
        snp2p_model = SNP2PhenotypeModel(tree_parser, args.hidden_dims,
                                         sys2pheno=args.sys2pheno, gene2pheno=args.gene2pheno, snp2pheno=args.snp2pheno,
                                         subtree_order=args.subtree_order,
                                         dropout=args.dropout, n_covariates=snp2p_dataset.n_cov,
                                         binary=(not args.regression), activation='softmax', input_format=args.input_format,
                                         poincare=args.poincare)
        args.start_epoch = 0

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.cuda is None:
            #torch.cuda.set_device(args.gpu)
            print("Distributed trainings are set up")
            snp2p_model.to(device)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.jobs = int((args.jobs + ngpus_per_node - 1) / ngpus_per_node)
            print(args.batch_size, args.jobs)
            snp2p_model = torch.nn.parallel.DistributedDataParallel(snp2p_model, device_ids=[gpu], find_unused_parameters=True)
        else:
            snp2p_model.to(device)
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            snp2p_model = torch.nn.parallel.DistributedDataParallel(snp2p_model, find_unused_parameters=True)
    elif args.cuda is not None:
        #torch.cuda.set_device(args.gpu)
        snp2p_model = snp2p_model.to(device)
        print("Model is loaded at GPU(%d)" % args.cuda)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        snp2p_model = torch.nn.DataParallel(snp2p_model).to(device)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                     and args.rank % torch.cuda.device_count() == 0):
        print("Summary of trainable parameters")
        count_parameters(snp2p_model)

    if args.distributed:
        if args.regression:
            if args.z_weight == 0:
                snp2p_sampler = None
            else:
                snp2p_sampler = DistributedCohortSampler(snp2p_dataset, num_replicas=args.world_size, rank=args.rank,
                                                     phenotype_col=1, sex_col=2, z_weight=args.z_weight)
            #snp2p_sampler = torch.utils.data.distributed.DistributedSampler(snp2p_dataset)
        else:
            if args.z_weight == 0:
                snp2p_sampler = None
            else:
                snp2p_sampler = DistributedBinaryCohortSampler(snp2p_dataset, num_replicas=args.world_size, rank=args.rank)
        shuffle = False
    else:
        shuffle = False
        if args.regression:
            if args.z_weight == 0:
                snp2p_sampler = None
            else:
                snp2p_sampler = CohortSampler(snp2p_dataset, phenotype_col='PHENOTYPE', sex_col='SEX', z_weight=args.z_weight)
        else:
            if args.z_weight == 0:
                snp2p_sampler = None
            else:
                snp2p_sampler = BinaryCohortSampler(snp2p_dataset)

    snp2p_dataloader = DataLoader(snp2p_dataset, batch_size=args.batch_size, collate_fn=snp2p_collator,
                                  num_workers=args.jobs, shuffle=shuffle, sampler=snp2p_sampler)

    if args.val_bfile is not None:
        val_snp2p_dataset = PLINKDataset(tree_parser, args.val_bfile, args.val_cov, args.val_pheno, cov_mean_dict=args.cov_mean_dict,
                                         cov_std_dict=args.cov_std_dict, flip=args.flip, input_format=args.input_format,
                                         cov_ids=args.cov_ids, target_phenotype=args.target_phenotype)
        val_snp2p_dataloader = DataLoader(val_snp2p_dataset, shuffle=False, batch_size=args.batch_size,
                                          num_workers=args.jobs, collate_fn=snp2p_collator)
    else:
        val_snp2p_dataloader = None

    snp2p_trainer = SNP2PTrainer(snp2p_model, snp2p_dataloader, device, args,
                                 validation_dataloader=val_snp2p_dataloader, fix_system=fix_system)
    snp2p_trainer.train(args.epochs, args.out)

if __name__ == '__main__':
    main()
