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

from src.utils.data.dataset import SNP2PDataset, SNP2PCollator, CohortSampler, DistributedCohortSampler
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
    # Participant Genotype file
    parser.add_argument('--genotype', help='Personal genotype file', type=str)
    # Indexing files
    parser.add_argument('--snp2id', help='SNP to ID mapping file', type=str)
    parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)
    # Hierarchy files
    parser.add_argument('--onto', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--snp2gene', help='SNP to gene mapping file', type=str)
    parser.add_argument('--subtree_order', help='Subtree cascading order', nargs='+', default=['default'])
    # Training, validation, test files
    parser.add_argument('--train', help='Training dataset', type=str)
    parser.add_argument('--val', help='Validation dataset', type=str, default=None)
    parser.add_argument('--test', help='Test dataset', type=str, default=None)
    # Propagation option
    parser.add_argument('--sys2env', action='store_true', default=False)
    parser.add_argument('--env2sys', action='store_true', default=False)
    parser.add_argument('--sys2gene', action='store_true', default=False)
    parser.add_argument('--regression', action='store_true', default=False)
    # Model parameters
    parser.add_argument('--hidden_dims', help='hidden dimension for model', default=256, type=int)
    # Training parameters
    parser.add_argument('--epochs', help='Training epochs for training', type=int, default=300)
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--wd', help='Weight decay', type=float, default=0.001)
    parser.add_argument('--z_weight', help='Sampling weight', type=float, default=1.0)
    parser.add_argument('--dropout', help='dropout ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=128)
    parser.add_argument('--val_step', help='Validation step', type=int, default=20)
    parser.add_argument('--jobs', help="The number of threads", type=int, default=0)
    # GPU option
    parser.add_argument('--cuda', help='Specify GPU', type=int, default=None)
    # Multi-GPU option
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=1)
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

    print("Loading Genotype dataset... at %s"%args.genotype)
    args.genotype = pd.read_csv(args.genotype, index_col=0, sep='\t')#.astype('int32')
    print("Loading done...")

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

    tree_parser = SNPTreeParser(args.onto, args.snp2gene, args.gene2id, args.snp2id)

    if args.model is not None:
        snp2p_model = torch.load(args.model, map_location=device)
    else:
        snp2p_model = SNP2PhenotypeModel(tree_parser, args.hidden_dims, subtree_order=args.subtree_order,
                                         dropout=args.dropout, n_covariates=4,
                                         binary=(not args.regression), activation='softmax')

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.cuda is None:
            #torch.cuda.set_device(args.gpu)
            print("Distributed training are set up")
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

    fix_system = False
    print("Summary of trainable parameters")
    if args.sys2env:
        print("Model will use Sys2Env")
    if args.env2sys:
        print("Model will use Env2Sys")
    if args.sys2gene:
        print("Model will use Sys2Gene")


    train_dataset = pd.read_csv(args.train, header=None, sep='\t')

    snp2p_dataset = SNP2PDataset(train_dataset, args.genotype, tree_parser)
    snp2p_collator = SNP2PCollator(tree_parser)

    if args.distributed:
        if args.regression:
            #snp2p_sampler = DistributedCohortSampler(train_dataset, num_replicas=args.world_size, rank=args.rank,
            #                                         phenotype_col=1, sex_col=2, z_weight=args.z_weight)
            snp2p_sampler = torch.utils.data.distributed.DistributedSampler(snp2p_dataset)
        else:
            snp2p_sampler = torch.utils.data.distributed.DistributedSampler(snp2p_dataset)
        shuffle = False
    else:
        shuffle = False
        if args.regression:
            snp2p_sampler = None#CohortSampler(train_dataset, phenotype_col=1, sex_col=2, z_weight=args.z_weight)
        else:
            snp2p_sampler = None

    snp2p_dataloader = DataLoader(snp2p_dataset, batch_size=args.batch_size, collate_fn=snp2p_collator,
                                  num_workers=args.jobs, shuffle=shuffle, sampler=snp2p_sampler)
    if args.val is not None:
        val_dataset = pd.read_csv(args.val, header=None, sep='\t')
        val_snp2p_dataset = SNP2PDataset(val_dataset, args.genotype, tree_parser, age_mean=snp2p_dataset.age_mean,
                                         age_std=snp2p_dataset.age_std)
        val_snp2p_dataloader = DataLoader(val_snp2p_dataset, shuffle=False, batch_size=args.batch_size,
                                              num_workers=args.jobs, collate_fn=snp2p_collator)
    else:
        val_snp2p_dataloader = None

    drug_response_trainer = SNP2PTrainer(snp2p_model, snp2p_dataloader, device, args,
                                                validation_dataloader=val_snp2p_dataloader, fix_system=fix_system)
    drug_response_trainer.train(args.epochs, args.out)




if __name__ == '__main__':
    main()
