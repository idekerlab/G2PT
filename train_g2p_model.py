import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import pandas as pd

import argparse
import torch
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from prettytable import PrettyTable

from src.model.genotype2phenotype_model import Genotype2PhenotypeModel

from src.utils.data import TreeParser
from src.utils.data.dataset import G2PDataset, G2PCollator
from src.utils.trainer import G2PTrainer
import numpy as np
import torch.nn as nn

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
    parser.add_argument('--onto', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--subtree_order', help='Subtree cascading order', nargs='+', default=['default'])
    parser.add_argument('--train', help='Training dataset', type=str)
    parser.add_argument('--val', help='Validation dataset', type=str, default=None)
    parser.add_argument('--test', help='Test dataset', type=str, default=None)
    parser.add_argument('--system_embedding', default=None)
    parser.add_argument('--gene_embedding', default=None)

    parser.add_argument('--epochs', help='Training epochs for training', type=int, default=300)
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--wd', help='Weight decay', type=float, default=0.001)
    parser.add_argument('--z_weight', help='Z weight for sampling', type=float, default=1.)

    parser.add_argument('--hidden_dims', help='hidden dimension for model', default=256, type=int)
    parser.add_argument('--dropout', help='dropout ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=128)
    parser.add_argument('--val_step', help='Batch size', type=int, default=20)

    parser.add_argument('--cuda', help='Specify GPU', type=int, default=None)
    parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)

    parser.add_argument('--l2_lambda', help='l1 lambda for l1 loss', type=float, default=0.001)

    parser.add_argument('--genotypes', help='Mutation information for cell lines', type=str)

    parser.add_argument('--model', help='model trained', default=None)

    parser.add_argument('--jobs', help="The number of threads", type=int, default=0)
    parser.add_argument('--out', help="output model path")

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--local_rank', default=1)
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    parser.add_argument('--sys2cell', action='store_true', default=False)
    parser.add_argument('--cell2sys', action='store_true', default=False)
    parser.add_argument('--sys2gene', action='store_true', default=False)

    args = parser.parse_args()
    if args.cuda is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
        args.gpu = args.cuda

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    
    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print("The world size is %d"%args.world_size)
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        print("GPU %d rank is %d" % (gpu, args.rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print("GPU %d process initialized" % (gpu))
        torch.cuda.empty_cache()


    args.genotypes = {genotype.split(":")[0]: genotype.split(":")[1] for genotype in args.genotypes.split(',')}

    if torch.cuda.is_available():
        device = torch.device("cuda:%d" % gpu)
    else:
        device = torch.device("cpu")

    tree_parser = TreeParser(args.onto, args.gene2id)

    if args.model is not None:
        g2p_model = torch.load(args.model, map_location=device)
    else:
        g2p_model = Genotype2PhenotypeModel(tree_parser, list(args.genotypes.keys()), args.hidden_dims, dropout=args.dropout)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            #torch.cuda.set_device(args.gpu)
            g2p_model.to(device)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.jobs = int((args.jobs + ngpus_per_node - 1) / ngpus_per_node)
            print(args.batch_size, args.jobs)
            g2p_model = torch.nn.parallel.DistributedDataParallel(g2p_model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            print("Distributed training are set up")
            g2p_model.to(device)
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set

            g2p_model = torch.nn.parallel.DistributedDataParallel(g2p_model, find_unused_parameters=True)
    elif args.gpu is not None:
        #torch.cuda.set_device(args.gpu)
        g2p_model = g2p_model.to(device)
        print("Model is loaded at GPU(%d)" % args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        g2p_model = torch.nn.DataParallel(g2p_model).to(device)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                     and args.rank % torch.cuda.device_count() == 0):
        print("Summary of trainable parameters")
        count_parameters(g2p_model)


    fix_system = False
    '''
    if args.system_embedding:
        system_embedding_dict = np.load(args.system_embedding, allow_pickle=True).item()
        #print("Loading System Embeddings :", args.system_embedding)
        #if "NEST" not in NeST_embedding_dict.keys():
        #    print("NEST root term does not exist!")
        #    system_embedding_dict["NEST"] = np.mean(
        #        np.stack([system_embedding_dict["NEST:1"], NeST_embedding_dict["NEST:2"], NeST_embedding_dict["NEST:3"]],
        #                 axis=0), axis=0, keepdims=False)
        system_embeddings = np.stack(
            [system_embedding_dict[key] for key, value in sorted(tree_parser.system2ind.items(), key=lambda a: a[1])])
        g2p_model.system_embedding.weight = nn.Parameter(torch.tensor(system_embeddings))
        print(g2p_model.system_embedding.weight)
        g2p_model.system_embedding.weight.requires_grad = False
        fix_system = True
    if args.gene_embedding:
        gene_embedding_dict = np.load(args.gene_embedding, allow_pickle=True).item()
        print("Loading Gene Embeddings :", args.gene_embedding)
        gene_embeddings = np.stack([gene_embedding_dict[key] for key, value in sorted(tree_parser.gene2ind.items(), key=lambda a: a[1])])
        g2p_model.gene_embedding.weight = nn.Parameter(torch.tensor(gene_embeddings))
        #drug_response_model.gene_embedding.weight.requires_grad = False
    '''
    print("Summary of trainable parameters")
    if args.sys2cell:
        print("Model will use Sys2Cell")
    if args.cell2sys:
        print("Model will use Cell2Sys")
    if args.sys2gene:
        print("Model will use Sys2Gene")


    train_dataset = pd.read_csv(args.train, header=None, sep='\t')
    g2p_dataset = G2PDataset(train_dataset, args.genotypes, tree_parser)
    g2p_collator = G2PCollator(list(args.genotypes.keys()))

    if args.distributed:
        #affinity_dataset = affinity_dataset.sample(frac=1).reset_index(drop=True)
        interaction_sampler = torch.utils.data.distributed.DistributedSampler(g2p_dataset)
        shuffle = False
    else:
        shuffle = True
        interaction_sampler = None
    g2p_dataloader = DataLoader(g2p_dataset, batch_size=args.batch_size, collate_fn=g2p_collator, num_workers = args.jobs, shuffle = shuffle, sampler = interaction_sampler)
    if args.val is not None:
        val_dataset = pd.read_csv(args.val, header=None, sep='\t')
        val_g2p_dataset = G2PDataset(val_dataset, args.genotypes, tree_parser)
        val_g2p_dataloader = DataLoader(val_g2p_dataset, shuffle=False, batch_size=args.batch_size,
                                              num_workers=args.jobs, collate_fn=g2p_collator)
    else:
        val_g2p_dataloader = None

    drug_response_trainer = G2PTrainer(g2p_model, g2p_dataloader, device, args,
                                                validation_dataloader=val_g2p_dataloader, fix_system=fix_system)
    drug_response_trainer.train(args.epochs, args.out)


if __name__ == '__main__':
    main()
