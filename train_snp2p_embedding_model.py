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

from torch.utils.data.distributed import DistributedSampler
from src.utils.data.dataset import SNP2PCollator, PLINKDataset, DynamicPhenotypeBatchIterableDataset, DynamicPhenotypeBatchIterableDatasetDDP, EmbeddingDataset
from src.utils.tree import SNPTreeParser
from src.utils.trainer import SNP2PTrainer
from datetime import timedelta

from torch.utils.data.dataloader import DataLoader
import zarr
import re
from glob import glob


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

def sort_key(path):
    m = re.search(r"chr(\w+)_block(\d+)", path)
    chr_str, blk_str = m.group(1), m.group(2)

    # handle X/Y as 23/24, or use dict if you want alphabetical
    chr_num = {"X": 23, "Y": 24}.get(chr_str, int(chr_str))
    blk_num = int(blk_str)
    return (chr_num, blk_num)

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
    '''
    if distributed and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",    # MASTER_ADDR / MASTER_PORT already exported
            world_size=world_size,
            rank=rank,
            timeout=timedelta(hours=3),
        )
    '''
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

    # Indexing files
    #parser.add_argument('--snp2id', help='SNP to ID mapping file', type=str)
    #parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)
    # Hierarchy files
    parser.add_argument('--onto', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--snp2gene', help='SNP to gene mapping file', type=str)
    parser.add_argument('--interaction-types', help='Subtree cascading order', nargs='+', default=['default'])

    # Train bfile format
    parser.add_argument('--embedding-dir', help='Training genotype dataset', type=str, default=None)
    parser.add_argument('--fam', type=str)

    parser.add_argument('--train-bfile', help='Training genotype dataset', type=str, default=None)
    parser.add_argument('--train-cov', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--train-pheno', help='Training covariates dataset', type=str, default=None)

    parser.add_argument('--val-bfile', help='Validation dataset', type=str, default=None)
    parser.add_argument('--val-cov', help='Validation covariates dataset', type=str, default=None)
    parser.add_argument('--val-pheno', help='Training covariates dataset', type=str, default=None)

    parser.add_argument('--test-cov', help='Validation covariates dataset', type=str, default=None)
    parser.add_argument('--test-pheno', help='Validation covariates dataset', type=str, default=None)
    parser.add_argument('--cov-ids', nargs='*', default=[])
    parser.add_argument('--flip', action='store_true', default=False)
    parser.add_argument('--pheno-ids', nargs='*', default=[])
    parser.add_argument('--bt', nargs='*', default=[])
    parser.add_argument('--qt', nargs='*', default=[])
    # Propagation option
    parser.add_argument('--sys2env', action='store_true', default=False)
    parser.add_argument('--env2sys', action='store_true', default=False)
    parser.add_argument('--sys2gene', action='store_true', default=False)
    parser.add_argument('--sys2pheno', action='store_true', default=True)
    parser.add_argument('--gene2pheno', action='store_true', default=False)
    parser.add_argument('--snp2pheno', action='store_true', default=False)

    parser.add_argument('--dense-attention', action='store_true', default=False)
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
    '''
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
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
    '''

    # Model input and output
    parser.add_argument('--model', help='path to trained model', default=None)
    parser.add_argument('--out', help="output model path")

    args = parser.parse_args()
    print('Start Process')
    if args.cuda is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    #args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node

    distributed, rank, world_size, local_rank, device = ddp_setup()

    #+  # -------- torchrun gives all ranks in env --------
    args.distributed = distributed
    args.rank = rank
    args.world_size = world_size
    args.local_rank = local_rank

    torch.cuda.set_device(args.local_rank)
    main_worker(args)  # no mp.spawn!
    '''
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
        print("The world size is %d" % args.world_size)
        main_worker(args.cuda, ngpus_per_node, args)
    '''

def main_worker(args):
    #global best_acc1
    #node_name = socket.gethostname()
    #print(f"Initialize main worker {rank} at node {node_name}")
    gpu = args.local_rank
    node_name = socket.gethostname()
    print(f"[{args.rank}/{args.world_size}] running on {node_name} GPU {gpu}, rank: {args.rank}, local_rank: {args.local_rank}", flush=True)
    if args.distributed and not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(hours=3),
        )
    '''
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
    '''
    print("Finish setup main worker", args.rank)

    if args.distributed:
        device = torch.device("cuda:%d" % gpu)
    elif args.cuda is not None:
        device = torch.device("cuda:%d" % args.cuda)
    else:
        device = torch.device("cpu")
    if (len(args.qt) + len(args.bt)) > 1:
        multiple_phenotypes = True
    else:
        multiple_phenotypes = False

    tree_parser = SNPTreeParser(args.onto, args.snp2gene, dense_attention=args.dense_attention, multiple_phenotypes=multiple_phenotypes)
    #embedding_zarrs = glob(os.path.join(args.zarr_path, "*"))
    #embedding_zarrs = sorted(embedding_zarrs, key=sort_key)
    #zarr_opened = zarr.open(args.zarr_path, mode="r")
    fix_system = False
    fam = pd.read_csv(args.fam, sep=' ', header=None)
    iid2ind = {iid: i for i, iid in enumerate(fam[1].map(str).values)}
    '''
    if args.train_bfile is None:
        print("Loading Genotype dataset... at %s" % args.genotype_csv)
        genotype = pd.read_csv(args.genotype_csv, index_col=0, sep='\t')  # .astype('int32')
        print("Loading done...")
        train_dataset = pd.read_csv(args.train_cov, header=None, sep='\t')
        snp2p_dataset = SNP2PDataset(train_dataset, genotype, tree_parser, n_cov=args.n_cov)
    else:
    '''
    #print("Loading PLINK bfile... at %s" % args.train_bfile)
    snp2p_dataset = EmbeddingDataset(tree_parser, embedding=args.embedding_dir, bfile=args.train_bfile, iid2ind=iid2ind, cov=args.train_cov, pheno=args.train_pheno,
                                 cov_ids=args.cov_ids, pheno_ids=args.pheno_ids, bt=args.bt, qt=args.qt)
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
    print("Loading done...")

    snp2p_collator = SNP2PCollator(tree_parser, input_format='embedding', pheno_ids=tree_parser.phenotypes)

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
                                         interaction_types=args.interaction_types,
                                         dropout=args.dropout, n_covariates=snp2p_dataset.n_cov,
                                         n_phenotypes=snp2p_dataset.n_pheno, activation='softmax', input_format='embedding')
        print(args.model, 'initialized')
        snp2p_model.load_state_dict(snp2p_model_dict['state_dict'])
        if args.model.split('.')[-1].isdigit():
            args.start_epoch = int(args.model.split('.')[-1])
        else:
            args.start_epoch = 0

    else:
        snp2p_model = SNP2PhenotypeModel(tree_parser, args.hidden_dims,
                                         sys2pheno=args.sys2pheno, gene2pheno=args.gene2pheno, snp2pheno=args.snp2pheno,
                                         interaction_types=args.interaction_types,
                                         dropout=args.dropout, n_covariates=snp2p_dataset.n_cov,
                                         activation='softmax', input_format='embedding',
                                         n_phenotypes=snp2p_dataset.n_pheno,
                                         phenotypes=snp2p_dataset.pheno_ids)
        #snp2p_model = snp2p_model.half()
        #snp2p_model = torch.compile(snp2p_model, fullgraph=True)
        #snp2p_model = torch.compile(snp2p_model, fullgraph=True)
        args.start_epoch = 0

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        print("Distributed trainings are set up")
        args.jobs = int((args.jobs) / args.world_size)
        snp2p_model = snp2p_model.to(device)
        snp2p_model = torch.nn.parallel.DistributedDataParallel(snp2p_model, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
    elif args.cuda is not None:
        snp2p_model = snp2p_model.to(device)
        print("Model is loaded at GPU(%d)" % args.cuda)
    else:
        print("Model is on cpu (not recommended)")

    if not args.distributed or (args.distributed and args.rank  == 0):
        print("Summary of trainable parameters")
        count_parameters(snp2p_model)
    args.dynamic_phenotype_sampling = False
    if args.distributed:
        if args.dynamic_phenotype_sampling:
            dataset = DynamicPhenotypeBatchIterableDatasetDDP(snp2p_dataset, snp2p_collator, args.batch_size, shuffle=True)
            snp2p_dataloader = DataLoader(dataset, batch_size=None,
                                          num_workers=args.jobs,
                                          prefetch_factor=2
                                          )
        else:
            print("No dynamic Sampling")
            snp2p_sampler = DistributedSampler(dataset=snp2p_dataset, shuffle=True)
            shuffle = False
            snp2p_dataloader = DataLoader(snp2p_dataset, batch_size=args.batch_size, collate_fn=snp2p_collator,
                                          num_workers=args.jobs, shuffle=shuffle, sampler=snp2p_sampler,
                                          pin_memory=True,
                                          persistent_workers=True,  # keep workers alive across epochs
                                          prefetch_factor=2
                                          )
    else:
        snp2p_sampler = None
        if args.dynamic_phenotype_sampling:
            #snp2p_batch_sampler = DynamicPhenotypeBatchSampler(dataset=snp2p_dataset, batch_size=args.batch_size)
            dataset = DynamicPhenotypeBatchIterableDataset(snp2p_dataset, snp2p_collator, args.batch_size, shuffle=True)
            snp2p_dataloader = DataLoader(dataset, batch_size=None,
                                          num_workers=args.jobs,
                                          prefetch_factor=2
                                          )
        else:
            print("No dynamic Sampling")
            #snp2p_sampler = DistributedSampler(dataset=snp2p_dataset, shuffle=True)
            snp2p_batch_sampler = None
            batch_size = None
            shuffle = True
            snp2p_dataloader = DataLoader(snp2p_dataset, batch_size=args.batch_size, collate_fn=snp2p_collator,
                                          num_workers=args.jobs, shuffle=shuffle, sampler=None,
                                          pin_memory=True,
                                          persistent_workers=True,  # keep workers alive across epochs
                                          #prefetch_factor=2
                                          )
    '''
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
    '''



    if args.val_cov is not None:
        val_snp2p_dataset = EmbeddingDataset(tree_parser, embedding=args.embedding_dir, bfile=args.val_bfile, iid2ind=iid2ind, cov=args.val_cov, pheno=args.val_pheno,
                                 cov_ids=args.cov_ids, pheno_ids=args.pheno_ids, bt=args.bt, qt=args.qt, cov_mean_dict=args.cov_mean_dict,
                                 cov_std_dict=args.cov_std_dict)
        #PLINKDataset(tree_parser, args.val_bfile, args.val_cov, args.val_pheno, cov_mean_dict=args.cov_mean_dict,
        #                                 cov_std_dict=args.cov_std_dict, flip=args.flip, input_format=args.input_format,
        #                                 cov_ids=args.cov_ids, pheno_ids=args.pheno_ids, bt=args.bt, qt=args.qt)
        val_snp2p_dataloader = DataLoader(val_snp2p_dataset, shuffle=False, batch_size=int(args.batch_size/(args.ngpus_per_node*2)),
                                          num_workers=args.jobs, collate_fn=snp2p_collator, pin_memory=True)
    else:
        val_snp2p_dataloader = None


    snp2p_trainer = SNP2PTrainer(snp2p_model, tree_parser, snp2p_dataloader, device, args,
                                 validation_dataloader=val_snp2p_dataloader, fix_system=fix_system)
    snp2p_trainer.train(args.epochs, args.out)

if __name__ == '__main__':
    print("Python __main__", flush=True)
    #print("NCCL socket name :", os.environ["NCCL_SOCKET_IFNAME"])
    main()
