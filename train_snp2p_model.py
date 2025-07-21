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
from src.utils.chunker import MaskBasedChunker, AttentionAwareChunker

from src.model.model.snp2phenotype import SNP2PhenotypeModel

from torch.utils.data.distributed import DistributedSampler
from src.utils.data.dataset import SNP2PCollator, PLINKDataset
from src.utils.data.dataset import DynamicPhenotypeBatchIterableDataset, DynamicPhenotypeBatchIterableDatasetDDP
from src.utils.data.dataset import ChunkSNP2PCollator
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
        #if parameter.requires_grad:
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

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
    # Participant Genotype file
    parser.add_argument('--genotype-csv', help='Personal genotype file', type=str, default=None)
    parser.add_argument('--n_cov', help='The number of covariates', type=int, default=4)
    # Indexing files
    #parser.add_argument('--snp2id', help='SNP to ID mapping file', type=str)
    #parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)
    # Hierarchy files
    parser.add_argument('--onto', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--snp2gene', help='SNP to gene mapping file', type=str)
    parser.add_argument('--interaction-types', help='Subtree cascading order', nargs='+', default=['default'])

    # Train bfile format
    parser.add_argument('--train-bfile', help='Training genotype dataset', type=str, default=None)
    parser.add_argument('--train-cov', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--train-pheno', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--val-bfile', help='Validation dataset', type=str, default=None)
    parser.add_argument('--val-cov', help='Validation covariates dataset', type=str, default=None)
    parser.add_argument('--val-pheno', help='Training covariates dataset', type=str, default=None)
    parser.add_argument('--test-bfile', help='Test dataset', type=str, default=None)
    parser.add_argument('--test-cov', help='Validation covariates dataset', type=str, default=None)
    parser.add_argument('--test-pheno', help='Validation covariates dataset', type=str, default=None)
    parser.add_argument('--cov-ids', nargs='*', default=[])
    parser.add_argument('--flip', action='store_true', default=False)
    parser.add_argument('--pheno-ids', nargs='*', default=[])
    parser.add_argument('--bt', nargs='*', default=[])
    parser.add_argument('--qt', nargs='*', default=[])
    parser.add_argument('--target-phenotype', type=str, )
    # Propagation option
    parser.add_argument('--cov-effect', default='pre')
    parser.add_argument('--sys2env', action='store_true', default=False)
    parser.add_argument('--env2sys', action='store_true', default=False)
    parser.add_argument('--sys2gene', action='store_true', default=False)
    parser.add_argument('--sys2pheno', action='store_true', default=True)
    parser.add_argument('--gene2pheno', action='store_true', default=False)
    parser.add_argument('--snp2pheno', action='store_true', default=False)

    parser.add_argument('--dynamic-phenotype-sampling', action='store_true', default=False)

    parser.add_argument('--poincare', action='store_true', default=False)

    parser.add_argument('--dense-attention', action='store_true', default=False)
    parser.add_argument('--input-format', default='indices', choices=["indices", "binary"])
    parser.add_argument('--regression', action='store_true', default=False)
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
    args.batch_size = args.batch_size // world_size
    #args.jobs = args.jobs // world_size

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
    print("Finish setup main worker", args.rank)

    if args.distributed:
        device = torch.device("cuda:%d" % gpu)
    elif args.cuda is not None:
        device = torch.device("cuda:%d" % args.cuda)
    elif args.world_size == 1 and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if (len(args.qt) + len(args.bt) > 1) :
        multiple_phenotypes = True
    else:
        multiple_phenotypes = False

    tree_parser = SNPTreeParser(args.onto, args.snp2gene, dense_attention=args.dense_attention, multiple_phenotypes=multiple_phenotypes)
    #chunker = MaskBasedChunker(snp2gene_mask=tree_parser.snp2gene_mask, gene2sys_mask=tree_parser.gene2sys_mask, target_chunk_size=20000)
    fix_system = False
    '''
    if args.train_bfile is None:
        print("Loading Genotype dataset... at %s" % args.genotype_csv)
        genotype = pd.read_csv(args.genotype_csv, index_col=0, sep='\t')  # .astype('int32')
        print("Loading done...")
        train_dataset = pd.read_csv(args.train_cov, header=None, sep='\t')
        snp2p_dataset = SNP2PDataset(train_dataset, genotype, tree_parser, n_cov=args.n_cov)
    else:
    '''
    print("Loading PLINK bfile... at %s" % args.train_bfile)


    snp2p_dataset = PLINKDataset(tree_parser, args.train_bfile, args.train_cov, args.train_pheno, flip=args.flip,
                                 input_format=args.input_format,
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

    snp2p_collator = SNP2PCollator(tree_parser, input_format=args.input_format)
    #snp2p_collator = ChunkSNP2PCollator(tree_parser, chunker=chunker, input_format=args.input_format)

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
                                         n_phenotypes=snp2p_dataset.n_pheno, activation='softmax', input_format=args.input_format)
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
                                         activation='softmax', input_format=args.input_format,
                                         n_phenotypes=snp2p_dataset.n_pheno)
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

    if args.distributed:
        if args.dynamic_phenotype_sampling:
            print("Dynamic phenotype sampling with DDP")
            dataset = DynamicPhenotypeBatchIterableDatasetDDP(tree_parser, snp2p_dataset, snp2p_collator, args.batch_size,
                                                              rank=args.rank,
                                                              world_size=args.world_size,
                                                              shuffle=True, n_phenotype2sample=1)
            snp2p_dataloader = DataLoader(dataset, batch_size=None,
                                          num_workers=args.jobs,
                                          prefetch_factor=2,
                                          persistent_workers = True,  # Add this
                                          pin_memory = True)  # Add this
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
            print("Dynamic phenotype sampling")
            #snp2p_batch_sampler = DynamicPhenotypeBatchSampler(dataset=snp2p_dataset, batch_size=args.batch_size)
            dataset = DynamicPhenotypeBatchIterableDataset(tree_parser, snp2p_dataset, snp2p_collator, args.batch_size, shuffle=True, n_phenotype2sample=1)
            snp2p_dataloader = DataLoader(dataset, batch_size=None,
                                          num_workers=args.jobs,
                                          prefetch_factor=2,
                                          persistent_workers = True,  # Add this
                                          pin_memory = True)  # Add this
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



    if args.val_bfile is not None:
        val_snp2p_dataset = PLINKDataset(tree_parser, args.val_bfile, args.val_cov, args.val_pheno, cov_mean_dict=args.cov_mean_dict,
                                         cov_std_dict=args.cov_std_dict, flip=args.flip, input_format=args.input_format,
                                         cov_ids=args.cov_ids, pheno_ids=args.pheno_ids, bt=args.bt, qt=args.qt)
        val_snp2p_dataloader = DataLoader(val_snp2p_dataset, shuffle=False, batch_size=int(args.batch_size/(args.ngpus_per_node*2)),
                                          num_workers=args.jobs, collate_fn=snp2p_collator, pin_memory=True)
    else:
        val_snp2p_dataloader = None


    snp2p_trainer = SNP2PTrainer(snp2p_model, tree_parser, snp2p_dataloader, device, args, args.target_phenotype,
                                 validation_dataloader=val_snp2p_dataloader, fix_system=fix_system)
    snp2p_trainer.train(args.epochs, args.out)

if __name__ == '__main__':
    #try:
    mp.set_start_method('spawn', force=True)
    #except RuntimeError:
    #    pass # Already set, ignore
    print("Python __main__", flush=True)
    #print("NCCL socket name :", os.environ["NCCL_SOCKET_IFNAME"])
    main()
