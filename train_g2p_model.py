import argparse
import torch
import pandas as pd
from prettytable import PrettyTable

from src.model.genotype2phenotype_model import G2PModel

from src.utils.data import TreeParser, G2PDataset, G2PCollator
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


if __name__ == '__main__':
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

    args = parser.parse_args()
    args.genotypes = {genotype.split(":")[0]: genotype.split(":")[1] for genotype in args.genotypes.split(',')}

    if args.cuda is not None:
        if torch.cuda.is_available():
            device = torch.device("cuda:%d" % args.cuda)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    tree_parser = TreeParser(args.onto, args.gene2id)

    train_dataset = pd.read_csv(args.train, header=None, sep='\t')

    if args.model is not None:
        g2p_model = torch.load(args.model, map_location=device)
    else:
        g2p_model = G2PModel(tree_parser, ["mutation"], args.hidden_dims, dropout=args.dropout)

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
    if args.gene_embedding:
        gene_embedding_dict = np.load(args.gene_embedding, allow_pickle=True).item()
        print("Loading Gene Embeddings :", args.gene_embedding)
        gene_embeddings = np.stack([gene_embedding_dict[key] for key, value in sorted(tree_parser.gene2ind.items(), key=lambda a: a[1])])
        g2p_model.gene_embedding.weight = nn.Parameter(torch.tensor(gene_embeddings))
        #drug_response_model.gene_embedding.weight.requires_grad = False

    print("Summary of trainable parameters")
    count_parameters(g2p_model)
    train_dataset = pd.read_csv(args.train, header=None, sep='\t')
    g2p_dataset = G2PDataset(train_dataset, args.genotypes, tree_parser)
    g2p_collator = G2PCollator(list(args.genotypes.keys()))
    g2p_dataloader = DataLoader(g2p_dataset, batch_size=args.batch_size, collate_fn=g2p_collator,
                                               num_workers=args.jobs)
    if args.val is not None:
        val_dataset = pd.read_csv(args.val, header=None, sep='\t')
        val_g2p_dataset = G2PDataset(val_dataset, args.genotypes, tree_parser)
        val_g2p_dataloader = DataLoader(val_g2p_dataset, shuffle=False, batch_size=args.batch_size,
                                              num_workers=args.jobs, collate_fn=g2p_collator)
    else:
        val_g2p_dataloader = None

    drug_response_trainer = G2PTrainer(g2p_model, g2p_dataloader, device, args,
                                                validation_dataloader=val_g2p_dataloader)
    drug_response_trainer.train(args.epochs, args.out)