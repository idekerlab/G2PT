import argparse
import torch
import pandas as pd
from prettytable import PrettyTable

from src.model.compound import ECFPCompoundModel, ChemBERTaCompoundModel
from src.model.drug_response_model import DrugResponseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer

from src.utils.data import CompoundEncoder, TreeParser, DrugResponseDataset, DrugResponseCollator, DrugResponseSampler, DrugBatchSampler, DrugDataset, CellLineBatchSampler
from src.utils.trainer import DrugResponseTrainer, DrugTrainer, DrugResponseFineTuner
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
    parser.add_argument('--nest_embedding', default=None)
    parser.add_argument('--gene_embedding', default=None)

    parser.add_argument('--epochs', help='Training epochs for training', type=int, default=300)
    parser.add_argument('--compound_epochs', type=int, default=10)
    parser.add_argument('--lr', help='Learning rate', type=float, default=0.001)
    parser.add_argument('--wd', help='Weight decay', type=float, default=0.001)
    parser.add_argument('--z_weight', help='Z weight for sampling', type=float, default=1.)

    parser.add_argument('--hidden_dims', help='hidden dimension for model', default=256, type=int)
    parser.add_argument('--dropout', help='dropout ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', help='Batch size', type=int, default=128)
    parser.add_argument('--val_step', help='Batch size', type=int, default=20)

    parser.add_argument('--cuda', help='Specify GPU', type=int, default=None)
    parser.add_argument('--gene2id', help='Gene to ID mapping file', type=str)
    parser.add_argument('--cell2id', help='Cell to ID mapping file', type=str)

    parser.add_argument('--genotypes', help='Mutation information for cell lines', type=str)

    parser.add_argument('--bert', help='huggingface repository for smiles parsing', default=None)
    parser.add_argument('--radius', help='ECFP radius', type=int, default=2)
    parser.add_argument('--n_bits', help='ECFP number of bits', type=int, default=512)
    parser.add_argument('--compound_layers', help='Compound_dense_layer', nargs='+', default=[256], type=int)
    parser.add_argument('--l2_lambda', help='l1 lambda for l1 loss', type=float, default=0.001)

    parser.add_argument('--model', help='model trained', default=None)

    parser.add_argument('--jobs', help="The number of threads", type=int, default=0)
    parser.add_argument('--out', help="output model path")

    args = parser.parse_args()

    if args.bert is not None:
        print(args.bert, "is used for compound encoding")
        tokenizer = AutoTokenizer.from_pretrained(args.bert)
        compound_encoder = CompoundEncoder('SMILES', tokenizer=tokenizer)
        compound_bert =  AutoModelForSequenceClassification.from_pretrained(args.bert)
        for name, param in compound_bert.named_parameters():
            param.requires_grad = False
            if len(name.split("."))>=5:
                if name.split(".")[1]=='encoder':
                    if int(name.split(".")[3])>=10 :
                        param.requires_grad = True
                        #print(name, param.requires_grad)
        compound_model = ChemBERTaCompoundModel(compound_bert, args.dropout)
    else:
        print("ECFP with radius %d, and %d bits used for compound encoding"%(args.radius, args.n_bits))
        compound_encoder = CompoundEncoder('Morgan', args.radius, args.n_bits)
        compound_model = ECFPCompoundModel(args.n_bits, args.compound_layers, args.dropout)

    tree_parser = TreeParser(args.onto, args.gene2id)

    train_dataset = pd.read_csv(args.train, header=None, sep='\t')

    if args.cuda is not None:
        device = torch.device("cuda:%d" % args.cuda)
    else:
        device = torch.device("cpu")

    compound_dataset = DrugDataset(train_dataset, compound_encoder)
    compound_datalorder = DataLoader(compound_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.jobs)

    compound_trainer = DrugTrainer(compound_model, compound_datalorder, device=device, args=args)



    args.genotypes = {genotype.split(":")[0]: genotype.split(":")[1] for genotype in args.genotypes.split(',')}
    if args.model is not None:
        drug_response_model = torch.load(args.model, map_location=device)
    else:
        compound_model = compound_trainer.train(args.compound_epochs)
        drug_response_model = DrugResponseModel(tree_parser, list(args.genotypes.keys()),
                                                compound_model, args.hidden_dims, dropout=args.dropout)

    if args.nest_embedding:
        NeST_embedding_dict = np.load(args.nest_embedding, allow_pickle=True).item()
        print("Loading System Embeddings :", args.nest_embedding)
        if "NEST" not in NeST_embedding_dict.keys():
            print("NEST root term does not exist!")
            NeST_embedding_dict["NEST"] = np.mean(
                np.stack([NeST_embedding_dict["NEST:1"], NeST_embedding_dict["NEST:2"], NeST_embedding_dict["NEST:3"]],
                         axis=0), axis=0, keepdims=False)
        system_embeddings = np.stack(
            [NeST_embedding_dict[key] for key, value in sorted(tree_parser.system2ind.items(), key=lambda a: a[1])])
        drug_response_model.system_embedding.weight = nn.Parameter(torch.tensor(system_embeddings))
        print(drug_response_model.system_embedding.weight)
        drug_response_model.system_embedding.weight.requires_grad = False
    if args.gene_embedding:
        gene_embedding_dict = np.load(args.gene_embedding, allow_pickle=True).item()
        print("Loading Gene Embeddings :", args.gene_embedding)
        gene_embeddings = np.stack([gene_embedding_dict[key] for key, value in sorted(tree_parser.gene2ind.items(), key=lambda a: a[1])])
        drug_response_model.gene_embedding.weight = nn.Parameter(torch.tensor(gene_embeddings))
        #drug_response_model.gene_embedding.weight.requires_grad = False

    print("Summary of trainable parameters")
    count_parameters(drug_response_model)
    drug_response_dataset = DrugResponseDataset(train_dataset, args.cell2id, args.genotypes, compound_encoder,
                                                tree_parser, args.subtree_order)
    if args.val is not None:
        val_dataset = pd.read_csv(args.val, header=None, sep='\t')
        val_drug_response_dataset = DrugResponseDataset(val_dataset, args.cell2id, args.genotypes, compound_encoder,
                                                    tree_parser, args.subtree_order)
        drug_response_collator = DrugResponseCollator(list(args.genotypes.keys()), compound_encoder)

        val_drug_response_dataloader = DataLoader(val_drug_response_dataset, shuffle=False, batch_size=args.batch_size,
                                              num_workers=args.jobs, collate_fn=drug_response_collator)
    else:
        val_drug_response_dataloader = None

    drug_response_collator = DrugResponseCollator(list(args.genotypes.keys()), compound_encoder)
    if args.model is not None:
        drug_response_dataloader = DataLoader(drug_response_dataset, batch_size=args.batch_size,
                                              collate_fn=drug_response_collator,
                                              num_workers=args.jobs)
        drug_response_trainer = DrugResponseFineTuner(drug_response_model, drug_response_dataloader, device, args,
                                                    validation_dataloader=val_drug_response_dataloader)
        drug_response_trainer.train(args.epochs, args.out)
    else:
        #drug_response_sampler = DrugResponseSampler(train_dataset, group_index=1, response_index=2, z_weights=args.z_weight)
        drug_batch_sampler = DrugBatchSampler(train_dataset, drug_response_dataset.drug_response_mean_dict,
                                              batch_size=args.batch_size, group_index=1, response_index=2,
                                              z_weights=args.z_weight)
        cellline_batch_sampler = CellLineBatchSampler(train_dataset, drug_response_dataset.drug_response_mean_dict,
                                                      batch_size=args.batch_size, group_index=0, drug_index=1,
                                                      response_index=2, z_weights=args.z_weight)
        drug_response_dataloader_drug = DataLoader(drug_response_dataset,
                                              #batch_size=args.batch_size,
                                              #sampler=drug_response_sampler,
                                              batch_sampler= drug_batch_sampler,
                                              collate_fn=drug_response_collator,
                                              num_workers=args.jobs)
        drug_response_dataloader_cellline = DataLoader(drug_response_dataset,
                                                   # batch_size=args.batch_size,
                                                   # sampler=drug_response_sampler,
                                                   batch_sampler=cellline_batch_sampler,
                                                   collate_fn=drug_response_collator,
                                                   num_workers=args.jobs)



        drug_response_trainer = DrugResponseTrainer(drug_response_model, drug_response_dataloader_drug, drug_response_dataloader_cellline, device, args,
                                                    validation_dataloader=val_drug_response_dataloader)
        drug_response_trainer.train(args.epochs, args.out)



    test_dataset = pd.read_csv(args.test, header=None, sep='\t')

    test_drug_response_dataset = DrugResponseDataset(test_dataset, args.cell2id, args.genotypes, compound_encoder,
                                                     tree_parser, args.subtree_order)
    drug_response_collator = DrugResponseCollator(list(args.genotypes.keys()), compound_encoder)

    test_drug_response_dataloader = DataLoader(test_drug_response_dataset, shuffle=False, batch_size=args.batch_size,
                                               num_workers=args.jobs, collate_fn=drug_response_collator)
    torch.cuda.empty_cache()
    #drug_response_trainer.evaluate(test_drug_response_dataloader, 0, name="Test")

    best_model = drug_response_trainer.get_best_model()
    drug_response_trainer.evaluate(best_model, test_drug_response_dataloader, 1, name="Test")
    print("Saving model to %s"%args.out)
    torch.save(drug_response_model, args.out)







