import sys
import numpy as np
import pandas as pd
import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum
import pandas as pd
import copy
import pickle

from pycox.models import CoxTime
from pycox.models import PCHazard
from pycox.models import LogisticHazard
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from pycox.models.cox_time import MLPVanillaCoxTime

import torch
import torchtuples as tt
import torch.nn.parallel
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from prettytable import PrettyTable

from src.model.compound import ECFPCompoundModel, ChemBERTaCompoundModel, DrugEmbeddingCompoundModel
from src.model.model.drug_response_model import DrugResponseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline, RobertaModel, RobertaTokenizer

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, BatchSampler
from torch.nn.utils.rnn import pad_sequence
from random import shuffle

from src.utils.data import CompoundEncoder
from src.utils.tree import MutTreeParser
from src.utils.data.dataset import DrugResponseDataset, DrugResponseCollator, DrugResponseSampler, DrugBatchSampler, DrugDataset, CellLineBatchSampler
from src.utils.trainer import DrugResponseTrainer, DrugTrainer, DrugResponseFineTuner
import torch.nn as nn
import pubchempy as pcp
import statistics as s
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
import math
from operator import itemgetter
from scipy.stats import ttest_ind
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler

import gc
from torch.optim.lr_scheduler import  StepLR
from torch.nn.utils import prune
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from src.utils.trainer import CCCLoss
from src.utils.data import move_to
from torch.nn import functional as F

class HiddenPrints:
    def __enter__(self):
        
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        
        sys.stdout.close()
        sys.stdout = self._original_stdout

class PatientResponseDataset(Dataset):

    def __init__(self, drug_response, patient2ind, patient2genotypes, compound_encoder, tree_parser:MutTreeParser, mut2gene=True, with_indices=True):

        self.compound_encoder = compound_encoder
        self.drug_response_df = drug_response
        self.compound_grouped = self.drug_response_df.groupby(1)
        self.drug_dict = {drug:self.compound_encoder.encode(drug) for drug in self.drug_response_df[1].unique()}
        self.mut2gene = mut2gene
        self.with_indices = with_indices

        self.patient2ind_df = pd.read_csv(patient2ind, sep='\t', header=None)
        self.patient2ind = {j:i for i, j in self.patient2ind_df.itertuples(0, None)}

        self.patient2genotype_dict = {genotype: pd.read_csv(mut_data, header=None).astype('int32') for genotype, mut_data in patient2genotypes.items()}
        self.tree_parser = tree_parser

    def __len__(self):
        
        return self.drug_response_df.shape[0]

    def __getitem__(self, index):
        
        patient, drug, time, source, age, sex, event = self.drug_response_df.iloc[index].values

        patient_ind = self.patient2ind[patient]

        if self.with_indices:
            if self.mut2gene:
                patient_mut_dict = {mut_type:self.tree_parser.get_mut2gene(np.where(mut_value.iloc[patient_ind].values == 1.0)[0],
                                                                           type_indices={1.0: np.where(mut_value.iloc[patient_ind].values == 1.0)[0]})
                                    for mut_type, mut_value in self.patient2genotype_dict.items()}
            else:
                patient_mut_dict = {mut_type:self.tree_parser.get_mut2sys(np.where(mut_value.iloc[patient_ind].values == 1.0)[0],
                                                                          type_indices={1.0: np.where(mut_value.iloc[patient_ind].values == 1.0)[0]})
                                    for mut_type, mut_value in self.patient2genotype_dict.items()}
        else:
            if self.mut2gene:
                patient_mut_dict = {mut_type: self.tree_parser.get_mutation2genotype_mask(
                    torch.tensor(mut_value.iloc[patient_ind].values, dtype=torch.float32))
                                    for mut_type, mut_value in self.patient2genotype_dict.items()}
            else:
                patient_mut_dict = {mut_type: self.tree_parser.get_system2genotype_mask(
                    torch.tensor(mut_value.iloc[patient_ind].values, dtype=torch.float32))
                                    for mut_type, mut_value in self.patient2genotype_dict.items()}

        result_dict = dict()
        result_dict['genotype'] = patient_mut_dict
        result_dict['drug'] = self.drug_dict[drug]
        result_dict['time'] = time
        result_dict['age'] = age
        result_dict['sex'] = sex
        result_dict['event'] = event
        
        return result_dict

class PatientResponseCollator(object):

    def __init__(self, tree_parser:MutTreeParser, genotypes, compound_encoder, mut2gene=True, with_indices=True):
        
        self.tree_parser = tree_parser
        self.genotypes = genotypes
        self.compound_encoder = compound_encoder
        self.compound_type = self.compound_encoder.feature
        self.mut2gene = mut2gene
        self.with_indices = with_indices

    def __call__(self, data):
        
        result_dict = dict()
        mutation_dict = dict()
        for genotype in self.genotypes:
            if self.with_indices:
                embedding_dict = {}
                if self.mut2gene:
                    #NOTE: On 3/19/2025 I found that padding_value was set to n_genes-1. I believe this is wrong, and if I recall I set this to 
                    # n_genes-1 thinking it was fixing an indexing error I was getting. I'm worried this is now a small bug in the model. 
                    # Changing it on this date for the transfer learning model. If bugs arrise, consult with Ingoo
                    embedding_dict['mut'] = pad_sequence([dr['genotype'][genotype]['mut'] for dr in data], 
                                                         padding_value=self.tree_parser.n_genes-1, batch_first=True).to(torch.long)
                    mut_max_len = embedding_dict['mut'].size(1)
                    embedding_dict['mask'] = torch.stack([dr['genotype'][genotype]['mask'] for dr in data])[:, :mut_max_len, :mut_max_len]
                    mutation_dict[genotype] = embedding_dict
                else:
                    embedding_dict['gene'] = pad_sequence([dr['genotype'][genotype]['gene'] for dr in data], 
                                                          padding_value=self.tree_parser.n_genes, batch_first=True).to(torch.long)
                    embedding_dict['sys'] = pad_sequence([dr['genotype'][genotype]['sys'] for dr in data], 
                                                         padding_value=self.tree_parser.n_systems, batch_first=True).to(torch.long)
                    gene_max_len = embedding_dict['gene'].size(1)
                    sys_max_len = embedding_dict['sys'].size(1)
                    embedding_dict['mask'] = torch.stack([dr['genotype'][genotype]['mask'] for dr in data])[:, :sys_max_len, :gene_max_len]
                    mutation_dict[genotype] = embedding_dict
            else:
                mutation_dict[genotype] = torch.stack([dr['genotype'][genotype] for dr in data])

        result_dict['genotype'] = mutation_dict
        result_dict['drug'] = self.compound_encoder.collate([dr['drug'] for dr in data])
        result_dict['time'] = torch.tensor([dr['time'] for dr in data])
        result_dict['age'] = torch.tensor([dr['age'] for dr in data])
        result_dict['sex'] = torch.tensor([dr['sex'] for dr in data])
        result_dict['event'] = torch.tensor([dr['event'] for dr in data])
        
        return result_dict

# NOTE: Not using this for now as I will be using pycox models for the hazard transform
class AUCtoHazardTransform(nn.Module):
    def __init__(self, hidden_dim):
        super(AUCtoHazardTransform, self).__init__()
        
        # Simple MLP transform
        self.fc1 = nn.Linear(1, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        return self.fc2(self.relu(self.fc1(x)))

class TransferCellResponse(object):
    def __init__(self,
                 cell_line_model="/cellar/users/zwallace/G2PT/models_nofolds_diff/rsi_model.pt",
                 cell_line_response="/cellar/users/zwallace/G2PT/data/train_dataset_rsi.txt",
                 cell_mutations="/cellar/users/zwallace/G2PT/data/cell2mutation_ctg_av.txt",
                 cell_amplifics="/cellar/users/zwallace/G2PT/data/old_copynumber/cell2cnamplification_ctg_av.txt",
                 cell_deletions="/cellar/users/zwallace/G2PT/data/old_copynumber/cell2cndeletion_ctg_av.txt",
                 cell_gene2ind="/cellar/users/zwallace/G2PT/data/gene2ind_ctg_av.txt",
                 ontology ="/cellar/users/zwallace/G2PT/data/ontology_ctg_av.txt",
                 epochs=50, batch_size=None, transform_hidden=16, lr=0.01, hidden_dims=128, dropout=0.2, jobs=16,
                 loss_function="cox_time", num_durations=10, num_features=3, transform_dropout = 0.1,
                 diff_transformer=True, unfreeze=None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.cell_line_response = pd.read_csv(cell_line_response, header=None, sep='\t')
        self.cell_genotypes = {'mutation': cell_mutations, 'cna': cell_amplifics, 'cnd': cell_deletions}
        self.ontology = ontology
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.jobs = jobs
        self.lr = lr
        self.unfreeze = unfreeze
        self.loss_function = loss_function
        self.num_durations = num_durations
        self.num_features = num_features
        self.hidden_layers = [transform_hidden, transform_hidden]
        self.transform_dropout = transform_dropout

        self.optimizer = tt.optim.Adam(self.lr)

        if self.loss_function == "cox_time":
            self.labtrans = CoxTime.label_transform()
        elif self.loss_function == "pc_hazard":
            self.labtrans = PCHazard.label_transform(self.num_durations)
        elif self.loss_function == "logistic_hazard":
            self.labtrans = LogisticHazard.label_transform(self.num_durations)

        with HiddenPrints():
            self.compound_encoder = CompoundEncoder('Embedding', dataset=self.cell_line_response, out="dummy/")
            self.compound_model = DrugEmbeddingCompoundModel(self.compound_encoder.num_drugs(), self.hidden_dims)

            self.tree_parser = MutTreeParser(self.ontology, cell_gene2ind)

            self.cell_response_model = DrugResponseModel(self.tree_parser, list(self.cell_genotypes.keys()), self.hidden_dims,
                                                         self.compound_model, dropout=self.dropout, diff_transformer=diff_transformer)
    
        model = torch.load(cell_line_model, map_location=torch.device(self.device))
        model = model["state_dict"]

        self.cell_response_model.load_state_dict(model)
        self.cell_response_model.to(self.device)
           
        # Freeze all parameters to begin with
        for param in self.cell_response_model.parameters():
            param.requires_grad = False

        # Only Unfreeze prediction layer 
        if self.unfreeze == "predictor":
            for param in self.cell_response_model.predictor_systems.parameters():
                param.requires_grad = True

        # Only unfreeze prediction layer and genetic translation layer (sys2comp)
        elif self.unfreeze == "translation":
            for param in self.cell_response_model.sys2comp.parameters():
                param.requires_grad = True
            for param in self.cell_response_model.predictor_systems.parameters():
                param.requires_grad = True

        # Unfreeze all but the mut2gene module
        elif self.unfreeze == "propagation":
            for param in self.cell_response_model.gene2sys.parameters():
                param.requires_grad = True
            for param in self.cell_response_model.sys2env.parameters():
                param.requires_grad = True
            for param in self.cell_response_model.env2sys.parameters():
                param.requires_grad = True
            for param in self.cell_response_model.sys2comp.parameters():
                param.requires_grad = True
            for param in self.cell_response_model.predictor_systems.parameters():
                param.requires_grad = True

        # Unfreeze entire model (update all parameters)
        elif self.unfreeze == "all":
            for param in self.cell_response_model.parameters():
                param.requires_grad = True 

    def FineTunePatients(self,
                         patient_response="/cellar/users/zwallace/G2PT/data/test_PFS_gemcitabine.txt",
                         patient_indicators="/cellar/users/zwallace/G2PT/data/test_gemcitabine.txt",
                         patient_mutations="/cellar/users/zwallace/G2PT/data/cell2mutation_gemcitabine.txt",
                         patient_amplifics="/cellar/users/zwallace/G2PT/data/cell2cnamplification_gemcitabine.txt",
                         patient_deletions="/cellar/users/zwallace/G2PT/data/cell2deletion_gemcitabine.txt",
                         patient2ind="/cellar/users/zwallace/G2PT/data/cell2ind_gemcitabine.txt",
                         patient_gene2ind="/cellar/users/zwallace/G2PT/data/gene2ind_tcga.txt"):

        # Merge reponse and indicators so that there is a column for the event idicator        
        patient_dataset = self.merge_patient_data(patient_response, patient_indicators)
        patient_genotypes = {'mutation': patient_mutations, 'cna': patient_amplifics, 'cnd': patient_deletions}
        
        # For optimizing a Cox loss functions, we typically need to use a full batch when the dataset is relatively small
        if self.batch_size == None:
            batch_size = len(patient_dataset)
        else:
            batch_size = self.batch_size

        with HiddenPrints():
            dataset = PatientResponseDataset(patient_dataset, patient2ind, patient_genotypes, 
                                             self.compound_encoder, self.tree_parser, 
                                             mut2gene=True, 
                                             with_indices=True)
            collator = PatientResponseCollator(self.tree_parser, list(patient_genotypes.keys()), 
                                               self.compound_encoder, 
                                               mut2gene=True, 
                                               with_indices=True)
            dataloader = DataLoader(dataset, shuffle=True, 
                                    batch_size=batch_size, 
                                    num_workers=self.jobs, 
                                    collate_fn=collator)

        if self.loss_function == "cox_time":
            self.train_pycox_CT(dataloader, batch_size)
        elif self.loss_function == "pc_hazard":
            self.train_pycox_PCH(dataloader, batch_size)
        elif self.loss_function == "logistic_hazard":
            self.train_pycox_LH(dataloader, batch_size)
    
    def train_pycox_CT(self, dataloader, batch_size):

        nested_subtrees_forward = move_to(dataloader.dataset.tree_parser.get_nested_subtree_mask(['default'], direction='forward'), self.device)
        nested_subtrees_backward = move_to(dataloader.dataset.tree_parser.get_nested_subtree_mask(['default'], direction='backward'), self.device)
        gene2system_mask = move_to(torch.tensor(dataloader.dataset.tree_parser.gene2sys_mask, dtype=torch.bool), self.device)
        system2gene_mask = move_to(torch.tensor(dataloader.dataset.tree_parser.sys2gene_mask, dtype=torch.bool), self.device)
        
        for i, batch in enumerate(dataloader):
            batch = move_to(batch, self.device)
            
            # Get AUC predictions from patient profile
            auc = self.cell_response_model(batch['genotype'], batch['drug'],
                                           nested_subtrees_forward, nested_subtrees_backward,
                                           gene2system_mask,
                                           system2gene_mask,
                                           sys2cell=True, 
                                           cell2sys=True, 
                                           sys2gene=False,
                                           gene2drug=False, 
                                           mut2gene=True, 
                                           with_indices=True)
            
            auc, age, sex = auc.squeeze().detach().cpu().numpy(), batch['age'].detach().cpu().numpy(), batch['sex'].detach().cpu().numpy()
            hazard_actual = self.labtrans.fit_transform(batch['time'].detach().cpu().numpy(), batch['event'].detach().cpu().numpy())

        # Standardized AUC and Age features
        df = pd.DataFrame({'AUC': auc, 'Age': age, 'Sex': sex}) 
        standards = [(['AUC'], StandardScaler()), (['Age'], StandardScaler())]
        binarized = [('Sex', None)]
        mapper = DataFrameMapper(standards + binarized)
        hazard_train = mapper.fit_transform(df).astype('float32')

        # Instantiate a Hazard Tranform Nueral Network Model
        func = MLPVanillaCoxTime(self.num_features, self.hidden_layers, dropout=self.transform_dropout)
        hazard_transform = CoxTime(func, self.optimizer, labtrans=self.labtrans, device=self.device)

        # Train the Hazard Transform Model
        log = hazard_transform.fit(hazard_train, hazard_actual, batch_size, verbose=True, epochs=self.epochs)

    def train_pycox_PCH(self, dataloader, batch_size):

        nested_subtrees_forward = move_to(dataloader.dataset.tree_parser.get_nested_subtree_mask(['default'], direction='forward'), self.device)
        nested_subtrees_backward = move_to(dataloader.dataset.tree_parser.get_nested_subtree_mask(['default'], direction='backward'), self.device)
        gene2system_mask = move_to(torch.tensor(dataloader.dataset.tree_parser.gene2sys_mask, dtype=torch.bool), self.device)
        system2gene_mask = move_to(torch.tensor(dataloader.dataset.tree_parser.sys2gene_mask, dtype=torch.bool), self.device)
        
        for i, batch in enumerate(dataloader):
            batch = move_to(batch, self.device)
            
            # Get AUC predictions from patient profile
            auc = self.cell_response_model(batch['genotype'], batch['drug'],
                                           nested_subtrees_forward, nested_subtrees_backward,
                                           gene2system_mask,
                                           system2gene_mask,
                                           sys2cell=True, 
                                           cell2sys=True, 
                                           sys2gene=False,
                                           gene2drug=False, 
                                           mut2gene=True, 
                                           with_indices=True)
            
            auc, age, sex = auc.squeeze().detach().cpu().numpy(), batch['age'].detach().cpu().numpy(), batch['sex'].detach().cpu().numpy()
            hazard_actual = self.labtrans.fit_transform(batch['time'].detach().cpu().numpy(), batch['event'].detach().cpu().numpy())

        # Standardized AUC and Age features
        df = pd.DataFrame({'AUC': auc, 'Age': age, 'Sex': sex}) 
        standards = [(['AUC'], StandardScaler()), (['Age'], StandardScaler())]
        binarized = [('Sex', None)]
        mapper = DataFrameMapper(standards + binarized)
        hazard_train = mapper.fit_transform(df).astype('float32')

        # Instantiate a Hazard Tranform Nueral Network Model
        out_features = self.labtrans.out_features
        func = tt.practical.MLPVanilla(self.num_features, self.hidden_layers, out_features, dropout=self.transform_dropout)
        hazard_transform = PCHazard(func, self.optimizer, duration_index=self.labtrans.cuts)

        # Train the Hazard Transform Modle
        log = hazard_transform.fit(hazard_train, hazard_actual, batch_size, epochs=self.epochs)
    
    def train_pycox_LH(self, dataloader, batch_size):

        nested_subtrees_forward = move_to(dataloader.dataset.tree_parser.get_nested_subtree_mask(['default'], direction='forward'), self.device)
        nested_subtrees_backward = move_to(dataloader.dataset.tree_parser.get_nested_subtree_mask(['default'], direction='backward'), self.device)
        gene2system_mask = move_to(torch.tensor(dataloader.dataset.tree_parser.gene2sys_mask, dtype=torch.bool), self.device)
        system2gene_mask = move_to(torch.tensor(dataloader.dataset.tree_parser.sys2gene_mask, dtype=torch.bool), self.device)
        
        for i, batch in enumerate(dataloader):
            batch = move_to(batch, self.device)
            
            # Get AUC predictions from patient profile
            auc = self.cell_response_model(batch['genotype'], batch['drug'],
                                           nested_subtrees_forward, nested_subtrees_backward,
                                           gene2system_mask,
                                           system2gene_mask,
                                           sys2cell=True, 
                                           cell2sys=True, 
                                           sys2gene=False,
                                           gene2drug=False, 
                                           mut2gene=True, 
                                           with_indices=True)
            
            auc, age, sex = auc.squeeze().detach().cpu().numpy(), batch['age'].detach().cpu().numpy(), batch['sex'].detach().cpu().numpy()
            hazard_actual = self.labtrans.fit_transform(batch['time'].detach().cpu().numpy(), batch['event'].detach().cpu().numpy())

        # Standardized AUC and Age features
        df = pd.DataFrame({'AUC': auc, 'Age': age, 'Sex': sex}) 
        standards = [(['AUC'], StandardScaler()), (['Age'], StandardScaler())]
        binarized = [('Sex', None)]
        mapper = DataFrameMapper(standards + binarized)
        hazard_train = mapper.fit_transform(df).astype('float32')

        # Instantiate a Hazard Tranform Nueral Network Model
        out_features = self.labtrans.out_features
        func = tt.practical.MLPVanilla(self.num_features , self.hidden_layers, out_features, dropout=self.transform_dropout)
        hazard_transform = LogisticHazard(func, self.optimizer, duration_index=self.labtrans.cuts, device=self.device)

        # Train the Hazard Transform Modle
        log = hazard_transform.fit(hazard_train, hazard_actual, batch_size, epochs=self.epochs)

    def merge_patient_data(self, response, indicators):

        response = pd.read_csv(response, header = None, sep = '\t')
        indicators = pd.read_csv(indicators, usecols = ['PATIENT_ID', 'Diagnosis Age', 'Sex', 'Progression Free Status'], sep = '\t')

        # Drop the patients that are missing a value for sex, age, or PFS status
        indicators = indicators.dropna()

        # Preprocess data
        sex = list(indicators['Sex'])
        sex = [1 if s == 'Male' else 0 for s in sex]
        indicators['Sex'] = sex
        status = list(indicators['Progression Free Status'])
        status = [int(s.split(":")[0]) for s in status]
        indicators['Progression Free Status'] = status

        # Merge response and indicators
        dataset = response.merge(indicators, left_on=0, right_on='PATIENT_ID', how='inner')
        dataset.drop(columns=['PATIENT_ID'], inplace=True)

        dataset.columns = range(dataset.shape[1])

        return dataset

    def get_transfered_models(self):

        return self.cell_response_model, self.hazard_transform
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Modeling hyperparameters
    parser.add_argument('--epochs', dest='epochs', type=int, default=200)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=None)
    parser.add_argument('--lr', dest='lr', type=float, default=0.01)
    parser.add_argument('--transform_hidden', dest='transform_hidden', type=int, default=32)
    parser.add_argument('--loss_function', dest='loss_function', type=str, default="cox_time")
    parser.add_argument('--num_durations', dest='num_durations', type=int, default=10)
    parser.add_argument('--num_features', dest='num_features', type=int, default=3)
    parser.add_argument('--transform_dropout', dest='transform_dropout', type=float, default=0.1)
    parser.add_argument('--unfreeze', dest='unfreeze', type=str, default=None)

    # Input cell-line datasets for model instantiation
    parser.add_argument('--cell_line_model', dest='cell_line_model', type=str, default="/cellar/users/zwallace/G2PT/models_nofolds_diff/rsi_model.pt")
    parser.add_argument('--cell_line_repsonse', dest='cell_line_response', type=str, default="/cellar/users/zwallace/G2PT/data/train_dataset_rsi.txt")
    parser.add_argument('--cell_mutations', dest='cell_mutations', type=str, default="/cellar/users/zwallace/G2PT/data/cell2mutation_ctg_av.txt")
    parser.add_argument('--cell_amplifics', dest='cell_amplifics', type=str, default="/cellar/users/zwallace/G2PT/data/old_copynumber/cell2cnamplification_ctg_av.txt")
    parser.add_argument('--cell_deletions', dest='cell_deletions', type=str, default="/cellar/users/zwallace/G2PT/data/old_copynumber/cell2cndeletion_ctg_av.txt")
    parser.add_argument('--cell_gene2ind', dest='cell_gene2ind', type=str, default="/cellar/users/zwallace/G2PT/data/gene2ind_ctg_av.txt")
    parser.add_argument('--ontology', dest='ontology', type=str, default="/cellar/users/zwallace/G2PT/data/ontology_ctg_av.txt")

    # Inpute patient datasets for model transfer
    parser.add_argument('--patient_repsonse', dest='patient_response', type=str, default="/cellar/users/zwallace/G2PT/data/test_PFS_gemcitabine.txt")
    parser.add_argument('--patient_indicators', dest='patient_indicators', type=str, default="/cellar/users/zwallace/G2PT/data/test_gemcitabine.txt")
    parser.add_argument('--patient_mutations', dest='patient_mutations', type=str, default="/cellar/users/zwallace/G2PT/data/cell2mutation_gemcitabine.txt")
    parser.add_argument('--patient_amplifics', dest='patient_amplifics', type=str, default="/cellar/users/zwallace/G2PT/data/cell2cnamplification_gemcitabine.txt")
    parser.add_argument('--patient_deletions', dest='patient_deletions', type=str, default="/cellar/users/zwallace/G2PT/data/cell2cndeletion_gemcitabine.txt")
    parser.add_argument('--patient2ind', dest='patient2ind', type=str, default="/cellar/users/zwallace/G2PT/data/cell2ind_gemcitabine.txt")
    parser.add_argument('--patient_gene2ind', dest='patient_gene2ind', type=str, default="/cellar/users/zwallace/G2PT/data/gene2ind_tcga.txt")

    # Model saving parameters
    parser.add_argument('--transfered_cell_model', dest='transfered_cell_model', type=str, default="/cellar/users/zwallace/G2PT/transfer/transfered_cell_model.pt")
    parser.add_argument('--hazard_transform_model', dest='hazard_transform_model', type=str, default="/cellar/users/zwallace/G2PT/transfer/hazard_transform_model.pt")
    
    args = parser.parse_args()

    print("Instaniating Cell-Line Model ...\n")
    patient_response_trainer = TransferCellResponse(cell_line_model=args.cell_line_model,
                                                    cell_line_response=args.cell_line_response,
                                                    cell_mutations=args.cell_mutations,
                                                    cell_amplifics=args.cell_amplifics,
                                                    cell_deletions=args.cell_deletions,
                                                    cell_gene2ind=args.cell_gene2ind,
                                                    ontology=args.ontology,
                                                    epochs=args.epochs,
                                                    batch_size=args.batch_size,
                                                    lr=args.lr,
                                                    transform_hidden=args.transform_hidden,
                                                    loss_function=args.loss_function,
                                                    num_durations=args.num_durations,
                                                    num_features=args.num_features,
                                                    transform_dropout=args.transform_dropout,
                                                    unfreeze=args.unfreeze)

    if args.unfreeze == None:
        print("Freezing all G2PT layers ...\n")
    elif args.unfreeze == "predictor":
        print("Unfreezing G2PT prediction layer ...\n")
    elif args.unfreeze == "translation":
        print("Unfreezing translation and prediction layer ...\n")
    elif args.unfreeze == "propagation":
        print("Unfreezing hiearchical propagation, translation, and prediction layer ...\n")
    elif args.unfreeze == "all":
        print("All G2PT layers are unfrozen (finetuning enitre model) ...\n")

    print("Finetuning the cell-line model on input patient datatset ...\n")
    patient_response_trainer.FineTunePatients(patient_response=args.patient_response,
                                              patient_indicators=args.patient_indicators,
                                              patient_mutations=args.patient_mutations,
                                              patient_amplifics=args.patient_amplifics,
                                              patient_deletions=args.patient_deletions,
                                              patient2ind=args.patient2ind,
                                              patient_gene2ind=args.patient_gene2ind)

    #transfered_cell_model, hazard_transform_model = patient_response_trainer.get_transfered_models()

    #print("Saving models to state_dict() ...")
    #torch.save(transfered_cell_model.state_dict(), args.transfered_cell_model)
    #torch.save(hazard_transform_model.state_dict(), args.hazard_transform_model)    
    