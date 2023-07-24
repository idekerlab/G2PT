import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import  StepLR
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
import copy
from transformers import get_linear_schedule_with_warmup
from src.utils.trainer import CCCLoss
from src.utils.data import move_to
import copy



class DrugResponseTrainer(object):

    def __init__(self, drug_response_model, drug_response_dataloader_drug, drug_response_dataloader_cellline, device, args, validation_dataloader=None, fix_system=False):
        self.device = device
        self.drug_response_model = drug_response_model.to(self.device)
        '''
        for name, param in self.drug_response_model.named_parameters():
            if "compound_encoder" in name:
                param.requires_grad = False
                print(name, param.requires_grad)
        '''
        self.drug_response_dataloader_drug = drug_response_dataloader_drug
        self.drug_response_dataloader_cellline = drug_response_dataloader_cellline

        #self.nested_subtrees = self.move_to(self.nested_subtrees, self.device)
        #self.gene2gene_mask = torch.tensor(self.drug_response_dataloader.dataset.tree_parser.gene2gene_mask, dtype=torch.float32)
        #self.gene2gene_mask = self.move_to(self.gene2gene_mask, self.device)
        self.compound_loss = nn.L1Loss()
        self.ccc_loss = CCCLoss()
        self.beta = 0.1
        self.drug_response_loss = nn.SmoothL1Loss(self.beta)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.drug_response_model.parameters()), lr=args.lr, weight_decay=args.wd)
        self.validation_dataloader = validation_dataloader
        self.l2_lambda = args.l2_lambda
        self.best_model = self.drug_response_model
        self.total_train_step = len(self.drug_response_dataloader_drug)*args.epochs# + len(self.drug_response_dataloader_cellline)*args.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(0.2 * self.total_train_step),
                                                          self.total_train_step)
        self.nested_subtrees_forward = self.drug_response_dataloader_drug.dataset.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='forward')
        self.nested_subtrees_forward = move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = self.drug_response_dataloader_drug.dataset.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='backward')
        self.nested_subtrees_backward = move_to(self.nested_subtrees_backward, device)
        self.system2gene_mask = move_to(torch.tensor(self.drug_response_dataloader_drug.dataset.tree_parser.sys2gene_mask, dtype=torch.bool), device)
        print("%d sys2gene in Dataloader" % self.drug_response_dataloader_drug.dataset.tree_parser.sys2gene_mask.sum())
        self.args = args
        #self.compound_encoder = copy.deepcopy(self.drug_response_model.compound_encoder)
        self.fix_system = fix_system
        self.g2p_module_names = ["Mut2Sys", "Sys2Cell", "Cell2Sys"]
        #self.system_embedding = copy.deepcopy(self.drug_response_model.system_embedding)
        #print(self.system2gene_mask)
        #print(self.nested_subtrees_forward)
        #print(self.nested_subtrees_backward)

    def train(self, epochs, output_path=None):

        self.best_model = self.drug_response_model
        best_performance = 0
        for epoch in range(epochs):
            self.train_epoch(epoch + 1)
            gc.collect()
            torch.cuda.empty_cache()

            if (epoch % self.args.val_step)==0 & (epoch != 0):
                if self.validation_dataloader is not None:
                    performance = self.evaluate(self.drug_response_model, self.validation_dataloader, epoch+1, name="Validation")
                    if performance > best_performance:
                        self.best_model = copy.deepcopy(self.drug_response_model).to('cpu')
                    torch.cuda.empty_cache()
                    gc.collect()
                if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                                                 and self.args.rank % torch.cuda.device_count() == 0):
                    if output_path:
                        output_path_epoch = output_path + ".%d"%epoch
                        print("Save to...", output_path_epoch)
                        if self.args.multiprocessing_distributed:
                            torch.save(self.drug_response_model.module, output_path_epoch)
                        else:
                            torch.save(self.drug_response_model, output_path_epoch)
            #self.lr_scheduler.step()

    def get_best_model(self):
        return self.best_model

    def evaluate(self, model, dataloader, epoch, name="Validation"):
        trues = []
        results = []
        dataloader_with_tqdm = tqdm(dataloader)

        test_df = dataloader.dataset.drug_response_df.reset_index()
        test_grouped = test_df.reset_index().groupby(1)
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader_with_tqdm):
                trues.append(batch['response_mean'] + batch['response_residual'])
                batch = move_to(batch, self.device)
                drug_response_predicted = model(batch['genotype'], batch['drug'],
                                                self.nested_subtrees_forward, self.nested_subtrees_backward, self.system2gene_mask,
                                                sys2cell=self.args.sys2cell,
                                                cell2sys=self.args.cell2sys,
                                                sys2gene=self.args.sys2gene)
                drug_response_predicted_detached = drug_response_predicted.detach().cpu().numpy()
                #compound_predicted_detached = compound_predicted.detach().cpu().numpy()
                #results.append(compound_predicted_detached+drug_response_predicted_detached)
                results.append(drug_response_predicted_detached)
                dataloader_with_tqdm.set_description("%s epoch: %d" % (name, epoch))
                del drug_response_predicted
                del drug_response_predicted_detached
                del batch
        trues = np.concatenate(trues)
        results = np.concatenate(results)[:, 0]
        r_square = metrics.r2_score(trues, results)
        pearson = pearsonr(trues, results)
        spearman = spearmanr(trues, results)
        print("R_square: ", r_square)
        print("Pearson R", pearson)
        print("Spearman Rho: ", spearman)

        r2_score_dict = {}
        pearson_dict = {}
        spearman_dict = {}
        for smiles, indice in test_grouped.groups.items():
            if len(indice) <= 1:
                continue
            #print(test_df.loc[indice][2])
            #print(results[indice])
            r2 = metrics.r2_score(test_df.loc[indice][2], results[indice])
            rho = spearmanr(test_df.loc[indice][2], results[indice]).correlation
            pearson = pearsonr(test_df.loc[indice][2], results[indice])[0]
            if np.isnan(r2):
                print(test_df.loc[indice][2], results[indice])
            else:
                r2_score_dict[smiles] = r2
                spearman_dict[smiles] = rho
                pearson_dict[smiles] = pearson
        pearson_per_drug = np.array(list(pearson_dict.values())).mean()
        spearman_per_drug = np.array(list(spearman_dict.values())).mean()
        print("Pearson per drug: ", pearson_per_drug)
        print("Spearman per drug: ", spearman_per_drug)

        return pearson_per_drug


    def train_epoch(self, epoch):
        self.drug_response_model.train()
        self.iter_minibatches(self.drug_response_dataloader_drug, epoch, name="DrugBatch", ccc=False)
        #self.iter_minibatches(self.drug_response_dataloader_cellline, epoch, name="CellLineBatch", ccc=False)


    def iter_minibatches(self, dataloader, epoch, name="", ccc=True):
        mean_comp_loss = 0.
        mean_response_loss = 0.
        mean_ccc_loss = 0.
        dataloader_with_tqdm = tqdm(dataloader)
        for i, batch in enumerate(dataloader_with_tqdm):
            batch = move_to(batch, self.device)
            drug_response_predicted = self.drug_response_model(batch['genotype'], batch['drug'],
                                                           self.nested_subtrees_forward, self.nested_subtrees_backward, self.system2gene_mask,
                                                               sys2cell=self.args.sys2cell,
                                                               cell2sys=self.args.cell2sys,
                                                               sys2gene=self.args.sys2gene)

            #compound_loss = self.compound_loss(compound_predicted[:, 0], batch['response_mean'].to(torch.float32))
            #drug_response_loss = self.drug_response_loss(compound_predicted[:, 0]+drug_response_predicted[:, 0], (batch['response_mean']+batch['response_residual']).to(torch.float32))
            drug_response_loss = self.drug_response_loss(drug_response_predicted[:, 0],(batch['response_mean'] + batch['response_residual']).to(torch.float32))
            #ccc_loss = self.ccc_loss((batch['response_mean']+batch['response_residual']).to(torch.float32), compound_predicted[:, 0]+drug_response_predicted[:, 0])
            ccc_loss = self.ccc_loss((batch['response_mean'] + batch['response_residual']).to(torch.float32), drug_response_predicted[:, 0])
            #drug_response_loss = self.drug_response_loss((drug_response_predicted[:, 0]), (batch['response_residual']).to(torch.float32))
            #ccc_loss = self.ccc_loss((batch['response_residual']).to(torch.float32),  drug_response_predicted[:, 0])

            #mean_comp_loss += float(compound_loss)
            mean_response_loss += float(drug_response_loss)
            mean_ccc_loss += float(ccc_loss)
            #print(compound_predicted, drug_response_predicted)
            #print(compound_loss, drug_response_loss, ccc_loss)
            #break
            loss =  drug_response_loss
            if ccc:
                loss = loss + ccc_loss * self.beta
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            #self.drug_response_model.compound_encoder = self.compound_encoder
            #if self.fix_system:
            #    self.drug_response_model.system_embedding = self.system_embedding
            dataloader_with_tqdm.set_description("%s Train epoch: %d, Compound loss %.3f,  Drug Response loss: %.3f, CCCLoss: %.3f" % (
            name, epoch, mean_comp_loss / (i + 1), mean_response_loss / (i + 1), mean_ccc_loss / (i+1) ))
            del loss
            del drug_response_loss, ccc_loss
            del drug_response_predicted
            del batch
        del mean_ccc_loss, mean_comp_loss, mean_response_loss

