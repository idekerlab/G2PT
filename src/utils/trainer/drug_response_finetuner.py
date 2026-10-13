import gc
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import  StepLR
from sklearn import metrics
from scipy.stats import spearmanr
from tqdm import tqdm
import numpy as np
import copy
from src.utils.trainer import CCCLoss


class DrugResponseFineTuner(object):

    def __init__(self, drug_response_model, drug_response_dataloader, device, args, validation_dataloader=None):
        self.device = device
        self.drug_response_model = drug_response_model.to(self.device)
        self.drug_response_dataloader = drug_response_dataloader
        self.nested_subtrees = self.drug_response_dataloader.dataset.tree_parser.get_nested_subtree_mask(args.subtree_order)
        self.nested_subtrees = self.move_to(self.nested_subtrees, self.device)
        self.compound_loss = nn.L1Loss()
        self.ccc_loss = CCCLoss()
        self.beta = 0.05
        self.drug_response_loss = nn.SmoothL1Loss(beta=self.beta)
        self.optimizer = optim.SGD(self.drug_response_model.parameters(), lr=args.lr, weight_decay=args.wd)
        self.validation_dataloader = validation_dataloader
        self.l2_lambda = args.l2_lambda
        self.best_model = self.drug_response_model
        for name, param in self.drug_response_model.named_parameters():
            if 'compound_encoder' in name:
                print(name)
                param.requires_grad = False

        self.args = args


    def train(self, epochs, output_path=None):

        self.best_model = self.drug_response_model
        best_performance = 0
        for epoch in range(epochs):
            self.train_epoch(epoch + 1)
            gc.collect()
            torch.cuda.empty_cache()
            if self.validation_dataloader is not None:
                if (epoch % self.args.val_step)==0 & (epoch != 0):
                    performance = self.evaluate(self.drug_response_model, self.validation_dataloader, epoch+1, name="Validation")
                    if performance > best_performance:
                        self.best_model = copy.deepcopy(self.drug_response_model).to('cpu')
                    torch.cuda.empty_cache()
                    gc.collect()
                    if output_path:
                        output_path_epoch = output_path + ".%d"%epoch
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
                trues.append(batch['response_mean']+batch['response_residual'])
                batch = self.move_to(batch, self.device)
                compound_predicted, drug_response_predicted = model(batch['genotype'], batch['drug'], self.nested_subtrees)
                drug_response_predicted_detached = drug_response_predicted.detach().cpu().numpy()
                compound_predicted_detached = compound_predicted.detach().cpu().numpy()
                results.append(compound_predicted_detached + drug_response_predicted_detached)
                dataloader_with_tqdm.set_description("%s epoch: %d" % (name, epoch))
                del compound_predicted
                del compound_predicted_detached
                del drug_response_predicted
                del drug_response_predicted_detached
                del batch
        trues = np.concatenate(trues)
        results = np.concatenate(results)[:, 0]
        r_square = metrics.r2_score(trues, results)
        spearman = spearmanr(trues, results)
        print("R_square: ", r_square)
        print("Spearman Rho: ", spearman)

        r2_score_dict = {}
        spearman_dict = {}
        for smiles, indice in test_grouped.groups.items():
            r2 = metrics.r2_score(test_df.loc[indice][2], results[indice])
            rho = spearmanr(test_df.loc[indice][2], results[indice]).correlation
            if np.isnan(r2):
                print(test_df.loc[indice][2], results[indice])
            else:
                r2_score_dict[smiles] = r2
                spearman_dict[smiles] = rho
        spearman_per_drug = np.array(list(spearman_dict.values())).mean()
        print("Spearman per drug: ", spearman_per_drug)

        return spearman_per_drug


    def train_epoch(self, epoch):
        self.drug_response_model.train()
        self.iter_minibatches(self.drug_response_dataloader, epoch, name="Batch")


    def iter_minibatches(self, dataloader, epoch, name=""):
        mean_comp_loss = 0
        mean_response_loss = 0
        mean_ccc_loss = 0
        dataloader_with_tqdm = tqdm(dataloader)
        for i, batch in enumerate(dataloader_with_tqdm):
            batch_moved = self.move_to(batch, self.device)
            compound_predicted, drug_response_predicted = self.drug_response_model(batch_moved['genotype'], batch_moved['drug'], self.nested_subtrees)

            compound_loss = self.compound_loss(compound_predicted[:, 0], batch_moved['response_mean'].to(torch.float32))
            drug_response_loss = self.drug_response_loss(drug_response_predicted[:, 0], batch_moved['response_residual'].to(torch.float32))
            #ccc_loss = self.ccc_loss((batch['response_mean']+batch['response_residual']).to(torch.float32), compound_predicted[:, 0]+drug_response_predicted[:, 0])
            mean_comp_loss += float(compound_loss)
            mean_response_loss += float(drug_response_loss)
            #mean_ccc_loss += ccc_loss
            loss = compound_loss + drug_response_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            dataloader_with_tqdm.set_description("%s Train epoch: %d, Compound loss %.3f,  Drug Response loss: %.3f" % (
            name, epoch, mean_comp_loss / (i + 1), mean_response_loss / (i + 1) ))
            del loss
            del compound_loss, drug_response_loss
            del drug_response_predicted, compound_predicted
            del batch, batch_moved
        del mean_ccc_loss, mean_comp_loss, mean_response_loss



    def move_to(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device)
        elif isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                res[k] = self.move_to(v, device)
            return res
        elif isinstance(obj, list):
            res = []
            for v in obj:
                res.append(self.move_to(v, device))
            return res
        else:
            return obj.to(device)
