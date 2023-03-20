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
import copy



class G2PTrainer(object):

    def __init__(self, g2p_model, g2p_dataloader, device, args, validation_dataloader=None):
        self.device = device
        self.g2p_model = g2p_model.to(self.device)
        '''
        for name, param in self.drug_response_model.named_parameters():
            if "compound_encoder" in name:
                param.requires_grad = False
                print(name, param.requires_grad)
        '''

        #self.nested_subtrees = self.move_to(self.nested_subtrees, self.device)
        #self.gene2gene_mask = torch.tensor(self.drug_response_dataloader.dataset.tree_parser.gene2gene_mask, dtype=torch.float32)
        #self.gene2gene_mask = self.move_to(self.gene2gene_mask, self.device)
        self.compound_loss = nn.L1Loss()
        self.ccc_loss = CCCLoss()
        self.beta = 0.1
        self.phenotype_loss = nn.SmoothL1Loss(self.beta)
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.g2p_model.parameters()), lr=args.lr, weight_decay=args.wd)
        self.validation_dataloader = validation_dataloader
        self.g2p_dataloader = g2p_dataloader
        self.l2_lambda = args.l2_lambda
        self.best_model = self.g2p_model

        self.total_train_step = len(self.g2p_dataloader)*args.epochs# + len(self.drug_response_dataloader_cellline)*args.epochs
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, int(0.2 * self.total_train_step),
                                                          self.total_train_step)
        self.nested_subtrees_forward = self.g2p_dataloader.dataset.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='forward')
        self.nested_subtrees_forward = self.move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = self.g2p_dataloader.dataset.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='backward')
        self.nested_subtrees_backward = self.move_to(self.nested_subtrees_backward, device)
        self.system2gene_mask = self.move_to(torch.tensor(self.g2p_dataloader.dataset.tree_parser.system2gene_mask, dtype=torch.bool), device)
        self.args = args
        self.system_positional_embedding = copy.deepcopy(self.g2p_model.system_embedding_pos_enc)

    def train(self, epochs, output_path=None):

        self.best_model = self.g2p_model
        best_performance = 0
        for epoch in range(epochs):
            self.train_epoch(epoch + 1)
            gc.collect()
            torch.cuda.empty_cache()

            if (epoch % self.args.val_step)==0 & (epoch != 0):
                if self.validation_dataloader is not None:
                    performance = self.evaluate(self.g2p_model, self.validation_dataloader, epoch+1, name="Validation")
                    if performance > best_performance:
                        self.best_model = copy.deepcopy(self.g2p_model).to('cpu')
                    torch.cuda.empty_cache()
                    gc.collect()
                if output_path:
                    output_path_epoch = output_path + ".%d"%epoch
                    print("Save to...", output_path_epoch)
                    torch.save(self.g2p_model, output_path_epoch)
            #self.lr_scheduler.step()

    def get_best_model(self):
        return self.best_model

    def evaluate(self, model, dataloader, epoch, name="Validation"):
        trues = []
        results = []
        dataloader_with_tqdm = tqdm(dataloader)

        test_df = dataloader.dataset.g2p_df.reset_index()
        test_grouped = test_df.reset_index().groupby(1)
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader_with_tqdm):
                trues.append(batch['phenotype'])
                batch = self.move_to(batch, self.device)
                phenotype_predicted = model(batch['genotype'], self.nested_subtrees_forward, self.nested_subtrees_backward,)
                phenotype_predicted_detached = phenotype_predicted.detach().cpu().numpy()
                #compound_predicted_detached = compound_predicted.detach().cpu().numpy()
                #results.append(compound_predicted_detached+drug_response_predicted_detached)
                results.append(phenotype_predicted_detached)
                dataloader_with_tqdm.set_description("%s epoch: %d" % (name, epoch))
                #del compound_predicted
                #del compound_predicted_detached
                del phenotype_predicted
                del phenotype_predicted_detached
                del batch
        trues = np.concatenate(trues)
        results = np.concatenate(results)[:, 0]
        r_square = metrics.r2_score(trues, results)
        pearson = pearsonr(trues, results)
        spearman = spearmanr(trues, results)
        print("R_square: ", r_square)
        print("Pearson R", pearson)
        print("Spearman Rho: ", spearman)

        return pearson[0]


    def train_epoch(self, epoch):
        self.g2p_model.train()
        self.iter_minibatches(self.g2p_dataloader, epoch, name="Batch", ccc=True)
        #self.iter_minibatches(self.drug_response_dataloader_cellline, epoch, name="CellLineBatch", ccc=False)


    def iter_minibatches(self, dataloader, epoch, name="", ccc=True):
        mean_comp_loss = 0.
        mean_response_loss = 0.
        mean_ccc_loss = 0.
        dataloader_with_tqdm = tqdm(dataloader)
        for i, batch in enumerate(dataloader_with_tqdm):
            batch = self.move_to(batch, self.device)
            phenotype_predicted = self.g2p_model(batch['genotype'], self.nested_subtrees_forward, self.nested_subtrees_backward)

            #compound_loss = self.compound_loss(compound_predicted[:, 0], batch['response_mean'].to(torch.float32))
            #drug_response_loss = self.drug_response_loss(compound_predicted[:, 0]+drug_response_predicted[:, 0], (batch['response_mean']+batch['response_residual']).to(torch.float32))
            phenotype_loss = self.phenotype_loss(phenotype_predicted[:, 0],(batch['phenotype']).to(torch.float32))
            #ccc_loss = self.ccc_loss((batch['response_mean']+batch['response_residual']).to(torch.float32), compound_predicted[:, 0]+drug_response_predicted[:, 0])
            ccc_loss = self.ccc_loss((batch['phenotype']).to(torch.float32), phenotype_predicted[:, 0])
            #drug_response_loss = self.drug_response_loss((drug_response_predicted[:, 0]), (batch['response_residual']).to(torch.float32))
            #ccc_loss = self.ccc_loss((batch['response_residual']).to(torch.float32),  drug_response_predicted[:, 0])

            mean_response_loss += float(phenotype_loss)
            mean_ccc_loss += float(ccc_loss)
            #print(compound_predicted, drug_response_predicted)
            #print(compound_loss, drug_response_loss, ccc_loss)
            #break
            loss =  phenotype_loss
            if ccc:
                loss = loss + ccc_loss * self.beta
            #print("Loss", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.g2p_model.system_positional_embedding = self.system_positional_embedding
            dataloader_with_tqdm.set_description("%s Train epoch: %d, Phenotype loss: %.3f, CCCLoss: %.3f" % (
            name, epoch, mean_response_loss / (i + 1), mean_ccc_loss / (i+1) ))
            del loss
            del phenotype_loss, ccc_loss
            del phenotype_predicted
            del batch
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
