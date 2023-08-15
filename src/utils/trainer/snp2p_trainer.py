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
from src.utils.data import move_to
from transformers import get_linear_schedule_with_warmup
from src.utils.trainer import CCCLoss
import copy



class SNP2PTrainer(object):

    def __init__(self, snp2p_model, snp2p_dataloader, device, args, validation_dataloader=None, fix_system=False):
        self.device = device
        self.snp2p_model = snp2p_model.to(self.device)
        '''
        for name, param in self.drug_response_model.named_parameters():
            if "compound_encoder" in name:
                param.requires_grad = False
                print(name, param.requires_grad)
        '''

        #self.nested_subtrees = self.move_to(self.nested_subtrees, self.device)
        #self.gene2gene_mask = torch.tensor(self.drug_response_dataloader.dataset.tree_parser.gene2gene_mask, dtype=torch.float32)
        #self.gene2gene_mask = self.move_to(self.gene2gene_mask, self.device)
        self.ccc_loss = CCCLoss(mean_diff=False)
        if args.regression:
            self.ccc = True
        else:
            self.ccc = False
        self.beta = 0.1
        if args.regression:
            self.phenotype_loss = nn.L1Loss()
        else:
            self.phenotype_loss = nn.BCELoss()
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.snp2p_model.parameters()), lr=args.lr, weight_decay=args.wd)
        self.validation_dataloader = validation_dataloader
        self.snp2p_dataloader = snp2p_dataloader
        self.l2_lambda = args.l2_lambda
        self.best_model = self.snp2p_model

        self.total_train_step = len(self.snp2p_dataloader)*args.epochs# + len(self.drug_response_dataloader_cellline)*args.epochs
        #self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10)
        self.nested_subtrees_forward = self.snp2p_dataloader.dataset.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='forward')
        self.nested_subtrees_forward = move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = snp2p_dataloader.dataset.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='backward')
        self.nested_subtrees_backward = move_to(self.nested_subtrees_backward, device)
        self.sys2gene_mask = move_to(torch.tensor(self.snp2p_dataloader.dataset.tree_parser.sys2gene_mask, dtype=torch.bool), device)
        self.gene2sys_mask = self.sys2gene_mask.T
        self.args = args
        self.fix_system = fix_system
        self.g2p_module_names = ["Mut2Sys","Sys2Cell", "Cell2Sys"]
        #self.system_embedding = copy.deepcopy(self.g2p_model.system_embedding)

    def train(self, epochs, output_path=None):

        self.best_model = self.snp2p_model
        best_performance = 0
        for epoch in range(epochs):
            self.train_epoch(epoch + 1)
            gc.collect()
            torch.cuda.empty_cache()

            if (epoch % self.args.val_step)==0 & (epoch != 0):
                if self.validation_dataloader is not None:
                    if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                                                     and self.args.rank % torch.cuda.device_count() == 0):
                        performance = self.evaluate(self.snp2p_model, self.validation_dataloader, epoch + 1,
                                                    name="Validation")
                        if performance > best_performance:
                            self.best_model = copy.deepcopy(self.snp2p_model).to('cpu')
                        torch.cuda.empty_cache()
                        gc.collect()
                        if output_path:
                            output_path_epoch = output_path + ".%d"%epoch
                            print("Save to...", output_path_epoch)
                            if self.args.multiprocessing_distributed:
                                torch.save(self.snp2p_model.module, output_path_epoch)
                            else:
                                torch.save(self.snp2p_model, output_path_epoch)
                #if output_path:
                #    output_path_epoch = output_path + ".%d"%epoch
                #    print("Save to...", output_path_epoch)
                #    torch.save(self.g2p_model, output_path_epoch)
            #self.lr_scheduler.step()

    def get_best_model(self):
        return self.best_model

    def evaluate(self, model, dataloader, epoch, name="Validation"):
        trues = []
        '''
        result_dic = {"Mut2Sys":[]}
        if self.args.sys2cell:
            result_dic["Sys2Cell"] = []
        if self.args.cell2sys:
            result_dic["Cell2Sys"] = []
        
        '''
        dataloader_with_tqdm = tqdm(dataloader)
        results = []
        model.to(self.device)
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader_with_tqdm):
                trues.append(batch['phenotype'])
                batch = move_to(batch, self.device)
                phenotype_predicted = model(batch['genotype'],
                                            self.nested_subtrees_forward,
                                            self.nested_subtrees_backward,
                                            gene2sys_mask=self.gene2sys_mask,
                                            sys2gene_mask=self.sys2gene_mask,
                                            sys2cell=self.args.sys2cell,
                                            cell2sys=self.args.cell2sys,
                                            sys2gene=self.args.sys2gene)
                #for phenotype_predicted_i, module_name in zip(phenotype_predicted, self.g2p_module_names):
                phenotype_predicted_detached = phenotype_predicted.detach().cpu().numpy()
                results.append(phenotype_predicted_detached)
                dataloader_with_tqdm.set_description("%s epoch: %d" % (name, epoch))
                del phenotype_predicted
                del phenotype_predicted_detached
                del batch
        trues = np.concatenate(trues)
        #for module_name, value in result_dic.items():
        #    if len(value)>0:
        results = np.concatenate(results)[:, 0]
        print(trues[:50])
        print(results[:50])
        if self.args.regression:
            r_square = metrics.r2_score(trues, results)
            pearson = pearsonr(trues, results)
            spearman = spearmanr(trues, results)
            performance = pearson[0]
            #print(module_name)
            print("R_square: ", r_square)
            print("Pearson R", pearson)
            print("Spearman Rho: ", spearman)
        else:
            performance = metrics.average_precision_score(trues, results)
            print("AUPR: ", performance)

        return performance


    def train_epoch(self, epoch):
        self.snp2p_model.train()
        if self.args.multiprocessing_distributed:
            self.snp2p_dataloader.sampler.set_epoch(epoch)
        self.iter_minibatches(self.snp2p_dataloader, epoch, name="Batch", snp_loss=False, ccc=self.ccc)


    def iter_minibatches(self, dataloader, epoch, name="", snp_loss=True, ccc=True):
        '''
        mean_response_loss_dict = {"Mut2Sys":0.}
        if self.args.sys2cell:
            mean_response_loss_dict["Sys2Cell"] = 0.
        if self.args.cell2sys:
            mean_response_loss_dict["Cell2Sys"] = 0.
        '''
        mean_response_loss = 0.
        mean_ccc_loss = 0.
        dataloader_with_tqdm = tqdm(dataloader)
        #phenotype_vector = copy.deepcopy(self.g2p_model.phenotype_vector.weight)
        #system_embedding = copy.deepcopy(self.g2p_model.system_embedding.weight)
        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        for i, batch in enumerate(dataloader_with_tqdm):
            batch = move_to(batch, self.device)
            #GPUtil.showUtilization()
            phenotype_predicted = self.snp2p_model(batch['genotype'],
                                                     self.nested_subtrees_forward,
                                                     self.nested_subtrees_backward,
                                                     gene2sys_mask=self.gene2sys_mask,
                                                     sys2gene_mask=self.sys2gene_mask,
                                                     sys2cell=self.args.sys2cell,
                                                     cell2sys=self.args.cell2sys,
                                                     sys2gene=self.args.sys2gene)
            #phenotype_vector_cur = copy.deepcopy(self.g2p_model.phenotype_vector.weight)
            #print(phenotype_vector_cur)
            #system_embedding_cur = copy.deepcopy(self.g2p_model.system_embedding.weight)
            #print((phenotype_vector - phenotype_vector_cur).sum(), (system_embedding - system_embedding_cur).sum())
            #phenotype_vector = phenotype_vector_cur
            #system_embedding = system_embedding_cur
            phenotype_loss = 0
            ccc_loss = 0
            #GPUtil.showUtilization()
            #for phenotype_predicted_i, module_name in zip(phenotype_predicted, self.g2p_module_names):
            phenotype_loss_result = self.phenotype_loss(phenotype_predicted[:, 0],(batch['phenotype']).to(torch.float32))
            phenotype_loss += phenotype_loss_result
            mean_response_loss += float(phenotype_loss_result)
            if ccc:
                ccc_loss_result = self.ccc_loss((batch['phenotype']).to(torch.float32), phenotype_predicted[:, 0])
                ccc_loss += ccc_loss_result
                mean_ccc_loss += float(ccc_loss_result)
            #loss_format = #" , ".join(["%s:%3f"%(module_name, value/(i + 1)) for module_name, value in mean_response_loss_dict.items()])
            #GPUtil.showUtilization()
            loss =  phenotype_loss
            cosine_loss = 0.
            ratio_loss = 0.
            if snp_loss:
                if self.args.multiprocessing_distributed:
                    cosine_loss = (1 - cos(self.snp2p_model.module.homozygous_a1_embedding.weight,self.snp2p_model.module.heterozygous_embedding.weight)).mean()
                    ratio_loss = torch.relu(torch.norm(self.snp2p_model.module.heterozygous_embedding.weight, dim=0) - torch.norm(self.snp2p_model.module.homozygous_a1_embedding.weight, dim=0)).mean()
                else:
                    cosine_loss = (1 - cos(self.snp2p_model.homozygous_a1_embedding.weight,self.snp2p_model.heterozygous_embedding.weight)).mean()
                    ratio_loss = torch.relu(torch.norm(self.snp2p_model.heterozygous_embedding.weight, dim=0) - torch.norm(self.snp2p_model.homozygous_a1_embedding.weight, dim=0)).mean()
                loss = loss + 0.1 * (cosine_loss + ratio_loss)
            if ccc:
                loss = loss + 0.1 * ccc_loss
            #GPUtil.showUtilization()
            #print("Loss", loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()
            #if self.fix_system:
            #    self.g2p_model.system_embedding = self.system_embedding
            dataloader_with_tqdm.set_description("%s Train epoch: %3.f, Phenotype loss: %.3f, CCCLoss: %.3f, CosLoss: %.3f, RatioLoss: %.3f, Data time %.3f, %.3f" % (
            name, epoch, mean_response_loss/(i + 1), mean_ccc_loss/(i + 1), cosine_loss, ratio_loss, batch['datatime'], batch['time']))
            del loss
            del phenotype_loss, ccc_loss
            if snp_loss:
                del cosine_loss, ratio_loss
            del phenotype_predicted
            del batch
        del mean_ccc_loss
