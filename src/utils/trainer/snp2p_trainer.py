import gc
import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
import copy
from src.utils.data import move_to
from src.utils.trainer import CCCLoss, FocalLoss, VarianceLoss, MultiplePhenotypeLoss
import torch.nn.functional as F
import copy

def _corrcoef(x):                         # x: [B, P]
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, unbiased=False) + 1e-6)
    return (x.T @ x) / (x.size(0) - 1)    # [P, P]

def correlation_matching_loss(pred, target, lam=0.05):
    """
    pred   : [B, P]  logits or probabilities
    target : [B, P]  0/1 labels (float32)
    """
    c_pred = _corrcoef(pred.detach())     # stop grad through corr
    c_true = _corrcoef(target)
    tri_mask = torch.triu(torch.ones_like(c_pred), diagonal=1).bool()
    diff = torch.abs(c_pred - c_true)[tri_mask]
    return lam * diff.mean()

def linear_temperature_schedule(epoch, total_epochs, T_init=1.0, T_final=0.1):
    return max(T_final, T_init - (epoch / total_epochs) * (T_init - T_final))

def get_param_groups(model, base_lr):
    lora_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'lora' in name.lower():
            lora_params.append(param)
        else:
            base_params.append(param)

    return [
        {'params': base_params, 'lr': base_lr},
        {'params': lora_params, 'lr': base_lr * 10}
    ]


class SNP2PTrainer(object):

    def __init__(self, snp2p_model, tree_parser, snp2p_dataloader, device, args, validation_dataloader=None, fix_system=False, pretrain_dataloader=None):
        self.args = args
        self.device = device
        self.snp2p_model = snp2p_model.to(self.device)
        self.ccc_loss = CCCLoss()
        self.beta = 0.1
        '''
        if args.regression:
            self.phenotype_loss = nn.MSELoss()
            self.variance_loss = VarianceLoss()
        else:
            self.phenotype_loss = nn.BCELoss()
        '''
        self.loss = MultiplePhenotypeLoss(args.bt_inds, args.qt_inds)
        self.phenotypes = args.pheno_ids
        self.qt = args.qt
        self.qt_inds = args.qt_inds
        self.bt = args.bt
        self.bt_inds = args.bt_inds
        self.optimizer = optim.AdamW(get_param_groups(self.snp2p_model, args.lr),
                                     weight_decay=args.wd)
        self.validation_dataloader = validation_dataloader
        self.snp2p_dataloader = snp2p_dataloader
        self.pretrain_dataloader = pretrain_dataloader
        self.pretrain_epochs = 1
        self.best_model = self.snp2p_model
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        #self.total_train_step = len(
        #    self.snp2p_dataloader) * args.epochs  # + len(self.drug_response_dataloader_cellline)*args.epochs
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 10)
        self.snp2gene_mask = move_to(torch.tensor(tree_parser.snp2gene_mask, dtype=torch.float32), device)
        self.nested_subtrees_forward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='forward', format='indices')#self.args.input_format)
        self.nested_subtrees_forward = move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = tree_parser.get_hierarchical_interactions(
            tree_parser.interaction_types, direction='backward', format='indices')#self.args.input_format)
        self.nested_subtrees_backward = move_to(self.nested_subtrees_backward, device)
        self.sys2gene_mask = move_to(
            torch.tensor(tree_parser.sys2gene_mask, dtype=torch.float32), device)
        self.gene2sys_mask = self.sys2gene_mask.T
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        self.fix_system = fix_system
        self.dynamic_phenotype_sampling = args.dynamic_phenotype_sampling

        n_sys2pad = int(np.ceil(len(tree_parser.sys2ind)/8)*8) - len(tree_parser.sys2ind)
        system_temp_tensor = [np.log(1+len(tree_parser.sys2gene_full[tree_parser.ind2sys[i]])) for i in range(len(tree_parser.sys2ind))] + [10.0] * n_sys2pad
        self.system_temp_tensor = move_to(torch.tensor(system_temp_tensor, dtype=torch.float32), device)

    def train(self, epochs, output_path=None):
        ccc = False
        '''
        self.dynamic_phenotype_sampling = False
        self.snp2p_dataloader.dataset.dataset.dynamic_phenotype_sampling = False
        for epoch in range(self.pretrain_epochs):
            self.snp2p_model.train()
            #self.pretrain_dataloader.dataset.dynamic_phenotype_sampling = False
            self.pretrain(self.snp2p_dataloader, epoch, name='Pretrain')
            gc.collect()
            torch.cuda.empty_cache()
        

        if not self.args.distributed or (self.args.distributed
                                                         and self.args.rank % torch.cuda.device_count() == 0):
            performance = self.evaluate(self.snp2p_model, self.validation_dataloader, 0, name='Validation', print_importance=False)
            gc.collect()
            torch.cuda.empty_cache()
        self.snp2p_dataloader.dataset.dataset.dynamic_phenotype_sampling = False
        self.dynamic_phenotype_sampling = True
        '''
        #self.best_model = self.snp2p_model
        best_performance = 0

        if self.dynamic_phenotype_sampling:
            self.snp2p_dataloader.dataset.dataset.dynamic_phenotype_sampling = True
        '''
        for epoch in range(5):
            self.iter_minibatches(self.snp2p_dataloader, epoch, name="SNP Adaptation :", snp_only=True)
        if not self.args.distributed or (self.args.distributed and self.args.rank == 0):
            self.evaluate(self.snp2p_model, self.validation_dataloader, epoch + 1,
                         name="SNP adapdation validation", print_importance=False, snp_only=True)

        if self.args.distributed:
            model = self.snp2p_model.module
        else:
            model = self.snp2p_model
        
        for name, param in model.named_parameters():
            if 'snp_embedding' in name:
                param.requires_grad = False
                print(name, " become freezed")
            if 'block_embedding' in name:
                param.requires_grad = False
                print(name, " become freezed")
        '''
        self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.snp2p_model.parameters()), lr=self.args.lr,
                                     weight_decay=self.args.wd)

        for epoch in range(self.args.start_epoch, epochs):
            self.train_epoch(epoch + 1, ccc=ccc)
            gc.collect()
            torch.cuda.empty_cache()

            if (epoch % self.args.val_step) == 0 & (epoch != 0):
                if self.validation_dataloader is not None:
                    if not self.args.distributed or (self.args.distributed and self.args.rank == 0):
                        performance = self.evaluate(self.snp2p_model, self.validation_dataloader, epoch + 1,
                                                    name="Validation", print_importance=False)
                        torch.cuda.empty_cache()
                        gc.collect()
                if output_path:
                    output_path_epoch = output_path + ".%d" % epoch
                    if self.args.distributed:
                        if self.args.rank == 0:
                            print("Save to...", output_path_epoch)
                            torch.save({"arguments": self.args,
                                    "state_dict": self.snp2p_model.module.state_dict()},
                                   output_path_epoch)
                    else:
                        print("Save to...", output_path_epoch)
                        torch.save(
                            {"arguments": self.args, "state_dict": self.snp2p_model.state_dict()},
                            output_path_epoch)
            self.scheduler.step()

    def evaluate(self, model, dataloader, epoch, name="Validation", print_importance=False, snp_only=False):
        trues = []
        dataloader_with_tqdm = tqdm(dataloader)
        results = []
        covariates = []
        sys_scores = []
        gene_scores = []
        #model.to(self.device)
        model.eval()
        '''
        if epoch > 5:
            ld = True
        else:
            ld = False
        '''
        with torch.no_grad():
            for i, batch in enumerate(dataloader_with_tqdm):
                trues.append(batch['phenotype'])
                covariates.append(batch['covariates'].detach().cpu().numpy())
                batch = move_to(batch, self.device)
                phenotype_predicted = model(batch['genotype'], batch['covariates'], batch['phenotype_indices'],
                                                   self.nested_subtrees_forward,
                                                   self.nested_subtrees_backward,
                                                   snp2gene_mask = self.snp2gene_mask,
                                                                   gene2sys_mask=self.gene2sys_mask,#batch['gene2sys_mask'],
                                                                   sys2gene_mask=self.sys2gene_mask,
                                                                   sys2env=self.args.sys2env,
                                                                   env2sys=self.args.env2sys,
                                                                   sys2gene=self.args.sys2gene,
                                                                   snp_only=snp_only, ld=True,)
                                                       #sys_temp= self.system_temp_tensor)
                #for phenotype_predicted_i, module_name in zip(phenotype_predicted, self.g2p_module_names):
                if len(phenotype_predicted.size())==3:
                    phenotype_predicted = phenotype_predicted[:, :, 0]
                else:
                    phenotype_predicted = phenotype_predicted
                phenotype_predicted_detached = phenotype_predicted.detach().cpu().numpy()

                #sys_scores.append(sys_score.detach().cpu().numpy())
                #gene_scores.append(gene_score.detach().cpu().numpy())

                results.append(phenotype_predicted_detached)
                dataloader_with_tqdm.set_description("%s epoch: %d" % (name, epoch))
                del phenotype_predicted
                del phenotype_predicted_detached
                del batch
        trues = np.concatenate(trues)
        covariates = np.concatenate(covariates)
        results = np.concatenate(results)

        for t, i in zip(self.qt, self.qt_inds):
            mask = (trues[:, i] != -9)
            r_square = metrics.r2_score(trues[mask, i], results[mask, i])
            pearson = pearsonr(trues[mask, i], results[mask, i])
            spearman = spearmanr(trues[mask, i], results[mask, i])
            performance = pearson[0]
            #print(module_name)
            print("Performance overall for %s"%t)
            print("R_square: ", r_square)
            print("Pearson R", pearson[0])
            print("Spearman Rho: ", spearman[0])

            print("Performance female")
            female_indices = (covariates[:, 0]==1) & (trues[:, i] != -9)
            r_square = metrics.r2_score(trues[female_indices, i], results[female_indices, i])
            pearson = pearsonr(trues[female_indices, i], results[female_indices, i])
            spearman = spearmanr(trues[female_indices, i], results[female_indices, i])
            female_performance = pearson[0]
            #print(module_name)
            print("R_square: ", r_square)
            print("Pearson R", pearson[0])
            print("Spearman Rho: ", spearman[0])

            print("Performance male")
            male_indices = (covariates[:, 1] == 1) & (trues[:, i] != -9)
            r_square = metrics.r2_score(trues[male_indices, i], results[male_indices, i])
            pearson = pearsonr(trues[male_indices, i], results[male_indices, i])
            spearman = spearmanr(trues[male_indices, i], results[male_indices, i])
            male_performance = pearson[0]
            #print(module_name)
            print("R_square: ", r_square)
            print("Pearson R", pearson[0])
            print("Spearman Rho: ", spearman[0])
            print(" ")
        for t, i in zip(self.bt, self.bt_inds):
            print(trues[:50, i])
            print(results[:50, i])
            auc_performance = metrics.roc_auc_score(trues[:, i], results[:, i])
            performance = metrics.average_precision_score(trues[:, i], results[:, i])

            print("Performance overall for %s"%t)
            print("AUC: ", auc_performance)
            print("AUPR: ", performance)
            print("Performance female")
            female_indices = covariates[:, 0] == 1
            female_auc_performance = metrics.roc_auc_score(trues[female_indices, i], results[female_indices, i])
            female_performance = metrics.average_precision_score(trues[female_indices, i], results[female_indices, i])
            print("AUC: ", female_auc_performance)
            print("AUPR: ", female_performance)

            print("Performance male")
            male_indices = covariates[:, 1] == 1
            male_auc_performance = metrics.roc_auc_score(trues[male_indices, i], results[male_indices, i])
            male_performance = metrics.average_precision_score(trues[male_indices, i], results[male_indices, i])
            print("AUC: ", male_auc_performance)
            print("AUPR: ", male_performance)
            print(" ")

        return performance

    def train_epoch(self, epoch, ccc=False, sex=False):


        self.snp2p_model.train()
        if self.args.distributed:
            if self.args.z_weight!=0:
                self.snp2p_dataloader.sampler.set_epoch(epoch)

        new_temperature = linear_temperature_schedule(epoch, self.args.epochs, T_init=1.0, T_final=0.25)
        new_block_sampling_prob = linear_temperature_schedule(epoch, self.args.epochs, T_init=0.1, T_final=1)
        '''
        if not self.args.distributed:
            self.snp2p_model.set_temperature(new_temperature)
            #self.snp2p_model.block_sampling_prob = new_block_sampling_prob
        else:
            self.snp2p_model.module.set_temperature(new_temperature)
            #self.snp2p_model.module.block_sampling_prob = new_block_sampling_prob
        '''
        self.iter_minibatches(self.snp2p_dataloader, epoch, name="Batch", sex=False)

    def iter_minibatches(self, dataloader, epoch, name="", snp_only=False, sex=False):
        mean_response_loss = 0.
        mean_ccc_loss = 0.
        mean_score_loss = 0.
        mean_snp_loss = 0.
        #print(batch['genotypoe'])
        #print(self.snp2gene_mask.shape, self.snp2gene_mask)
        #print(np.where(self.snp2gene_mask.detach().cpu().numpy()==0), self.snp2gene_mask)
        #print(self.gene2sys_mask.shape, self.gene2sys_mask)
        #print(self.nested_subtrees_forward)
        '''
        if epoch>5:
            ld = True
        else:
            ld = False
        '''
        if self.dynamic_phenotype_sampling:
            num_batches = np.ceil(len(dataloader.dataset.dataset)/(self.args.batch_size*self.args.world_size))
            dataloader_with_tqdm = tqdm(dataloader, total=num_batches)
        else:
            dataloader_with_tqdm = tqdm(dataloader)
        for i, batch in enumerate(dataloader_with_tqdm):
            batch = move_to(batch, self.device)
            #print(batch)
            if self.dynamic_phenotype_sampling:

                nested_subtrees_forward = batch['mask']['subtree_forward']
                nested_subtrees_backward = batch['mask']['subtree_backward']
                gene2sys_mask = self.gene2sys_mask#batch['mask']['gene2sys_mask']
                sys2gene_mask = self.sys2gene_mask#batch['mask']['sys2gene_mask']
            else:
                nested_subtrees_forward = self.nested_subtrees_forward
                nested_subtrees_backward = self.nested_subtrees_backward
                gene2sys_mask = self.gene2sys_mask
                sys2gene_mask = self.sys2gene_mask
            #print(batch['genotype']['snp'].shape)
            #print(batch['genotype']['snp'])
            phenotype_predicted = self.snp2p_model(batch['genotype'], batch['covariates'], batch['phenotype_indices'],
                                                   nested_subtrees_forward,
                                                   nested_subtrees_backward,
                                                   #sys_temp= self.system_temp_tensor,
                                                   snp2gene_mask = self.snp2gene_mask,
                                                   gene2sys_mask=gene2sys_mask,#batch['gene2sys_mask'],
                                                   sys2gene_mask=sys2gene_mask,
                                                   sys2env=self.args.sys2env,
                                                   env2sys=self.args.env2sys,
                                                   sys2gene=self.args.sys2gene, snp_only=snp_only,
                                                   ld=True, predict_snp=False)

            if len(phenotype_predicted.size())==3:
                predictions = phenotype_predicted[:, :, 0]
            else:
                predictions = phenotype_predicted

            if self.dynamic_phenotype_sampling:
                pheno_inds = batch['phenotype_indices'][0].detach().cpu().tolist()
                phenos = [self.args.ind2pheno[ind] for ind in pheno_inds]
                dynamic_qt_inds = []
                dynamic_bt_inds = []

                for ind, pheno in enumerate(phenos):
                    if self.args.pheno2type[pheno]=='qt':
                        dynamic_qt_inds.append(ind)
                    elif self.args.pheno2type[pheno]=='bt':
                        dynamic_bt_inds.append(ind)
                #print(dynamic_bt_inds, dynamic_qt_inds)
                loss = MultiplePhenotypeLoss(dynamic_bt_inds, dynamic_qt_inds)

            else:
                loss = self.loss

            phenotype_loss = 0
            phenotype_loss_result = loss(predictions, batch['phenotype'])


            '''
            labels = batch['genotype']['snp_label']
            logits = F.log_softmax(snp_predicted, dim=-1)
            snp_loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            print(batch['genotype']['snp'])
            print(labels)
            print(logits)
            print(logits.size())
            print(loss)
            asdf
            '''

            phenotype_loss += phenotype_loss_result #+ correlation_matching_loss(predictions, batch['phenotype'])
            mean_response_loss += float(phenotype_loss_result)
            #mean_snp_loss += float(snp_loss)
            loss =  phenotype_loss #+ 0.01 * snp_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            dataloader_with_tqdm.set_description(
                "%s Train epoch: %3.f, Phenotype loss: %.3f, SNPLoss: %.3f" % (
                    name, epoch, mean_response_loss / (i + 1), mean_snp_loss / (i + 1)))
            del loss
            del phenotype_loss, phenotype_loss_result
            del phenotype_predicted
            del batch

        del mean_response_loss, mean_ccc_loss
