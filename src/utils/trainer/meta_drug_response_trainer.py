import gc
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import numpy as np
from src.utils.data import move_to
import copy
import pickle
import torch.distributed as dist


class MetaDrugResponseTrainer(object):

    def __init__(self, drug_response_model, tree_parser, drug_response_meta_sampler, device,
                 args, validation_dataloader=None, fix_embedding=False):
        self.device = device
        self.drug_response_model = drug_response_model.to(self.device)
        self.inner_lr = args.inner_lr
        self.outer_lr = args.outer_lr
        self.inner_loop_steps = args.inner_loop_steps
        self.drug_response_meta_sampler = drug_response_meta_sampler
        self.meta_optimizer = optim.Adam(self.drug_response_model.parameters(), lr=self.outer_lr)
        self.drug_response_loss = nn.SmoothL1Loss(0.1)
        self.validation_dataloader = validation_dataloader
        self.l2_lambda = args.l2_lambda
        self.best_model = self.drug_response_model
        self.tree_parser = tree_parser
        self.nested_subtrees_forward = self.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='forward')
        self.nested_subtrees_forward = move_to(self.nested_subtrees_forward, device)
        self.nested_subtrees_backward = self.tree_parser.get_nested_subtree_mask(
            args.subtree_order, direction='backward')
        self.nested_subtrees_backward = move_to(self.nested_subtrees_backward, device)
        self.gene2system_mask = move_to(
            torch.tensor(self.tree_parser.gene2sys_mask, dtype=torch.bool),
            device)
        self.system2gene_mask = move_to(
            torch.tensor(self.tree_parser.sys2gene_mask, dtype=torch.bool),
            device)
        print("%d sys2gene in Dataloader" % self.tree_parser.sys2gene_mask.sum())
        self.args = args
        self.fix_embedding = fix_embedding
        self.g2p_module_names = ["Mut2Sys", "Sys2Cell", "Cell2Sys"]
        self.performance = {}

        if fix_embedding:
            if self.args.multiprocessing_distributed:
                self.system_embedding = copy.deepcopy(self.drug_response_model.module.system_embedding)
                self.gene_embedding = copy.deepcopy(self.drug_response_model.module.gene_embedding)
            else:
                self.system_embedding = copy.deepcopy(self.drug_response_model.system_embedding)
                self.gene_embedding = copy.deepcopy(self.drug_response_model.gene_embedding)

    def train(self, epochs, output_path=None):

        self.best_model = self.drug_response_model
        best_performance = 0
        epoch_iterator = tqdm(range(epochs))
        for epoch in epoch_iterator:
            self.drug_response_meta_sampler._shuffle_classes()
            support_loss, query_loss = self.iter_tasks(self.drug_response_meta_sampler)
            epoch_iterator.set_description(
                "Train epoch: %d, Support loss %.3f,  Query loss: %.3f" % (epoch, support_loss, query_loss))
            gc.collect()
            torch.cuda.empty_cache()

            if (epoch % self.args.val_step) == 0 & (self.args.rank==0):
                if self.validation_dataloader is not None:
                    performance = self.evaluate(self.drug_response_model, self.validation_dataloader, epoch + 1,
                                                name="Validation")
                    if performance > best_performance:
                        self.best_model = copy.deepcopy(self.drug_response_model).to('cpu')
                    torch.cuda.empty_cache()
                    gc.collect()
                if not self.args.multiprocessing_distributed or (self.args.multiprocessing_distributed
                                                                 and self.args.rank % torch.cuda.device_count() == 0):
                    if output_path:
                        output_path_epoch = output_path + ".%d" % epoch
                        print("Save to...", output_path_epoch)
                        if self.args.multiprocessing_distributed:
                            torch.save(
                                {"arguments": self.args, "state_dict": self.drug_response_model.module.state_dict()},
                                output_path_epoch)
                        else:
                            torch.save({"arguments": self.args, "state_dict": self.drug_response_model.state_dict()},
                                       output_path_epoch)


        out = output_path.split("/")
        folder = out[0]
        fold = out[1].split("_")[2]
        with open(folder + '/val_performance_' + fold + '.pkl', 'wb') as handle:
            pickle.dump(self.performance, handle)

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
                                                self.nested_subtrees_forward, self.nested_subtrees_backward,
                                                self.gene2system_mask, self.system2gene_mask,
                                                sys2cell=self.args.sys2cell,
                                                cell2sys=self.args.cell2sys,
                                                sys2gene=self.args.sys2gene,
                                                gene2drug=self.args.gene2drug,
                                                mut2gene=self.args.mut2gene,
                                                with_indices=self.args.with_indices)
                drug_response_predicted_detached = drug_response_predicted.detach().cpu().numpy()
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

        self.performance[epoch] = {"pearson_per_drug": pearson_dict, "spearman_per_drug": spearman_dict}
        return pearson_per_drug

    def iter_tasks(self, task_sampler):

        meta_weights = {name: param.clone() for name, param in self.drug_response_model.state_dict().items()}
        mean_support_loss = 0.
        mean_query_loss = 0.

        task_model = copy.deepcopy(self.drug_response_model).to(self.device)
        task_optimizer = optim.Adam(task_model.parameters(), lr=self.inner_lr)

        support, query = task_sampler.sample_task()
        for _ in range(self.inner_loop_steps):
            batch = move_to(support, self.device)
            drug_response_predicted_inner = task_model(batch['genotype'], batch['drug'],
                                                               self.nested_subtrees_forward,
                                                               self.nested_subtrees_backward,
                                                               self.gene2system_mask, self.system2gene_mask,
                                                               sys2cell=self.args.sys2cell,
                                                               cell2sys=self.args.cell2sys,
                                                               sys2gene=self.args.sys2gene,
                                                               gene2drug=self.args.gene2drug,
                                                               mut2gene=self.args.mut2gene,
                                                               with_indices=self.args.with_indices)
            support_loss = self.drug_response_loss(drug_response_predicted_inner[:, 0],
                                                         (batch['response_mean'] + batch['response_residual']).to(
                                                             torch.float32))
            task_optimizer.zero_grad()
            support_loss.backward()
            task_optimizer.step()
            del drug_response_predicted_inner
            mean_support_loss += float(support_loss)
            del support_loss
        del support
        mean_support_loss = mean_support_loss/self.inner_loop_steps
        batch = move_to(query, self.device)
        drug_response_predicted_outer = task_model(batch['genotype'], batch['drug'],
                                                   self.nested_subtrees_forward,
                                                   self.nested_subtrees_backward,
                                                   self.gene2system_mask, self.system2gene_mask,
                                                   sys2cell=self.args.sys2cell,
                                                   cell2sys=self.args.cell2sys,
                                                   sys2gene=self.args.sys2gene,
                                                   gene2drug=self.args.gene2drug,
                                                   mut2gene=self.args.mut2gene,
                                                   with_indices=self.args.with_indices)
        query_loss = self.drug_response_loss(drug_response_predicted_outer[:, 0],
                                               (batch['response_mean'] + batch['response_residual']).to(
                                                   torch.float32))
        #print(drug_response_predicted_outer[:, 0])
        mean_query_loss = float(query_loss)
        del batch
        del query_loss
        task_weights = task_model.state_dict()



        for name in meta_weights:
            meta_weights[name] += self.outer_lr * (task_weights[name] - meta_weights[name])

        for name, param in meta_weights.items():
            dist.all_reduce(param, op=dist.ReduceOp.SUM)
            param /= self.args.world_size
        del task_model

        self.meta_optimizer.zero_grad()
        self.drug_response_model.load_state_dict(meta_weights)
        if self.fix_embedding:
            self.drug_response_model.system_embedding = self.system_embedding
            self.drug_response_model.gene_embedding = self.gene_embedding
        return mean_support_loss, mean_query_loss

