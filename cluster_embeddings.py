import pickle
import argparse
import math
import numpy as np
import pandas as pd
from operator import itemgetter

import igraph as ig
import leidenalg as la

from scipy.stats import wilcoxon
from scipy.stats import fisher_exact
from scipy.sparse import coo_matrix
from statsmodels.stats.multitest import multipletests

import torch
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from torch_geometric.nn.pool import knn

class ClusterEmbeddings(object):
    def __init__(self, results, system_genes, nest_to_name, index_to_nest,
                 mutations="data/cell2mutation_ctg_av.txt",
                 deletions="data/old_copynumber/cell2cndeletion_ctg_av.txt",
                 amplifics="data/old_copynumber/cell2cnamplification_ctg_av.txt",
                 gene2ind= "data/gene2ind_ctg_av.txt",
                 cell2ind="data/cell2ind_av.txt",
                ):

        self.mutations = list(pd.read_csv(mutations, sep = "\t", header = None)[0])
        self.deletions = list(pd.read_csv(deletions, sep = "\t", header = None)[0])
        self.amplifics = list(pd.read_csv(amplifics, sep = "\t", header = None)[0])
        gene2ind = pd.read_csv(gene2ind, sep = "\t", header = None)
        cell2ind = pd.read_csv(cell2ind, sep = "\t", header = None)
        self.gene2ind = dict(zip(gene2ind[1], gene2ind[0]))
        self.cell2ind = dict(zip(cell2ind[1], cell2ind[0])) 
        
        self.results = results
        self.system_genes = system_genes
        self.nest_to_name = nest_to_name
        self.index_to_nest = index_to_nest
        self.drugs = list(results.keys())

    def analyze(self):

        name_to_nest = dict((v,k) for k,v in self.nest_to_name.items())
        nest_to_index = dict((v,k) for k,v in self.index_to_nest.items())
        systems = list(self.system_genes.keys())

        system_embeddings = {}
        for drug in self.drugs:
            cells = self.results[drug]["celllines"]
            for system in systems:
                system_index = nest_to_index[name_to_nest[system]]
                embeddings = self.results[drug]["system_embeddings"][:, system_index, :]
                embeddings = pd.DataFrame(embeddings, index = cells)
                if system in system_embeddings.keys():
                    merged_embeddings = pd.concat([system_embeddings[system], embeddings], axis=0, join='outer')
                    merged_embeddings = merged_embeddings[~merged_embeddings.index.duplicated(keep='first')]
                    system_embeddings[system] = merged_embeddings
                else:
                    system_embeddings[system] = embeddings

        system_count = 0
        cluster_results = {}
        for system in systems:
            N = len(system_embeddings[system])
            embeddings = torch.from_numpy(system_embeddings[system].values).to(device="cuda")
            cell_labels = system_embeddings[system].index.tolist()

            # Create KNN cosine similarity matrix from the embeddings
            row, col = knn(x=embeddings, y=embeddings, k=15, cosine=True)
            cos_sim = F.cosine_similarity(embeddings[row], embeddings[col], dim=1)
            cos_sim = torch.clamp(cos_sim, min=0.0, max=1.0)
            row_cpu = row.detach().cpu().numpy()
            col_cpu = col.detach().cpu().numpy()
            sim_cpu = cos_sim.detach().cpu().numpy()
            knn_dist_mat = coo_matrix((sim_cpu, (row_cpu, col_cpu)), shape=(N, N))
            
            # Convert to igraph object and run the Leiden algorithm
            graph = ig.Graph.Weighted_Adjacency(knn_dist_mat, mode="directed", loops=False)
            partition = la.find_partition(graph, la.RBConfigurationVertexPartition, weights="weight", 
                                          resolution_parameter=0.8, seed=42)

            # Create dict mapping cluster label to list of cells
            cluster_info = {}
            cluster_labels = partition.membership 
            for cell, cluster in zip(cell_labels, cluster_labels):
                if cluster_info.get(cluster):
                    cluster_info[cluster]["Cells"].append(cell)
                else:
                    cluster_info[cluster] = {"Cells": [cell]}

            cluster_info = self.analyze_cluster_genes(system, cluster_info)
            cluster_info = self.analyze_cluster_response(cluster_info)

            cluster_results[system] = {}
            cluster_results[system]["Modularity"] = partition.quality()
            cluster_results[system]["Clusters"] = cluster_info

            system_count += 1
            print("Done with  system", system_count, ":", system)

        return(cluster_results)


    def analyze_cluster_genes(self, system, cluster_info):
        
        genes = list(self.system_genes[system])
        clusters = list(cluster_info.keys())

        contingency_tables = {}
        for gene in genes:
            contingency_tables[gene] = np.zeros((len(clusters), 2), dtype=int)
        
        for cluster in clusters:
            cluster_idx = clusters.index(cluster)
            cells = cluster_info[cluster]["Cells"]
            for cell in cells:
                cell_index = self.cell2ind[cell]
                muts = self.mutations[cell_index].split(",")
                cnds = self.deletions[cell_index].split(",")
                cnas = self.amplifics[cell_index].split(",")
                for gene in genes:
                    gene_index = self.gene2ind[gene]
                    if ((muts[gene_index] == "1") or (cnds[gene_index] == "1") or (cnas[gene_index] == "1")):
                        contingency_tables[gene][cluster_idx, 0] += 1
                    else:
                        contingency_tables[gene][cluster_idx, 1] += 1

        cluster_gene_sig = []
        for gene in genes:
            table = contingency_tables[gene]
            for cluster in clusters:
                cluster_idx = clusters.index(cluster)
                in_cluster_counts = table[cluster_idx]
                out_cluster_counts = [0, 0]
                for row_idx, row in enumerate(table):
                    if row_idx != cluster_idx:
                        out_cluster_counts = [x + y for x, y in zip(row, out_cluster_counts)]
                
                two_by_two_table = np.asarray([in_cluster_counts, out_cluster_counts])
                odds_ratio, pval =  fisher_exact(two_by_two_table, alternative="greater")
                data = pd.DataFrame({"Cluster": [cluster], "Gene": [gene], "P-Value": [pval]})
                cluster_gene_sig.append(data)

        cluster_gene_sig = pd.concat(cluster_gene_sig, ignore_index = True)

        rejected, corrected, _, _ = multipletests(cluster_gene_sig["P-Value"], alpha=0.05, method="fdr_bh")
        corrected_df = cluster_gene_sig.copy()
        corrected_df["FDR"] = corrected
        corrected_df = corrected_df.drop(columns = ["P-Value"])

        cluster_profiles = corrected_df.groupby("Cluster").apply(lambda g: dict(zip(g["Gene"], g["FDR"]))).to_dict()
        
        for cluster in cluster_profiles:
            profiles = cluster_profiles[cluster]
            cluster_info[cluster]["Profiles"] = profiles

        return(cluster_info)

    def analyze_cluster_response(self, cluster_info):

        clusters = list(cluster_info.keys())

        cluster_drug_sig = []
        for drug in self.drugs:
            drug_cells = self.results[drug]["celllines"]
            response = self.results[drug]["predictions"]
            for cluster in clusters:
                cluster_cells = cluster_info[cluster]["Cells"]
                cluster_cell_indices = [drug_cells.index(cell) for cell in cluster_cells if cell in drug_cells]
                cluster_cell_response = [response[i] for i in cluster_cell_indices]
                    
                pval_res, pval_sen = self.cluster_response_wilcoxon(response, cluster_cell_response)
                pval_acc = self.cluster_response_fisher(drug, cluster_cells)
                
                data = pd.DataFrame({"Cluster": [cluster], "Drug": [drug], "Res-P": [pval_res], "Sen-P": [pval_sen], "Acc-P": [pval_acc]})
                cluster_drug_sig.append(data)

        cluster_drug_sig = pd.concat(cluster_drug_sig, ignore_index = True)

        rejected, res_corrected, _, _ = multipletests(cluster_drug_sig["Res-P"], alpha=0.05, method="fdr_bh")
        rejected, sen_corrected, _, _ = multipletests(cluster_drug_sig["Sen-P"], alpha=0.05, method="fdr_bh")
        rejected, acc_corrected, _, _ = multipletests(cluster_drug_sig["Acc-P"], alpha=0.05, method="fdr_bh")
        corrected_df = cluster_drug_sig.copy()
        corrected_df["Res FDR"] = res_corrected
        corrected_df["Sen FDR"] = sen_corrected
        corrected_df["Acc FDR"] = acc_corrected
        corrected_df = corrected_df.drop(columns = ["Res-P", "Sen-P", "Acc-P"])

        cluster_response = corrected_df.groupby("Cluster").apply(lambda g: {drug: {"Res FDR": res_fdr, "Sen FDR": sen_fdr, "Acc FDR": acc_fdr} 
                                                                        for drug, res_fdr, sen_fdr, acc_fdr 
                                                                            in zip(g["Drug"], g["Res FDR"], g["Sen FDR"], g["Acc FDR"])}
                                                                ).to_dict()
        
        for cluster in cluster_response:
            response = cluster_response[cluster]
            cluster_info[cluster]["Response"] = response

        return(cluster_info)

    def cluster_response_wilcoxon(self, all_response, cluster_response):

        pval_res = wilcoxon(cluster_response - np.median(all_response), alternative='greater')[1]
        pval_sen = wilcoxon(cluster_response - np.median(all_response), alternative='less')[1]

        return(pval_res, pval_sen)

    def cluster_response_fisher(self, drug, cluster_cells):

        def cluster_res_sen(cluster_cells, drug_cell_auc, percentile):
            
            N = math.floor(percentile*len(drug_cell_auc))
            resistant_cells = list(dict(sorted(drug_cell_auc.items(), key=itemgetter(1), reverse=True)[:N]).keys())
            sensitive_cells = list(dict(sorted(drug_cell_auc.items(), key=itemgetter(1), reverse=False)[:N]).keys())

            cluster_resistant = list(set(resistant_cells).intersection(set(cluster_cells)))
            cluster_sensitive = list(set(sensitive_cells).intersection(set(cluster_cells)))

            return(cluster_resistant, cluster_sensitive)

        drug_cell_true = dict(zip(self.results[drug]["celllines"], self.results[drug]["actual"]))
        drug_cell_pred = dict(zip(self.results[drug]["celllines"], self.results[drug]["predictions"]))

        cluster_res_true, cluster_sen_true = cluster_res_sen(cluster_cells, drug_cell_true, 0.25)
        cluster_res_pred, cluster_sen_pred = cluster_res_sen(cluster_cells, drug_cell_pred, 0.50)

        true_res_pred_res = len(list(set(cluster_res_true).intersection(set(cluster_res_pred))))
        true_res_pred_sen = len(list(set(cluster_res_true).intersection(set(cluster_sen_pred))))
        true_sen_pred_res = len(list(set(cluster_sen_true).intersection(set(cluster_res_pred))))
        true_sen_pred_sen = len(list(set(cluster_sen_true).intersection(set(cluster_sen_pred))))

        contingency_table = np.asarray([[true_res_pred_res, true_res_pred_sen], [true_sen_pred_res, true_sen_pred_sen]])
        odds_ratio, pval = fisher_exact(contingency_table)

        return(pval)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--results', dest='results', type=str, default="results_diff_embeddings.pkl")
    parser.add_argument('--system_to_genes', dest='system_genes', type=str, default="system_to_genes.pkl")
    parser.add_argument('--nest_to_name', dest='nest_to_name', type=str, default="nest_to_old_name.pkl")
    parser.add_argument('--index_to_nest', dest='index_to_nest', type=str, default="index_to_nest.pkl")
    parser.add_argument('--out', dest='out', type=str, default="clustered_embeddings.pkl")
    args = parser.parse_args()

    with open(args.results, "rb") as handle:
        results = pickle.load(handle)

    with open(args.system_genes, "rb") as handle:
        system_genes = pickle.load(handle)

    with open(args.nest_to_name, "rb") as handle:
        nest_to_name = pickle.load(handle)

    with open(args.index_to_nest, "rb") as handle:
        index_to_nest = pickle.load(handle)

    cluster_embeddings = ClusterEmbeddings(results, system_genes, nest_to_name, index_to_nest).analyze()

    with open(args.out, "wb") as handle:
        pickle.dump(cluster_embeddings, handle)