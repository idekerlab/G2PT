import multiprocessing
import sys
import os
import argparse
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.analysis.epistasis import EpistasisFinder
from src.utils.tree import SNPTreeParser

import argparse
import json
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.analysis.epistasis import EpistasisFinder
from src.utils.tree import SNPTreeParser
import multiprocessing
from functools import partial
import itertools
import statsmodels.api as sm
import statsmodels.formula.api as smf
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt



def process_system(finder_instance, system, causal_info, quantile=0.9):
    """
    Worker function to process a single system.
    """
    print(f"--- Searching in System: {system} with quantile {quantile}---")
    
    system_snps = set(finder_instance.tree_parser.sys2snp.get(system, []))

    true_additive_snps = set(map(int, causal_info.get('additive_snps', [])))
    true_epistatic_pairs_raw = causal_info.get('epistatic_pairs', [])
    true_epistatic_pairs = {tuple(sorted(map(int, pair))) for pair in true_epistatic_pairs_raw}
    
    additive_in_system = system_snps.intersection(true_additive_snps)
    epistasis_in_system = {pair for pair in true_epistatic_pairs if set(pair).issubset(system_snps)}

    print(f"  System SNPs (including hierarchy): {len(system_snps)}")
    print(f"  True Additive SNPs in System: {len(additive_in_system)} {list(additive_in_system)}")
    print(f"  True Epistatic Pairs in System: {len(epistasis_in_system)} {list(epistasis_in_system)}")

    chi_candidates, fisher_candidates, _ = finder_instance.search_epistasis_on_system(
        system=system,
        target='phenotype',
        return_significant_only=False,
        sex=2,
        verbose=0,
        check_inheritance=False,
        interaction_test=False,
        quantile=quantile
    )

    called_snp_pairs = []#set(map(lambda p: tuple(sorted(p)), chi_candidates + fisher_candidates))

    local_pvalues_chi = defaultdict(lambda: 1.0)
    local_pvalues_fisher = defaultdict(lambda: 1.0)

    #print(chi_candidates)
    #print(fisher_candidates)

    for snp1, snp2, corrected_p in chi_candidates:
        pair = tuple(sorted((int(snp1), int(snp2))))
        if corrected_p < local_pvalues_chi[pair]:
            local_pvalues_chi[pair] = corrected_p
        if corrected_p < 0.05:
            called_snp_pairs.append(pair)


    for snp1, snp2, corrected_p in fisher_candidates:
        pair = tuple(sorted((int(snp1), int(snp2))))
        if corrected_p < local_pvalues_fisher[pair]:
            local_pvalues_fisher[pair] = corrected_p
        if corrected_p < 0.05:
            called_snp_pairs.append(pair)

    print(f"  Called SNP Pairs (candidates for regression): {len(called_snp_pairs)} {list(called_snp_pairs)}")
            
    return dict(local_pvalues_chi), dict(local_pvalues_fisher)

def parallel_worker(system, quantile, causal_info, args):
    """
    A wrapper function for parallel execution. It initializes its own EpistasisFinder.
    """
    # Each worker initializes its own finder to avoid sharing complex objects
    # between processes, which is safer and cleaner.
    tree_parser = SNPTreeParser(ontology=args.onto, snp2gene=args.snp2gene)
    finder_instance = EpistasisFinder(
        tree_parser=tree_parser,
        tsv=args.tsv,
        attention_results=args.attention_results,
        cov=args.cov,
        pheno=args.pheno
    )
    # Call the original processing function
    return process_system(finder_instance, system, causal_info, quantile=quantile)

def plot_curves(y_true, y_score, auc, aupr, prefix):
    """Generates and saves ROC and PR curves."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Receiver Operating Characteristic (ROC)')
    ax1.legend(loc="lower right")
    ax2.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUPR = {aupr:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall (PR) Curve')
    ax2.legend(loc="lower left")
    plt.tight_layout()
    plot_filename = f"{prefix}_curves.png"
    plt.savefig(plot_filename)
    print(f"--- ROC and PR curves saved to {plot_filename} ---")
    plt.close()

if __name__ == '__main__':
    try:
        print("--- [1/9] Parsing command line arguments ---")
        parser = argparse.ArgumentParser(description="Evaluate epistasis detection performance using AUC and AUPR.")
        parser.add_argument('--causal-info', required=True, help='Path to the JSON file with causal SNP info.')
        parser.add_argument('--system-importance', required=True, help='Path to the system importance CSV file.')
        parser.add_argument('--attention-results', required=True, help='Path to the attention results CSV file.')
        parser.add_argument('--tsv', required=True, help='Path to the genotype TSV file.')
        parser.add_argument('--pheno', required=True, help='Path to the phenotype file.')
        parser.add_argument('--cov', required=True, help='Path to the covariate file.')
        parser.add_argument('--onto', required=True, help='Path to the ontology file.')
        parser.add_argument('--snp2gene', required=True, help='Path to the snp2gene mapping file.')
        parser.add_argument('--top-n-systems', type=int, default=20, help='Number of top systems to search within.')
        parser.add_argument('--output-prefix', required=True, help='Prefix for output files.')
        parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel processes to use.')
        args = parser.parse_args()
        print("--- [2/9] Arguments parsed successfully ---")

        print(f"--- [3/9] Loading causal info from {args.causal_info} ---")
        with open(args.causal_info, 'r') as f:
            causal_info = json.load(f)
        true_epistatic_pairs = {tuple(sorted(map(int, pair))) for pair in causal_info['epistatic_pairs']}
        true_additive_snps = set(map(int, causal_info.get('additive_snps', [])))
        all_causal_snps = true_additive_snps.union(set(itertools.chain.from_iterable(true_epistatic_pairs)))
        print(f"--- [4/9] Loaded {len(true_epistatic_pairs)} epistatic pairs and {len(true_additive_snps)} additive SNPs ---")
        
        print(f"--- [5/9] Loading system importance from {args.system_importance} ---")
        system_importance_df = pd.read_csv(args.system_importance)
        print(f"--- [6/9] System importance loaded. Shape: {system_importance_df.shape} ---")
        
        ranked_systems = system_importance_df.sort_values(by='corr_mean_abs', ascending=False)
        top_systems = ranked_systems.head(args.top_n_systems)['System'].tolist()
        print(f"--- [7/9] Identified top {len(top_systems)} systems ---")

        # +++++ START DIAGNOSTIC +++++
        print("\n--- [8/9] Running Pre-flight Diagnostic Check ---")
        print("Initializing a temporary parser to check data mapping...")
        temp_parser = SNPTreeParser(ontology=args.onto, snp2gene=args.snp2gene)
        print("Parser initialized successfully.")
        
        parser_snps = set(temp_parser.snp2ind.keys())
        causal_snps_in_parser = all_causal_snps.intersection(parser_snps)
        print(f"Found {len(causal_snps_in_parser)} of {len(all_causal_snps)} causal SNPs in the parser's master list.")
        if len(causal_snps_in_parser) < len(all_causal_snps):
            print(f"WARNING: Missing causal SNPs: {sorted(list(all_causal_snps - causal_snps_in_parser))}")

        print("\nChecking for causal SNPs within the top-ranked systems...")
        found_any = False
        for system in top_systems:
            if system not in temp_parser.sys2snp:
                print(f"  - System '{system}' not found in parser's sys2snp map. Skipping.")
                continue
            
            system_snps = set(temp_parser.sys2snp[system])
            intersection = system_snps.intersection(causal_snps_in_parser)
            
            if intersection:
                print(f"  - SUCCESS: System '{system}' contains {len(intersection)} causal SNP(s): {list(intersection)}")
                found_any = True
        
        if not found_any and top_systems:
            print("  - WARNING: None of the top-ranked systems contain any of the known causal SNPs.")
        
        root_nodes = [node for node, degree in temp_parser.sys_graph.in_degree() if degree == 0]
        if root_nodes:
            print(f"\nChecking root node(s): {root_nodes}")
            for root in root_nodes:
                root_snps = set(temp_parser.sys2snp.get(root, []))
                intersection = root_snps.intersection(causal_snps_in_parser)
                print(f"  - Root '{root}' contains {len(intersection)} causal SNP(s).")
        print("--- End of Diagnostic Check ---\n")
        # +++++ END DIAGNOSTIC +++++

        print(f"--- [9/9] Starting parallel processing with {args.num_workers} workers ---")
        
        # Create a list of tasks for the process pool
        tasks = []
        for system in top_systems:
            for quantile in [0.1, 0.9]:
                tasks.append((system, quantile, causal_info, args))

        # Use a multiprocessing Pool to execute tasks in parallel
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            results = pool.starmap(parallel_worker, tasks)

        print("\n--- Merging results from parallel workers ---")
        global_pvalues_chi = defaultdict(lambda: 1.0)
        global_pvalues_fisher = defaultdict(lambda: 1.0)

        for local_pvalues_chi, local_pvalues_fisher in results:
            for pair, p_val in local_pvalues_chi.items():
                if p_val < global_pvalues_chi[pair]:
                    global_pvalues_chi[pair] = p_val
            for pair, p_val in local_pvalues_fisher.items():
                if p_val < global_pvalues_fisher[pair]:
                    global_pvalues_fisher[pair] = p_val

        global_pvalues_union = defaultdict(lambda: 1.0)
        for pair, p_val in global_pvalues_chi.items():
            if p_val < global_pvalues_union[pair]:
                global_pvalues_union[pair] = p_val
        for pair, p_val in global_pvalues_fisher.items():
            if p_val < global_pvalues_union[pair]:
                global_pvalues_union[pair] = p_val

        print("--- Preparing final ranked lists for evaluation ---")
        all_snps = pd.read_csv(args.snp2gene, sep='\t')['snp'].astype(str).tolist()
        all_possible_pairs = list(itertools.combinations(all_snps, 2))
        all_possible_pairs = [tuple(sorted((int(snp1), int(snp2)))) for snp1, snp2 in all_possible_pairs]
        
        y_true, y_scores_chi, y_scores_fisher, y_scores_union, y_scores_intersection = [], [], [], [], []
        EPSILON = 1e-300

        for pair in all_possible_pairs:
            sorted_pair = tuple(sorted(pair))
            y_true.append(1 if sorted_pair in true_epistatic_pairs else 0)
            p_chi = global_pvalues_chi[sorted_pair]
            p_fisher = global_pvalues_fisher[sorted_pair]
            y_scores_chi.append(-np.log10(p_chi + EPSILON))
            y_scores_fisher.append(-np.log10(p_fisher + EPSILON))
            y_scores_union.append(-np.log10(min(p_chi, p_fisher) + EPSILON))
            y_scores_intersection.append(-np.log10(max(p_chi, p_fisher) + EPSILON))

        print("\n--- Calculating Precision & Recall at FDR < 0.05 ---")
        tested_pairs = list(global_pvalues_union.keys())
        p_values = [global_pvalues_union[pair] for pair in tested_pairs]

        if not p_values:
            precision, recall, tp, fp = 0.0, 0.0, 0, 0
        else:
            from statsmodels.stats.multitest import multipletests
            reject, q_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            called_positives = {pair for pair, rejected in zip(tested_pairs, reject) if rejected}
            true_positives_set = called_positives.intersection(true_epistatic_pairs)
            tp = len(true_positives_set)
            fp = len(called_positives) - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / len(true_epistatic_pairs) if len(true_epistatic_pairs) > 0 else 0.0
            
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        print("\n--- Final Performance Metrics ---")
        auc_chi = roc_auc_score(y_true, y_scores_chi)
        aupr_chi = average_precision_score(y_true, y_scores_chi)
        auc_fisher = roc_auc_score(y_true, y_scores_fisher)
        aupr_fisher = average_precision_score(y_true, y_scores_fisher)
        auc_union = roc_auc_score(y_true, y_scores_union)
        aupr_union = average_precision_score(y_true, y_scores_union)
        auc_intersection = roc_auc_score(y_true, y_scores_intersection)
        aupr_intersection = average_precision_score(y_true, y_scores_intersection)

        summary_data = {
            "Metric": ["AUC_Chi_Path", "AUPR_Chi_Path", "AUC_Fisher_Path", "AUPR_Fisher_Path", "AUC_Union", "AUPR_Union", "AUC_Intersection", "AUPR_Intersection", "Precision", "Recall"],
            "Value": [auc_chi, aupr_chi, auc_fisher, aupr_fisher, auc_union, aupr_union, auc_intersection, aupr_intersection, precision, recall]
        }
        summary_df = pd.DataFrame(summary_data)
        print(summary_df)
        summary_df.to_csv(f"{args.output_prefix}_summary_aupr.csv", index=False)
        raw_result = pd.DataFrame(
            {'snp1': [pair[0] for pair in all_possible_pairs], 'snp2': [pair[1] for pair in all_possible_pairs],
             'score': y_scores_union})
        raw_result.to_csv(f"{args.output_prefix}_p_values.csv", index=False)
        
        plot_curves(y_true, y_scores_union, auc_union, aupr_union, f"{args.output_prefix}_union")
        
        print(f"\nEvaluation complete. Summary saved to {args.output_prefix}_summary_aupr.csv")

    except Exception as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
