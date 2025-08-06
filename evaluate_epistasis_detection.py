import json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import argparse

def evaluate_epistasis_detection(causal_info_path, snp2gene_path, ontology_path, system_importance_path, output_prefix):
    # 1. Load Causal Information
    with open(causal_info_path, 'r') as f:
        causal_info = json.load(f)
    epistatic_snp_pairs = causal_info['epistatic_pairs']
    causal_snps = set()
    for pair in epistatic_snp_pairs:
        causal_snps.add(f"snp_{pair[0]}")
        causal_snps.add(f"snp_{pair[1]}")

    # 2. Load SNP-to-Gene Mapping
    snp2gene_df = pd.read_csv(snp2gene_path, sep='\t')
    snp_to_gene_map = dict(zip(snp2gene_df['snp'], snp2gene_df['gene']))

    # Identify causal genes based on causal SNPs
    causal_genes = set()
    for snp_id in causal_snps:
        if snp_id in snp_to_gene_map:
            causal_genes.add(snp_to_gene_map[snp_id])

    # 3. Load Ontology and Map Genes to Systems
    ontology_df = pd.read_csv(ontology_path, sep='\t', header=None, names=['parent', 'child', 'interaction'])
    
    # Filter for gene-to-system mappings (where interaction is 'gene')
    gene_to_system_df = ontology_df[ontology_df['interaction'] == 'gene']
    
    # Create a mapping from gene to systems (a gene can belong to multiple systems)
    gene_to_systems_map = {}
    for _, row in gene_to_system_df.iterrows():
        gene = row['child']
        system = row['parent']
        if gene not in gene_to_systems_map:
            gene_to_systems_map[gene] = set()
        gene_to_systems_map[gene].add(system)

    # Identify causal systems
    causal_systems = set()
    for causal_gene in causal_genes:
        if causal_gene in gene_to_systems_map:
            causal_systems.update(gene_to_systems_map[causal_gene])

    print(f"Number of causal epistatic SNPs: {len(causal_snps)}")
    print(f"Number of unique causal genes associated with epistatic SNPs: {len(causal_genes)}")
    print(f"Number of unique causal systems associated with epistatic SNPs: {len(causal_systems)}")

    # 4. Load System Importance Scores
    system_importance_df = pd.read_csv(system_importance_path)
    
    # Ensure the 'System' column is present
    if 'System' not in system_importance_df.columns:
        raise ValueError(f"'{system_importance_path}' must contain a 'System' column.")
    
    # Assuming the importance score is 'corr_with_phenotype'
    importance_col = [col for col in system_importance_df.columns if 'corr_with_phenotype' in col]
    if not importance_col:
        raise ValueError(f"No 'corr_with_phenotype' column found in {system_importance_path}. Please check the column name.")
    importance_col = importance_col[0]

    # 5. Prepare data for AUC/AUPRC calculation
    # Create a 'is_causal' column: 1 if system is causal, 0 otherwise
    system_importance_df['is_causal'] = system_importance_df['System'].apply(lambda x: 1 if x in causal_systems else 0)

    # Filter out systems with NaN importance scores if any
    system_importance_df.dropna(subset=[importance_col], inplace=True)

    # Check if there are both positive and negative samples for AUC/AUPRC
    if len(system_importance_df['is_causal'].unique()) < 2:
        print("Warning: Not enough unique classes (causal/non-causal) to compute AUC/AUPRC.")
        print(f"Causal systems found in importance file: {system_importance_df['is_causal'].sum()}")
        print(f"Total systems in importance file: {len(system_importance_df)}")
        return

    y_true = system_importance_df['is_causal']
    y_scores = system_importance_df[importance_col].abs() # Use absolute correlation as importance

    # 6. Calculate AUC and AUPRC
    auc_score = roc_auc_score(y_true, y_scores)
    auprc_score = average_precision_score(y_true, y_scores)

    print(f"AUC for epistatic system detection: {auc_score:.4f}")
    print(f"AUPRC for epistatic system detection: {auprc_score:.4f}")

    # Optionally, save the results
    results_df = pd.DataFrame({
        'Metric': ['AUC', 'AUPRC'],
        'Score': [auc_score, auprc_score]
    })
    results_df.to_csv(f"{output_prefix}_epistasis_system_evaluation_results.csv", index=False)
    print(f"Evaluation results saved to {output_prefix}_epistasis_system_evaluation_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate epistatic interaction detection performance at the system level.')
    parser.add_argument('--causal_info', type=str, required=True,
                        help='Path to the causal_info.json file.')
    parser.add_argument('--snp2gene', type=str, required=True,
                        help='Path to the snp2gene.tsv file.')
    parser.add_argument('--ontology', type=str, required=True,
                        help='Path to the ontology.tsv file.')
    parser.add_argument('--system_importance', type=str, required=True,
                        help='Path to the system_importance.csv file (output from predict_attention.py).')
    parser.add_argument('--output_prefix', type=str, default='epistasis_system_evaluation',
                        help='Prefix for the output evaluation results file.')
    
    args = parser.parse_args()

    evaluate_epistasis_detection(args.causal_info, args.snp2gene, args.ontology, args.system_importance, args.output_prefix)
