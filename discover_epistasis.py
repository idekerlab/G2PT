import multiprocessing
import sys
import os
import argparse
import pandas as pd
from functools import partial
from collections import defaultdict
from statsmodels.stats.multitest import multipletests

# Ensure the source directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.analysis.epistasis import EpistasisFinder
from src.utils.tree import SNPTreeParser

def process_system_for_discovery(system, args):
    """
    Worker function to process a single system for epistasis discovery.
    It initializes its own EpistasisFinder and runs the search.
    """
    print(f"--- Searching in System: {system} ---")
    try:
        # Each worker initializes its own finder to avoid sharing complex objects
        tree_parser = SNPTreeParser(ontology=args.onto, snp2gene=args.snp2gene)
        finder_instance = EpistasisFinder(
            tree_parser=tree_parser,
            bfile=args.bfile,
            tsv=args.tsv,
            attention_results=args.attention_results,
            cov=args.cov,
            pheno=args.pheno
        )

        # Run the search. We want all tested pairs to perform a global FDR correction later.
        # We disable check_inheritance for speed and to reduce assumptions.
        chi_path_results, fisher_path_results, _, _, _ = finder_instance.search_epistasis_on_system(
            system=system,
            target='phenotype',
            return_significant_only=False,  # Get p-values for all tested pairs
            sex=2,                          # Analyze all individuals
            verbose=0,
            check_inheritance=False,        # Skip inheritance model fitting
            interaction_test=True,          # CRITICAL: Perform regression-based interaction test
            quantile=args.quantile
        )

        # Combine results from both internal search paths (Chi-Square and Fisher)
        all_tested_pairs_in_system = chi_path_results + fisher_path_results

        # Extract the raw interaction p-value for each unique pair
        discovered_pairs = []
        for snp1, snp2, raw_p, _ in all_tested_pairs_in_system:
            pair = tuple(sorted((snp1, snp2)))
            discovered_pairs.append((pair, raw_p))

        #pairs = list(all_results.keys())
        if len(discovered_pairs)==0:
            return []
        p_values = [p for pair, p in discovered_pairs]
        reject, q_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        final_results = []

        for i, pair in enumerate(discovered_pairs):
            if reject[i]: # Only save pairs that are significant after global correction
                final_results.append({
                    'snp1': pair[0],
                    'snp2': pair[1],
                    'p_value': p_values[i],
                    'q_value': q_values[i]
                })

        print(f"  Tested {len(discovered_pairs)} pairs in {system}")
        return final_results

    except Exception as e:
        print(f"--- ERROR processing system {system}: {e} ---")
        import traceback
        traceback.print_exc()
        return []


if __name__ == '__main__':
    try:
        print("--- [1/7] Parsing command line arguments for Epistasis Discovery ---")
        parser = argparse.ArgumentParser(description="Discover statistically significant epistatic pairs from real data.")
        parser.add_argument('--system-importance', required=True, help='Path to the system importance CSV file.')
        parser.add_argument('--attention-results', required=True, help='Path to the attention results CSV file.')
        parser.add_argument('--tsv', required=False, help='Path to the genotype TSV file.')
        parser.add_argument('--bfile', required=False, help='Path to the genotype bfile prefix.')
        parser.add_argument('--pheno', required=True, help='Path to the phenotype file.')
        parser.add_argument('--cov', required=True, help='Path to the covariate file.')
        parser.add_argument('--onto', required=True, help='Path to the ontology file.')
        parser.add_argument('--snp2gene', required=True, help='Path to the snp2gene mapping file.')
        parser.add_argument('--max-system-genes', type=int, default=50, help='Maximum number of genes allowed in a system to be searched.')
        parser.add_argument('--quantile', type=float, default=0.9, help='Quantile for attention score filtering.')
        parser.add_argument('--output-prefix', required=True, help='Prefix for output files.')
        parser.add_argument('--num-workers', type=int, default=4, help='Number of parallel processes to use.')
        args = parser.parse_args()
        if not args.tsv and not args.bfile:
            parser.error("Either --tsv or --bfile must be provided.")
        print("--- [2/7] Arguments parsed successfully ---")

        print(f"--- [3/7] Initializing SNPTreeParser to filter systems ---")
        tree_parser = SNPTreeParser(ontology=args.onto, snp2gene=args.snp2gene)
        
        snp_to_gene_df = pd.read_csv(args.snp2gene, sep='\t')
        snp_to_gene = dict(zip(snp_to_gene_df['snp'].astype(str), snp_to_gene_df['gene']))

        print(f"--- [4/7] Loading and filtering systems based on gene count (max={args.max_system_genes}) ---")
        system_importance_df = pd.read_csv(args.system_importance)
        all_systems = system_importance_df['System'].tolist()

        systems_to_search = []
        for system in all_systems:
            system_snps = tree_parser.sys2snp.get(system, [])
            if not system_snps:
                continue
            
            system_genes = {snp_to_gene.get(str(snp)) for snp in system_snps if str(snp) in snp_to_gene}
            system_genes.discard(None)

            if len(system_genes) <= args.max_system_genes:
                systems_to_search.append(system)
            else:
                print(f"  - Skipping system '{system}': has {len(system_genes)} genes (>{args.max_system_genes})")

        print(f"--- [5/7] Identified {len(systems_to_search)} systems to search after filtering ---")

        print(f"--- [6/7] Starting parallel processing with {args.num_workers} workers ---")
        with multiprocessing.Pool(processes=args.num_workers) as pool:
            worker_func = partial(process_system_for_discovery, args=args)
            results_list = pool.map(worker_func, systems_to_search)

        print("\n--- [7/7] Aggregating results and performing global FDR correction ---")

        all_results = sum(results_list, [])
        #all_results = defaultdict(lambda: 1.0)
        #for system_results in results_list:
        #    for pair, p_val in system_results:
        #        if p_val < all_results[pair]:
        #            all_results[pair] = p_val
        
        #if not all_results:
        #    print("--- No candidate pairs were tested. Exiting. ---")
        #    pd.DataFrame(columns=['snp1', 'snp2', 'p_value', 'q_value']).to_csv(f"{args.output_prefix}_final_epistasis_results.csv", index=False)
        #    sys.exit(0)

        # Perform global FDR correction
        #pairs = list(all_results.keys())
        #p_values = [all_results[p] for p in pairs]
        #reject, q_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

        #final_results = 
        #for i, pair in enumerate(pairs):
        #    #if reject[i]: # Only save pairs that are significant after global correction
        #    final_results.append({
        #            'snp1': pair[0],
        #            'snp2': pair[1],
        #            'p_value': p_values[i],
        #            'q_value': q_values[i]
        #        })
        
        final_df = pd.DataFrame(all_results)
        if not final_df.empty:
            final_df = final_df.sort_values(by='q_value', ascending=True)

        output_filename = f"{args.output_prefix}_final_epistasis_results.csv"
        final_df.to_csv(output_filename, index=False)

        print(f"\n--- Discovery complete. Found {len(final_df)} significant pairs (FDR < 0.05). ---")
        print(f"--- Results saved to {output_filename} ---")

    except Exception as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
