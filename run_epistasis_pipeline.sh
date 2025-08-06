#!/bin/bash
#SBATCH -A nrnb-gpup
#SBATCH -p nrnb-gpu
#SBATCH --job-name=epistasis_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=epistasis_pipeline_%j.log

set -e # Exit immediately if a command exits with a non-zero status.

# --- 0. Setup ---
echo "--- Setting up Environment ---"
# Activate Conda Environment
source ~/.bashrc
conda activate G2PT_github

# Set Project Root and PYTHONPATH
export PROJECT_ROOT=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
echo "Project Root: $PROJECT_ROOT"

# Configuration
OUTPUT_DIR="epistasis_simulation_output_slurm"
SYNTH_SCRIPT="src/utils/analysis/epistasis_simulation.py"
TRAIN_SCRIPT="train_snp2p_model.py"
PREDICT_SCRIPT="predict_attention.py"
EVAL_SCRIPT="evaluate_epistasis_search.py"
SEED=123

# Create output directory
mkdir -p $OUTPUT_DIR
echo "Output directory: $OUTPUT_DIR"

# --- 1. Generate Synthetic Data ---
echo -e "\n--- Step 1: Generating Synthetic Data ---"
WRAPPER_SCRIPT="${OUTPUT_DIR}/generate_data.py"
cat > $WRAPPER_SCRIPT << EOL
import pandas as pd
import numpy as np
import json
from src.utils.analysis.epistasis_simulation import simulate_epistasis, build_hierarchical_ontology

output_dir = "$OUTPUT_DIR"
seed = $SEED

print("Running simulation...")
sim = simulate_epistasis(n_samples=2000, n_snps=2000, seed=seed)

print("Building ontology...")
snp_df, gene_df, system_df = build_hierarchical_ontology(sim, seed=seed)

# Create consistent sample IDs
iids = [f'sample_{i}' for i in range(sim['y'].shape[0])]

# Save genotypes with IID as index and SNP IDs as columns
geno_df = pd.DataFrame(sim['G'], columns=snp_df['snp'].values)
geno_df.index = iids
geno_df.index.name = 'IID'
geno_df.to_csv(f"{output_dir}/genotypes.tsv", sep='\t')

# Save phenotypes and covariates with the same IIDs
pheno_df = pd.DataFrame({'FID': iids, 'IID': iids, 'phenotype': sim['y']})
pheno_df.to_csv(f"{output_dir}/simulation.pheno", index=False, sep='\t')

cov_df = pd.DataFrame({
    'FID': iids,
    'IID': iids,
    'SEX': np.random.randint(0, 2, size=sim['y'].shape[0]),
    'AGE': np.random.randint(40, 70, size=sim['y'].shape[0])
})
cov_df.to_csv(f"{output_dir}/simulation.cov", index=False, sep='\t')

# Save mappings
snp_df.to_csv(f"{output_dir}/snp2gene.tsv", index=False, sep='\t')

# Combine gene->system and system->supersystem for the model's expected format
gene_df = gene_df.rename(columns={'gene_id': 'child', 'system_id': 'parent'})
gene_df['interaction'] = 'gene'
system_df = system_df.rename(columns={'system_id': 'child', 'supersystem_id': 'parent'})
system_df['interaction'] = 'default'

ontology_df = pd.concat([
    gene_df[['parent', 'child', 'interaction']],
    system_df[['parent', 'child', 'interaction']]
])
ontology_df.to_csv(f"{output_dir}/ontology.tsv", index=False, sep='\t', header=False)

# Causal Info for Evaluation
causal_info = {
    'epistatic_pairs': [list(map(int, p)) for p in sim['pair_idx']],
    'additive_snps': list(map(int, sim['additive_idx']))
}
with open(f"{output_dir}/causal_info.json", 'w') as f:
    json.dump(causal_info, f, indent=2)

print("Synthetic data generation complete.")
EOL

python $WRAPPER_SCRIPT
echo "--- Synthetic Data Generation Finished ---"

# --- Sanity Check ---
echo -e "\n--- Sanity Check: Initializing SNPTreeParser ---"
SANITY_CHECK_SCRIPT="${OUTPUT_DIR}/sanity_check.py"
cat > $SANITY_CHECK_SCRIPT << EOL
from src.utils.tree.snp_tree import SNPTreeParser
import os

print("Running sanity check...")
try:
    parser = SNPTreeParser(
        ontology=os.path.join("$OUTPUT_DIR", "ontology.tsv"),
        snp2gene=os.path.join("$OUTPUT_DIR", "snp2gene.tsv")
    )
    print("SNPTreeParser initialized successfully!")
    print(f"Found {parser.n_snps} SNPs, {parser.n_genes} Genes, and {parser.n_systems} Systems.")
    print("Sanity check passed.")
except Exception as e:
    print(f"Sanity check FAILED: {e}")
    exit(1)
EOL

python $SANITY_CHECK_SCRIPT
echo "--- Sanity Check Finished ---"

# --- 2. Train G2PT Model (using TSVDataset) ---
echo -e "\n--- Step 2: Training G2PT Model ---"
MODEL_OUTPUT_PREFIX="${OUTPUT_DIR}/g2pt_model"
python $TRAIN_SCRIPT \
    --train-tsv-path "${OUTPUT_DIR}" \
    --train-pheno "${OUTPUT_DIR}/simulation.pheno" \
    --train-cov "${OUTPUT_DIR}/simulation.cov" \
    --onto "${OUTPUT_DIR}/ontology.tsv" \
    --snp2gene "${OUTPUT_DIR}/snp2gene.tsv" \
    --out "$MODEL_OUTPUT_PREFIX" \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-4 \
    --qt "phenotype"

# Find the last saved model file
MODEL_FILE=$(ls -t ${OUTPUT_DIR}/g2pt_model* | head -n 1)
echo "Training complete. Model saved to: $MODEL_FILE"

# --- 3. Predict System Importance ---
echo -e "\n--- Step 3: Predicting System Importance ---"
PREDICTIONS_PREFIX="${OUTPUT_DIR}/attention_scores"
python $PREDICT_SCRIPT \
    --model "$MODEL_FILE" \
    --tsv-path "${OUTPUT_DIR}" \
    --pheno "${OUTPUT_DIR}/simulation.pheno" \
    --cov "${OUTPUT_DIR}/simulation.cov" \
    --onto "${OUTPUT_DIR}/ontology.tsv" \
    --snp2gene "${OUTPUT_DIR}/snp2gene.tsv" \
    --out "$PREDICTIONS_PREFIX" \
    --batch-size 256 \
    --cuda 0

# Rename the output file for the evaluation script
mv "${PREDICTIONS_PREFIX}.phenotype.head_sum.sys_importance.csv" "${PREDICTIONS_PREFIX}_sys_importance.csv"

echo "Prediction complete. Attention scores saved."

# --- 4. Evaluate Epistasis Detection ---
echo -e "\n--- Step 4: Evaluating Epistasis Detection ---"
EVAL_OUTPUT_PREFIX="${OUTPUT_DIR}/epistasis_evaluation"
python $EVAL_SCRIPT \
    --causal-info "${OUTPUT_DIR}/causal_info.json" \
    --attention-results "${OUTPUT_DIR}/attention_scores.phenotype.head_sum.csv" \
    --system-importance "${OUTPUT_DIR}/attention_scores_sys_importance.csv" \
    --tsv-path "${OUTPUT_DIR}" \
    --pheno "${OUTPUT_DIR}/simulation.pheno" \
    --cov "${OUTPUT_DIR}/simulation.cov" \
    --onto "${OUTPUT_DIR}/ontology.tsv" \
    --snp2gene "${OUTPUT_DIR}/snp2gene.tsv" \
    --output-prefix "$EVAL_OUTPUT_PREFIX" \
    --top-n-systems 10

echo -e "\n--- Pipeline Finished Successfully ---"