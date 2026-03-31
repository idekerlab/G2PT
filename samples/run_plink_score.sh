#!/usr/bin/env bash
# run_plink_score.sh
#
# Pipeline:
#   1. Generate synthetic train / val / test data (bed/bim/fam/pheno/cov)
#   2. Run PLINK linear GWAS on train set
#   3. Build a score file from GWAS results (SNPs with p < P_THRESH)
#   4. Compute polygenic scores on the test set with `plink --score`
#   5. Evaluate PRS vs. true phenotype (Pearson r²)
#
# Usage:
#   bash samples/run_plink_score.sh
# or to override defaults:
#   P_THRESH=1e-3 N_TRAIN=2000 bash samples/run_plink_score.sh

set -euo pipefail

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/cellar/users/i5lee/miniconda3/envs/G2PT_github/bin/python
PLINK=/cellar/users/i5lee/bin/plink
OUT_DIR="${SCRIPT_DIR}"
WORK_DIR="${SCRIPT_DIR}/plink_score_tmp"

# ── parameters ────────────────────────────────────────────────────────────────
SEED=${SEED:-42}
N_TRAIN=${N_TRAIN:-1000}
N_VAL=${N_VAL:-500}
N_TEST=${N_TEST:-300}
N_SNPS=${N_SNPS:-293}
P_THRESH=${P_THRESH:-5e-2}          # p-value threshold for SNP inclusion in score
PREFIX_TRAIN=synthetic_train
PREFIX_VAL=synthetic_val
PREFIX_TEST=synthetic_test

mkdir -p "${WORK_DIR}"

echo "============================================================"
echo " Step 1: Generate synthetic data"
echo "============================================================"
"${PYTHON}" "${SCRIPT_DIR}/generate_synthetic_data.py" \
    --seed      "${SEED}"    \
    --n_train   "${N_TRAIN}" \
    --n_val     "${N_VAL}"   \
    --n_test    "${N_TEST}"  \
    --n_snps    "${N_SNPS}"  \
    --out_dir   "${OUT_DIR}"

echo ""
echo "============================================================"
echo " Step 2: GWAS on train set (plink --linear)"
echo "============================================================"
# PLINK expects pheno without header when using --pheno; we strip it here.
# The pheno file has columns: FID  IID  PHENOTYPE
"${PLINK}" \
    --bfile   "${OUT_DIR}/${PREFIX_TRAIN}"       \
    --pheno   "${OUT_DIR}/${PREFIX_TRAIN}.pheno" \
    --pheno-name PHENOTYPE                       \
    --linear                                     \
    --covar   "${OUT_DIR}/${PREFIX_TRAIN}.cov"   \
    --covar-name SEX,AGE,AGE2,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10 \
    --allow-no-sex                               \
    --out     "${WORK_DIR}/gwas_train"           \
    2>&1 | tail -20

ASSOC_FILE="${WORK_DIR}/gwas_train.assoc.linear"
if [ ! -f "${ASSOC_FILE}" ]; then
    echo "ERROR: GWAS output not found at ${ASSOC_FILE}"
    exit 1
fi

echo ""
echo "Total GWAS results: $(tail -n +2 "${ASSOC_FILE}" | wc -l) rows"

echo ""
echo "============================================================"
echo " Step 3: Build score file (SNPs with p < ${P_THRESH}, ADD test only)"
echo "============================================================"
SCORE_FILE="${WORK_DIR}/score.txt"

# assoc.linear columns: CHR SNP BP A1 TEST NMISS BETA STAT P
# Keep only ADD rows (skip interaction / covariate rows), filter by p-value,
# output: SNP  A1  BETA
awk -v p="${P_THRESH}" '
    NR == 1 { next }                     # skip header
    $5 == "ADD" && $9+0 < p+0 {         # ADD test, p < threshold
        print $2, $4, $7                 # SNP  A1  BETA
    }
' "${ASSOC_FILE}" > "${SCORE_FILE}"

N_SCORE=$(wc -l < "${SCORE_FILE}")
echo "SNPs in score file: ${N_SCORE}  (p < ${P_THRESH})"

if [ "${N_SCORE}" -eq 0 ]; then
    echo "WARNING: No SNPs passed the p-value threshold. Relaxing to p < 0.5 …"
    P_THRESH=0.5
    awk -v p="${P_THRESH}" '
        NR == 1 { next }
        $5 == "ADD" && $9+0 < p+0 {
            print $2, $4, $7
        }
    ' "${ASSOC_FILE}" > "${SCORE_FILE}"
    N_SCORE=$(wc -l < "${SCORE_FILE}")
    echo "SNPs in score file after relaxing: ${N_SCORE}"
fi

echo "Top 5 SNPs in score file:"
head -5 "${SCORE_FILE}"

echo ""
echo "============================================================"
echo " Step 4: Compute PRS on test set (plink --score)"
echo "============================================================"
# score file columns: 1=SNP  2=A1(effect allele)  3=BETA
# --score-no-mean-imputation: use 0 for missing genotypes
# header flag not needed (no header in score file)
"${PLINK}" \
    --bfile  "${OUT_DIR}/${PREFIX_TEST}"  \
    --score  "${SCORE_FILE}" 1 2 3        \
    --out    "${WORK_DIR}/prs_test"       \
    --allow-no-sex                        \
    2>&1 | tail -10

PRS_FILE="${WORK_DIR}/prs_test.profile"
if [ ! -f "${PRS_FILE}" ]; then
    echo "ERROR: PRS output not found at ${PRS_FILE}"
    exit 1
fi

echo ""
echo "PRS file preview (first 5 rows):"
head -6 "${PRS_FILE}"

echo ""
echo "============================================================"
echo " Step 5: Evaluate PRS — Pearson r² vs. true phenotype"
echo "============================================================"
"${PYTHON}" - "${WORK_DIR}" "${OUT_DIR}/${PREFIX_TEST}.pheno" <<'PYEOF'
import sys
import pandas as pd
import numpy as np

work, pheno_path = sys.argv[1], sys.argv[2]

prs   = pd.read_csv(f"{work}/prs_test.profile", sep=r"\s+")
pheno = pd.read_csv(pheno_path, sep="\t")

merged = prs.merge(pheno, left_on=["FID","IID"], right_on=["FID","IID"])
r = np.corrcoef(merged["SCORE"], merged["PHENOTYPE"])[0, 1]
r2 = r ** 2

print(f"  Test samples with PRS : {len(merged)}")
print(f"  PRS mean ± SD         : {merged['SCORE'].mean():.4f} ± {merged['SCORE'].std():.4f}")
print(f"  Pearson r             : {r:.4f}")
print(f"  R²  (variance explained) : {r2:.4f}  ({r2*100:.1f}%)")
PYEOF

echo ""
echo "All outputs in: ${WORK_DIR}"
echo "  gwas_train.assoc.linear   — full GWAS summary statistics"
echo "  score.txt                 — SNP weights used for scoring"
echo "  prs_test.profile          — per-sample polygenic scores"
