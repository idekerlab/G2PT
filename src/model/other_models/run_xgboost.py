import argparse
import os
import sys

# Avoid shadowing the third-party xgboost package with the local xgboost.py file.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

import pandas as pd
import numpy as np
from sgkit.io import plink
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit


def main():
    parser = argparse.ArgumentParser(description='Some beautiful description')
    parser.add_argument('--train-bfile', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--train-cov', type=str)
    parser.add_argument('--train-pheno', type=str)
    parser.add_argument('--val-bfile', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--val-cov', type=str)
    parser.add_argument('--val-pheno', type=str)
    parser.add_argument('--test-bfile', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--test-cov', type=str)
    parser.add_argument('--test-pheno', type=str)
    parser.add_argument('--phenotype-name', help='Name of the phenotype column in the phenotype file', type=str)
    parser.add_argument('--regression', action='store_true')

    parser.add_argument('--out', help='output csv')

    args = parser.parse_args()

    print("Reading Input File....")

    print("\t Reading", args.train_bfile, flush=True)
    train_plink_data = plink.read_plink(path=args.train_bfile)
    train_genotype = pd.DataFrame(train_plink_data.call_genotype.as_numpy().sum(axis=-1).T)
    train_genotype.index = train_plink_data.sample_id.values
    train_genotype.columns = train_plink_data.variant_id.values
    train_cov = pd.read_csv(args.train_cov, sep='\t')
    train_pheno = pd.read_csv(args.train_pheno, sep='\t')

    print("\t Reading", args.val_bfile, flush=True)
    val_plink_data = plink.read_plink(path=args.val_bfile)
    val_genotype = pd.DataFrame(val_plink_data.call_genotype.as_numpy().sum(axis=-1).T)
    val_genotype.index = val_plink_data.sample_id.values
    val_genotype.columns = val_plink_data.variant_id.values
    val_cov = pd.read_csv(args.val_cov, sep='\t')
    val_pheno = pd.read_csv(args.val_pheno, sep='\t')

    print("\t Reading", args.test_bfile, flush=True)
    test_plink_data = plink.read_plink(path=args.test_bfile)
    test_genotype = pd.DataFrame(test_plink_data.call_genotype.as_numpy().sum(axis=-1).T)
    test_genotype.index = test_plink_data.sample_id.values
    test_genotype.columns = test_plink_data.variant_id.values
    test_cov = pd.read_csv(args.test_cov, sep='\t')
    test_pheno = pd.read_csv(args.test_pheno, sep='\t')

    print("Formatting Input Files....")

    train_cov['FID'] = train_cov['FID'].astype(str)
    train_cov['IID'] = train_cov['IID'].astype(str)
    train_pheno['FID'] = train_pheno['FID'].astype(str)
    train_pheno['IID'] = train_pheno['IID'].astype(str)

    val_cov['FID'] = val_cov['FID'].astype(str)
    val_cov['IID'] = val_cov['IID'].astype(str)
    val_pheno['FID'] = val_pheno['FID'].astype(str)
    val_pheno['IID'] = val_pheno['IID'].astype(str)

    test_cov['FID'] = test_cov['FID'].astype(str)
    test_cov['IID'] = test_cov['IID'].astype(str)
    test_pheno['FID'] = test_pheno['FID'].astype(str)
    test_pheno['IID'] = test_pheno['IID'].astype(str)

    # Explicitly define cov_ids to prevent data leakage
    pcs = [f'PC{i}' for i in range(1, 11)]
    cov_ids = ['SEX', 'AGE'] + pcs
    print("Using covariates:", cov_ids)

    train_cov = train_cov.set_index('IID').loc[train_genotype.index]
    val_cov = val_cov.set_index('IID').loc[val_genotype.index]
    test_cov = test_cov.set_index('IID').loc[test_genotype.index]

    train_pheno = train_pheno.set_index('IID').loc[train_genotype.index]
    val_pheno = val_pheno.set_index('IID').loc[val_genotype.index]
    test_pheno = test_pheno.set_index('IID').loc[test_genotype.index]

    train_cov['PHENOTYPE'] = train_pheno[args.phenotype_name]
    val_cov['PHENOTYPE'] = val_pheno[args.phenotype_name]
    test_cov['PHENOTYPE'] = test_pheno[args.phenotype_name]

    def filter_valid_labels(genotype_df, cov_df):
        valid_labels = pd.to_numeric(cov_df['PHENOTYPE'], errors='coerce')
        valid_mask = valid_labels.notna() & np.isfinite(valid_labels)
        cov_df = cov_df.loc[valid_mask].copy()
        cov_df['PHENOTYPE'] = valid_labels.loc[valid_mask]
        genotype_df = genotype_df.loc[cov_df.index]
        return genotype_df, cov_df

    train_genotype, train_cov = filter_valid_labels(train_genotype, train_cov)
    val_genotype, val_cov = filter_valid_labels(val_genotype, val_cov)
    test_genotype, test_cov = filter_valid_labels(test_genotype, test_cov)

    print("Merging genotype and covariates", flush=True)
    train_genotype_cov_merged = pd.concat([train_genotype, train_cov], axis=1)
    val_genotype_cov_merged = pd.concat([val_genotype, val_cov], axis=1)
    test_genotype_cov_merged = pd.concat([test_genotype, test_cov], axis=1)

    feature_cols = train_plink_data.variant_id.values.tolist() + cov_ids

    train_val_merged = pd.concat([train_genotype_cov_merged, val_genotype_cov_merged])

    val_indices = [-1] * train_genotype_cov_merged.shape[0] + [0] * val_genotype_cov_merged.shape[0]
    pds = PredefinedSplit(test_fold=val_indices)

    print("Building and training XGBoost model...", flush=True)
    if args.regression:
        xgboost_model = XGBRegressor(objective='reg:squarederror', random_state=42)
        scoring = 'r2'
    else:
        xgboost_model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", random_state=42)
        scoring = 'roc_auc'

    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }

    grid_search = GridSearchCV(estimator=xgboost_model, param_grid=param_grid, cv=pds,
                               scoring=scoring, verbose=1, n_jobs=1)

    x_train, y_train = train_val_merged[feature_cols].values, train_val_merged['PHENOTYPE'].values
    print("Training model...", flush=True)
    grid_search.fit(x_train, y_train)

    print("Best parameters for model:", grid_search.best_params_)
    best_model = grid_search.best_estimator_

    print("\nEvaluating final model...", flush=True)
    x_val, y_val = val_genotype_cov_merged[feature_cols].values, val_genotype_cov_merged['PHENOTYPE'].values
    val_predicted = best_model.predict(x_val)
    val_score = r2_score(y_val, val_predicted)
    print(f"Final Validation R2: {val_score:.4f}", flush=True)

    x_test, y_test = test_genotype_cov_merged[feature_cols].values, test_genotype_cov_merged['PHENOTYPE'].values
    test_predicted = best_model.predict(x_test)
    test_score = r2_score(y_test, test_predicted)
    print(f"Final Test R2: {test_score:.4f}", flush=True)

    x_train_full = train_genotype_cov_merged[feature_cols].values
    train_predicted = best_model.predict(x_train_full)

    train_df = train_cov[cov_ids].copy()
    val_df = val_cov[cov_ids].copy()
    test_df = test_cov[cov_ids].copy()

    train_df['prediction'] = train_predicted
    val_df['prediction'] = val_predicted
    test_df['prediction'] = test_predicted

    whole_df = pd.concat([train_df, val_df, test_df])
    whole_df.to_csv(args.out)

    print("Saving to ... ", args.out, flush=True)


if __name__ == '__main__':
    main()
