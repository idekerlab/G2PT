import argparse
import pandas as pd
import numpy as np
from sgkit.io import plink

from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, KFold, PredefinedSplit



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

    parser.add_argument('--out', help='output csv')

    args = parser.parse_args()

    print("Reading Input File....")



    print("\t Reading",args.train_bfile, flush=True)
    train_plink_data = plink.read_plink(path=args.train_bfile)
    train_genotype = pd.DataFrame(train_plink_data.call_genotype.as_numpy().sum(axis=-1).T)
    train_genotype.index = train_plink_data.sample_id.values
    train_genotype.columns = train_plink_data.variant_id.values
    train_cov = pd.read_csv(args.train_cov, sep='\t')
    train_pheno = pd.read_csv(args.train_pheno, sep='\t')

    print(train_genotype)
    print(train_cov)
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

    print("Merging genotype and covariates", flush=True)
    #train_genotype_cov_merged = train_genotype.merge(train_cov.set_index("IID"), left_index=True, right_index=True)
    #val_genotype_cov_merged = val_genotype.merge(val_cov.set_index("IID"), left_index=True, right_index=True)
    #test_genotype_cov_merged = test_genotype.merge(test_cov.set_index("IID"), left_index=True, right_index=True)
    train_genotype_cov_merged = pd.concat([train_genotype, train_cov], axis=1)
    val_genotype_cov_merged = pd.concat([val_genotype, val_cov], axis=1)
    test_genotype_cov_merged = pd.concat([test_genotype, test_cov], axis=1)

    snp_cols = train_plink_data.variant_id.values.tolist()

    train_val_genotype_merged = pd.concat([train_genotype, val_genotype])
    train_val_phenotype_merged = pd.concat([train_cov['PHENOTYPE'], val_cov['PHENOTYPE']])

    val_indices = [-1] * train_genotype.shape[0] + [0] * val_genotype.shape[0]

    pds = PredefinedSplit(test_fold=val_indices)

    print("STEP 1: Building and training genetics-only model...", flush=True)
    elastic_net = ElasticNet()
    param_grid = {
        'alpha': [0.1, 1, 10, 0.01],
        'l1_ratio': np.arange(0.40, 1.00, 0.10),
        'tol': [0.0001, 0.001]
    }

    # Perform grid search with cross-validation on genetics data only
    grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=pds, scoring='r2',
                               verbose=1, n_jobs=1)

    x_train_step1, y_train_step1 = train_val_genotype_merged[snp_cols].values, train_val_phenotype_merged.values
    print("Training genetics-only model...", flush=True)
    grid_search.fit(x_train_step1, y_train_step1)

    print("Best parameters for genetics-only model:", grid_search.best_params_)
    best_genetic_model = grid_search.best_estimator_

    # Generate genetic predictions (step 1 output)
    print("Generating predictions from genetics-only model...", flush=True)
    train_genetic_pred = best_genetic_model.predict(train_genotype[snp_cols].values)
    val_genetic_pred = best_genetic_model.predict(val_genotype[snp_cols].values)
    test_genetic_pred = best_genetic_model.predict(test_genotype[snp_cols].values)

    # --- STEP 2: Train model with covariates and genetic prediction ---
    print("\nSTEP 2: Building and training combined model...", flush=True)

    # Create new dataframes for step 2
    train_df_step2 = train_cov[cov_ids].copy()
    train_df_step2['genetic_prediction'] = train_genetic_pred
    
    val_df_step2 = val_cov[cov_ids].copy()
    val_df_step2['genetic_prediction'] = val_genetic_pred

    test_df_step2 = test_cov[cov_ids].copy()
    test_df_step2['genetic_prediction'] = test_genetic_pred

    # Combine train and validation sets for final model training
    train_val_df_step2 = pd.concat([train_df_step2, val_df_step2])
    
    feature_cols_step2 = cov_ids + ['genetic_prediction']
    
    x_train_step2 = train_val_df_step2[feature_cols_step2].values
    y_train_step2 = train_val_phenotype_merged.values

    # Fit a simple linear regression model
    final_model = LinearRegression()
    final_model.fit(x_train_step2, y_train_step2)

    # --- Evaluation ---
    print("\nEvaluating final model...", flush=True)
    x_val_step2, y_val_step2 = val_df_step2[feature_cols_step2].values, val_cov['PHENOTYPE'].values
    val_predicted = final_model.predict(x_val_step2)
    val_r2 = r2_score(y_val_step2, val_predicted)
    print("Final Validation R2 with best parameters:", val_r2, flush=True)

    x_test_step2, y_test_step2 = test_df_step2[feature_cols_step2].values, test_cov['PHENOTYPE'].values
    test_predicted = final_model.predict(x_test_step2)
    test_r2 = r2_score(y_test_step2, test_predicted)
    print("Final Test R2 with best parameters:", test_r2, flush=True)

    # Generate final predictions for all sets
    x_train_full_step2 = train_df_step2[feature_cols_step2].values
    train_predicted = final_model.predict(x_train_full_step2)

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
