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
    parser.add_argument('--val-bfile', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--val-cov', type=str)
    parser.add_argument('--test-bfile', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--test-cov', type=str)

    parser.add_argument('--out', help='output csv')

    args = parser.parse_args()

    print("Reading Input File....")



    print("\t Reading",args.train_bfile, flush=True)
    train_plink_data = plink.read_plink(path=args.train_bfile)
    train_genotype = pd.DataFrame(train_plink_data.call_genotype.as_numpy().sum(axis=-1).T)
    train_genotype.index = train_plink_data.sample_id.values
    train_genotype.columns = train_plink_data.variant_id.values
    train_cov = pd.read_csv(args.train_cov, sep='\t')
    print(train_genotype)
    print(train_cov)
    print("\t Reading", args.val_bfile, flush=True)
    val_plink_data = plink.read_plink(path=args.val_bfile)
    val_genotype = pd.DataFrame(val_plink_data.call_genotype.as_numpy().sum(axis=-1).T)
    val_genotype.index = val_plink_data.sample_id.values
    val_genotype.columns = val_plink_data.variant_id.values
    val_cov = pd.read_csv(args.val_cov, sep='\t')

    print("\t Reading", args.test_bfile, flush=True)
    test_plink_data = plink.read_plink(path=args.test_bfile)
    test_genotype = pd.DataFrame(test_plink_data.call_genotype.as_numpy().sum(axis=-1).T)
    test_genotype.index = test_plink_data.sample_id.values
    test_genotype.columns = test_plink_data.variant_id.values
    test_cov = pd.read_csv(args.test_cov, sep='\t')
    print("Formatting Input Files....")

    train_cov['FID'] = train_cov['FID'].astype(str)
    train_cov['IID'] = train_cov['IID'].astype(str)

    val_cov['FID'] = val_cov['FID'].astype(str)
    val_cov['IID'] = val_cov['IID'].astype(str)

    test_cov['FID'] = test_cov['FID'].astype(str)
    test_cov['IID'] = test_cov['IID'].astype(str)

    cov_ids = [cov for cov in train_cov.columns[2:] if cov != 'PHENOTYPE']

    train_cov = train_cov.set_index('IID').loc[train_genotype.index]
    val_cov = val_cov.set_index('IID').loc[val_genotype.index]
    test_cov = test_cov.set_index('IID').loc[test_genotype.index]

    print("Merging genotype and covariates", flush=True)
    #train_genotype_cov_merged = train_genotype.merge(train_cov.set_index("IID"), left_index=True, right_index=True)
    #val_genotype_cov_merged = val_genotype.merge(val_cov.set_index("IID"), left_index=True, right_index=True)
    #test_genotype_cov_merged = test_genotype.merge(test_cov.set_index("IID"), left_index=True, right_index=True)
    train_genotype_cov_merged = pd.concat([train_genotype, train_cov], axis=1)
    val_genotype_cov_merged = pd.concat([val_genotype, val_cov], axis=1)
    test_genotype_cov_merged = pd.concat([test_genotype, test_cov], axis=1)

    feature_cols = train_plink_data.variant_id.values.tolist() + cov_ids

    train_val_merged = pd.concat([train_genotype_cov_merged, val_genotype_cov_merged])

    val_indices = [-1] * train_genotype_cov_merged.shape[0] + [0] * val_genotype_cov_merged.shape[0]

    pds = PredefinedSplit(test_fold=val_indices)

    print("Building Model..", flush=True)
    elastic_net = ElasticNet()
    param_grid = {
                    'alpha'     : [0.1,1,10,0.01],
                'l1_ratio'  :  np.arange(0.40,1.00,0.10),
                'tol'       : [0.0001,0.001]
    }

# Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=elastic_net, param_grid=param_grid, cv=pds, scoring='neg_mean_squared_error',
                               verbose=1, n_jobs=-1)

    x_train, y_train = train_val_merged[feature_cols].values, train_val_merged['PHENOTYPE'].values
    print("Training Model..", flush=True)
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    bst = grid_search.best_estimator_

    x_val, y_val = val_genotype_cov_merged[feature_cols].values, val_genotype_cov_merged['PHENOTYPE'].values

    val_predicted = bst.predict(x_val)
    val_mse = r2_score(y_val, val_predicted)
    print("Validation MSE with best parameters:", val_mse, flush=True)


    print("Testing Model..")
    x_test, y_test = test_genotype_cov_merged[feature_cols].values, test_genotype_cov_merged['PHENOTYPE'].values
    test_predicted = bst.predict(x_test)
    test_mse = r2_score(y_test, test_predicted)
    print("Test MSE with best parameters:", test_mse, flush=True)

    x_train, y_train = train_genotype_cov_merged[feature_cols].values, train_genotype_cov_merged['PHENOTYPE'].values
    train_predicted = bst.predict(x_train)

    train_df = train_cov[cov_ids]
    val_df = val_cov[cov_ids]
    test_df = test_cov[cov_ids]

    train_df['prediction'] = train_predicted
    val_df['prediction'] = val_predicted
    test_df['prediction'] = test_predicted

    whole_df = pd.concat([train_df, val_df, test_df])
    whole_df.to_csv(args.out)

    print("Saving to ... ", args.out, flush=True)


if __name__ == '__main__':
    main()
