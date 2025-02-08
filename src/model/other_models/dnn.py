import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
from sgkit.io import plink
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, TensorDataset
from sklearn.metrics import r2_score
from tqdm import tqdm


def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = move_to(v, device)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(move_to(v, device))
        return res
    else:
        return obj.to(device)


def predict(model, data_loader, device):
    model.eval()
    prediction_result = []
    with torch.no_grad():
        val_loss = 0
        for data, target in data_loader:
            data = move_to(data, device)
            target = move_to(target, device)
            predicted_y = model(data)
            prediction_result.append(predicted_y.detach().cpu().numpy())
    return np.concatenate(prediction_result, axis=0)[:, 0]


class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleLinearModel, self).__init__()
        self.linear_0 = nn.Linear(input_dim, hidden_dim)
        self.linear_1 = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.linear_0(x))
        prediction = self.linear_1(x)
        return prediction


def main():
    parser = argparse.ArgumentParser(description='Some beautiful description')
    parser.add_argument('--train-bfile', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--train-cov', type=str)
    parser.add_argument('--val-bfile', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--val-cov', type=str)
    parser.add_argument('--test-bfile', help='Ontology file used to guide the neural network', type=str)
    parser.add_argument('--test-cov', type=str)
    parser.add_argument('--cuda', type=int, default=None)
    parser.add_argument('--epoch', type=int, default=100)

    parser.add_argument('--out', help='output csv')

    args = parser.parse_args()

    print("Reading Input File....")



    print("\t Reading",args.train_bfile, flush=True)
    train_plink_data = plink.read_plink(path=args.train_bfile)
    train_genotype = pd.DataFrame(train_plink_data.call_genotype.as_numpy().sum(axis=-1).T)
    train_genotype.index = train_plink_data.sample_id.values
    train_genotype.columns = train_plink_data.variant_id.values
    train_cov = pd.read_csv(args.train_cov, sep='\t')
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


    if args.cuda is not None:
        device = torch.device('cuda:%d'%args.cuda)
    else:
        device = torch.cpu()
    train_genotype_cov_merged = pd.concat([train_genotype, train_cov], axis=1)
    val_genotype_cov_merged = pd.concat([val_genotype, val_cov], axis=1)
    test_genotype_cov_merged = pd.concat([test_genotype, test_cov], axis=1)

    feature_cols = train_plink_data.variant_id.values.tolist() + cov_ids

    x_train, y_train = train_genotype_cov_merged[feature_cols].values, train_genotype_cov_merged['PHENOTYPE'].values
    x_train, y_train = torch.tensor(x_train, dtype=torch.float), torch.tensor(y_train, dtype=torch.float)

    x_val, y_val = val_genotype_cov_merged[feature_cols].values, val_genotype_cov_merged['PHENOTYPE'].values
    x_val, y_val = torch.tensor(x_val, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    best_params = None
    best_val_loss = float('inf')

    param_grid = {
        'hidden_dim': [32, 64, 128, 256],
        'lr': [0.001, 0.001, 0.01, 0.1],
        'num_epochs': [10, 20, 30]
    }

    for hidden_dim in param_grid['hidden_dim']:
        for lr in param_grid['lr']:
            for num_epochs in param_grid['num_epochs']:
                print(f"Training with hidden_dim={hidden_dim}, lr={lr}, num_epochs={num_epochs}")

                # Initialize model, loss function, and optimizer
                model = SimpleLinearModel(input_dim=len(feature_cols), hidden_dim=hidden_dim)
                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                model = model.to(device)
                # Training loop
                for epoch in range(num_epochs):
                    model.train()
                    for X_batch, y_batch in tqdm(train_loader):
                        X_batch = move_to(X_batch, device)
                        y_batch = move_to(y_batch, device)
                        optimizer.zero_grad()
                        predictions = model(X_batch).squeeze()
                        loss = criterion(predictions, y_batch)
                        loss.backward()
                        optimizer.step()

                # Validation loop
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = move_to(X_batch, device)
                        y_batch = move_to(y_batch, device)
                        predictions = model(X_batch).squeeze()
                        val_loss = criterion(predictions, y_batch)
                        val_losses.append(val_loss.item())

                avg_val_loss = np.mean(val_losses)
                print(f"Validation Loss: {avg_val_loss}")

                # Update best parameters if current model is better
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_params = {'hidden_dim': hidden_dim, 'lr': lr, 'num_epochs': num_epochs}

    best_hidden_dim = best_params['hidden_dim']
    best_lr = best_params['lr']
    best_num_epochs = best_params['num_epochs']

    model = SimpleLinearModel(input_dim=len(feature_cols), hidden_dim=best_hidden_dim)
    model = model.to(device)
    # Define the loss function and optimizer with the best learning rate
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr)

    # Train the model using the best parameters
    for epoch in range(best_num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = move_to(X_batch, device)
            y_batch = move_to(y_batch, device)
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()

    x_test, y_test = test_genotype_cov_merged[feature_cols].values, test_genotype_cov_merged['PHENOTYPE'].values
    x_test, y_test = torch.tensor(x_test, dtype=torch.float), torch.tensor(y_test, dtype=torch.float)
    test_dataset = TensorDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

    train_predicted = predict(model, train_loader, device)
    val_predicted = predict(model, val_loader, device)
    test_predicted = predict(model, test_loader, device)

    val_mse = r2_score(y_val, val_predicted)
    print("Validation MSE with best parameters:", val_mse, flush=True)

    test_mse = r2_score(y_test, test_predicted)
    print("Test MSE with best parameters:", test_mse, flush=True)


    train_df = train_cov[cov_ids]
    val_df = val_cov[cov_ids]
    test_df = test_cov[cov_ids]

    train_df['prediction'] = train_predicted
    val_df['prediction'] = val_predicted
    test_df['prediction'] = test_predicted

    whole_df = pd.concat([train_df, val_df, test_df])
    whole_df.to_csv(args.out)

    print("Saving to ... ", args.out)


if __name__ == '__main__':
    main()
