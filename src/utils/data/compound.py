from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data.dataset import Dataset
from scipy.stats import zscore, skewnorm
import numpy as np
import pandas as pd
import torch
import pickle


def skew_normal_mode(data):
    a, loc, scale = skewnorm.fit(data)
    m, v, s, k = skewnorm.stats(a, loc=loc, scale=scale, moments='mvsk')
    delta = a / np.sqrt(1+a**2)
    u_z = np.sqrt(2/np.pi)*delta
    sigma_z = np.sqrt(1-u_z**2)
    m_z = u_z - s*sigma_z/2 + np.sign(a)*np.exp(-2*np.pi/np.abs(a))/2
    mode = loc + scale*m_z
    return mode, v, s, k


class CompoundEncoder(object):

    def __init__(self, feature="Morgan", radius=2, n_bits=2048, tokenizer=None, dataset=[], out=None, **kwargs):
        self.feature = feature
        self.radius = radius
        self.n_bits = n_bits
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.out = out

        if isinstance(self.dataset, pd.DataFrame):
            self.drugs = list(self.dataset.groupby(1).groups)
            self.smiles2ind = {k: v for v, k in enumerate(self.drugs)}

            dir = self.out.split("/")[0]
            with open(dir+'/smiles2ind.pkl', 'wb') as handle:
                pickle.dump(self.smiles2ind, handle)
            
    def get_type(self):
        return self.feature

    def num_drugs(self):
        return len(self.drugs)

    def mol_to_morgan(self, mol):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        return fp

    def encode(self, smiles):
        if self.feature =='Embedding':
            return(torch.tensor(self.smiles2ind[smiles]))
        elif self.feature=='Morgan':
            mol = Chem.MolFromSmiles(smiles)
            morgan_feature = torch.tensor(self.mol_to_morgan(mol), dtype=torch.float32)
            return morgan_feature
        elif self.feature=='SMILES':
            return smiles

    def collate(self, encoded_list):
        if self.feature == 'Embedding':
            return torch.stack(encoded_list)
        if self.feature=='Morgan':
            return torch.stack(encoded_list)
        elif self.feature=="SMILES":
            return self.tokenizer(encoded_list, padding=True, return_tensors='pt')["input_ids"]

class DrugDataset(Dataset):

    def __init__(self, drug_response, compound_encoder):
        self.drug_response_df = drug_response
        self.drug_grouped = self.drug_response_df.groupby(1)
        self.drugs = list(self.drug_grouped.groups)
        self.compound_encoder = compound_encoder
        #self.drug_response_mean_dict = {drug: skew_normal_mode(self.drug_grouped.get_group(drug)[2])[0] for drug in
        #                                self.drug_response_df[1].unique()}
        self.drug_response_mean_dict = {drug: self.drug_grouped.get_group(drug)[2].mean() for drug in
                                        self.drug_response_df[1].unique()}

    def __len__(self):
        return len(self.drugs)

    def __getitem__(self, index):
        drug = self.drugs[index]
        return self.compound_encoder.encode(drug), self.drug_response_mean_dict[drug]
