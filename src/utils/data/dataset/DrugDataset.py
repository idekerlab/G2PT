from torch.utils.data.dataset import Dataset


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
