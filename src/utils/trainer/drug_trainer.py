import gc
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.utils.data import move_to
import numpy as np
import copy
from transformers import get_linear_schedule_with_warmup


class DrugTrainer(object):

    def __init__(self, compound_model, compound_dataloader, device, lr, wd):
        self.device = device
        self.compound_model = compound_model.to(self.device)
        self.compound_dataloader = compound_dataloader
        self.optimizer = optim.AdamW(self.compound_model.parameters(), lr=lr, weight_decay=wd)
        self.loss_func = nn.MSELoss()


    def train(self, epochs):

        for epoch in range(epochs):
            dataloader_with_tqdm = tqdm(self.compound_dataloader)
            self.compound_model.train()
            mean_loss = 0
            for i, batch in enumerate(dataloader_with_tqdm):
                drug_feature, drug_mean_response = move_to(batch, self.device)
                drug_prediction = self.compound_model.predict(drug_feature.to(torch.float32))
                loss = self.loss_func(drug_prediction[:, 0], drug_mean_response.to(torch.float32))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                mean_loss = mean_loss + loss
                dataloader_with_tqdm.set_description("Train epoch: %d, Compound loss %.3f" % (epoch, mean_loss/(i+1)))
                del loss
                del drug_prediction
                del drug_feature, drug_mean_response, batch
            torch.cuda.empty_cache()
        return self.compound_model
