import torch.nn as nn

class ECFPCompoundModel(nn.Module):

    def __init__(self, n_bits:int =1024, hidden_layers:tuple = (256, ), dropout=0.2):
        super().__init__()
        self.hidden_layers = [n_bits] + list(hidden_layers)
        self.layers = nn.ModuleList([nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]) for i in range(len(self.hidden_layers)-1)])
        self.activation = nn.GELU()
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden_layers[i+1]) for i in range(len(self.hidden_layers)-1)])
        self.dropout = nn.Dropout(dropout)
        self.compound_prediction = nn.Linear(hidden_layers[-1], 1)

    def forward(self, fps):
        x = fps
        for layer, norm in zip(self.layers, self.norms):
            x = layer(x)
            x = self.activation(x)
            x = norm(x)
            x = self.dropout(x)

        return x

    def predict(self, fps):
        x = self.forward(fps)
        compound_prediction = self.compound_prediction(x)
        return compound_prediction

class ChemBERTaCompoundModel(nn.Module):

    def __init__(self, chemberta_model, dropout=0.2):
        super().__init__()
        self.model = chemberta_model
        self.hidden_layers = [self.model.config.hidden_size]
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids):
        outputs = self.model(input_ids, output_hidden_states=True)
        return self.dropout(outputs.hidden_states[-1][:, 0, :])



