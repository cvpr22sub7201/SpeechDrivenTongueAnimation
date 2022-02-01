import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'LSTM'
]

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout=0.2, bidirectional=False, embedding_dim=0):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.bidir_factor = 2 if bidirectional else 1
        
        self.feat_fc = nn.Linear(input_dim, embedding_dim) if embedding_dim > 0 else None
        lstm_input_dim = embedding_dim if embedding_dim > 0 else input_dim
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim * self.bidir_factor, output_dim)
        self.params = dict(input_dim=self.input_dim,
                            hidden_dim=self.hidden_dim, 
                            output_dim=self.output_dim,
                            embedding_dim=self.embedding_dim,
                            n_layers=self.n_layers,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
        
    def forward(self, x, hc):
        if self.feat_fc is not None:
            x = self.feat_fc(x)
        out, hc = self.lstm(x, hc)
        out = self.fc(self.relu(out))
        return out, hc

    def infer(self, x):
        hc0 = self.init_hidden(x.shape[0])
        out, _ = self.forward(x, hc0)
        return out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers * self.bidir_factor, batch_size, self.hidden_dim).zero_()
        context = weight.new(self.n_layers * self.bidir_factor, batch_size, self.hidden_dim).zero_()
        return hidden, context
    
    def save_model(self, save_path):
        torch.save({'model_params': self.params,
                    'model_state_dict': self.state_dict()
                }, save_path)
    
    @staticmethod
    def load_model(checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=str(device))
        model = LSTM(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        return model