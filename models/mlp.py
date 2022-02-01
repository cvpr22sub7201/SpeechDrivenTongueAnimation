import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import SelfAttentionModule


__all__ = [
    'MLP', 'MLPInputAtt'
]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 dropouts: list):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.n_layers = len(hidden_dims)
        self.dropouts = dropouts
        all_dims = [input_dim] + hidden_dims + [output_dim]

        self.net = nn.Sequential()
        for i in range(1, len(all_dims)):
            self.net.add_module(f'fc_{i}',
                                nn.Linear(all_dims[i-1], all_dims[i]))

            if i < len(all_dims)-1:
                self.net.add_module(f'relu_{i}', nn.ReLU())
                self.net.add_module(f'dropout_{i}', nn.Dropout(dropouts[i-1]))

        self.params = dict(input_dim=input_dim,
                           hidden_dims=hidden_dims,
                           output_dim=output_dim,
                           dropouts=dropouts)

    def forward(self, x):
        return self.net(x)

    def save_model(self, save_path):
        torch.save({'model_params': self.params,
                    'model_state_dict': self.state_dict()
                    }, save_path)

    @staticmethod
    def load_model(checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=str(device))
        model = MLP(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model


class MLPInputAtt(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 dropouts: list, att_dim: int, proj_dim: int):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.att_dim = att_dim  # feature dim
        self.proj_dim = proj_dim  # feature projection dim
        self.dropouts = dropouts
        self.n_layers = len(hidden_dims)
        self.seq_len = input_dim // att_dim

        self.att_module = SelfAttentionModule(input_dim=att_dim,
                                              proj_dim=proj_dim)

        self.mlp = MLP(input_dim=input_dim,
                       hidden_dims=hidden_dims,
                       output_dim=output_dim,
                       dropouts=dropouts)

        self.params = dict(input_dim=input_dim,
                           hidden_dims=hidden_dims,
                           output_dim=output_dim,
                           dropouts=dropouts,
                           att_dim=att_dim,
                           proj_dim=proj_dim)

    def forward(self, x):
        x = self.att_module(x.transpose(1, 2))
        x = torch.flatten(x.transpose(1, 2), 1, 2)

        return self.mlp(x)

    def save_model(self, save_path):
        torch.save({'model_params': self.params,
                    'model_state_dict': self.state_dict()
                    }, save_path)

    @staticmethod
    def load_model(checkpoint_path, device):
        checkpoint = torch.load(checkpoint_path, map_location=str(device))
        model = MLPInputAtt(**checkpoint['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        return model
