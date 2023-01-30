import torch
import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, 
                 seq_len=8, 
                 class_nums=10, 
                 channels_num=1, 
                 activation_func=nn.ReLU,
                 n_layers=3, 
                 hid_layer_dim=64):
        super().__init__()

        self.n_layers = n_layers
        self.activation = activation_func()
        if n_layers == 1:
            layers = [nn.Linear(seq_len * channels_num, class_nums), self.activation]
        elif n_layers == 2:
            layers = [nn.Linear(seq_len * channels_num, hid_layer_dim), self.activation,
                     nn.Linear(hid_layer_dim, class_nums)]
        elif n_layers > 2:
            layers = [nn.Linear(seq_len * channels_num, hid_layer_dim)]
            layers.append(self.activation)
            for _ in range(0, n_layers - 2):
                layers.append(nn.Linear(hid_layer_dim, hid_layer_dim))
                layers.append(self.activation)
            layers.append(nn.Linear(hid_layer_dim, class_nums))
        elif n_layers < 1:
            print("Invalid layers quantity")
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        scores = self.layers(x.float())
        return scores
    
    def predict(self, x, device='cpu'):
        with torch.no_grad():
            outputs = self(x.to(device))
            _, predicted = torch.max(outputs.data.cpu(), 1)
        return predicted