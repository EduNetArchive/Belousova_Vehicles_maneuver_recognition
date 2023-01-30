import torch
import torch.nn as nn

class FewLayerCNN(nn.Module):
    def __init__(self, 
                 conv_layer_nums, 
                 seq_len=8, 
                 class_nums=10, 
                 channels_num=1, 
                 kernel_size=3,
                 activation_func=nn.ReLU):
        super().__init__() 
        self.conv_layer_num = conv_layer_nums
        self.af1 = nn.Sigmoid()
        self.conv = nn.ModuleList()
        self.channels = channels_num
        lin_dim = seq_len
        for i in range(1, conv_layer_nums + 1):
            self.conv.append(nn.Conv1d(self.channels, 2 * self.channels, kernel_size))
            self.channels *= 2
            # l_out = (l_in + 2 * padding - dilation * (kernel_size - 1) - 1)/stride + 1
            lin_dim = lin_dim - (kernel_size - 1)
        lin_dim *= self.channels
        
        self.lin1 = nn.Linear(lin_dim, class_nums)
#         self.lin2 =  nn.Linear(64, class_nums)
        self.activation = activation_func()

    def forward(self, x):
        """
          Remember that first dimension is the batch dimension
          
        """
        x = torch.unsqueeze(x, 2)
        x = torch.transpose(x, 1, 2).float()
        for i in range(0, self.conv_layer_num):
            x = (self.conv[i])(x)
            x = self.af1(x)
        x = torch.flatten(x, 1)
#         x = self.lin1(x)
#         x = self.activation(x)
        scores = self.lin1(x)
        return scores
    
    def predict(self, x, device='cpu'):
        with torch.no_grad():
            outputs = self(x.to(device))
            _, predicted = torch.max(outputs.data.cpu(), 1)
        return predicted