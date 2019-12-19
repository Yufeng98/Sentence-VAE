import torch
import torch.nn as nn
import torch.nn.functional as F




class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1) #self.softmax(x)

