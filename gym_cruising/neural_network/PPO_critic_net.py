import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_sizes = [256, 256]


class CriticHead(nn.Module):
    def __init__(self, token_dim=16):
        super(CriticHead, self).__init__()
        self.fl1 = nn.Linear(token_dim, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], 1)

    def forward(self, token):  # x: (B, U, D)
        x = token.mean(dim=1)  # (B, D) aggrega tutti gli UAV
        x = F.relu(self.fl1(x))
        x = F.relu(self.fl2(x))
        return self.fl3(x) # (B, 1)
