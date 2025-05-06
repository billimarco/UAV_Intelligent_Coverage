import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_sizes = [256, 256]


class ActorHead(nn.Module):
    def __init__(self, token_dim=16):
        super(ActorHead, self).__init__()
        self.fl1 = nn.Linear(token_dim, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], 2)
        self.log_std = nn.Parameter(torch.zeros(2))  # Deviazione standard appresa (globale)

    def forward(self, token):
        x = F.relu(self.fl1(token))
        x = F.relu(self.fl2(x))
        mean = self.fl3(x)
        std = torch.exp(self.log_std).expand_as(mean)  # (B, U, 2)
        return mean, std
